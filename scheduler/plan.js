// The planner: which devices run the model, in what order, holding which layers.
//
// Five strategies behind one interface, so the room can switch between them live and
// the comparison in the pitch is a measurement rather than an argument:
//
//   solo      everything on the fastest capable device        (the honest ceiling)
//   even      equal split, join order                          (naive baseline)
//   memory    proportional to memory, sorted by memory desc    (what exo does)
//   compute   proportional to measured speed, join order       (network-blind)
//   optimal   subset + chain order + layer cut + host, solved exactly
//
// `memory` is a faithful reproduction of exo's documented ring memory-weighted
// partitioning, not a strawman: it is the thing to beat, so it is implemented
// properly and given the same feasibility repair as everything else.
//
// The optimal solver decomposes cleanly:
//
//   - The chain is a CYCLE (host -> workers -> host), and a cycle's cost does not
//     depend on where you start. So the tour is solved once per subset and reused
//     for every host choice within it.
//   - With uniform per-layer cost -- true for every dense model, including this one --
//     the layer allocation depends only on WHICH devices are present, not their
//     order. So ordering and allocation separate, and each is solved exactly.
//
// Exact, with no solver dependency: Held-Karp for the tour (O(2^n n^2)) and a DP over
// cut points for the layers (O(n L^2)). prima.cpp's Halda needs an ILP solver and
// reports 10-12 ms; this runs in single-digit milliseconds for a room of six.

import { modelSpec, layerCap, canHost, predict, hostOverheadMs, hopMs, memoryFor } from "./cost.js";

export const STRATEGIES = ["solo", "even", "memory", "compute", "optimal"];

// Rooms bigger than this fall back to a heuristic tour. 2^12 subsets x Held-Karp is
// still milliseconds, but the growth is exponential and a hung planner during a demo
// is worse than a slightly suboptimal chain.
const EXACT_LIMIT = 12;

// ---------------------------------------------------------------- allocation

// Exact contiguous layer allocation over an ordered chain.
//
// DP[i][l] = cheapest way to cover the first l layers with the first i devices.
// Every device in the chain must take at least one layer: a device holding zero pays
// its hops for no work, which is strictly worse than not being in the chain at all.
// Dropping it is the subset search's job, not this function's.
//
// `weights` lets a model charge different costs for different layers (a hybrid model
// with periodic full-attention blocks, say). For a uniform model every weight is 1
// and the result is provably order-independent -- which the tests assert.
//
// NOT WIRED UP. No production caller supplies `weights`: planOptimal calls
// `allocate(spec, chain, spec.layers)` with it defaulted to null, nothing computes
// per-layer costs, and `layerCap()` still divides a memory budget by one uniform
// `layerBytes`. This parameter is groundwork, not hybrid support.
//
// Supporting a hybrid model (Qwen 3.8's Gated-DeltaNet blocks interleaved with
// full-attention ones) needs four things, and this is one:
//   1. a per-layer cost source -- the model spec must describe each layer's type
//   2. the profiler measuring each type, not one `msPerLayer` per device
//   3. `layerCap`/`memoryFor` taking per-layer bytes instead of a single figure
//   4. planOptimal threading the resulting weights through to here
// Do not read "the DP accepts weights" as "the scheduler handles hybrid models".

// Tie-break weight. With a linear objective, equally fast devices make [1,1,28] and
// [10,10,10] cost exactly the same, and a plain DP will happily return the first.
// They are not equally good: the lopsided one puts 28 layers of memory on one device
// and collapses if that device turns out slightly slower than measured. Adding a
// small penalty on the sum of squared loads makes the DP prefer balance among ties.
//
// The weight has to be small enough that it can never outvote a real difference in
// predicted time. Worst-case penalty spread is ~L^2 = 900, so at 1e-6 the largest
// influence is 9e-4 ms, far below any difference worth acting on.
const BALANCE_EPS = 1e-6;

export function allocate(spec, chain, L, weights = null) {
  const n = chain.length;
  const caps = chain.map((d, i) => layerCap(spec, d, i === 0));
  if (caps.reduce((a, b) => a + b, 0) < L) return null;      // cannot hold the model
  if (n > L) return null;                                    // more devices than layers

  const W = new Float64Array(L + 1);
  for (let l = 0; l < L; l++) W[l + 1] = W[l] + (weights ? weights[l] : 1);

  const INF = Infinity;
  const dp = Array.from({ length: n + 1 }, () => new Float64Array(L + 1).fill(INF));
  const take = Array.from({ length: n + 1 }, () => new Int32Array(L + 1).fill(-1));
  dp[0][0] = 0;

  for (let i = 0; i < n; i++) {
    const ms = chain[i].msPerLayer;
    const cap = caps[i];
    for (let l = 0; l <= L; l++) {
      const base = dp[i][l];
      if (base === INF) continue;
      const most = Math.min(cap, L - l);
      for (let c = 1; c <= most; c++) {
        // leave at least one layer for each device still to come
        if (L - (l + c) < n - i - 1) break;
        const cost = base + ms * (W[l + c] - W[l]) + BALANCE_EPS * c * c;
        if (cost < dp[i + 1][l + c]) { dp[i + 1][l + c] = cost; take[i + 1][l + c] = c; }
      }
    }
  }
  if (dp[n][L] === INF) return null;

  const counts = new Array(n);
  for (let i = n, l = L; i > 0; i--) { counts[i - 1] = take[i][l]; l -= take[i][l]; }

  // Report the true predicted cost, not the tie-broken one the search minimised.
  let costMs = 0;
  for (let i = 0, l = 0; i < n; i++) { costMs += chain[i].msPerLayer * (W[l + counts[i]] - W[l]); l += counts[i]; }
  return { counts, costMs, caps };
}

// ---------------------------------------------------------------- tour

// Held-Karp: the exact minimum-cost Hamiltonian cycle. Returns the visiting order.
// Cycle cost is independent of the starting node, so the caller rotates the result to
// put whichever device it elected host at the front.
export function bestTour(ids, dist) {
  const n = ids.length;
  if (n <= 1) return { cost: 0, order: ids.slice() };
  if (n === 2) return { cost: dist(ids[0], ids[1]) + dist(ids[1], ids[0]), order: ids.slice() };

  const full = (1 << n) - 1;
  const INF = Infinity;
  const dp = new Float64Array((full + 1) * n).fill(INF);
  const parent = new Int32Array((full + 1) * n).fill(-1);
  const at = (mask, j) => mask * n + j;

  dp[at(1, 0)] = 0;
  for (let mask = 1; mask <= full; mask++) {
    if (!(mask & 1)) continue;                       // every path starts at node 0
    for (let j = 0; j < n; j++) {
      const cur = dp[at(mask, j)];
      if (cur === INF || !(mask & (1 << j))) continue;
      for (let k = 0; k < n; k++) {
        if (mask & (1 << k)) continue;
        const nm = mask | (1 << k);
        const cost = cur + dist(ids[j], ids[k]);
        if (cost < dp[at(nm, k)]) { dp[at(nm, k)] = cost; parent[at(nm, k)] = j; }
      }
    }
  }

  let best = INF, endAt = -1;
  for (let j = 1; j < n; j++) {
    const c = dp[at(full, j)] + dist(ids[j], ids[0]);
    if (c < best) { best = c; endAt = j; }
  }
  if (endAt < 0) return { cost: INF, order: ids.slice() };

  const order = [];
  for (let mask = full, j = endAt; j >= 0; ) {
    order.push(ids[j]);
    const p = parent[at(mask, j)];
    mask ^= 1 << j;
    j = p;
  }
  order.reverse();
  return { cost: best, order };
}

// Nearest-neighbour then 2-opt, for rooms too large to solve exactly.
function heuristicTour(ids, dist) {
  const left = ids.slice(1);
  const order = [ids[0]];
  while (left.length) {
    let bi = 0, bd = Infinity;
    for (let i = 0; i < left.length; i++) {
      const d = dist(order[order.length - 1], left[i]);
      if (d < bd) { bd = d; bi = i; }
    }
    order.push(left.splice(bi, 1)[0]);
  }
  const cycle = (o) => o.reduce((s, id, i) => s + dist(id, o[(i + 1) % o.length]), 0);
  let best = cycle(order), improved = true;
  while (improved) {
    improved = false;
    for (let i = 1; i < order.length - 1; i++) {
      for (let j = i + 1; j < order.length; j++) {
        const cand = order.slice(0, i).concat(order.slice(i, j + 1).reverse(), order.slice(j + 1));
        const c = cycle(cand);
        if (c < best - 1e-9) { order.splice(0, order.length, ...cand); best = c; improved = true; }
      }
    }
  }
  return { cost: best, order };
}

// ---------------------------------------------------------------- helpers

const rangesFrom = (counts) => {
  const out = [];
  let at = 0;
  for (const c of counts) { out.push([at, at + c]); at += c; }
  return out;
};

// Largest-remainder apportionment, then repaired against real memory caps. Used by the
// baselines so they produce plans that actually run -- comparing against a strategy
// that crashes would prove nothing.
function proportional(spec, chain, L, weightOf) {
  const n = chain.length;
  const caps = chain.map((d, i) => layerCap(spec, d, i === 0));
  if (caps.reduce((a, b) => a + b, 0) < L) return null;

  const w = chain.map(weightOf);
  const tot = w.reduce((a, b) => a + b, 0) || 1;
  const exact = w.map((x) => (L * x) / tot);
  const counts = exact.map(Math.floor);
  const order = exact.map((e, i) => ({ i, frac: e - counts[i] })).sort((a, b) => b.frac - a.frac);
  let rem = L - counts.reduce((a, b) => a + b, 0);
  for (let k = 0; rem > 0; k++, rem--) counts[order[k % n].i]++;

  // every device in the chain must hold something, and nothing may exceed its cap
  let repaired = false;
  for (let i = 0; i < n; i++) if (counts[i] === 0 && caps[i] > 0) { counts[i] = 1; repaired = true; }
  for (let guard = 0; guard < 1000; guard++) {
    let over = -1, under = -1;
    for (let i = 0; i < n; i++) {
      if (counts[i] > caps[i]) over = i;
      if (counts[i] < caps[i] && under < 0) under = i;
    }
    if (over < 0) break;
    if (under < 0) return null;
    counts[over]--; counts[under]++; repaired = true;
  }
  if (counts.reduce((a, b) => a + b, 0) !== L) return null;
  if (counts.some((c, i) => c > caps[i] || c < 1)) return null;
  return { counts, repaired };
}

function build(spec, chain, counts, all, rtt, why, extra = {}) {
  const byId = new Map(all.map((d) => [d.id, d]));
  const ids = chain.map((d) => d.id);
  const pred = predict(spec, { chain: ids, counts }, byId, rtt);
  const kept = new Set(ids);
  return {
    host: ids[0],
    chain: ids,
    counts,
    ranges: rangesFrom(counts),
    dropped: all.filter((d) => !kept.has(d.id)).map((d) => d.id),
    predicted: pred,
    why,
    ...extra,
  };
}

// ---------------------------------------------------------------- strategies

function planSolo(spec, devices, rtt) {
  const hosts = devices.filter((d) => canHost(spec, d) && layerCap(spec, d, true) >= spec.layers);
  if (!hosts.length) return null;
  const h = hosts.slice().sort((a, b) => a.msPerLayer - b.msPerLayer)[0];
  return build(spec, [h], [spec.layers], devices, rtt, [
    `${h.name} holds all ${spec.layers} layers alone, so there are no hops at all.`,
  ]);
}

function planWeighted(spec, devices, rtt, { order, weightOf, label }) {
  const chain = order(devices.slice());
  if (!chain.length) return null;
  if (!canHost(spec, chain[0])) {
    const h = chain.findIndex((d) => canHost(spec, d));
    if (h < 0) return null;
    chain.unshift(chain.splice(h, 1)[0]);
  }
  const a = proportional(spec, chain, spec.layers, weightOf);
  if (!a) return null;
  const why = [label];
  if (a.repaired) why.push("Some shares did not fit in memory and were moved; this is what the strategy does in practice.");
  return build(spec, chain, a.counts, devices, rtt, why);
}

function planOptimal(spec, devices, rtt) {
  const n = devices.length;
  if (!n) return null;
  const dist = (a, b) => hopMs(spec, rtt(a, b), rtt.mbps ? rtt.mbps(a, b) : 0);
  const exact = n <= EXACT_LIMIT;

  let best = null;
  const considered = [];

  for (let mask = 1; mask < 1 << n; mask++) {
    const subset = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) subset.push(devices[i]);
    if (subset.length > spec.layers) continue;
    if (!subset.some((d) => canHost(spec, d))) continue;

    const ids = subset.map((d) => d.id);
    const tour = subset.length <= 2
      ? { cost: subset.length === 1 ? 0 : dist(ids[0], ids[1]) + dist(ids[1], ids[0]), order: ids }
      : exact ? bestTour(ids, dist) : heuristicTour(ids, dist);
    if (!Number.isFinite(tour.cost)) continue;

    const byId = new Map(subset.map((d) => [d.id, d]));
    for (const h of subset) {
      if (!canHost(spec, h)) continue;
      // rotate the cycle so the host leads; a cycle's cost is start-independent
      const at = tour.order.indexOf(h.id);
      const rotated = tour.order.slice(at).concat(tour.order.slice(0, at)).map((id) => byId.get(id));
      const alloc = allocate(spec, rotated, spec.layers);
      if (!alloc) continue;

      const total = hostOverheadMs(spec, h.msPerLayer) + alloc.costMs + tour.cost;
      const cand = { total, chain: rotated, counts: alloc.counts, tourMs: tour.cost, host: h };
      considered.push({ size: subset.length, host: h.name, total });
      if (!best || total < best.total) best = cand;
    }
  }
  if (!best) return null;

  const why = explain(spec, best, devices, rtt, considered);
  return build(spec, best.chain, best.counts, devices, rtt, why, {
    searched: considered.length,
    exactTour: exact,
  });
}

// The reasoning, in words. A plan nobody can interrogate is a plan nobody trusts, and
// on a stage "why did it drop my phone" is the first question asked.
function explain(spec, best, devices, rtt, considered) {
  const why = [];
  const hostCap = devices.filter((d) => canHost(spec, d));
  const faster = hostCap.filter((d) => d.msPerLayer < best.host.msPerLayer).length;

  why.push(
    `Host: ${best.host.name} at ${best.host.msPerLayer.toFixed(2)} ms/layer — ` +
    (faster === 0
      ? `the fastest of ${hostCap.length} device${hostCap.length > 1 ? "s" : ""} able to hold the embedding table.`
      : `chosen over ${faster} faster device${faster > 1 ? "s" : ""} because of memory or link cost.`) +
    ` The LM head alone is ${spec.headRatio.toFixed(1)} layers of work and runs here.`
  );

  if (best.chain.length === 1) {
    why.push(`Running solo. Splitting would add hops without reducing total compute, because decode is a sum of stages, not a max.`);
  } else {
    why.push(`Chain: ${best.chain.map((d) => d.name).join(" → ")} → back to host. ${best.chain.length} devices, ${best.chain.length} hops, ${best.tourMs.toFixed(1)} ms of network per token.`);
  }

  const kept = new Set(best.chain.map((d) => d.id));
  for (const d of devices) {
    if (kept.has(d.id)) continue;
    // What would it have cost to include this device? One extra hop against the
    // compute it would have taken off the others.
    const nearest = best.chain.reduce((m, c) => Math.min(m, rtt(d.id, c.id)), Infinity);
    const hopCost = Number.isFinite(nearest) ? nearest : 50;
    why.push(
      `Dropped ${d.name}: adding it costs about ${hopCost.toFixed(0)} ms of extra hop per token, ` +
      `more than the ${d.msPerLayer.toFixed(1)} ms/layer of work it could take off the chain.`
    );
  }

  const slowest = best.chain.reduce((a, b) => (a.msPerLayer > b.msPerLayer ? a : b));
  const fastest = best.chain.reduce((a, b) => (a.msPerLayer < b.msPerLayer ? a : b));
  if (best.chain.length > 1 && slowest.msPerLayer > fastest.msPerLayer * 1.25) {
    const i = best.chain.indexOf(slowest), j = best.chain.indexOf(fastest);
    why.push(
      `${fastest.name} is ${(slowest.msPerLayer / fastest.msPerLayer).toFixed(1)}× faster than ${slowest.name}, ` +
      `so it holds ${best.counts[j]} layers against ${best.counts[i]}.`
    );
  }
  return why;
}

// ---------------------------------------------------------------- entry point

// devices: [{ id, name, msPerLayer, budgetBytes }]
// rtt:     (a, b) -> ms, optionally with an .mbps(a, b) for link bandwidth
export function plan(spec, devices, rtt, { strategy = "optimal" } = {}) {
  if (!devices.length) return null;
  const t0 = (typeof performance !== "undefined" ? performance : Date).now();
  let p = null;

  switch (strategy) {
    case "solo":
      p = planSolo(spec, devices, rtt);
      break;

    case "even":
      p = planWeighted(spec, devices, rtt, {
        order: (d) => d,                                  // join order, as found
        weightOf: () => 1,
        label: "Equal split in join order. Ignores how fast each device is and how far apart they are.",
      });
      break;

    case "memory":
      p = planWeighted(spec, devices, rtt, {
        order: (d) => d.sort((a, b) => b.budgetBytes - a.budgetBytes),
        weightOf: (d) => d.budgetBytes,
        label: "Layers in proportion to memory, biggest device first — exo's ring memory-weighted partitioning. Nothing here measures speed or distance.",
      });
      break;

    case "compute":
      p = planWeighted(spec, devices, rtt, {
        order: (d) => d,
        weightOf: (d) => 1 / Math.max(d.msPerLayer, 1e-6),
        label: "Layers in proportion to measured speed, but in join order and blind to the network.",
      });
      break;

    case "optimal":
      p = planOptimal(spec, devices, rtt);
      break;

    default:
      throw new Error("unknown strategy: " + strategy);
  }

  if (!p) return null;
  p.strategy = strategy;
  p.solveMs = (typeof performance !== "undefined" ? performance : Date).now() - t0;
  return p;
}

// Run every strategy over the same room. This is the benchmark from the deck: same
// devices, same model, one number each.
export function compareAll(spec, devices, rtt) {
  const rows = [];
  for (const s of STRATEGIES) {
    let p = null;
    try { p = plan(spec, devices, rtt, { strategy: s }); } catch { /* infeasible */ }
    rows.push({
      strategy: s,
      feasible: !!p,
      totalMs: p?.predicted.totalMs ?? null,
      tokensPerSec: p?.predicted.tokensPerSec ?? null,
      devices: p?.chain.length ?? 0,
      host: p ? p.predicted.perDevice[0].name : null,
      plan: p,
    });
  }
  const feasible = rows.filter((r) => r.feasible);
  const bestMs = Math.min(...feasible.map((r) => r.totalMs));
  for (const r of rows) r.relative = r.feasible ? r.totalMs / bestMs : null;
  return rows.sort((a, b) => (a.totalMs ?? Infinity) - (b.totalMs ?? Infinity));
}

export { modelSpec, layerCap, canHost, memoryFor, predict };
