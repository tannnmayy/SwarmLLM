// Gates for the scheduler. This is the contribution, so it gets the hardest tests.
//
//   node tests/plan.test.mjs
//
// Two ideas run through this file:
//   1. Where an exact algorithm exists, check it against an independent brute force.
//      The DP is checked against greedy; Held-Karp against every permutation.
//   2. Where a claim appears in the pitch, there is a test named after the claim.
//      "optimal is never worse than any baseline" is a test, not an assertion.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  modelSpec, plan, compareAll, allocate, bestTour, layerCap, canHost, predict, STRATEGIES,
} from "../scheduler/plan.js";
import { rttLookup, hopMs, hostOverheadMs } from "../scheduler/cost.js";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const MB = 2 ** 20;

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};
const near = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

const manifest = JSON.parse(await readFile(join(ROOT, "models/smollm2-135m/manifest.json"), "utf8"));
const spec = modelSpec(manifest, { precision: "f32", maxSeq: 512 });

console.log(`\nmodel: ${spec.label}`);
console.log(`  ${spec.layers} layers · layer ${(spec.layerBytes / MB).toFixed(1)} MB + ${(spec.kvBytesPerLayer / MB).toFixed(2)} MB KV · embed ${(spec.embedBytes / MB).toFixed(0)} MB`);
console.log(`  LM head = ${spec.headRatio.toFixed(2)} layers of arithmetic, host-serial`);
console.log(`  hidden state on the wire: ${spec.wireBytes} B`);

// A flat LAN: everyone 8 ms from everyone.
const lan = (ms = 8) => {
  const f = () => ms;
  return f;
};
const dev = (id, name, msPerLayer, gb) => ({ id, name, msPerLayer, budgetBytes: gb * 2 ** 30 });

// ---------------------------------------------------------------- allocation
console.log("\nallocation (DP)");
{
  const chain = [dev("a", "A", 4, 8), dev("b", "B", 4, 8), dev("c", "C", 4, 8)];
  const a = allocate(spec, chain, 30);
  ok("splits 30 layers across 3 equal devices", a && a.counts.reduce((x, y) => x + y, 0) === 30);
  ok("equal devices get an equal share", a && Math.max(...a.counts) - Math.min(...a.counts) <= 1,
     a && JSON.stringify(a.counts));

  // Independent check: for a uniform model the objective is linear, so give everyone
  // one layer and then hand the rest to the fastest devices in order. That is provably
  // optimal for a linear objective with box constraints, and must match the DP.
  const greedy = (chain, L) => {
    const caps = chain.map((d, i) => layerCap(spec, d, i === 0));
    if (caps.reduce((x, y) => x + y, 0) < L || chain.length > L) return null;
    const counts = caps.map((c) => (c > 0 ? 1 : 0));
    if (counts.some((c) => c === 0)) return null;
    let rem = L - counts.reduce((x, y) => x + y, 0);
    const order = chain.map((d, i) => i).sort((x, y) => chain[x].msPerLayer - chain[y].msPerLayer);
    for (const i of order) {
      const room = Math.min(caps[i] - counts[i], rem);
      counts[i] += room; rem -= room;
      if (!rem) break;
    }
    if (rem) return null;
    return counts.reduce((s, c, i) => s + c * chain[i].msPerLayer, 0);
  };

  let mismatch = null;
  for (let trial = 0; trial < 400; trial++) {
    const n = 2 + Math.floor(Math.random() * 5);
    const c = Array.from({ length: n }, (_, i) =>
      dev("d" + i, "D" + i, 1 + Math.random() * 20, 1.5 + Math.random() * 8));
    const a2 = allocate(spec, c, spec.layers);
    const g = greedy(c, spec.layers);
    if (!a2 && g == null) continue;
    if (!a2 || g == null || !near(a2.costMs, g, 1e-6)) {
      mismatch = { counts: a2?.counts, dp: a2?.costMs, greedy: g, ms: c.map((d) => +d.msPerLayer.toFixed(2)) };
      break;
    }
  }
  ok("DP matches an independent greedy optimum over 400 random rooms", !mismatch, JSON.stringify(mismatch));

  ok("respects memory caps", (() => {
    // 1.5 GB host: embed is 113 MB, each layer 14.9 MB -> cannot hold many
    const c = [dev("h", "H", 4, 1.5), dev("w", "W", 4, 8)];
    const a3 = allocate(spec, c, 30);
    if (!a3) return false;
    return a3.counts[0] <= layerCap(spec, c[0], true) && a3.counts[1] <= layerCap(spec, c[1], false);
  })());

  ok("every device in the chain gets at least one layer", (() => {
    const c = [dev("h", "H", 1, 8), dev("w", "W", 900, 8)];
    const a4 = allocate(spec, c, 30);
    return a4 && a4.counts.every((x) => x >= 1);
  })());

  ok("refuses a chain that cannot hold the model", allocate(spec, [dev("t", "T", 4, 0.3)], 30) === null);
  ok("refuses more devices than layers", allocate(spec,
     Array.from({ length: 31 }, (_, i) => dev("x" + i, "X" + i, 4, 8)), 30) === null);

  ok("allocation is order-independent for a uniform model", (() => {
    const c = [dev("a", "A", 3, 8), dev("b", "B", 11, 8), dev("c", "C", 6, 8)];
    const base = allocate(spec, c, 30).costMs;
    const perms = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    // the host slot pays the embedding table, so compare like with like: rotate only
    // among the non-host positions
    return perms.filter((p) => p[0] === 0).every((p) => near(allocate(spec, p.map((i) => c[i]), 30).costMs, base, 1e-9));
  })());

  ok("a faster device is given more layers", (() => {
    const c = [dev("h", "H", 2, 8), dev("s", "S", 20, 8)];
    const a5 = allocate(spec, c, 30);
    return a5 && a5.counts[0] > a5.counts[1];
  })());
}

// ---------------------------------------------------------------- tour
console.log("\nchain order (Held-Karp)");
{
  const perms = (a) => a.length <= 1 ? [a] :
    a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map((r) => [x, ...r]));

  let worstGap = 0, bad = null;
  for (let trial = 0; trial < 120; trial++) {
    const n = 3 + Math.floor(Math.random() * 4);            // 3..6
    const ids = Array.from({ length: n }, (_, i) => "n" + i);
    const m = {};
    for (const a of ids) { m[a] = {}; for (const b of ids) m[a][b] = a === b ? 0 : 1 + Math.random() * 200; }
    for (const a of ids) for (const b of ids) m[b][a] = m[a][b];   // symmetric
    const d = (a, b) => m[a][b];

    const hk = bestTour(ids, d);
    // brute force: fix the first node (a cycle has no preferred start)
    let brute = Infinity;
    for (const p of perms(ids.slice(1))) {
      const o = [ids[0], ...p];
      brute = Math.min(brute, o.reduce((s, id, i) => s + d(id, o[(i + 1) % o.length]), 0));
    }
    const gap = hk.cost - brute;
    if (gap > 1e-6) { bad = { n, hk: hk.cost, brute }; break; }
    worstGap = Math.max(worstGap, Math.abs(gap));
  }
  ok("Held-Karp equals brute force over 120 random metrics", !bad, JSON.stringify(bad));
  ok("returned order really costs what it claims", (() => {
    const ids = ["a", "b", "c", "d", "e"];
    const m = {}; ids.forEach((a) => { m[a] = {}; ids.forEach((b) => { m[a][b] = a === b ? 0 : 5 + ((a.charCodeAt(0) * b.charCodeAt(0)) % 90); }); });
    ids.forEach((a) => ids.forEach((b) => { m[b][a] = m[a][b]; }));
    const d = (a, b) => m[a][b];
    const t = bestTour(ids, d);
    const walked = t.order.reduce((s, id, i) => s + d(id, t.order[(i + 1) % t.order.length]), 0);
    return near(walked, t.cost, 1e-9) && new Set(t.order).size === ids.length;
  })());

  ok("two sites: the chain crosses the link exactly twice", (() => {
    // A and B on site 1, C and D on site 2. A good cycle is A B | C D and back:
    // two crossings. A bad one (A C B D) crosses four times.
    const site = { a: 0, b: 0, c: 1, d: 1 };
    const d = (x, y) => (site[x] === site[y] ? 3 : 90);
    const t = bestTour(["a", "b", "c", "d"], d);
    let crossings = 0;
    t.order.forEach((id, i) => { if (site[id] !== site[t.order[(i + 1) % t.order.length]]) crossings++; });
    return crossings === 2;
  })());
}

// ---------------------------------------------------------------- decisions
console.log("\nplanner decisions");
{
  // Big fast laptop that can hold everything, plus a slow phone on a slow link.
  const room = [dev("pc", "Gaming PC", 3, 8), dev("ph", "Phone", 30, 3)];
  const rtt = rttLookup({ pc: { ph: 40 }, ph: { pc: 40 } }, { fallback: 40 });
  const p = plan(spec, room, rtt, { strategy: "optimal" });
  ok("runs solo when one device can hold the model and the other is a net loss",
     p.chain.length === 1 && p.host === "pc", JSON.stringify({ chain: p.chain, counts: p.counts }));
  ok("says why it dropped the phone", p.why.some((w) => /Dropped Phone/.test(w)), p.why.join(" | "));

  // Model does not fit on any one device: splitting is forced.
  const tight = [dev("a", "A", 4, 0.42), dev("b", "B", 4, 0.42), dev("c", "C", 4, 0.42)];
  const p2 = plan(spec, tight, lan(6), { strategy: "optimal" });
  ok("splits when no single device can hold the model", p2 && p2.chain.length > 1,
     p2 ? `chain ${p2.chain.length}` : "infeasible");

  // Host election: the fastest capable device should take the LM head.
  const mixed = [
    dev("slow", "Slow", 18, 8),
    dev("fast", "Fast", 3, 8),
    dev("mid", "Mid", 9, 8),
  ];
  const p3 = plan(spec, mixed, lan(9), { strategy: "optimal" });
  ok("elects the fastest capable device as host", p3.host === "fast", "elected " + p3.host);

  // A phone too small for the embedding table must never be host -- even when it is
  // the fastest device in the room. 0.1 GB holds 7 layers as a worker but cannot fit
  // the 108 MB embedding table plus a layer, so it is worker-only.
  const withPhone = [dev("phone", "Phone", 2, 0.1), dev("pc", "PC", 9, 8)];
  ok("never elects a device too small for the embedding table, even if it is fastest",
     !canHost(spec, withPhone[0]) && layerCap(spec, withPhone[0], false) >= 1 &&
     plan(spec, withPhone, lan(5), { strategy: "optimal" }).host === "pc");

  // Chain order across two sites. Budgets are set so no three devices can hold the
  // model (3 + 10 + 10 = 23 < 30), which forces all four into the chain and makes the
  // ordering decision the only thing left to get right.
  const twoSite = [
    dev("a1", "A1", 5, 0.15), dev("a2", "A2", 5, 0.15),
    dev("b1", "B1", 5, 0.15), dev("b2", "B2", 5, 0.15),
  ];
  const site = { a1: 0, a2: 0, b1: 1, b2: 1 };
  const wan = rttLookup(Object.fromEntries(twoSite.map((x) =>
    [x.id, Object.fromEntries(twoSite.map((y) => [y.id, site[x.id] === site[y.id] ? 2 : 120]))])));
  const p4 = plan(spec, twoSite, wan, { strategy: "optimal" });
  const cross = p4.chain.filter((id, i) => site[id] !== site[p4.chain[(i + 1) % p4.chain.length]]).length;
  ok("uses all four devices when three cannot hold the model", p4.chain.length === 4,
     "chain " + p4.chain.join(" -> "));
  ok("orders a two-site chain to cross the slow link only twice", cross === 2,
     p4.chain.join(" -> ") + " crossings=" + cross);
  // join order here would be a1, a2, b1, b2 -- already good. The interesting case is
  // when the ids interleave, which is what a real room's random peer ids do.
  const shuffled = [twoSite[0], twoSite[2], twoSite[1], twoSite[3]];
  const p5 = plan(spec, shuffled, wan, { strategy: "optimal" });
  const cross5 = p5.chain.filter((id, i) => site[id] !== site[p5.chain[(i + 1) % p5.chain.length]]).length;
  const joinOrderCross = shuffled.filter((d, i) => site[d.id] !== site[shuffled[(i + 1) % shuffled.length].id]).length;
  ok("re-orders an interleaved room that join order would get wrong",
     cross5 === 2 && joinOrderCross === 4,
     `optimal ${p5.chain.join(" -> ")} (${cross5}) vs join order (${joinOrderCross})`);
}

// ---------------------------------------------------------------- invariants
console.log("\ninvariants (every strategy, many random rooms)");
{
  let bad = null, checked = 0, optimalLosses = 0, worstLoss = 0;
  for (let trial = 0; trial < 300 && !bad; trial++) {
    const n = 1 + Math.floor(Math.random() * 5);
    const room = Array.from({ length: n }, (_, i) =>
      dev("d" + i, "D" + i, 1 + Math.random() * 30, 0.3 + Math.random() * 8));
    const m = {};
    for (const a of room) { m[a.id] = {}; for (const b of room) m[a.id][b.id] = a.id === b.id ? 0 : 1 + Math.random() * 150; }
    for (const a of room) for (const b of room) m[b.id][a.id] = m[a.id][b.id];
    const rtt = rttLookup(m);

    const results = {};
    for (const s of STRATEGIES) {
      let p = null;
      try { p = plan(spec, room, rtt, { strategy: s }); } catch (e) { bad = { s, e: e.message }; break; }
      if (!p) continue;
      results[s] = p;
      checked++;

      const total = p.counts.reduce((a, b) => a + b, 0);
      if (total !== spec.layers) { bad = { s, why: "layers do not sum to " + spec.layers, counts: p.counts }; break; }
      if (p.counts.some((c) => c < 1)) { bad = { s, why: "a device holds zero layers", counts: p.counts }; break; }
      if (new Set(p.chain).size !== p.chain.length) { bad = { s, why: "a device appears twice in the chain" }; break; }
      const flat = p.ranges.flat();
      if (p.ranges[0][0] !== 0 || p.ranges.at(-1)[1] !== spec.layers) { bad = { s, why: "ranges do not cover the model" }; break; }
      for (let i = 1; i < p.ranges.length; i++) if (p.ranges[i][0] !== p.ranges[i - 1][1]) { bad = { s, why: "ranges are not contiguous" }; break; }

      // memory feasibility, per device, for real
      for (let i = 0; i < p.chain.length; i++) {
        const d = room.find((x) => x.id === p.chain[i]);
        const cap = layerCap(spec, d, i === 0);
        if (p.counts[i] > cap) { bad = { s, why: `${d.name} over memory: ${p.counts[i]} > ${cap}` }; break; }
      }
      if (bad) break;
      if (!canHost(spec, room.find((x) => x.id === p.host))) { bad = { s, why: "host cannot hold the embedding table" }; break; }
    }
    if (bad) break;

    // The claim from the deck, tested rather than asserted.
    if (results.optimal) {
      for (const s of STRATEGIES) {
        if (s === "optimal" || !results[s]) continue;
        const gap = results.optimal.predicted.totalMs - results[s].predicted.totalMs;
        if (gap > 1e-6) { optimalLosses++; worstLoss = Math.max(worstLoss, gap); }
      }
    }
  }
  ok(`every strategy produces a valid, feasible, contiguous plan (${checked} plans)`, !bad, JSON.stringify(bad));
  ok("optimal is never beaten by any baseline", optimalLosses === 0,
     optimalLosses ? `${optimalLosses} losses, worst ${worstLoss.toFixed(3)} ms` : "");
}

console.log("\ncost model consistency");
{
  const room = [dev("a", "A", 4, 8), dev("b", "B", 7, 4), dev("c", "C", 12, 2)];
  const rtt = lan(11);
  const p = plan(spec, room, rtt, { strategy: "optimal" });
  const byId = new Map(room.map((d) => [d.id, d]));
  const re = predict(spec, { chain: p.chain, counts: p.counts }, byId, rtt);
  ok("predict() reproduces the planner's own number", near(re.totalMs, p.predicted.totalMs, 1e-9));

  const parts = p.predicted.breakdown;
  ok("breakdown sums to the total",
     near(parts.computeMs + parts.hopMs + parts.hostMs, p.predicted.totalMs, 1e-9));
  ok("hop count equals chain length for a multi-device chain",
     p.chain.length === 1 ? p.predicted.hops.length === 0 : p.predicted.hops.length === p.chain.length);
  ok("a solo plan has no hops at all", (() => {
    const s = plan(spec, [dev("x", "X", 4, 8)], lan(50), { strategy: "optimal" });
    return s.predicted.hops.length === 0 && s.predicted.breakdown.hopMs === 0;
  })());
  ok("host overhead really is the LM head",
     near(p.predicted.breakdown.hostMs, hostOverheadMs(spec, byId.get(p.host).msPerLayer), 1e-9));
}

// ---------------------------------------------------------------- the real room
console.log("\nthe actual demo room (5 devices, one Wi-Fi)");
{
  // ms/layer estimated from the measured two-device run: 30 layers + an 8-layer head
  // in ~200 ms per token. Replace with probe.html numbers when they land.
  const room = [
    dev("pc1", "Tanmay PC", 5.0, 6),
    dev("pc2", "PC 2", 6.5, 6),
    dev("and1", "Android 1", 20, 2),
    dev("and2", "Android 2", 24, 2),
    dev("ios", "iPhone 14+", 12, 2.5),
  ];
  const rtt = lan(9);
  const rows = compareAll(spec, room, rtt);
  console.log("\n    strategy   devices  pred tok/s   per token   host");
  for (const r of rows) {
    if (!r.feasible) { console.log(`    ${r.strategy.padEnd(9)}  infeasible`); continue; }
    console.log(`    ${r.strategy.padEnd(9)}  ${String(r.devices).padStart(4)}   ${r.tokensPerSec.toFixed(2).padStart(9)}   ${r.totalMs.toFixed(0).padStart(7)} ms   ${r.host}`);
  }
  const best = rows[0], mem = rows.find((r) => r.strategy === "memory");
  ok("optimal is the fastest strategy in the real room", best.strategy === "optimal" || near(best.totalMs, rows.find((r) => r.strategy === "optimal").totalMs, 1e-9),
     "winner was " + best.strategy);
  const opt = rows.find((r) => r.strategy === "optimal");
  console.log(`\n    optimal vs memory-weighted: ${(mem.totalMs / opt.totalMs).toFixed(2)}x`);
  for (const w of opt.plan.why) console.log("    · " + w);
  ok("solves fast enough to re-plan live", opt.plan.solveMs < 250, opt.plan.solveMs.toFixed(1) + " ms");
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
