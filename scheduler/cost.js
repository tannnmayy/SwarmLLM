// The cost model: what one token actually costs on a given arrangement of devices.
//
// Everything the planner decides rests on this file, so it states its assumptions
// out loud rather than burying them in a solver.
//
// THE CENTRAL FACT. Single-stream decode is a *sum* of stages plus hops, not a max.
// One token is in flight at a time and it walks the whole chain, so splitting a model
// over more devices makes a single stream SLOWER, never faster. exo measures this
// directly: 1, 2 and 3 x M4 Pro give 49.3, 44.4 and 39.7 tok/s. prima.cpp's 70B win
// (674 ms/token against 10,120 on one machine) comes from escaping memory exhaustion,
// not from parallelism.
//
// A swarm therefore buys CAPACITY, not SPEED. The scheduler's job is to know when not
// to split at all, and to minimise the damage when the model genuinely does not fit.
// A planner built on the opposite belief -- that more devices means faster -- would
// happily add a phone that costs more in hops than it returns in compute.
//
// Units: milliseconds and bytes throughout. No implicit seconds anywhere.

// ---------------------------------------------------------------- model spec

// Derive everything the planner needs from a model manifest.
//
// `precision` is the width weights occupy IN MEMORY, which is not the width on disk.
// Our CPU engine widens f16 shards to f32 arrays, so it costs twice the download.
// A future WebGPU path keeping f16 in buffers passes "f16" and the caps double.
export function modelSpec(manifest, { precision = "f32", maxSeq = 512 } = {}) {
  const c = manifest.config;
  const D = c.hiddenSize;
  const kvDim = c.kvHeads * c.headDim;
  const bpw = precision === "f16" ? 2 : 4;

  // Multiply-accumulates in one transformer block, decoding a single token.
  // Attention over the KV cache is O(seq) and small next to the projections at
  // these sizes, so it is folded into the measured ms/layer rather than modelled.
  const layerMACs =
    D * D +                 // q
    D * kvDim +             // k
    D * kvDim +             // v
    D * D +                 // o
    3 * D * c.intermediate; // gate, up, down

  const headMACs = c.vocab * D;
  const layerParams = layerMACs + 2 * D;             // + the two RMSNorm vectors
  const embedParams = c.vocab * D;

  return {
    label: manifest.label,
    layers: c.layers,
    hidden: D,
    maxSeq,
    precision,

    layerMACs,
    headMACs,
    // The LM head is one matvec over the whole vocabulary, and it runs on the host
    // alone. At 49,152 x 576 that is ~8 layers of arithmetic -- the single largest
    // serial term in the model, and the reason host election matters at all.
    headRatio: headMACs / layerMACs,

    layerBytes: layerParams * bpw,
    embedBytes: embedParams * bpw,
    // K and V for every position we might reach. Grows with the layer count, so a
    // device holding more layers pays for more cache.
    kvBytesPerLayer: 2 * maxSeq * kvDim * 4,
    // Hidden state on the wire, f16. This is what crosses the network per hop.
    wireBytes: D * 2,
    // Activations, logits and the odd scratch vector. Small and roughly fixed.
    scratchBytes: (c.vocab + 8 * D + 4 * c.intermediate) * 4,
    tiedEmbeddings: !!c.tiedEmbeddings,
  };
}

// ---------------------------------------------------------------- memory

export function memoryFor(spec, layerCount, isHost) {
  return (
    layerCount * (spec.layerBytes + spec.kvBytesPerLayer) +
    (isHost ? spec.embedBytes : 0) +
    spec.scratchBytes
  );
}

// How many layers this device could hold, as host or as a worker.
export function layerCap(spec, device, isHost) {
  const free = device.budgetBytes - (isHost ? spec.embedBytes : 0) - spec.scratchBytes;
  if (free <= 0) return 0;
  return Math.max(0, Math.floor(free / (spec.layerBytes + spec.kvBytesPerLayer)));
}

// A host must hold the embedding table (which is also the LM head when tied) and at
// least one layer. Devices that cannot are workers only -- typically phones.
export function canHost(spec, device) {
  return layerCap(spec, device, true) >= 1;
}

// ---------------------------------------------------------------- latency

// One hop, one way: half the round trip, plus the time to push the bytes.
// Slicing keeps a hop to a single one-way trip (see room/wire.js), so this does not
// carry the extra round trip an unsliced 10 KB send would pay.
export function hopMs(spec, rttMs, linkMbps) {
  const serialize = linkMbps > 0 ? (spec.wireBytes * 8) / (linkMbps * 1e6) * 1000 : 0;
  return rttMs / 2 + serialize;
}

// What the host does that nobody else can: embed, final norm, the LM head matvec,
// and sampling. Charged at the host's own measured speed.
export function hostOverheadMs(spec, hostMsPerLayer) {
  return spec.headRatio * hostMsPerLayer;
}

// Predict the per-token latency of a concrete arrangement.
//
//   chain    device ids in layer order, host first
//   counts   layers held by each, same order
//   The hidden state travels host -> chain[1] -> ... -> chain[k] -> host, so a chain
//   of k workers pays k+1 hops. A solo host pays none.
export function predict(spec, { chain, counts }, byId, rtt) {
  const host = byId.get(chain[0]);
  const compute = chain.map((id, i) => counts[i] * byId.get(id).msPerLayer);
  const computeMs = compute.reduce((a, b) => a + b, 0);

  const hops = [];
  for (let i = 0; i < chain.length; i++) {
    const from = chain[i];
    const to = chain[(i + 1) % chain.length];   // last one wraps back to the host
    if (chain.length === 1) break;              // solo: no hops at all
    hops.push({ from, to, ms: hopMs(spec, rtt(from, to), rtt.mbps ? rtt.mbps(from, to) : 0) });
  }
  const hopMsTotal = hops.reduce((a, h) => a + h.ms, 0);
  const hostMs = hostOverheadMs(spec, host.msPerLayer);
  const totalMs = computeMs + hopMsTotal + hostMs;

  return {
    totalMs,
    tokensPerSec: totalMs > 0 ? 1000 / totalMs : 0,
    breakdown: {
      computeMs,
      hopMs: hopMsTotal,
      hostMs,
      hopShare: totalMs > 0 ? hopMsTotal / totalMs : 0,
    },
    perDevice: chain.map((id, i) => ({
      id,
      name: byId.get(id).name,
      layers: counts[i],
      computeMs: compute[i],
      msPerLayer: byId.get(id).msPerLayer,
      memoryBytes: memoryFor(spec, counts[i], i === 0),
    })),
    hops,
  };
}

// Convenience: build an rtt(a, b) lookup from a flat matrix or a Map of maps.
// Unknown pairs fall back to `fallback` rather than zero -- assuming a free link to a
// device we have never measured is exactly the optimism that produces bad plans.
export function rttLookup(matrix, { fallback = 25, selfMs = 0 } = {}) {
  const fn = (a, b) => {
    if (a === b) return selfMs;
    const row = matrix instanceof Map ? matrix.get(a) : matrix[a];
    const v = row instanceof Map ? row.get(b) : row?.[b];
    if (v == null || !Number.isFinite(v)) {
      const back = matrix instanceof Map ? matrix.get(b) : matrix[b];
      const w = back instanceof Map ? back.get(a) : back?.[a];
      return w == null || !Number.isFinite(w) ? fallback : w;
    }
    return v;
  };
  return fn;
}
