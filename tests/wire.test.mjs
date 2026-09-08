// Correctness gate for the wire. Run: node tests/wire.test.mjs
// A silent bug here shows up as garbage text three hops away, so this runs first.

import {
  f32ToF16, f16ToF32, packF16, unpackF16, looksBad,
  encodeFrame, decodeSlice, makeReassembler, SLICE_BYTES,
} from "../room/wire.js";

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "  " + extra : "")); }
};

// performance.now exists in Node 23, but be explicit for older runtimes
if (typeof performance === "undefined") globalThis.performance = { now: () => Date.now() };

console.log("\nf16 round trip");
{
  const exact = [0, -0, 1, -1, 0.5, -0.5, 2, 1024, -2048, 65504, -65504];
  let allExact = true;
  for (const v of exact) {
    const r = f16ToF32(f32ToF16(v));
    if (!Object.is(r, v)) { allExact = false; console.log("     " + v + " -> " + r); }
  }
  ok("values representable in f16 survive exactly", allExact);

  // f16 has an 11-bit significand, so relative error is bounded by 2^-11 -- but only
  // for normals. Below 2^-14 (6.1e-5) the format goes subnormal and trades precision
  // for range, which is correct behaviour, not a bug. Assert each regime honestly.
  let worst = 0, worstAt = 0;
  for (let i = 0; i < 50000; i++) {
    const mag = Math.pow(10, Math.random() * 8 - 4);           // 1e-4 .. 1e4
    const v = (Math.random() < 0.5 ? -1 : 1) * mag;
    if (Math.abs(v) > 65504) continue;                          // beyond f16 range
    const r = f16ToF32(f32ToF16(v));
    const rel = Math.abs(r - v) / Math.abs(v);
    if (rel > worst) { worst = rel; worstAt = v; }
  }
  ok("relative error <= 2^-11 for f16 normals", worst <= 4.88e-4,
     "worst " + worst.toExponential(2) + " at " + worstAt.toExponential(3));

  // subnormals: relative error grows, but absolute error must stay under one ulp
  // of the subnormal grid (2^-24), or small activations would drift rather than
  // simply lose resolution.
  let worstAbs = 0;
  for (let i = 0; i < 20000; i++) {
    const v = (Math.random() * 2 - 1) * 6.1e-5;
    worstAbs = Math.max(worstAbs, Math.abs(f16ToF32(f32ToF16(v)) - v));
  }
  ok("subnormal absolute error within one ulp (2^-24)", worstAbs <= Math.pow(2, -24),
     "worst " + worstAbs.toExponential(2));

  ok("subnormals survive", (() => {
    for (const v of [6e-8, 1e-7, 5.96e-8, -6e-8]) {
      const r = f16ToF32(f32ToF16(v));
      if (Math.abs(r - v) / Math.abs(v) > 0.5) return false;
    }
    return true;
  })());

  ok("Infinity and NaN survive", (() => {
    if (f16ToF32(f32ToF16(Infinity)) !== Infinity) return false;
    if (f16ToF32(f32ToF16(-Infinity)) !== -Infinity) return false;
    if (!Number.isNaN(f16ToF32(f32ToF16(NaN)))) return false;
    return true;
  })());

  ok("overflow saturates to Infinity, not garbage", f16ToF32(f32ToF16(1e30)) === Infinity);
  ok("negative zero keeps its sign", Object.is(f16ToF32(f32ToF16(-0)), -0));
}

console.log("\npack / unpack");
{
  const n = 576;                                   // SmolLM2-135M hidden size
  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = Math.sin(i * 0.37) * 3.5;
  const back = unpackF16(packF16(src));
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(back[i] - src[i]));
  ok("hidden state survives pack/unpack", worst < 2e-3, "worst abs " + worst.toExponential(2));
  ok("packed size is half", packF16(src).byteLength === src.byteLength / 2);
  ok("looksBad passes a clean vector", !looksBad(back));
  const dirty = back.slice(); dirty[300] = NaN;
  ok("looksBad catches an injected NaN", looksBad(dirty));
}

console.log("\nframing");
{
  const mk = (n) => {
    const f = new Float32Array(n);
    for (let i = 0; i < n; i++) f[i] = (i % 211) * 0.013 - 1.3;
    return packF16(f);
  };
  const roundTrip = (data, { shuffle = false, pos = 7, n = 1 } = {}) => {
    const slices = encodeFrame({ t: "hidden", pos, n, data }, 42);
    const order = slices.map((_, i) => i);
    if (shuffle) order.reverse();
    const rs = makeReassembler();
    let out = null;
    for (const i of order) out = decodeSlice(rs, slices[i]) || out;
    return { out, count: slices.length };
  };

  for (const n of [1, 576, 2048, 5120]) {
    const data = mk(n);
    const { out, count } = roundTrip(data);
    const same = out && out.data.length === data.length && out.data.every((v, i) => v === data[i]);
    ok(`n=${n} (${data.byteLength} B, ${count} slice${count > 1 ? "s" : ""}) round-trips exactly`, !!same);
  }

  // exactly on the slice boundary is where off-by-ones live
  const per = SLICE_BYTES - 24;
  const edge = new Uint16Array(per / 2);
  edge.fill(0x3c00);
  const { out: e1, count: c1 } = roundTrip(edge);
  ok(`exact slice boundary (${per} B) is one slice`, c1 === 1 && !!e1);

  const edge2 = new Uint16Array(per / 2 + 1);
  edge2.fill(0x3c00);
  const { count: c2 } = roundTrip(edge2);
  ok("one byte over the boundary becomes two slices", c2 === 2);

  const big = mk(5120);
  ok("out-of-order slices reassemble", (() => {
    const { out } = roundTrip(big, { shuffle: true });
    return out && out.data.every((v, i) => v === big[i]);
  })());

  ok("duplicate slices are ignored", (() => {
    const slices = encodeFrame({ t: "hidden", pos: 3, data: big }, 9);
    const rs = makeReassembler();
    let done = null, completions = 0;
    for (const s of [...slices, ...slices]) {
      const r = decodeSlice(rs, s);
      if (r) { completions++; done = r; }
    }
    return completions === 1 && done.data.every((v, i) => v === big[i]);
  })());

  ok("every slice fits the SCTP budget", (() => {
    const slices = encodeFrame({ t: "hidden", pos: 0, data: mk(5120) }, 1);
    return slices.every((s) => s.byteLength <= SLICE_BYTES);
  })());

  ok("header fields survive", (() => {
    const slices = encodeFrame({ t: "hidden-ret-b", pos: 123456, n: 8, flags: 1, data: mk(576) }, 5);
    const rs = makeReassembler();
    let out = null;
    for (const s of slices) out = decodeSlice(rs, s) || out;
    return out.t === "hidden-ret-b" && out.pos === 123456 && out.n === 8 && out.flags === 1;
  })());

  ok("a foreign buffer is rejected, not misparsed", (() => {
    const rs = makeReassembler();
    return decodeSlice(rs, new ArrayBuffer(64)) === null && decodeSlice(rs, new ArrayBuffer(4)) === null;
  })());
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
