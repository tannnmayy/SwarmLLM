// Writes a tiny, real, Q8_0-quantized GGUF file for the GPU adapter's own
// numerical self-test (gpu-adapter-test.html) to load over an actual HTTP range
// request — the same code path a real Qwen3 GGUF goes through, at a size that
// fits in a repo and loads in milliseconds.
//
//   node tools/build-test-gguf.mjs
//
// Deliberately reuses the same tiny shape as engine/upstream/selftest.js's
// gpuSelfTest (hidden=64, 4 heads, 2 KV heads, head_dim=16, intermediate=128,
// vocab=96, 2 layers) so a mismatch between "the kernels are right" (Phase 1)
// and "the loader wires them up right" (Phase 2) cannot hide behind different
// shapes being exercised by each test.
//
// This writes GGUF's actual on-disk formats (not the engine's in-memory ones):
// KV metadata with real GGUF type tags, tensor infos with ggml's
// [innermost, ..., outermost] dimension order, and real Q8_0 blocks (an f16
// scale followed by 32 signed int8 values, little-endian) — the same 34-byte
// block layout engine/upstream/gguf.js's q8Repack/ggufToF32 read back.

import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const OUT = join(ROOT, "tests/fixtures/tiny-qwen3-q8.gguf");

// ---- f32 <-> f16, duplicated from engine/upstream/gguf.js rather than imported:
// this tool has to keep working even if that file's internals change shape,
// since its whole job is to be an independent producer for that file's consumer. ---
const _f16buf = new Float32Array(1), _f16u32 = new Uint32Array(_f16buf.buffer);
function f32ToF16(v) {
  _f16buf[0] = v;
  const x = _f16u32[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) { if (e < -10) return sign; m = (m | 0x800000) >> (1 - e); return sign | ((m + 0x1000) >> 13); }
  return sign | ((e << 10) + ((m + 0x1000) >> 13));
}
function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// ---- deterministic synthetic weights (same seed style as selftest.js) ----
const cfg = {
  hidden_size: 64, num_attention_heads: 4, num_key_value_heads: 2, head_dim: 16,
  intermediate_size: 128, vocab_size: 96, num_hidden_layers: 2,
  rms_norm_eps: 1e-5, rope_theta: 10000,
};
const { hidden_size: dim, num_attention_heads: nH, num_key_value_heads: nKV, head_dim: hd, intermediate_size: inter, vocab_size: vocab, num_hidden_layers: L } = cfg;
const qDim = nH * hd, kvDim = nKV * hd;

let seed = 424242;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
const mat = (r, c, s) => { const a = new Float32Array(r * c); for (let i = 0; i < a.length; i++) a[i] = rnd() * s; return a; };
const vec1 = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = 1 + rnd() * 0.2; return a; };

const embed = mat(vocab, dim, 1);
const finalNorm = vec1(dim);
const layers = [];
for (let l = 0; l < L; l++) layers.push({
  inNorm: vec1(dim), postNorm: vec1(dim),
  q: mat(qDim, dim, 0.3), k: mat(kvDim, dim, 0.3), v: mat(kvDim, dim, 0.3), o: mat(dim, qDim, 0.3),
  gate: mat(inter, dim, 0.3), up: mat(inter, dim, 0.3), down: mat(dim, inter, 0.3),
});

// Exported alongside the file (as JSON) so the browser test page has an
// independent copy of the exact floats used, for its own CPU reference —
// it must NOT recover them by re-reading the quantized GGUF, or a bug that
// corrupts both the file and the reference the same way would go undetected.
const weightsJSON = {
  cfg, embed: Array.from(embed), finalNorm: Array.from(finalNorm),
  layers: layers.map((l) => Object.fromEntries(Object.entries(l).map(([k, v]) => [k, Array.from(v)]))),
};

// ---- Q8_0 block encoding: f16 scale + 32 signed int8, 34 bytes/block ----
function q8_0Bytes(f32) {
  const n = f32.length, nb = Math.ceil(n / 32);
  const out = new Uint8Array(nb * 34);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    let amax = 0;
    for (let i = b * 32; i < Math.min(n, b * 32 + 32); i++) amax = Math.max(amax, Math.abs(f32[i]));
    const scaleF16 = f32ToF16(amax / 127 || 1);
    const scale = f16ToF32(scaleF16) || 1;
    dv.setUint16(b * 34, scaleF16, true);
    for (let i = 0; i < 32; i++) {
      const v = f32[b * 32 + i] || 0;
      const q = Math.max(-127, Math.min(127, Math.round(v / scale)));
      out[b * 34 + 2 + i] = q & 0xff;
    }
  }
  return out;
}

// ---- GGUF value type tags (must match engine/upstream/gguf.js's parser) ----
const T = { U32: 4, F32: 6, STR: 8 };

class Writer {
  constructor() { this.chunks = []; this.length = 0; }
  push(u8) { this.chunks.push(u8); this.length += u8.byteLength; }
  u32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); this.push(b); }
  u64(v) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); this.push(b); }
  f32(v) { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this.push(b); }
  str(s) { const bytes = new TextEncoder().encode(s); this.u64(bytes.length); this.push(bytes); }
  kv(key, type, writeValue) { this.str(key); this.u32(type); writeValue(); }
  toBuffer() {
    const out = new Uint8Array(this.length);
    let o = 0;
    for (const c of this.chunks) { out.set(c, o); o += c.byteLength; }
    return out;
  }
}

function layerNames(i) {
  const p = `blk.${i}.`;
  return {
    inNorm: p + "attn_norm.weight", q: p + "attn_q.weight", k: p + "attn_k.weight", v: p + "attn_v.weight",
    o: p + "attn_output.weight", postNorm: p + "ffn_norm.weight",
    gate: p + "ffn_gate.weight", up: p + "ffn_up.weight", down: p + "ffn_down.weight",
  };
}

async function main() {
  const w = new Writer();
  w.push(new Uint8Array([0x47, 0x47, 0x55, 0x46])); // "GGUF"
  w.u32(3);                                          // version 3

  // ---- collect tensors first (name -> {f32, rows, cols, kind}) so tensor count is known ----
  const tensors = [];
  const add1D = (name, f32) => tensors.push({ name, f32, shape: [f32.length], kind: "f32" });
  const add2D = (name, f32, rows, cols) => tensors.push({ name, f32, shape: [rows, cols], kind: "q8" });

  add2D("token_embd.weight", embed, vocab, dim);
  add1D("output_norm.weight", finalNorm);
  for (let i = 0; i < L; i++) {
    const N = layerNames(i), Lw = layers[i];
    add1D(N.inNorm, Lw.inNorm); add1D(N.postNorm, Lw.postNorm);
    add2D(N.q, Lw.q, qDim, dim); add2D(N.k, Lw.k, kvDim, dim); add2D(N.v, Lw.v, kvDim, dim);
    add2D(N.o, Lw.o, dim, qDim);
    add2D(N.gate, Lw.gate, inter, dim); add2D(N.up, Lw.up, inter, dim); add2D(N.down, Lw.down, dim, inter);
  }

  w.u64(tensors.length);   // tensor count
  w.u64(9);                // KV count

  w.kv("general.architecture", T.STR, () => w.str("qwen3"));
  w.kv("qwen3.embedding_length", T.U32, () => w.u32(dim));
  w.kv("qwen3.block_count", T.U32, () => w.u32(L));
  w.kv("qwen3.feed_forward_length", T.U32, () => w.u32(inter));
  w.kv("qwen3.attention.head_count", T.U32, () => w.u32(nH));
  w.kv("qwen3.attention.head_count_kv", T.U32, () => w.u32(nKV));
  w.kv("qwen3.attention.key_length", T.U32, () => w.u32(hd));
  w.kv("qwen3.attention.layer_norm_rms_epsilon", T.F32, () => w.f32(cfg.rms_norm_eps));
  w.kv("qwen3.rope.freq_base", T.F32, () => w.f32(cfg.rope_theta));

  // ---- tensor infos: name, n_dims, dims (ggml order: ne0=innermost=cols first), ggmlType, offset ----
  const GGML_F32 = 0, GGML_Q8_0 = 8;
  const bodies = tensors.map((t) => t.kind === "q8" ? q8_0Bytes(t.f32) : (() => {
    const b = new Uint8Array(t.f32.length * 4);
    const dv = new DataView(b.buffer);
    for (let i = 0; i < t.f32.length; i++) dv.setFloat32(i * 4, t.f32[i], true);
    return b;
  })());

  let offset = 0;
  const offsets = bodies.map((b) => { const o = offset; offset += b.byteLength; return o; });

  for (const t of tensors) {
    w.str(t.name);
    const ggmlDims = t.shape.length === 1 ? [t.shape[0]] : [t.shape[1], t.shape[0]]; // [cols, rows]
    w.u32(ggmlDims.length);
    for (const d of ggmlDims) w.u64(d);
    w.u32(t.kind === "q8" ? GGML_Q8_0 : GGML_F32);
    w.u64(offsets[tensors.indexOf(t)]);
  }

  const headerBytes = w.length;
  const align = 32;
  const dataStart = Math.ceil(headerBytes / align) * align;
  const pad = new Uint8Array(dataStart - headerBytes);
  w.push(pad);
  for (const b of bodies) w.push(b);

  const buf = w.toBuffer();
  await mkdir(join(ROOT, "tests/fixtures"), { recursive: true });
  await writeFile(OUT, buf);
  await writeFile(OUT.replace(/\.gguf$/, ".reference.json"), JSON.stringify(weightsJSON));
  console.log(`wrote ${OUT} (${buf.byteLength} bytes) and its .reference.json`);
}

main();
