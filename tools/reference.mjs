// CPU reference forward pass. Slow, obvious, and correct by construction.
//
//   node tools/reference.mjs "Once upon a time" 40
//
// This exists to be the thing the GPU engine must agree with. Every optimisation on
// the WebGPU side is gated against these numbers, so a kernel that is fast and wrong
// gets caught here rather than three hops into a swarm.
//
// It also proves the shard layout written by fetch-model.mjs is readable and correctly
// ordered, which is the other thing that would be miserable to debug later.

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "./tokenizer.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIR = join(ROOT, "models", process.env.MODEL || "smollm2-135m");

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
function f16ToF32(h) {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) {
    if (man === 0) { u32buf[0] = sign; return f32buf[0]; }
    let e = -1, m = man;
    do { e++; m <<= 1; } while (!(m & 0x400));
    u32buf[0] = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    return f32buf[0];
  }
  if (exp === 0x1f) { u32buf[0] = sign | 0x7f800000 | (man << 13); return f32buf[0]; }
  u32buf[0] = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return f32buf[0];
}
const widen = (u16) => {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]);
  return out;
};

async function loadShard(entry) {
  const buf = await readFile(join(DIR, entry.file));
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const out = {};
  for (const t of entry.index) {
    out[t.name] = { data: widen(u16.subarray(t.offset / 2, t.offset / 2 + t.length)), shape: t.shape };
  }
  return out;
}

const manifest = JSON.parse(await readFile(join(DIR, "manifest.json"), "utf8"));
const C = manifest.config;

console.error(`  ${manifest.label}: ${C.layers} layers, dim ${C.hiddenSize}, ${C.heads} heads / ${C.kvHeads} kv`);
console.error("  loading weights...");
const embed = (await loadShard(manifest.shards.embed)).embed;
const final = await loadShard(manifest.shards.final);
const layers = [];
for (const s of manifest.shards.layers) layers.push(await loadShard(s));

// ---------------------------------------------------------------- ops
// y = W @ x, W row-major [rows, cols]
function matvec(W, x, rows, cols, out) {
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) acc += W[base + c] * x[c];
    out[r] = acc;
  }
  return out;
}

function rmsnorm(x, w, eps, out) {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale * w[i];
  return out;
}

// Non-interleaved (GPT-NeoX style) RoPE: the pair for index i is (i, i + headDim/2).
function rope(v, nHeads, headDim, pos, theta) {
  const half = headDim >> 1;
  for (let h = 0; h < nHeads; h++) {
    const off = h * headDim;
    for (let i = 0; i < half; i++) {
      const freq = 1 / Math.pow(theta, (2 * i) / headDim);
      const a = pos * freq;
      const c = Math.cos(a), s = Math.sin(a);
      const x0 = v[off + i], x1 = v[off + i + half];
      v[off + i] = x0 * c - x1 * s;
      v[off + i + half] = x0 * s + x1 * c;
    }
  }
}

const silu = (x) => x / (1 + Math.exp(-x));

// ---------------------------------------------------------------- state
const D = C.hiddenSize, H = C.heads, KVH = C.kvHeads, HD = C.headDim, FF = C.intermediate;
const KVD = KVH * HD;
const MAXSEQ = 512;

const kCache = layers.map(() => new Float32Array(MAXSEQ * KVD));
const vCache = layers.map(() => new Float32Array(MAXSEQ * KVD));

const x = new Float32Array(D);
const xn = new Float32Array(D);
const q = new Float32Array(H * HD);
const k = new Float32Array(KVD);
const v = new Float32Array(KVD);
const attnOut = new Float32Array(H * HD);
const proj = new Float32Array(D);
const gate = new Float32Array(FF);
const up = new Float32Array(FF);

function forward(tokenId, pos) {
  x.set(embed.data.subarray(tokenId * D, tokenId * D + D));

  for (let l = 0; l < C.layers; l++) {
    const L = layers[l];
    rmsnorm(x, L.inNorm.data, C.rmsEps, xn);

    matvec(L.wq.data, xn, H * HD, D, q);
    matvec(L.wk.data, xn, KVD, D, k);
    matvec(L.wv.data, xn, KVD, D, v);

    rope(q, H, HD, pos, C.ropeTheta);
    rope(k, KVH, HD, pos, C.ropeTheta);

    kCache[l].set(k, pos * KVD);
    vCache[l].set(v, pos * KVD);

    const T = pos + 1;
    const scale = 1 / Math.sqrt(HD);
    for (let h = 0; h < H; h++) {
      const kvh = Math.floor(h / (H / KVH));           // grouped-query attention
      const qo = h * HD, ko = kvh * HD;
      const scores = new Float32Array(T);
      let max = -Infinity;
      for (let t = 0; t < T; t++) {
        let dot = 0;
        const kb = t * KVD + ko;
        for (let d = 0; d < HD; d++) dot += q[qo + d] * kCache[l][kb + d];
        scores[t] = dot * scale;
        if (scores[t] > max) max = scores[t];
      }
      let sum = 0;
      for (let t = 0; t < T; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
      for (let d = 0; d < HD; d++) {
        let acc = 0;
        for (let t = 0; t < T; t++) acc += scores[t] * vCache[l][t * KVD + ko + d];
        attnOut[qo + d] = acc / sum;
      }
    }

    matvec(L.wo.data, attnOut, D, D, proj);
    for (let i = 0; i < D; i++) x[i] += proj[i];

    rmsnorm(x, L.postNorm.data, C.rmsEps, xn);
    matvec(L.wGate.data, xn, FF, D, gate);
    matvec(L.wUp.data, xn, FF, D, up);
    for (let i = 0; i < FF; i++) gate[i] = silu(gate[i]) * up[i];
    matvec(L.wDown.data, gate, D, FF, proj);
    for (let i = 0; i < D; i++) x[i] += proj[i];
  }

  rmsnorm(x, final.finalNorm.data, C.rmsEps, xn);
  // tied embeddings: the LM head is the embedding matrix
  const head = C.tiedEmbeddings ? embed.data : final.lmHead.data;
  const logits = new Float32Array(C.vocab);
  matvec(head, xn, C.vocab, D, logits);
  return logits;
}

// ---------------------------------------------------------------- run
const tok = await Tokenizer.load(join(DIR, "tokenizer.json"));
const prompt = process.argv[2] || "Once upon a time";
const nNew = Number(process.argv[3] || 20);

const ids = tok.encode(prompt);
console.error(`  prompt: ${JSON.stringify(prompt)} -> ${ids.length} tokens [${ids.slice(0, 12).join(", ")}${ids.length > 12 ? ", ..." : ""}]`);
console.error("  generating...\n");

let pos = 0, logits = null;
const t0 = Date.now();
for (const id of ids) logits = forward(id, pos++);

const out = [];
process.stdout.write(prompt);
for (let n = 0; n < nNew; n++) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  if (best === C.eos) break;
  out.push(best);
  process.stdout.write(tok.decode([best]));
  logits = forward(best, pos++);
}
const dt = (Date.now() - t0) / 1000;
console.log();
console.error(`\n  ${out.length} tokens in ${dt.toFixed(1)}s (${(out.length / dt).toFixed(2)} tok/s, CPU reference)`);

// golden vector for the GPU engine to match
if (process.env.GOLDEN) {
  const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 10);
  console.error("  top-10 logits at final position:");
  for (const i of top) console.error(`    ${String(i).padStart(6)} ${logits[i].toFixed(6)}  ${JSON.stringify(tok.decode([i]))}`);
}
