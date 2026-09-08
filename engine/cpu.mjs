// Sliceable CPU engine.
//
// The whole point of this file is the interface, not the speed. A device holds a
// contiguous range of layers and exposes four calls:
//
//   embedRun(tokenId, pos)  host      : embed a token, then run my layers
//   runHidden(x, pos)       worker    : run my layers on an incoming hidden state
//   headFromHidden(x)       host      : final norm + LM head -> logits
//   reset()                 everyone  : new conversation, caches back to position 0
//
// That is the entire contract between devices. Everything the swarm does -- the
// placement solver, the recovery path, the dashboard -- is built on top of these
// four, so they are worth getting right before any GPU or network code exists.
//
// A slice loads only the layer files it was dealt: `download only your layers` is a
// property of the loader, not a claim in a slide.

import { readFile } from "node:fs/promises";
import { join } from "node:path";

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
export function f16ToF32(h) {
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

function widen(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16ToF32(u16[i]);
  return out;
}

async function loadShard(dir, entry) {
  const buf = await readFile(join(dir, entry.file));
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
  const out = {};
  for (const t of entry.index) out[t.name] = widen(u16.subarray(t.offset / 2, t.offset / 2 + t.length));
  return out;
}

export class CpuEngine {
  static async load(dir, { layerRange, hasEmbed = false, hasHead = false, maxSeq = 512 } = {}) {
    const manifest = JSON.parse(await readFile(join(dir, "manifest.json"), "utf8"));
    const C = manifest.config;
    const [lo, hi] = layerRange || [0, C.layers];

    const e = new CpuEngine();
    e.cfg = C;
    e.lo = lo; e.hi = hi;
    e.hasEmbed = hasEmbed;
    e.hasHead = hasHead;
    e.maxSeq = maxSeq;
    e.bytesLoaded = 0;

    e.layers = [];
    for (let i = lo; i < hi; i++) {
      e.layers.push(await loadShard(dir, manifest.shards.layers[i]));
      e.bytesLoaded += manifest.shards.layers[i].bytes;
    }
    if (hasEmbed || (hasHead && C.tiedEmbeddings)) {
      e.embed = (await loadShard(dir, manifest.shards.embed)).embed;
      e.bytesLoaded += manifest.shards.embed.bytes;
    }
    if (hasHead) {
      const f = await loadShard(dir, manifest.shards.final);
      e.finalNorm = f.finalNorm;
      e.lmHead = C.tiedEmbeddings ? e.embed : f.lmHead;
      e.bytesLoaded += manifest.shards.final.bytes;
    }

    const D = C.hiddenSize, KVD = C.kvHeads * C.headDim;
    e.kCache = e.layers.map(() => new Float32Array(maxSeq * KVD));
    e.vCache = e.layers.map(() => new Float32Array(maxSeq * KVD));
    e.x = new Float32Array(D);
    e.xn = new Float32Array(D);
    e.q = new Float32Array(C.heads * C.headDim);
    e.k = new Float32Array(KVD);
    e.v = new Float32Array(KVD);
    e.attn = new Float32Array(C.heads * C.headDim);
    e.proj = new Float32Array(D);
    e.gate = new Float32Array(C.intermediate);
    e.up = new Float32Array(C.intermediate);
    return e;
  }

  reset() {
    for (const c of this.kCache) c.fill(0);
    for (const c of this.vCache) c.fill(0);
  }

  get layerCount() { return this.hi - this.lo; }

  _matvec(W, x, rows, cols, out) {
    for (let r = 0; r < rows; r++) {
      let acc = 0;
      const base = r * cols;
      for (let c = 0; c < cols; c++) acc += W[base + c] * x[c];
      out[r] = acc;
    }
    return out;
  }

  _rms(x, w, out) {
    let ss = 0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const s = 1 / Math.sqrt(ss / x.length + this.cfg.rmsEps);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * s * w[i];
    return out;
  }

  _rope(v, nHeads, pos) {
    const HD = this.cfg.headDim, half = HD >> 1, theta = this.cfg.ropeTheta;
    for (let h = 0; h < nHeads; h++) {
      const off = h * HD;
      for (let i = 0; i < half; i++) {
        const a = pos / Math.pow(theta, (2 * i) / HD);
        const c = Math.cos(a), s = Math.sin(a);
        const x0 = v[off + i], x1 = v[off + i + half];
        v[off + i] = x0 * c - x1 * s;
        v[off + i + half] = x0 * s + x1 * c;
      }
    }
  }

  // Run this device's layer range in place on this.x
  _layers(pos) {
    const C = this.cfg;
    const D = C.hiddenSize, H = C.heads, KVH = C.kvHeads, HD = C.headDim;
    const KVD = KVH * HD, FF = C.intermediate;
    const perKV = H / KVH;
    const scale = 1 / Math.sqrt(HD);
    const T = pos + 1;

    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li];
      this._rms(this.x, L.inNorm, this.xn);

      this._matvec(L.wq, this.xn, H * HD, D, this.q);
      this._matvec(L.wk, this.xn, KVD, D, this.k);
      this._matvec(L.wv, this.xn, KVD, D, this.v);

      this._rope(this.q, H, pos);
      this._rope(this.k, KVH, pos);

      this.kCache[li].set(this.k, pos * KVD);
      this.vCache[li].set(this.v, pos * KVD);

      const kC = this.kCache[li], vC = this.vCache[li];
      for (let h = 0; h < H; h++) {
        const kvh = Math.floor(h / perKV);
        const qo = h * HD, ko = kvh * HD;
        const scores = new Float32Array(T);
        let max = -Infinity;
        for (let t = 0; t < T; t++) {
          let dot = 0;
          const kb = t * KVD + ko;
          for (let d = 0; d < HD; d++) dot += this.q[qo + d] * kC[kb + d];
          scores[t] = dot * scale;
          if (scores[t] > max) max = scores[t];
        }
        let sum = 0;
        for (let t = 0; t < T; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
        for (let d = 0; d < HD; d++) {
          let acc = 0;
          for (let t = 0; t < T; t++) acc += scores[t] * vC[t * KVD + ko + d];
          this.attn[qo + d] = acc / sum;
        }
      }

      this._matvec(L.wo, this.attn, D, D, this.proj);
      for (let i = 0; i < D; i++) this.x[i] += this.proj[i];

      this._rms(this.x, L.postNorm, this.xn);
      this._matvec(L.wGate, this.xn, FF, D, this.gate);
      this._matvec(L.wUp, this.xn, FF, D, this.up);
      for (let i = 0; i < FF; i++) this.gate[i] = (this.gate[i] / (1 + Math.exp(-this.gate[i]))) * this.up[i];
      this._matvec(L.wDown, this.gate, D, FF, this.proj);
      for (let i = 0; i < D; i++) this.x[i] += this.proj[i];
    }
  }

  // --- the four calls ------------------------------------------------------

  embedRun(tokenId, pos) {
    if (!this.embed) throw new Error("this slice has no embedding table");
    const D = this.cfg.hiddenSize;
    this.x.set(this.embed.subarray(tokenId * D, tokenId * D + D));
    this._layers(pos);
    return this.x.slice();
  }

  runHidden(xIn, pos) {
    this.x.set(xIn);
    this._layers(pos);
    return this.x.slice();
  }

  headFromHidden(xIn) {
    if (!this.lmHead) throw new Error("this slice has no LM head");
    const C = this.cfg;
    this._rms(xIn, this.finalNorm, this.xn);
    const logits = new Float32Array(C.vocab);
    this._matvec(this.lmHead, this.xn, C.vocab, C.hiddenSize, logits);
    return logits;
  }
}

export function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}
