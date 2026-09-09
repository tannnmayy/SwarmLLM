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

// Isomorphic: Node reads files, the browser fetches URLs, and `dir` is a path in one
// case and a URL prefix in the other. Everything below this boundary is identical, so
// the reference run in Node and the slice running on a phone execute the same code.
const inNode = typeof window === "undefined";

// Weight shards are immutable for a given model build, so a device should download
// each one exactly once and never again. Without this a phone re-fetches its whole
// slice every time it joins a room -- and during a recovery, every time the layers
// move. The Cache API is the right store: it holds Responses, survives a reload, and
// is evicted by the browser under storage pressure rather than by us.
//
// Guarded by a byte-length check against the manifest. A truncated or stale entry is
// worse than no cache at all: it would load as silently wrong weights, and the model
// would produce plausible nonsense with nothing pointing at the cause.
const CACHE = "aiswarm-weights-v1";
let cacheStats = { hits: 0, misses: 0, bytesFromCache: 0, stored: 0, storeFailed: false };

export function cacheStatsSnapshot() { return { ...cacheStats }; }
export function resetCacheStats() { cacheStats = { hits: 0, misses: 0, bytesFromCache: 0, stored: 0, storeFailed: false }; }

export async function clearWeightCache() {
  if (inNode || typeof caches === "undefined") return false;
  return caches.delete(CACHE);
}

async function readBytes(dir, file, expectBytes = 0) {
  if (inNode) {
    const { readFile } = await import("node:fs/promises");
    const { join } = await import("node:path");
    const b = await readFile(join(dir, file));
    return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  }

  const url = dir.replace(/\/$/, "") + "/" + file;
  let store = null;
  try { if (typeof caches !== "undefined") store = await caches.open(CACHE); } catch { /* private mode */ }

  if (store) {
    try {
      const hit = await store.match(url);
      if (hit) {
        const buf = await hit.arrayBuffer();
        if (!expectBytes || buf.byteLength === expectBytes) {
          cacheStats.hits++;
          cacheStats.bytesFromCache += buf.byteLength;
          return new Uint8Array(buf);
        }
        await store.delete(url);          // stale build: drop it and refetch
      }
    } catch { /* fall through to the network */ }
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${file}`);
  const buf = await res.arrayBuffer();
  if (expectBytes && buf.byteLength !== expectBytes) {
    throw new Error(`${file}: expected ${expectBytes} bytes, got ${buf.byteLength}`);
  }
  cacheStats.misses++;
  if (store) {
    // Quota refusals are normal on a phone. Losing the cache is a slower next join,
    // never a wrong answer, so swallow it and carry on.
    try {
      await store.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
      cacheStats.stored++;
    } catch { cacheStats.storeFailed = true; }
  }
  return new Uint8Array(buf);
}

async function readJSON(dir, file) {
  if (inNode) {
    const { readFile } = await import("node:fs/promises");
    const { join } = await import("node:path");
    return JSON.parse(await readFile(join(dir, file), "utf8"));
  }
  const res = await fetch(dir.replace(/\/$/, "") + "/" + file);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${file}`);
  return res.json();
}

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
  const bytes = await readBytes(dir, entry.file, entry.bytes || 0);
  // .slice() rather than a view: the fetched buffer may not be 2-byte aligned
  const u16 = new Uint16Array(bytes.slice().buffer);
  const out = {};
  for (const t of entry.index) out[t.name] = widen(u16.subarray(t.offset / 2, t.offset / 2 + t.length));
  return out;
}

export class CpuEngine {
  static async load(dir, { layerRange, hasEmbed = false, hasHead = false, maxSeq = 512, onProgress = null } = {}) {
    const manifest = await readJSON(dir, "manifest.json");
    const C = manifest.config;
    const [lo, hi] = layerRange || [0, C.layers];

    resetCacheStats();
    const e = new CpuEngine();
    e.cfg = C;
    e.manifest = manifest;
    e.lo = lo; e.hi = hi;
    e.hasEmbed = hasEmbed;
    e.hasHead = hasHead;
    e.maxSeq = maxSeq;
    e.bytesLoaded = 0;

    // total up front so progress is a real fraction, not a spinner
    let todo = 0;
    for (let i = lo; i < hi; i++) todo += manifest.shards.layers[i].bytes;
    if (hasEmbed || (hasHead && C.tiedEmbeddings)) todo += manifest.shards.embed.bytes;
    if (hasHead) todo += manifest.shards.final.bytes;
    const tick = () => onProgress && onProgress(e.bytesLoaded / todo, e.bytesLoaded, todo);

    e.layers = [];
    for (let i = lo; i < hi; i++) {
      e.layers.push(await loadShard(dir, manifest.shards.layers[i]));
      e.bytesLoaded += manifest.shards.layers[i].bytes;
      tick();
    }
    if (hasEmbed || (hasHead && C.tiedEmbeddings)) {
      e.embed = (await loadShard(dir, manifest.shards.embed)).embed;
      e.bytesLoaded += manifest.shards.embed.bytes;
      tick();
    }
    if (hasHead) {
      const f = await loadShard(dir, manifest.shards.final);
      e.finalNorm = f.finalNorm;
      e.lmHead = C.tiedEmbeddings ? e.embed : f.lmHead;
      e.bytesLoaded += manifest.shards.final.bytes;
      tick();
    }
    e.cache = cacheStatsSnapshot();

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
