// GpuEngineAdapter: the boundary between the room runtime and the imported
// upstream DenseEngine (engine/upstream/dense.js).
//
// room/room.js and the scheduler only know CpuEngine's four-call contract:
//
//   embedRun(tokenId, pos)   host   : embed a token, then run my layers
//   runHidden(x, pos)        worker : run my layers on an incoming hidden state
//   headFromHidden(x)        host   : final norm + LM head -> logits
//   reset()                  all    : new conversation, caches back to position 0
//
// DenseEngine already exposes exactly this shape (see UPSTREAM.md), so most of
// this file is not numerical glue -- it is everything DenseEngine.create() does
// NOT do for you: acquiring a WebGPU device with a classified error instead of a
// thrown string, fetching only a GGUF's header before committing to a multi-GB
// download, mapping GGUF metadata onto the {hidden_size, num_attention_heads, ...}
// shape the engine expects, range-fetching only the assigned layers plus
// host-only tensors, validating that what the file actually contains matches
// what the model descriptor (models/registry.mjs) claims, and caching validated
// byte ranges by (source URL, byte offset, byte length) so a rejoin does not
// re-download a device's own layers.
//
// Kept deliberately separate from engine/upstream/: everything in this file is
// locally owned and can change freely; everything it imports is vendored,
// unmodified, third-party code (see THIRD_PARTY_NOTICES.md).

import { DenseEngine } from "./upstream/dense.js";
import { parseGGUFHeader, ggufWeights, ggufShardBytes, GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_0 } from "./upstream/gguf.js";

const CACHE = "aiswarm-gguf-v1";

// Thrown for anything a browser/device is responsible for, as opposed to a bad
// model file or a bad plan. The `kind` lets the UI show a distinct message per
// failure instead of one generic "something went wrong" (blueprint section 6.3,
// "Unavailable on this device" must say why).
export class GpuCapabilityError extends Error {
  constructor(kind, message) {
    super(message);
    this.name = "GpuCapabilityError";
    this.kind = kind;
  }
}

// Phase 1's device-acquisition probe (gpu-test.html), reused rather than
// duplicated: the adapter needs the exact same adapter/device/feature dance the
// self-test already does, with the same distinct failure messages.
export async function acquireDevice({ requireShaderF16 = false } = {}) {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new GpuCapabilityError("no-webgpu", "This browser has no WebGPU. Chrome or Edge on desktop is the tested path.");
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (e) {
    throw new GpuCapabilityError("adapter-error", String(e?.message || e));
  }
  if (!adapter) {
    throw new GpuCapabilityError("no-adapter", "navigator.gpu exists but no adapter was returned. WebGPU may be disabled or the GPU is blocklisted.");
  }

  const shaderF16 = adapter.features.has("shader-f16");
  if (requireShaderF16 && !shaderF16) {
    throw new GpuCapabilityError("no-shader-f16", "This GPU/driver does not support the shader-f16 feature this model requires.");
  }

  let device;
  try {
    device = await adapter.requestDevice({ requiredFeatures: shaderF16 ? ["shader-f16"] : [] });
  } catch (e) {
    throw new GpuCapabilityError("device-error", String(e?.message || e));
  }

  let info = {};
  try { info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}); } catch { /* masked */ }

  return {
    device,
    caps: {
      webgpu: true,
      shaderF16,
      vendor: info.vendor || "?",
      architecture: info.architecture || "?",
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  };
}

// Everything else in this module is pure (no fetch, no GPU) and Node-testable —
// see tests/gpu-adapter.test.mjs. Only acquireDevice() and GpuEngineAdapter.load()
// need a real browser, which is why they are kept to the smallest surface possible.

const ARCH_TYPE_NAME = { [GGML_F32]: "F32", [GGML_F16]: "F16", [GGML_Q8_0]: "Q8_0", [GGML_Q4_0]: "Q4_0" };

// Map GGUF metadata onto the shape engine/upstream/dense.js expects. GGUF keys
// upstream are namespaced by architecture ("qwen3.embedding_length", not a fixed
// schema), so this has to read the architecture out of the file rather than
// assume one — see llama.cpp's GGUF key convention, which is what every
// converter (including the official Qwen3 GGUF releases) follows.
//
// Takes the whole parsed header (`{ meta, tensors }`), not just `meta`: header
// fetches always parse with `skipTokenizer: true` (a 150k-entry token array is
// pure waste for a value this adapter only wants once per load), which drops
// `tokenizer.ggml.tokens` — so vocab_size cannot depend on that array. The
// reliable source is the embedding tensor's own shape, which is always present.
export function cfgFromGGUFMeta(G, { archHint } = {}) {
  const meta = G.meta || G;               // tolerate a bare meta object, e.g. from a unit test
  const tensors = G.tensors || {};
  const arch = meta["general.architecture"] || archHint;
  if (!arch) throw new Error("GGUF file has no general.architecture key and no architecture hint was given");
  const p = (suffix) => meta[`${arch}.${suffix}`];

  const hidden_size = p("embedding_length");
  const num_hidden_layers = p("block_count");
  const intermediate_size = p("feed_forward_length");
  const num_attention_heads = p("attention.head_count");
  const num_key_value_heads = p("attention.head_count_kv") ?? num_attention_heads;
  const head_dim = p("attention.key_length") || (hidden_size && num_attention_heads ? hidden_size / num_attention_heads : undefined);
  const rms_norm_eps = p("attention.layer_norm_rms_epsilon") ?? 1e-6;
  const rope_theta = p("rope.freq_base") ?? 10000;

  let vocab_size = p("vocab_size");
  if (vocab_size == null && tensors["token_embd.weight"]) vocab_size = tensors["token_embd.weight"].shape[0];
  if (vocab_size == null && Array.isArray(meta["tokenizer.ggml.tokens"])) vocab_size = meta["tokenizer.ggml.tokens"].length;

  const fields = { hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, vocab_size };
  const missing = Object.entries(fields).filter(([, v]) => v == null).map(([k]) => k);
  if (missing.length) throw new Error(`GGUF metadata for architecture "${arch}" is missing: ${missing.join(", ")}`);

  return { ...fields, rms_norm_eps, rope_theta, architecture: arch };
}

export function eosFromMeta(meta) {
  const id = meta["tokenizer.ggml.eos_token_id"];
  return Number.isFinite(id) ? id : null;
}

// Refuse a range that does not fit the model this GGUF actually describes, and a
// quantization the file does not actually contain — a wrong scale layout
// produces plausible but incorrect language (blueprint section 3, Phase 7 task 3),
// so this must fail loudly before any weight bytes are fetched.
export function validateDescriptor(descriptor, G, cfg) {
  const gotArch = G.meta["general.architecture"];
  if (descriptor.architecture && gotArch !== descriptor.architecture) {
    throw new Error(`architecture mismatch: descriptor "${descriptor.id}" expects "${descriptor.architecture}", the GGUF file reports "${gotArch}"`);
  }

  if (descriptor.expectedFormat) {
    const probe = G.tensors["blk.0.attn_q.weight"] || G.tensors["blk.0." + "ffn_gate.weight"];
    if (probe) {
      const gotFormat = ARCH_TYPE_NAME[probe.ggmlType] || `ggml-type-${probe.ggmlType}`;
      if (gotFormat !== descriptor.expectedFormat) {
        throw new Error(`quantization mismatch: descriptor "${descriptor.id}" expects ${descriptor.expectedFormat}, the GGUF file's weights are ${gotFormat}`);
      }
    }
  }
}

export function validateLayerRange(layerRange, cfg) {
  const [lo, hi] = layerRange || [0, cfg.num_hidden_layers];
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 0 || hi > cfg.num_hidden_layers || lo >= hi) {
    throw new Error(`layer range [${lo}, ${hi}) is invalid for a ${cfg.num_hidden_layers}-layer model`);
  }
  return [lo, hi];
}

// Range-fetch one tensor's bytes, backed by the Cache API keyed by (source URL,
// byte offset, byte length) — not a friendly model name — so a truncated or
// stale entry cannot be silently reused as though it were a different range or
// a different model build (blueprint section 4.2 step 5).
function openRange(url, cacheStore) {
  return async (info) => {
    const key = `${url}?range=${info.byteOffset}-${info.byteOffset + info.byteLength - 1}`;
    if (cacheStore) {
      try {
        const hit = await cacheStore.match(key);
        if (hit) {
          const buf = await hit.arrayBuffer();
          if (buf.byteLength === info.byteLength) return new Uint8Array(buf);
          await cacheStore.delete(key); // stale/truncated: fall through to refetch
        }
      } catch { /* private mode */ }
    }
    const res = await fetch(url, { headers: { Range: `bytes=${info.byteOffset}-${info.byteOffset + info.byteLength - 1}` } });
    if (!res.ok && res.status !== 206) throw new Error(`HTTP ${res.status} range-fetching ${info.name}`);
    const buf = await res.arrayBuffer();
    if (buf.byteLength !== info.byteLength) throw new Error(`${info.name}: expected ${info.byteLength} bytes, got ${buf.byteLength}`);
    if (cacheStore && res.status === 206) {
      try { await cacheStore.put(key, new Response(buf.slice(0), { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota */ }
    }
    return new Uint8Array(buf);
  };
}

// The header (metadata + tensor table) sits at the front of the file. 16 MiB is
// comfortably more than any Qwen3-family GGUF header observed upstream, and a
// short range request means a device never pulls the multi-GB tensor payload
// just to read metadata (blueprint section 4.2 step 1).
const HEADER_PROBE_BYTES = 16 * 2 ** 20;

async function fetchHeader(url) {
  const res = await fetch(url, { headers: { Range: `bytes=0-${HEADER_PROBE_BYTES - 1}` } });
  // A file smaller than the probe window (real GGUFs never are; test fixtures can
  // be) makes a spec-compliant server answer 416 rather than clamp the range —
  // that is proof the server DOES support Range, and "the whole file is the
  // header", not "range-fetch is broken". Anything else that is not 206 means the
  // server ignored Range outright, which for a real multi-GB model is exactly the
  // failure this check exists to catch.
  if (res.status === 416) {
    const whole = await fetch(url);
    if (!whole.ok) throw new Error(`HTTP ${whole.status} fetching GGUF header from ${url}`);
    return parseGGUFHeader(await whole.arrayBuffer(), { skipTokenizer: true });
  }
  if (!res.ok && res.status !== 206) throw new Error(`HTTP ${res.status} fetching GGUF header from ${url}`);
  if (res.status !== 206) {
    throw new Error(`server ignored the Range request for ${url} (got a full ${res.status} response) — ` +
      "range-fetch is required so a device only downloads its own assigned layers");
  }
  const buf = await res.arrayBuffer();
  return parseGGUFHeader(buf, { skipTokenizer: true });
}

export class GpuEngineAdapter {
  // opts: { layerRange, hasEmbed, hasHead, maxSeq, onProgress, device?, caps? }
  // A caller may pass an already-acquired `device`/`caps` (e.g. one probed once
  // at join and reused for every model on that device); otherwise the adapter
  // acquires its own.
  static async load(descriptor, opts = {}) {
    const { layerRange, hasEmbed = false, hasHead = false, maxSeq = null, onProgress = null } = opts;
    let { device, caps } = opts;
    if (!device) {
      const acquired = await acquireDevice({ requireShaderF16: !!descriptor.capabilityRequirements?.shaderF16 });
      device = acquired.device;
      caps = acquired.caps;
    }

    if (!descriptor.modelUrl) {
      throw new Error(`${descriptor.label} has no modelUrl configured yet — it is a roadmap entry, not a loadable model`);
    }

    const G = await fetchHeader(descriptor.modelUrl);
    const cfg = cfgFromGGUFMeta(G, { archHint: descriptor.architecture });
    validateDescriptor(descriptor, G, cfg);
    const [lo, hi] = validateLayerRange(layerRange, cfg);

    let cacheStore = null;
    try { if (typeof caches !== "undefined") cacheStore = await caches.open(CACHE); } catch { /* private mode */ }
    const bytesOf = openRange(descriptor.modelUrl, cacheStore);

    const total = ggufShardBytes(G, { lo, hi, hasEmbed, hasHead });
    let fetched = 0;
    const tick = () => onProgress && onProgress(total ? fetched / total : 1, fetched, total);
    const weights = await ggufWeights(G, bytesOf, { lo, hi, hasEmbed, hasHead }, (f) => { fetched = f; tick(); });

    const resolvedMaxSeq = maxSeq || descriptor.maxSeqDefault || 512;

    device.pushErrorScope?.("out-of-memory");
    let dense;
    try {
      dense = await DenseEngine.create({ device, cfg, weights, layerRange: [lo, hi], hasEmbed, hasHead, maxSeq: resolvedMaxSeq });
    } finally {
      const oom = await device.popErrorScope?.();
      if (oom) throw new Error(`GPU ran out of memory loading ${descriptor.label} layers [${lo}, ${hi}) — this device pledged more than its GPU can hold`);
    }

    const e = new GpuEngineAdapter();
    e.dense = dense;
    e.device = device;
    e.caps = caps;
    e.descriptor = descriptor;
    e.lo = lo; e.hi = hi;
    e.maxSeq = resolvedMaxSeq;
    e.bytesLoaded = total;
    e.cache = null; // per-tensor Cache API identity here, not one shard blob like CpuEngine's
    e.cfg = { ...cfg, eos: eosFromMeta(G.meta) };
    return e;
  }

  get layerCount() { return this.hi - this.lo; }
  reset() { this.dense.reset(); }
  async embedRun(tokenId, pos) { return this.dense.embedRun(tokenId, pos); }
  async runHidden(x, pos) { return this.dense.runHidden(x, pos); }
  async headFromHidden(x) { return this.dense.headFromHidden(x); }
  dispose() { this.device?.destroy?.(); }
}
