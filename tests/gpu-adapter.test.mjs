// Phase 2 gate (the engine factory / GPU adapter boundary), the part of it that
// does not need a real GPU:
//
//   node tests/gpu-adapter.test.mjs
//
// GGUF-metadata mapping, descriptor/range validation, and factory dispatch are
// pure functions with no fetch and no WebGPU device, so they belong in `npm test`
// like everything else. The numerical GPU path (does a loaded model actually
// compute the right thing) cannot run in Node — there is no WebGPU here — so it
// has its own browser page: gpu-adapter-test.html, run manually against real
// hardware the same way gpu-test.html is.

import {
  cfgFromGGUFMeta, eosFromMeta, validateDescriptor, validateLayerRange, GpuCapabilityError,
  probeHeader, modelSpecFromGGUF, checkDeviceLost,
} from "../engine/gpu-adapter.mjs";
import { createEngine, EngineNotAvailableError } from "../engine/factory.mjs";
import { MODELS, STATUS, availableModels, getModel } from "../models/registry.mjs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};
const throws = async (name, fn, matches) => {
  try {
    await fn();
    ok(name, false, "did not throw");
  } catch (e) {
    ok(name, !matches || matches.test(e.message), e.message);
  }
};

// ---------------------------------------------------------------- registry
console.log("\nmodel registry");
{
  ok("smollm2-135m is verified", MODELS["smollm2-135m"].status === STATUS.VERIFIED);
  ok("qwen3-0.6b is verified (exact token-ID match vs an independent reference, Phase B)",
     MODELS["qwen3-0.6b"].status === STATUS.VERIFIED);
  ok("qwen3.8-27b is planned", MODELS["qwen3.8-27b"].status === STATUS.PLANNED);
  ok("qwen3.8-27b names its blocker rather than a fake ETA", /qwen35\.js/.test(MODELS["qwen3.8-27b"].blockedOn));
  ok("every descriptor declares an engineKind", Object.values(MODELS).every((m) => m.engineKind));
  ok("getModel throws on an unknown id rather than returning undefined", (() => {
    try { getModel("not-a-real-model"); return false; } catch { return true; }
  })());

  const noGpu = availableModels({ webgpu: false });
  const q06 = noGpu.find((m) => m.id === "qwen3-0.6b");
  ok("a device with no WebGPU sees Qwen3 0.6B as unavailable, regardless of its base status",
     q06.effectiveStatus === "unavailable" && /WebGPU/.test(q06.unavailableReason));
  const smol = noGpu.find((m) => m.id === "smollm2-135m");
  ok("the CPU model stays usable on a device with no WebGPU", smol.effectiveStatus === STATUS.VERIFIED);

  const noF16 = availableModels({ webgpu: true, shaderF16: false });
  ok("a WebGPU device lacking shader-f16 still sees Qwen3 0.6B as unavailable",
     noF16.find((m) => m.id === "qwen3-0.6b").effectiveStatus === "unavailable");

  // Phase D: a reasoning model needs more generation headroom than SmolLM2's demo
  // default, or its own <think> block gets cut off mid-thought (observed live).
  ok("SmolLM2 keeps its original, speed-tuned generation cap",
     MODELS["smollm2-135m"].maxTokensDefault === 60);
  ok("every dense-gguf Qwen3 descriptor sets a generation cap higher than 60",
     ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"].every((id) => MODELS[id].maxTokensDefault > 60));

  // Phase F: 1.7B passed the same full gate 0.6B did.
  ok("qwen3-1.7b is verified (full Phase F gate, incl. an external reference match)",
     MODELS["qwen3-1.7b"].status === STATUS.VERIFIED);
  ok("qwen3-1.7b points at a local mirror, not a remote CDN, like 0.6B does",
     MODELS["qwen3-1.7b"].modelUrl.startsWith("/models/"));
  ok("qwen3-1.7b records a real byte length and sha256, not a placeholder",
     /\d{9,}\s*bytes/.test(MODELS["qwen3-1.7b"].sourceRevision) &&
     /sha256:[0-9a-f]{64}/.test(MODELS["qwen3-1.7b"].sourceRevision));

  // Guards a bug fixed on 2026-09-10: qwen3-0.6b declared wireDtype "f16" when
  // room.js actually sends f32 for hidden=1024. Nothing reads the field, so the
  // error was invisible -- and it inverted the stated reason 0.6B's split output
  // is bit-identical to its solo output. tools/probe-gguf.mjs checks the declared
  // value against chooseEncoding() for real model files; this pins the one whose
  // value was wrong. See docs/gpu-reports/2026-09-10-phase-f-qwen3-1.7b.json.
  ok("qwen3-0.6b declares the f32 wire it actually uses (hidden 1024 fits one slice)",
     MODELS["qwen3-0.6b"].wireDtype === "f32");
  ok("qwen3-1.7b declares the lossy f16 wire it actually uses (hidden 2048)",
     MODELS["qwen3-1.7b"].wireDtype === "f16");
}

// ---------------------------------------------------------------- GGUF metadata mapping
console.log("\nGGUF metadata -> DenseEngine cfg");
{
  const G = {
    meta: {
      "general.architecture": "qwen3",
      "qwen3.embedding_length": 1024,
      "qwen3.block_count": 28,
      "qwen3.feed_forward_length": 3072,
      "qwen3.attention.head_count": 16,
      "qwen3.attention.head_count_kv": 8,
      "qwen3.attention.key_length": 128,
      "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
      "qwen3.rope.freq_base": 1000000,
    },
    tensors: { "token_embd.weight": { shape: [151936, 1024] } },
  };
  const cfg = cfgFromGGUFMeta(G, { archHint: "qwen3" });
  ok("hidden_size", cfg.hidden_size === 1024);
  ok("num_hidden_layers", cfg.num_hidden_layers === 28);
  ok("num_attention_heads", cfg.num_attention_heads === 16);
  ok("num_key_value_heads", cfg.num_key_value_heads === 8);
  ok("head_dim from attention.key_length, not hidden/heads", cfg.head_dim === 128);
  ok("vocab_size derived from the embedding tensor's own shape", cfg.vocab_size === 151936,
     "got " + cfg.vocab_size);
  ok("rms_norm_eps passed through", cfg.rms_norm_eps === 1e-6);
  ok("rope_theta passed through", cfg.rope_theta === 1000000);

  // The header is always parsed with skipTokenizer:true (see fetchHeader in
  // gpu-adapter.mjs), so tokenizer.ggml.tokens is never present at this point —
  // vocab_size must not silently depend on it.
  const G2 = { meta: { ...G.meta }, tensors: {} };
  await throws("vocab_size cannot be found with neither an explicit key nor an embed tensor",
    () => cfgFromGGUFMeta(G2, { archHint: "qwen3" }), /vocab_size/);

  const noHeadDimKey = { meta: { ...G.meta }, tensors: G.tensors };
  delete noHeadDimKey.meta["qwen3.attention.key_length"];
  const cfg2 = cfgFromGGUFMeta(noHeadDimKey, { archHint: "qwen3" });
  ok("head_dim falls back to hidden/heads when attention.key_length is absent",
     cfg2.head_dim === 1024 / 16);

  await throws("missing general.architecture and no hint throws, rather than mapping garbage",
    () => cfgFromGGUFMeta({ meta: {}, tensors: {} }),
    /architecture/);

  const missingBlockCount = { meta: { ...G.meta }, tensors: G.tensors };
  delete missingBlockCount.meta["qwen3.block_count"];
  await throws("a missing required field is named in the error, not just \"invalid\"",
    () => cfgFromGGUFMeta(missingBlockCount, { archHint: "qwen3" }),
    /num_hidden_layers/);

  ok("eosFromMeta reads tokenizer.ggml.eos_token_id",
     eosFromMeta({ "tokenizer.ggml.eos_token_id": 151645 }) === 151645);
  ok("eosFromMeta returns null rather than NaN/undefined when absent",
     eosFromMeta({}) === null);
}

// ---------------------------------------------------------------- descriptor / range validation
console.log("\ndescriptor and range validation");
{
  const descriptor = { id: "qwen3-0.6b", architecture: "qwen3", expectedFormat: "Q8_0" };
  const cfg = { num_hidden_layers: 28 };
  const G = { meta: { "general.architecture": "qwen3" }, tensors: { "blk.0.attn_q.weight": { ggmlType: 8 } } }; // 8 = GGML_Q8_0

  ok("matching architecture and format pass silently",
     (() => { try { validateDescriptor(descriptor, G, cfg); return true; } catch { return false; } })());

  await throws("architecture mismatch is rejected before any weight bytes are fetched",
    () => validateDescriptor(descriptor, { meta: { "general.architecture": "llama" }, tensors: G.tensors }, cfg),
    /architecture mismatch/);

  await throws("quantization mismatch is rejected (a Q4_0 file loaded as a Q8_0 descriptor)",
    () => validateDescriptor(descriptor, { meta: G.meta, tensors: { "blk.0.attn_q.weight": { ggmlType: 2 } } }, cfg), // 2 = GGML_Q4_0
    /quantization mismatch/);

  const [lo, hi] = validateLayerRange([4, 12], cfg);
  ok("a valid range passes through unchanged", lo === 4 && hi === 12);
  ok("layerRange defaults to the whole model when omitted",
     validateLayerRange(null, cfg).join(",") === "0,28");
  await throws("a range past the model's own layer count is rejected",
    () => validateLayerRange([20, 40], cfg), /invalid/);
  await throws("an empty/reversed range is rejected",
    () => validateLayerRange([10, 10], cfg), /invalid/);
}

// ---------------------------------------------------------------- header probing + real-byte spec
console.log("\nprobeHeader + modelSpecFromGGUF (against the real tiny Q8_0 fixture)");
{
  const fixture = await readFile(join(ROOT, "tests/fixtures/tiny-qwen3-q8.gguf"));
  const ab = fixture.buffer.slice(fixture.byteOffset, fixture.byteOffset + fixture.byteLength);
  const realFetch = globalThis.fetch;
  // A minimal stand-in for an HTTP range server, backed by the on-disk fixture —
  // exercises the exact fetchHeader()/probeHeader() code path room.js calls at
  // join, without needing a running dev server for a unit test.
  globalThis.fetch = async (url, opts) => {
    const range = opts?.headers?.Range;
    if (!range) return { status: 200, ok: true, arrayBuffer: async () => ab };
    const m = /bytes=(\d+)-(\d+)/.exec(range);
    const start = +m[1], end = Math.min(+m[2], ab.byteLength - 1);
    if (start >= ab.byteLength) return { status: 416, ok: false };
    const body = ab.slice(start, end + 1);
    return { status: 206, ok: true, arrayBuffer: async () => body };
  };
  try {
    const descriptor = { id: "test-tiny", label: "tiny test fixture", architecture: "qwen3", expectedFormat: "Q8_0", modelUrl: "http://fixture.test/tiny-qwen3-q8.gguf" };
    const { header, cfg } = await probeHeader(descriptor);
    ok("probeHeader falls back past a 416 on a file smaller than the probe window", cfg.num_hidden_layers === 2);
    ok("probeHeader's cfg matches the fixture's known shape", cfg.hidden_size === 64 && cfg.vocab_size === 96);

    const spec = modelSpecFromGGUF(header, cfg, descriptor, { maxSeq: 8 });
    ok("spec.layers / spec.hidden match the fixture", spec.layers === 2 && spec.hidden === 64);
    ok("spec.tiedEmbeddings is true (the fixture has no output.weight)", spec.tiedEmbeddings === true);
    ok("spec.layerBytes comes from real tensor sizes, not a bytes-per-weight guess",
       spec.layerBytes > 0 && Number.isInteger(spec.layerBytes));
    ok("spec.embedBytes matches the embedding tensor's real byte length", spec.embedBytes === 6528,
       "got " + spec.embedBytes);
    ok("spec.headRatio is headMACs/layerMACs, not a placeholder",
       Math.abs(spec.headRatio - spec.headMACs / spec.layerMACs) < 1e-9);
    ok("spec.kvBytesPerLayer scales with the maxSeq passed in", spec.kvBytesPerLayer === 2 * 8 * (2 * 16) * 4,
       "got " + spec.kvBytesPerLayer);

    await throws("modelSpecFromGGUF refuses a header with no embedding tensor",
      () => modelSpecFromGGUF({ tensors: {} }, cfg, descriptor, { maxSeq: 8 }),
      /token_embd/);
  } finally {
    globalThis.fetch = realFetch;
  }
}

// ---------------------------------------------------------------- factory dispatch
console.log("\nengine factory dispatch");
{
  await throws("the hybrid Qwen 3.8 engine names why it cannot run, not just \"unsupported\"",
    () => createEngine(getModel("qwen3.8-27b"), {}),
    /not vendored yet/);
  const err = await (async () => { try { await createEngine(getModel("qwen3.8-27b"), {}); } catch (e) { return e; } })();
  ok("that failure is a typed EngineNotAvailableError, not a generic throw",
     err instanceof EngineNotAvailableError);

  await throws("an unknown engine kind is rejected rather than silently doing nothing",
    () => createEngine({ id: "x", label: "x", engineKind: "made-up-kind" }, {}),
    /unknown engine kind/);

  ok("GpuCapabilityError carries a machine-readable kind, not just prose",
     new GpuCapabilityError("no-webgpu", "no gpu").kind === "no-webgpu");
}

// ---------------------------------------------------------------- device-lost surfacing (Phase A)
console.log("\ncheckDeviceLost (surfacing device.lost mid-generation, not just at load)");
{
  ok("a device that was never lost passes silently", (() => {
    try { checkDeviceLost({}); return true; } catch { return false; }
  })());
  await throws("a lost device throws a message naming the reason",
    () => checkDeviceLost({ __lost: { reason: "destroyed", message: "out of memory" } }),
    /GPU device lost \(destroyed\): out of memory/);
  await throws("a lost device with no message still throws something actionable",
    () => checkDeviceLost({ __lost: { reason: "unknown" } }),
    /no detail/);
  ok("the thrown message says the worker must be re-planned around, not just 'lost'",
     await (async () => {
       try { checkDeviceLost({ __lost: { reason: "destroyed" } }); return false; }
       catch (e) { return /re-planned around/.test(e.message); }
     })());
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
