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
} from "../engine/gpu-adapter.mjs";
import { createEngine, EngineNotAvailableError } from "../engine/factory.mjs";
import { MODELS, STATUS, availableModels, getModel } from "../models/registry.mjs";

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
  ok("qwen3-0.6b is experimental, not verified", MODELS["qwen3-0.6b"].status === STATUS.EXPERIMENTAL);
  ok("qwen3.8-27b is planned", MODELS["qwen3.8-27b"].status === STATUS.PLANNED);
  ok("qwen3.8-27b names its blocker rather than a fake ETA", /qwen35\.js/.test(MODELS["qwen3.8-27b"].blockedOn));
  ok("every descriptor declares an engineKind", Object.values(MODELS).every((m) => m.engineKind));
  ok("getModel throws on an unknown id rather than returning undefined", (() => {
    try { getModel("not-a-real-model"); return false; } catch { return true; }
  })());

  const noGpu = availableModels({ webgpu: false });
  const q06 = noGpu.find((m) => m.id === "qwen3-0.6b");
  ok("a device with no WebGPU sees Qwen3 0.6B as unavailable, not experimental",
     q06.effectiveStatus === "unavailable" && /WebGPU/.test(q06.unavailableReason));
  const smol = noGpu.find((m) => m.id === "smollm2-135m");
  ok("the CPU model stays usable on a device with no WebGPU", smol.effectiveStatus === STATUS.VERIFIED);

  const noF16 = availableModels({ webgpu: true, shaderF16: false });
  ok("a WebGPU device lacking shader-f16 still sees Qwen3 0.6B as unavailable",
     noF16.find((m) => m.id === "qwen3-0.6b").effectiveStatus === "unavailable");
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

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
