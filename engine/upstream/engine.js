// Imported from SwarmLLM — https://github.com/Nehanth/swarmllm
// Copyright (c) 2026 Nehanth Narendrula. MIT License; see THIRD_PARTY_NOTICES.md.
// Upstream commit 1c9763fcd7eb9c42865cd5937dceba678fe9d9a5. Functionally unmodified; this header is the only addition.

// SwarmLLM engine: public entry point. The implementation lives in focused
// modules; this file re-exports them so `import { ... } from "./engine.js"`
// keeps working for the room page, tests and benchmarks. See docs/kernels.md.
export { WGSL } from "./wgsl/base.js";
export { coopWGSL, probeUnpack } from "./wgsl/coop.js";
export { DenseEngine } from "./dense.js";
export { autotuneCoop } from "./autotune.js";
export { makeTokenizer } from "./tokenizer.js";
export { argmax } from "./sampling.js";
export { quantizeQ4 } from "./quant.js";
export { parseSafetensors, tensorF32, shardTensorNames, fetchModelShard, weightsFromSafetensors } from "./safetensors.js";
export { gpuSelfTest, kernelMicroTests } from "./selftest.js";
