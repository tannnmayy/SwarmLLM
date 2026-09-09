// The model ladder: one descriptor per model the room can be asked to run.
//
// This replaces room.js's old single-entry `MODELS = { "smollm2-135m": {...} }`.
// A descriptor is the source of truth for how to load a model, not a hardcoded
// layer count or byte size — engine/gpu-adapter.mjs still parses the real GGUF
// header at load time and refuses to proceed if what it finds disagrees with
// what is declared here.
//
// Status is a claim about evidence, not a wish:
//   verified     - self-test + one-device golden + split golden + recovery all
//                  passed on recorded hardware (see CHECKLIST.md / README.md).
//   experimental - the loader/engine can be tried; no recorded cross-device
//                  proof yet.
//   planned      - visible as a roadmap item; the engine it needs is not
//                  vendored, so it cannot be started at all.
// `unavailable` is not stored here — it is computed per-device by availableModels()
// below, because it depends on what the browser in front of it can actually do.

export const STATUS = {
  VERIFIED: "verified",
  EXPERIMENTAL: "experimental",
  PLANNED: "planned",
};

const GB = 2 ** 30;

export const MODELS = {
  "smollm2-135m": {
    id: "smollm2-135m",
    label: "SmolLM2 135M · CPU",
    status: STATUS.VERIFIED,
    engineKind: "cpu-smollm",
    dir: "/models/smollm2-135m",
    layers: 30,
    maxSeqDefault: 512,
    wireDtype: "f32",
    capabilityRequirements: { webgpu: false },
    why: "The working two-device demo: 134 tests, a recorded live run at 4.99 tok/s, " +
         "and a recovery that finishes the same sentence after a worker drops.",
  },

  "qwen3-0.6b": {
    id: "qwen3-0.6b",
    label: "Qwen3 0.6B · Q8_0",
    status: STATUS.EXPERIMENTAL,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    modelUrl: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    configUrl: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json",
    tokenizerUrl: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json",
    expectedFormat: "Q8_0",
    maxSeqDefault: 512,
    wireDtype: "f16",
    sourceRevision: "not yet fetched in this build — record exact byte length/SHA-256 on first successful load",
    minRoomEnvelopeBytes: Math.round(0.8 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "First real Qwen/GGUF/WebGPU milestone: 28 dense layers, and the imported " +
         "DenseEngine already handles Qwen3's QK-norm.",
  },

  "qwen3-1.7b": {
    id: "qwen3-1.7b",
    label: "Qwen3 1.7B · Q8_0",
    status: STATUS.PLANNED,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    modelUrl: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    configUrl: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json",
    tokenizerUrl: "https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/tokenizer.json",
    expectedFormat: "Q8_0",
    maxSeqDefault: 512,
    wireDtype: "f16",
    sourceRevision: "not yet fetched in this build — record exact byte length/SHA-256 on first successful load",
    minRoomEnvelopeBytes: Math.round(2.0 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "Same dense code path as 0.6B; the first real multi-device capacity demonstration.",
    blockedOn: "Qwen3 0.6B must reach Verified first (Phase 6 of the blueprint promotes one rung at a time).",
  },

  "qwen3-4b": {
    id: "qwen3-4b",
    label: "Qwen3 4B · Q8_0",
    status: STATUS.PLANNED,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    modelUrl: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    configUrl: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/config.json",
    tokenizerUrl: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json",
    expectedFormat: "Q8_0",
    maxSeqDefault: 512,
    wireDtype: "f16",
    sourceRevision: "not yet fetched in this build — record exact byte length/SHA-256 on first successful load",
    minRoomEnvelopeBytes: Math.round(4.6 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "Still dense; 36 layers. A scale-up of the same runtime, not a new architecture.",
    blockedOn: "Qwen3 1.7B must reach Verified first.",
  },

  "qwen3.8-27b": {
    id: "qwen3.8-27b",
    label: "Qwen 3.8 27B · Q4_0 (hybrid)",
    status: STATUS.PLANNED,
    engineKind: "qwen35-gguf",
    architecture: "qwen35",
    modelUrl: null,
    configUrl: null,
    tokenizerUrl: null,
    expectedFormat: "Q4_0",
    maxSeqDefault: 512,
    wireDtype: "f16",
    sourceRevision: null,
    minRoomEnvelopeBytes: Math.round(16.5 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "The end-state capacity demo: 64 blocks (48 Gated DeltaNet + 16 full attention) with MTP.",
    blockedOn: "engine/upstream/qwen35.js and wgsl/qwen35.js are not vendored yet (see UPSTREAM.md), " +
               "and the scheduler only models uniform dense layers (see scheduler/plan.js). " +
               "Cannot be started at all until both land — this is Phase 7/8 of the blueprint, not a URL swap.",
  },
};

// A model can be nominally "verified" or "experimental" and still be unusable on
// *this* device — a phone with no WebGPU cannot try Qwen3 0.6B no matter how well
// tested it is elsewhere. This is the per-device view the UI should render from,
// per the blueprint's "Unavailable on this device" tier (section 6.3).
export function availableModels(caps = {}) {
  return Object.values(MODELS).map((m) => {
    const req = m.capabilityRequirements || {};
    let unavailableReason = null;
    if (req.webgpu && caps.webgpu === false) unavailableReason = "this device has no WebGPU";
    else if (req.shaderF16 && caps.webgpu && caps.shaderF16 === false) unavailableReason = "this GPU/driver lacks shader-f16";

    return {
      ...m,
      effectiveStatus: unavailableReason ? "unavailable" : m.status,
      unavailableReason,
    };
  });
}

export function getModel(id) {
  const m = MODELS[id];
  if (!m) throw new Error(`unknown model id: ${id}`);
  return m;
}
