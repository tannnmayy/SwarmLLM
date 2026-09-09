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
    maxTokensDefault: 60,
    wireDtype: "f32",
    capabilityRequirements: { webgpu: false },
    why: "The working two-device demo: 134 tests, a recorded live run at 4.99 tok/s, " +
         "and a recovery that finishes the same sentence after a worker drops.",
  },

  "qwen3-0.6b": {
    id: "qwen3-0.6b",
    label: "Qwen3 0.6B · Q8_0",
    // Promoted from EXPERIMENTAL on 2026-09-10 (IMPLEMENTATION_PLAN.md Phase B):
    // this room's real Q8_0/WebGPU output matched an independent reference
    // implementation exactly, both on prompt tokenization (36/36 tokens) and on
    // 12/12 greedy-decoded output tokens. See the `why` field below for the one
    // caveat worth carrying forward, and
    // docs/gpu-reports/2026-09-10-phase-b-reference-check.json for full detail.
    status: STATUS.VERIFIED,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    // Served from this host's own static files, not huggingface.co directly. The
    // blueprint explicitly permits this ("permit a local static mirror for
    // demos" — section 9): a venue room does not want every device pulling 610 MB
    // from a remote CDN individually, and a browser sandboxed away from arbitrary
    // external hosts (as this development environment's preview browser is) can
    // still range-fetch from same-origin static files. Swap these three URLs back
    // to the canonical ones below for a deployment whose browsers can reach HF.
    modelUrl: "/models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf",
    configUrl: "/models/qwen3-0.6b/config.json",
    tokenizerUrl: "/models/qwen3-0.6b/tokenizer.json",
    upstreamUrl: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
    expectedFormat: "Q8_0",
    // The blueprint's suggested starting range (512-2048), and confirmed to
    // actually load — solo, all 28 layers plus the LM head, on this hardware —
    // once engine/gpu-adapter.mjs's acquireDevice() requests real device limits
    // instead of accepting WebGPU's ~128 MB storage-buffer-binding default. That
    // default (not this context length) was the cause of an earlier device-lost
    // crash here; see the comment on requiredLimits in gpu-adapter.mjs.
    maxSeqDefault: 512,
    // Higher than SmolLM2's 60: Qwen3 reasons in a <think> block before it
    // answers, and 60 tokens routinely cut that block off mid-thought in live
    // testing (see IMPLEMENTATION_PLAN.md Phase D). Calibrated against a real
    // measurement, not a guess: a live run of "Why is the sky blue? Explain
    // briefly." needed 275 tokens for a complete think-plus-answer turn ending
    // naturally at <|im_end|> (220 cut the same prompt off mid-think). 320 gives
    // that headroom while still leaving most of the 512-position window free
    // for the rest of a short conversation.
    maxTokensDefault: 320,
    // Derived, not chosen: room.js sets its wire encoding from
    // chooseEncoding(spec.hidden), never from this field. At hidden=1024 a f32
    // hidden state still fits in one SCTP slice, so this model rides the wire
    // LOSSLESSLY — which is the real reason its split output is token-identical
    // to its solo output (see the Phase 3 golden report). Every wider model
    // (hidden >= 1536, i.e. 1.7B and up) falls to lossy f16, where that
    // equality becomes an empirical question rather than a guarantee. This
    // field read "f16" until 2026-09-10 and was simply wrong.
    wireDtype: "f32",
    // Recorded from the real fetch, not guessed: HTTP response headers off
    // huggingface.co on 2026-09-09 (X-Repo-Commit, X-Linked-Size, X-Linked-ETag),
    // then independently confirmed by hashing the downloaded file — the local
    // SHA-256 matches HF's recorded ETag exactly, so the mirror under
    // models/qwen3-0.6b/ is byte-identical to the published artifact.
    sourceRevision: "huggingface.co/Qwen/Qwen3-0.6B-GGUF @ 23749fefcc72300e3a2ad315e1317431b06b590a, " +
      "639446688 bytes, sha256:9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
    minRoomEnvelopeBytes: Math.round(0.8 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "First real Qwen/GGUF/WebGPU milestone: 28 dense layers, the imported " +
         "DenseEngine already handles Qwen3's QK-norm, a live worker-drop recovery " +
         "produces a token-identical continuation, and greedy output matches an " +
         "independent transformers reference exactly for a fixed prompt. One caveat: " +
         "that reference ran the unquantized (fp32) weights, not a Q8_0-native tool " +
         "(llama.cpp) — no compiler or llama-cpp-python wheel was available in this " +
         "environment to run that stronger check. Q8_0 correctness is well-supported " +
         "but not proven against a quantization-aware reference.",
  },

  "qwen3-1.7b": {
    id: "qwen3-1.7b",
    label: "Qwen3 1.7B · Q8_0",
    status: STATUS.EXPERIMENTAL,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    // Local mirror, same reasoning as 0.6B above: a room full of devices should
    // pull 1.8 GB off the host on the LAN, not off a CDN one device at a time.
    modelUrl: "/models/qwen3-1.7b/Qwen3-1.7B-Q8_0.gguf",
    configUrl: "/models/qwen3-1.7b/config.json",
    tokenizerUrl: "/models/qwen3-1.7b/tokenizer.json",
    upstreamUrl: "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
    expectedFormat: "Q8_0",
    maxSeqDefault: 512,
    // Higher than SmolLM2's 60: Qwen3 reasons in a <think> block before it
    // answers, and 60 tokens routinely cut that block off mid-thought in live
    // testing (see IMPLEMENTATION_PLAN.md Phase D). Calibrated against a real
    // measurement, not a guess: a live run of "Why is the sky blue? Explain
    // briefly." needed 275 tokens for a complete think-plus-answer turn ending
    // naturally at <|im_end|> (220 cut the same prompt off mid-think). 320 gives
    // that headroom while still leaving most of the 512-position window free
    // for the rest of a short conversation.
    maxTokensDefault: 320,
    // Derived from chooseEncoding(2048), not chosen here. Unlike 0.6B, this model
    // is wide enough that an f32 hidden state would cost a second SCTP slice per
    // hop, so it rides the wire as LOSSY f16 -- which means split-vs-solo output
    // equality is an empirical result for this model, not a guarantee the way it
    // is at 0.6B. See the Phase F report.
    wireDtype: "f16",
    // Recorded from the real fetch on 2026-09-10, then independently confirmed by
    // hashing the downloaded file: the local SHA-256 matches HF's X-Linked-ETag
    // exactly, so the mirror under models/qwen3-1.7b/ is byte-identical to the
    // published artifact.
    sourceRevision: "huggingface.co/Qwen/Qwen3-1.7B-GGUF @ 90862c4b9d2787eaed51d12237eafdfe7c5f6077, " +
      "1834426016 bytes, sha256:061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
    minRoomEnvelopeBytes: Math.round(2.1 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "Same dense code path as 0.6B, twice as wide (hidden 2048, 28 layers). The first " +
         "rung where one mid-range GPU cannot comfortably hold the whole model, so the " +
         "capacity-through-more-devices claim is doing real work rather than being optional.",
    blockedOn: "Qwen3 0.6B is Verified; this rung has not been started yet — it needs its own " +
      "download, GGUF probe, and live gate (Phase F of the blueprint promotes one rung at a time).",
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
    // Higher than SmolLM2's 60: Qwen3 reasons in a <think> block before it
    // answers, and 60 tokens routinely cut that block off mid-thought in live
    // testing (see IMPLEMENTATION_PLAN.md Phase D). Calibrated against a real
    // measurement, not a guess: a live run of "Why is the sky blue? Explain
    // briefly." needed 275 tokens for a complete think-plus-answer turn ending
    // naturally at <|im_end|> (220 cut the same prompt off mid-think). 320 gives
    // that headroom while still leaving most of the 512-position window free
    // for the rest of a short conversation.
    maxTokensDefault: 320,
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
    // Higher than SmolLM2's 60: Qwen3 reasons in a <think> block before it
    // answers, and 60 tokens routinely cut that block off mid-thought in live
    // testing (see IMPLEMENTATION_PLAN.md Phase D). Calibrated against a real
    // measurement, not a guess: a live run of "Why is the sky blue? Explain
    // briefly." needed 275 tokens for a complete think-plus-answer turn ending
    // naturally at <|im_end|> (220 cut the same prompt off mid-think). 320 gives
    // that headroom while still leaving most of the 512-position window free
    // for the rest of a short conversation.
    maxTokensDefault: 320,
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
