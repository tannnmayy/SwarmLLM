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

// Where a device can get the weight bytes from. A descriptor declares both and
// models/delivery.mjs decides which one THIS page uses, per device, at load time.
//
//   upstream  the published artifact on huggingface.co, pinned to a commit.
//             Serves cross-origin range requests (access-control-allow-origin: *,
//             and `Range` is a CORS-safelisted request header for a single
//             `bytes=N-M`, so there is no preflight to fail). This is what makes a
//             static deployment possible at all: the page can be 200 KB on a CDN
//             while the 1.8 GB it needs comes from HF's CDN, not from the host.
//   mirror    this origin's own /models/ directory. Faster on a LAN, works with no
//             internet at all, and is the only source for a model whose files are
//             a local derivative rather than a published artifact (SmolLM2's
//             per-layer f16 shards are built by tools/fetch-model.mjs and exist
//             nowhere upstream).
//
// Both are byte-identical for every GGUF here — that is what `provenance` records
// and what tools/verify-delivery.mjs re-checks against the live upstream.
export const ORIGIN = { MIRROR: "mirror", UPSTREAM: "upstream" };

const GB = 2 ** 30;

// One place that knows how a Hugging Face download URL is spelled, so a repo,
// revision and filename cannot drift apart from the provenance record they came
// from. Pinned to a commit rather than `main` on purpose: the registry claims a
// specific sha256 for each file, and `main` is a mutable ref that can stop
// matching it without anything here changing.
const hf = (repo, revision, file) =>
  `https://huggingface.co/${repo}/resolve/${revision}/${file}`;

// The prose `sourceRevision` string every existing consumer already parses
// (tools/probe-gguf.mjs regexes a byte count and a sha256 out of it), generated
// from the structured record instead of typed out beside it. Same text as before;
// one source of truth now.
const describeProvenance = (p) =>
  `huggingface.co/${p.repo} @ ${p.revision}, ${p.bytes} bytes, sha256:${p.sha256}`;

// A GGUF model's three URLs at one origin. The weights live in the *-GGUF repo;
// config.json and tokenizer.json do not exist there (HF returns 404) and come
// from the base model repo, which is a different repo at a different revision.
// Exactly what each GGUF is, recorded from the real fetch rather than assumed.
//
// `sha256` is the file's own SHA-256 and is checkable two ways: hash the local
// mirror (tools/probe-gguf.mjs --hash), or read `X-Linked-ETag` off huggingface.co.
// One trap worth writing down, because it cost an hour: that header only exists on
// the 302 *before* the redirect to the CDN. Follow the redirect and the `ETag` you
// get back is the Xet content hash (`X-Xet-Hash`), a different number entirely —
// so a verifier must fetch with redirect: "manual". tools/verify-delivery.mjs does.
//
// `revision` is the commit the weights were taken at, and is what the upstream URL
// pins to. The base repo is a separate repo with its own revision: the *-GGUF repos
// carry only .gguf files, and return 404 for config.json and tokenizer.json.
const PROVENANCE = {
  "qwen3-0.6b": {
    repo: "Qwen/Qwen3-0.6B-GGUF",
    revision: "23749fefcc72300e3a2ad315e1317431b06b590a",
    file: "Qwen3-0.6B-Q8_0.gguf",
    bytes: 639446688,
    sha256: "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
    baseRepo: "Qwen/Qwen3-0.6B",
    baseRevision: "c1899de289a04d12100db370d81485cdf75e47ca",
    mirrorDir: "/models/qwen3-0.6b",
  },
  "qwen3-1.7b": {
    repo: "Qwen/Qwen3-1.7B-GGUF",
    revision: "90862c4b9d2787eaed51d12237eafdfe7c5f6077",
    file: "Qwen3-1.7B-Q8_0.gguf",
    bytes: 1834426016,
    sha256: "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
    baseRepo: "Qwen/Qwen3-1.7B",
    baseRevision: "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    mirrorDir: "/models/qwen3-1.7b",
  },
  "qwen3-4b": {
    repo: "Qwen/Qwen3-4B-GGUF",
    revision: "bc640142c66e1fdd12af0bd68f40445458f3869b",
    file: "Qwen3-4B-Q8_0.gguf",
    bytes: 4280404704,
    sha256: "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300",
    baseRepo: "Qwen/Qwen3-4B",
    baseRevision: "1cfa9a7208912126459214e8b04321603b3df60c",
    mirrorDir: "/models/qwen3-4b",
  },
};

function ggufSources(p) {
  return {
    [ORIGIN.UPSTREAM]: {
      model: hf(p.repo, p.revision, p.file),
      config: hf(p.baseRepo, p.baseRevision, "config.json"),
      tokenizer: hf(p.baseRepo, p.baseRevision, "tokenizer.json"),
    },
    [ORIGIN.MIRROR]: {
      model: `${p.mirrorDir}/${p.file}`,
      config: `${p.mirrorDir}/config.json`,
      tokenizer: `${p.mirrorDir}/tokenizer.json`,
    },
  };
}

export const MODELS = {
  "smollm2-135m": {
    id: "smollm2-135m",
    label: "SmolLM2 135M · CPU",
    status: STATUS.VERIFIED,
    engineKind: "cpu-smollm",
    // Mirror-only, and not for a convenience reason: what this engine loads is not
    // a published artifact. tools/fetch-model.mjs downloads SmolLM2's safetensors
    // and reshapes them into per-layer f16 shards (embed.bin, layer-NN.bin,
    // manifest.json) that exist only on the machine that ran it. There is no
    // upstream URL to fall back to, so a deployment that does not carry
    // models/smollm2-135m/ cannot offer this model at all — models/delivery.mjs
    // reports that rather than letting it 404 halfway through a join.
    sources: {
      [ORIGIN.MIRROR]: {
        dir: "/models/smollm2-135m",
        manifest: "/models/smollm2-135m/manifest.json",
        tokenizer: "/models/smollm2-135m/tokenizer.json",
      },
    },
    mirrorOnlyReason:
      "its per-layer f16 shards are built locally by tools/fetch-model.mjs and are not published anywhere upstream",
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
    // Two origins for the same bytes; models/delivery.mjs picks one per device.
    // The local mirror is preferred where it exists (LAN speed, works offline);
    // huggingface.co is the fallback, and is what a static deployment actually
    // runs on — see ORIGIN above.
    provenance: PROVENANCE["qwen3-0.6b"],
    sources: ggufSources(PROVENANCE["qwen3-0.6b"]),
    expectedFormat: "Q8_0",
    // The blueprint's suggested starting range (512-2048), and confirmed to
    // actually load — solo, all 28 layers plus the LM head, on this hardware —
    // once engine/gpu-adapter.mjs's acquireDevice() requests real device limits
    // instead of accepting WebGPU's ~128 MB storage-buffer-binding default. That
    // default (not this context length) was the cause of an earlier device-lost
    // crash here; see the comment on requiredLimits in gpu-adapter.mjs.
    maxSeqDefault: 2048,
    // Qwen3 reasons in a <think> block before it answers, so the cap has to fit a
    // whole thought plus the answer after it or the turn dies mid-sentence.
    // Measured, not guessed: 0.6B needed 275 tokens to finish "Why is the sky blue?
    // Explain briefly." (60 and 220 both cut it off), and 1.7B was still inside its
    // <think> block at 320. 640 clears both with room to spare, and against the
    // 2048-position window above still leaves ~1400 positions for follow-up turns.
    // room.js lowers this further, per turn, when less context than that remains.
    maxTokensDefault: 640,
    // Derived, not chosen: room.js sets its wire encoding from
    // chooseEncoding(spec.hidden), never from this field. At hidden=1024 a f32
    // hidden state still fits in one SCTP slice, so this model rides the wire
    // LOSSLESSLY — which is the real reason its split output is token-identical
    // to its solo output (see the Phase 3 golden report). Every wider model
    // (hidden >= 1536, i.e. 1.7B and up) falls to lossy f16, where that
    // equality becomes an empirical question rather than a guarantee. This
    // field read "f16" until 2026-09-10 and was simply wrong.
    wireDtype: "f32",
    // Derived from `provenance` above rather than typed beside it — the same prose
    // this field has always carried, minus the chance of the two disagreeing.
    sourceRevision: describeProvenance(PROVENANCE["qwen3-0.6b"]),
    // Sized against the real solo cost at this model's default window (1053 MB at
    // maxSeq 2048), with headroom. tools/probe-gguf.mjs fails if this drops below
    // the real figure, which is how the 512 -> 2048 change was caught.
    minRoomEnvelopeBytes: Math.round(1.2 * GB),
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
    // Promoted 2026-09-10 (IMPLEMENTATION_PLAN.md Phase F) after the full gate:
    // byte-verified download, header probe against config.json, adapter self-test,
    // one-device golden, split equivalence, a live two-device WebRTC room with
    // range-only downloads, worker-loss recovery, and an exact token-ID match
    // against an independent transformers reference.
    // docs/gpu-reports/2026-09-10-phase-f-qwen3-1.7b.json
    status: STATUS.VERIFIED,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    provenance: PROVENANCE["qwen3-1.7b"],
    sources: ggufSources(PROVENANCE["qwen3-1.7b"]),
    expectedFormat: "Q8_0",
    maxSeqDefault: 2048,
    // Qwen3 reasons in a <think> block before it answers, so the cap has to fit a
    // whole thought plus the answer after it or the turn dies mid-sentence.
    // Measured, not guessed: 0.6B needed 275 tokens to finish "Why is the sky blue?
    // Explain briefly." (60 and 220 both cut it off), and 1.7B was still inside its
    // <think> block at 320. 640 clears both with room to spare, and against the
    // 2048-position window above still leaves ~1400 positions for follow-up turns.
    // room.js lowers this further, per turn, when less context than that remains.
    maxTokensDefault: 640,
    // Derived from chooseEncoding(2048), not chosen here. Unlike 0.6B, this model
    // is wide enough that an f32 hidden state would cost a second SCTP slice per
    // hop, so it rides the wire as LOSSY f16 -- which means split-vs-solo output
    // equality is an empirical result for this model, not a guarantee the way it
    // is at 0.6B. See the Phase F report.
    wireDtype: "f16",
    sourceRevision: describeProvenance(PROVENANCE["qwen3-1.7b"]),
    // Real solo cost is 2193 MB at maxSeq 2048; this carries headroom above it.
    minRoomEnvelopeBytes: Math.round(2.4 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "Same dense code path as 0.6B, twice as wide (hidden 2048, 28 layers). Greedy " +
         "output matches an independent transformers reference exactly, and a live " +
         "two-device split survives losing a worker mid-answer. Two caveats worth " +
         "knowing: the reference was unquantized fp32, not a Q8_0-native tool; and this " +
         "is the first rung whose hidden state crosses the wire as LOSSY f16, so its " +
         "split-vs-solo token equality is a measured result rather than a guarantee.",
  },

  "qwen3-4b": {
    id: "qwen3-4b",
    label: "Qwen3 4B · Q8_0",
    status: STATUS.EXPERIMENTAL,
    engineKind: "dense-gguf",
    architecture: "qwen3",
    provenance: PROVENANCE["qwen3-4b"],
    sources: ggufSources(PROVENANCE["qwen3-4b"]),
    expectedFormat: "Q8_0",
    maxSeqDefault: 2048,
    // Qwen3 reasons in a <think> block before it answers, so the cap has to fit a
    // whole thought plus the answer after it or the turn dies mid-sentence.
    // Measured, not guessed: 0.6B needed 275 tokens to finish "Why is the sky blue?
    // Explain briefly." (60 and 220 both cut it off), and 1.7B was still inside its
    // <think> block at 320. 640 clears both with room to spare, and against the
    // 2048-position window above still leaves ~1400 positions for follow-up turns.
    // room.js lowers this further, per turn, when less context than that remains.
    maxTokensDefault: 640,
    wireDtype: "f16",
    sourceRevision: describeProvenance(PROVENANCE["qwen3-4b"]),
    minRoomEnvelopeBytes: Math.round(4.6 * GB),
    capabilityRequirements: { webgpu: true, shaderF16: true },
    why: "Still dense, but 36 layers at hidden 2560 — about 4.2 GB to hold. This is the rung " +
         "where one mid-range GPU stops being able to run the model alone, so a split is no " +
         "longer a demonstration of the idea but the only way to run it at all. The kernels " +
         "are proven correct at this model's dimensions (half the model, 2235 MB, loads and " +
         "computes finite non-zero hidden states); what is missing is memory, not correctness.",
    blockedOn: "Needs a second physical GPU. On the one machine tested so far the full model " +
      "allocates (4076 MB) and then loses the device on the first inference pass — and because " +
      "every 'device' in a single-machine test shares that one GPU, no split arrangement avoids " +
      "the ~4.2 GB total. A 2-way split needs ~2308 MB (host) + ~1914 MB (worker), both close to " +
      "the 2235 MB this GPU has already run successfully. " +
      "See docs/gpu-reports/2026-09-10-phase-g-qwen3-4b.json.",
  },

  "qwen3.8-27b": {
    id: "qwen3.8-27b",
    label: "Qwen 3.8 27B · Q4_0 (hybrid)",
    status: STATUS.PLANNED,
    engineKind: "qwen35-gguf",
    architecture: "qwen35",
    // No sources at all, deliberately: this rung is blocked on an engine that is
    // not vendored, so naming a URL would imply it could be loaded by swapping one.
    sources: null,
    provenance: null,
    expectedFormat: "Q4_0",
    maxSeqDefault: 2048,
    // Qwen3 reasons in a <think> block before it answers, so the cap has to fit a
    // whole thought plus the answer after it or the turn dies mid-sentence.
    // Measured, not guessed: 0.6B needed 275 tokens to finish "Why is the sky blue?
    // Explain briefly." (60 and 220 both cut it off), and 1.7B was still inside its
    // <think> block at 320. 640 clears both with room to spare, and against the
    // 2048-position window above still leaves ~1400 positions for follow-up turns.
    // room.js lowers this further, per turn, when less context than that remains.
    maxTokensDefault: 640,
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
// The one place the "can this device run this model" rule lives. Used by
// availableModels() below to grey out a picker entry, and by room.js to decide
// whether to tell the room this device is worth dealing layers to at all --
// those two answers must never disagree, or a device the picker called
// unavailable still gets handed a layer range and then fails to load it.
//
// `caps.webgpu === false` rather than `!caps.webgpu` on purpose: undefined means
// "not probed yet", which is not the same as "absent", and must not disqualify a
// device before its adapter request has resolved.
export function capabilityGap(descriptor, caps = {}) {
  const req = descriptor?.capabilityRequirements || {};
  if (req.webgpu && caps.webgpu === false) return "this device has no WebGPU";
  if (req.shaderF16 && caps.webgpu && caps.shaderF16 === false) return "this GPU/driver lacks shader-f16";
  return null;
}

export function availableModels(caps = {}) {
  return Object.values(MODELS).map((m) => {
    const unavailableReason = capabilityGap(m, caps);

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

// The URLs for one model at one origin, or null if it does not publish that origin.
// Callers should go through models/delivery.mjs rather than calling this directly —
// this is the lookup, that is the decision.
export function sourcesFor(descriptor, origin) {
  return descriptor?.sources?.[origin] || null;
}

// Which origins this model could be served from, best first. Order is the policy:
// a mirror on the same origin beats a CDN across the internet when it exists, so
// a LAN room stays a LAN room and an offline laptop keeps working.
export function deliveryOrigins(descriptor) {
  return [ORIGIN.MIRROR, ORIGIN.UPSTREAM].filter((o) => !!sourcesFor(descriptor, o));
}
