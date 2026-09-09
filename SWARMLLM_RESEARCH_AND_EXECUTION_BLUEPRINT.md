# AI Swarm: Browser-Native Distributed LLM Inference

## Research, current-state assessment, and execution blueprint

**Prepared:** 2026-09-09  
**Local project:** `E:\swarmllm`  
**Upstream implementation audited:** [`Nehanth/swarmllm` at `5781bc6`](https://github.com/Nehanth/swarmllm/tree/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366)  
**Purpose:** turn the present CPU distributed-inference proof of concept into a browser-only, WebGPU-powered multi-device LLM demo, then grow it safely through the requested model ladder.

---

## 1. The product being built

AI Swarm is not a cloud inference service and it is not ordinary “one model in each browser” chat. It is a **pipeline-parallel LLM runtime**:

1. A person opens the web app, creates a room, and gets a short code/QR.
2. Other nearby devices open the same site, enter that code, and join. No native app, account, model server, or per-device installation is required.
3. Each browser reports whether WebGPU is usable, a user-selected **capacity budget**, measured local inference throughput, and measured network links.
4. The planner chooses a host, decides which devices are worth using, orders them into a chain, and gives every selected browser one contiguous layer range.
5. Every browser downloads only the weight bytes for the range it owns; its GPU executes that range’s model math.
6. A token’s hidden state makes a lap through the layer chain, returning to the host for final normalization, logits, sampling, and display.
7. The question, streaming answer, layer map, health, and timing appear on every screen. A worker leaving triggers a new plan, reload/reuse of layer ranges, cache replay, and continued generation when the remaining room still has capacity.

The intended experience is:

```text
Creator/host browser                          Worker browser
─────────────────────                         ──────────────
Create room → code/QR ───── join code ────►   joins room
Choose model + capacity                       chooses capacity
     │                                              │
     └── planner assigns [0, 18)              [18, 28)
               │                                      │
prompt → tokenize → embed → GPU layers ─ hidden state ─► GPU layers
  ▲                                                             │
  └──── logits ← final norm + LM head ← hidden state ◄─────────┘
                         │
                   sample token
                         │
              broadcast token to every screen
```

For a 64-block model, use unambiguous **half-open ranges** in the UI and protocol: `[0, 62)` means layers 0–61 and `[62, 64)` means layers 62–63. This avoids the off-by-one ambiguity in phrases such as “0–62 and 62–7.” The UI can show friendly inclusive labels alongside the exact protocol form: `Layers 0–61` / `Layers 62–63`.

### What “no server in the middle” must mean honestly

The correct product claim is **no inference server and no account-based model backend**. There are still network services in a browser deployment:

| Component | Required? | Sees prompt/model computation? | Role |
|---|---:|---:|---|
| Static web host | Yes | No, aside from ordinary HTTPS request metadata | Serves HTML, JavaScript, CSS, icons, and optional local model files. |
| Signaling broker | Yes for WebRTC introduction | No model frames after peers connect | Exchanges connection offers/candidates so browsers can discover one another. |
| STUN | Usually | No application data | Helps browsers find routes through NAT. |
| TURN relay | Only when direct P2P fails | Relays encrypted WebRTC packets, but carries traffic metadata | Connectivity fallback; it does not execute inference. |
| Model repository / local static model host | Yes on first model load | It serves public weight bytes, not room activations | Each browser range-fetches only its assigned weight bytes and caches them. |

After WebRTC data channels are established, the activation frames travel browser-to-browser and the static/signaling services are not in the inference loop. WebRTC data channels use DTLS encryption hop by hop.[^upstream-security] This is materially different from sending a prompt to an API, but it is **not** a guarantee that untrusted room peers cannot infer prompts from activations. The upstream security document correctly treats a room as a trusted shared conversation, not a private computation enclave.[^upstream-security]

---

## 2. The target model ladder

The application should ultimately expose these model descriptors. It must not make an unverified model selectable as if it works: show it as **Available**, **Experimental**, or **Coming after validation**, with the reason and hardware requirement visible.

| UI key | Model / format | Why it is in the ladder | Architecture class | Upstream planner starting envelope* | Recommended first real-room test |
|---|---|---|---|---:|---|
| `smollm-135m` | SmolLM2 135M, original BF16 weights (GPU buffers use F16) | Smallest correctness and room-recovery demo | Dense Llama-family | 0.6 GB | Two browsers, 0.45–0.60 GB soft cap each. |
| `qwen3-0.6b` | Qwen3 0.6B, Q8_0 GGUF | First real WebGPU/Qwen milestone; 28 layers | Dense Qwen3 | 0.8 GB | Host 0.75–1.0 GB, worker 0.75–1.0 GB. |
| `qwen3-1.7b` | Qwen3 1.7B, Q8_0 GGUF | First meaningful multi-device capacity demonstration | Dense Qwen3 | 2.0 GB | Two devices around 1.5 GB each, or three around 1.0–1.25 GB each. |
| `qwen3-4b` | Qwen3 4B, Q8_0 GGUF | Stronger demo quality; 36 layers | Dense Qwen3 | 4.6 GB | Two capable laptops: host 3.0–3.5 GB, worker 2.5–3.0 GB; three devices around 2 GB each. |
| `qwen3.8-27b` | Qwen 3.8 27B, Q4_0 GGUF | End-state capacity demo | Hybrid: Gated DeltaNet + full attention | 16.5 GB | Two desktops around 10 GB + 8 GB, or three devices around 7 GB + 6 GB + 6 GB. Phones should initially receive only a very small validated range. |

\*These are upstream **planning envelopes**, not guaranteed physical VRAM requirements. The upstream model catalogue uses exactly 0.8, 2.0, 4.6, 16.5, and 0.6 GB as its room-level estimates.[^upstream-models] Actual feasible values depend on the GGUF, browser, GPU driver, KV-cache context length, non-layer weights, scratch buffers, and WebGPU allocation limits.

### Crucial terminology: capacity budget, not “giving RAM to the GPU”

A browser cannot reserve a guaranteed amount of GPU memory from the operating system. WebGPU deliberately does not expose a reliable “dedicated VRAM” dial. The slider should therefore be named something like:

> **GPU capacity budget (soft cap): 1.0 GB**  
> “Used by the room planner to limit this device’s assigned weights. The browser validates the actual allocation while loading.”

It must never promise that moving a slider allocates or isolates that memory. The implementation should:

1. Default to a conservative value based on a successful allocation probe, not a guessed hardware total.
2. Keep at least 20–30% headroom for browser/UI/compositor use and transient buffers.
3. Reserve extra room on the host for embeddings, final norm, LM head, tokenizer data, logits/readback, and cache.
4. Treat loader allocation failure as authoritative: lower the device’s usable budget, re-plan, and explain the change.
5. Persist a device’s successful value only as a hint, since background tabs, thermal limits, and other applications change real availability.

The recommended figures in the table are **starting test budgets**, not hardware recommendations. Start with lower-context rooms (512–2,048 tokens), a plugged-in laptop, a visible foreground tab, and a hard reload between failed allocation experiments.

### Why the order matters

The Qwen3 0.6B Q8 GGUF is an appropriate first GPU target: it has 28 layers, a Q8_0 artifact, a published 639 MB size in the official catalogue, and the exact dense model family is supported by the imported `DenseEngine` design.[^qwen-06][^upstream-dense] Qwen3 4B is still dense, but its official Q8 artifact is about 4.28 GB and it has 36 layers, so it is a scale-up of the same runtime rather than a new architecture.[^qwen-4b]

Qwen 3.8 27B is intentionally last. It is not merely a larger dense Qwen: upstream documents 64 blocks—48 recurrent Gated DeltaNet blocks and 16 full-attention blocks—with MTP/speculative decoding and different runtime state.[^upstream-architecture] It requires a separate engine path and a heterogeneous cost/memory scheduler.

---

## 3. Current project status: evidence, not aspiration

### Executive verdict

The project is **not off the rails**. It has already solved more of the distributed-systems product than a model-only demo normally does: room join, WebRTC tensor transport, exact dense-model placement, host election, a testable wire format, mirrored chat, and recovery/replay are present in the local codebase.

The honest boundary is equally important:

- A two-device **CPU SmolLM2 135M** distributed demo is the working product today.
- A WebGPU dense engine has been imported as source, but the room still loads `CpuEngine`; there is no local end-to-end Qwen GPU demonstration yet.
- The Qwen 3.8 hybrid engine is not in `engine/upstream/`, and the current scheduler only fully models uniform dense layers.

So the next win is not “build distributed inference from scratch.” It is “connect, verify, and productize the already-imported dense WebGPU path for Qwen3 0.6B, without weakening the CPU demo that already works.”

### What works locally now

| Area | Present local evidence | Status |
|---|---|---|
| Browser room and join flow | `room.html`, `room/room.js`, QR and mesh modules | Working dense-demo path. |
| WebRTC transport | `room/mesh.js`, `room/wire.js`; bit-exact bounce and wire tests | Working for current hidden-state protocol. |
| Current model | `SmolLM2-135M-Instruct`, 30 layers, hidden size 576, BF16 source reshaped to F16 layer shards | Working CPU model path. The model’s published configuration confirms 30 layers and hidden size 576.[^smollm-config] |
| Distributed token lap | Host embeds/runs its range; worker runs a contiguous range; last worker returns hidden state; host applies head/sampling | Working in `room/room.js`. |
| Mirrored chat | Host broadcasts generation start, every streamed token, and generation completion | Working on current path. |
| Dense scheduler | Exact subset search, Held–Karp chain order, contiguous allocation DP, host election, and reasons for excluded devices | Working for uniform dense layers. |
| Recovery | Worker departure triggers re-plan, redeal, cache replay, and token retry | Working current path; recovery regression suite exists. |
| Automated coverage | `npm test` is configured for wire, split, planner, recovery, QR, and Markdown suites | **134 passing tests** reported in the project README at this audit point. |
| HTTPS LAN development | Local server and certificate tooling | Working development support; required for real WebRTC/WebGPU on LAN. |

The local README records a two-browser run of 60 tokens at 4.99 tok/s, 63.6 ms median network lap, and a live recovery after a worker dropped, completing recovery in about 13.1 seconds.[^local-readme] Those are useful CPU-demo results. They do **not** establish GPU or Qwen performance.

### What has been added but is not integrated

| Asset | Local location | What it provides | What is still missing |
|---|---|---|---|
| Dense WebGPU engine | `engine/upstream/dense.js` | Dense Llama/Qwen3/Smol inference over WebGPU; async per-slice calls | A local adapter and room selection path. |
| GGUF loader | `engine/upstream/gguf.js` | GGUF parsing, Q4/Q8 repacking, range-based upload | Model descriptor, range-fetch orchestration, CORS/error handling, cache identity. |
| WGSL kernels | `engine/upstream/wgsl/base.js`, `coop.js`, `gemm.js` | Quantized matvec/GEMM, attention, RoPE, normalization, glue kernels | Actual browser validation on target hardware and integration into room execution. |
| GPU correctness harness | `gpu-test.html` | Tiny synthetic kernel and end-to-end self-tests | Must run in a real WebGPU-capable Chrome/Edge browser; a browser pane without an adapter cannot certify it. |
| Async room calls | `room/room.js` now awaits engine compute calls | Removes a real Promise/value mismatch | Does not by itself connect `DenseEngine.create()` to `CpuEngine.load()` call sites. |

The current room model registry contains only `smollm2-135m`, and `_load()` imports and invokes `CpuEngine.load(...)`. That is the direct, verifiable reason Qwen is not a local UI option yet.[^local-room]

### The imported-code snapshot is older than the upstream audit

The local `UPSTREAM.md` pins the imported dense engine to `1c9763f…`. The source audit used upstream commit `5781bc6…`, which contains the current engine, room protocol, benchmarks, model registry, and roadmap. The most material missing upstream files for the 27B path are:

```text
engine/qwen35.js       # hybrid Gated DeltaNet / attention / MTP engine
engine/wgsl/qwen35.js  # corresponding WebGPU shader generators
```

That is an architectural gap, not a small model-file change. Maintain the existing license/provenance notices when taking any additional upstream code; record the exact source commit and retain the upstream MIT notice.[^local-upstream]

### Scheduler status: a real differentiator, with a clear limitation

The local scheduler is ahead of the upstream roadmap in one important respect. Upstream still marks measured throughput placement, RTT-aware chain order, peer dropping, and host election as planned work.[^upstream-roadmap-27] Local `scheduler/plan.js` already searches device subsets, uses Held–Karp for the tour, performs a contiguous allocation DP, and can exclude a device that makes a lap worse.

However, it is ready **only for uniform dense layers** today:

- `allocate(..., weights)` accepts optional weights, but the production planner calls it without weights.
- the profiler produces one `msPerLayer` per device;
- `layerCap()` and `memoryFor()` assume one `layerBytes` figure.

That is excellent groundwork for the Qwen3 dense ladder. It is not valid hybrid Qwen 3.8 scheduling yet. The local source now documents this explicitly, which is the right engineering posture.[^local-plan]

---

## 4. What the upstream repository contributes

The upstream project is not a vague inspiration; it contains an implementable reference stack. The audit covered its engine interfaces, GGUF/safetensors loaders, room lifecycle, binary transport, security statement, benchmark log, test layout, model catalogue, architecture document, and roadmap at the pinned commit above.

### 4.1 Engine and shader implementation

Upstream separates two engines:

| Engine | Models | Important capabilities | Local state |
|---|---|---|---|
| `DenseEngine` | Qwen3 dense variants and SmolLM | Q8/Q4 GGUF, safetensors, RoPE, QK norm, range-local layer ranges, batch/prefill helpers | Imported, not connected. |
| `Qwen35Engine` | Qwen 3.8 27B hybrid family | Gated DeltaNet recurrent state, full attention/KV cache, batched prefill, MTP draft/verify, rollback | Not imported and not connected. |

The upstream WebGPU implementation is much more than a single generic matrix multiplication. Its documented kernel families include cooperative Q4/Q8 GEMV, batched GEMV, prefill GEMM, fused feed-forward operations, RMSNorm, RoPE, attention score/softmax/output, DeltaNet recurrence, and GPU argmax.[^upstream-kernels] Those are the concrete shaders that satisfy the product requirement that model math—not merely networking—runs on WebGPU.

The correct reuse strategy is **not** to rewrite those kernels. Treat upstream engine directories as vendored, attributed source; add a thin, locally owned adapter around their public engine interface. This keeps upstream updates auditable and preserves AI Swarm’s own differentiator: adaptive profiling and placement.

### 4.2 Weight loading and only-downloading owned layers

The intended Qwen path uses a GGUF file as a remote byte-addressable tensor container:

1. Fetch header and tensor metadata.
2. Build an index from layer number to tensor byte spans.
3. For a device’s assigned range `[a, b)`, request only tensors belonging to those layers plus required host-only tensors when that device is host.
4. Repack Q4/Q8 blocks while streaming into GPU buffers.
5. Cache complete validated response objects in Cache API, with a source URL/version/byte-size identity.

Upstream documents this exact pattern and its Cache API size-stamp handling.[^upstream-architecture][^upstream-kernels] The current local SmolLM path instead materializes one F16 file per layer ahead of time. Keep that path for the CPU oracle; use the upstream range loader for GPU GGUF models. Do not make every browser download an entire Qwen GGUF and then discard most of it—the product promise is range-local downloads.

### 4.3 Room protocol and live UI behavior

The upstream lifecycle already names the messages the target room needs: wait, load, progress, ready, reset, generate-start, token, generate-done, ask, and busy. It also defines hidden-state frames, batched prefill frames, and rollback for rejected speculative drafts.[^upstream-protocol]

AI Swarm should retain its existing message names where sensible or version a deliberate migration. The important invariant is behavioral, not the prefix:

```text
PLAN → DEAL → LOAD RANGE → READY → READY-ALL
     → PREFILL/DECODE HIDDEN-STATE LAPS → TOKEN BROADCAST
     → RESET or RECOVER/REPLAN
```

Every model runtime must conform to one room-facing contract:

```js
await engine.embedRun(tokenId, pos);        // host: embed + its layer range
await engine.runHidden(hidden, pos);        // worker: its layer range
await engine.headFromHidden(hidden);        // host: final norm + LM head → logits
engine.reset();                             // clear KV/recurrent state
```

Dense models can use that token-by-token interface first. The 27B phase extends it with batched prefill and speculative verification while preserving the plain decode path as the correctness baseline.

### 4.4 Transport details worth retaining

For Qwen 3.8, a hidden state at width 5,120 is 10 KB when packed as F16. Upstream’s transport slices binary frames at 4,600 bytes and can stripe traffic across negotiated WebRTC data channels to avoid SCTP burst behavior on higher-latency links.[^upstream-transport][^upstream-bench]

For the first Qwen3 0.6B milestone, the existing local wire layer may be sufficient after a byte-exact compatibility test. Do not replace a known working transport merely for aesthetic similarity. Introduce upstream’s slicing/striping only after measuring that the current wire framing harms a real Wi-Fi room, and retain deterministic reassembly tests.

### 4.5 Benchmarks: useful targets, not promises

Upstream reports WebGPU measurements including Qwen 3.8 Q4 decode around 9 tok/s plain and up to roughly 16 tok/s with self-speculation on a DGX Spark-class system, plus lower Mac results. It also records a two-device Mac+iPhone arrangement and detailed network transport experiments.[^upstream-bench]

Those figures are **upstream hardware-specific results**, not results obtained on this project’s devices. Use their benchmark methodology, prompt names, model URLs, context limits, and device reporting format. Only display AI Swarm performance claims after running the same tests in this room implementation.

---

## 5. The target architecture for AI Swarm

### 5.1 Responsibilities by device

| Responsibility | Host browser | Worker browser |
|---|---|---|
| Owns room/session transcript | Yes | Receives mirrored transcript. |
| Tokenizer, ChatML/Qwen chat template | Yes | No requirement. |
| Embedding and final LM head | Yes | No, unless a future architecture intentionally redistributes them. |
| Transformer blocks | Its assigned contiguous range | Its assigned contiguous range. |
| WebGPU inference | Yes | Yes. |
| Model download | Host-only tensors + its range | Its range only. |
| Sampling and streamed text authority | Yes | Receives authoritative token stream. |
| Network role | Control hub plus first/last chain link | Direct chain neighbor link(s) plus host control link. |

The host need not always be the device that created the room. The creator starts as provisional controller, but a capability-aware plan may elect a different host. During an active answer, host migration is a separate reliability feature; the existing code correctly avoids casually moving the conversation-owning host during mid-answer recovery.

### 5.2 Chain topology

The compute graph should be a directed pipeline with a return edge:

```text
                control / mirrored chat (host ↔ every room peer)
      ┌──────────────────────────────────────────────────────────┐
      │                                                          │
      ▼                                                          │
Host [0,a) ─hidden→ Worker A [a,b) ─hidden→ Worker B [b,L) ─hidden→ Host
      │                                                          ▲
      └──── token, plan, progress, reset, recovery events ──────┘
```

This has one activation hop per stage, whereas tensor parallelism would require collective communication inside every layer and is unsuitable for ordinary Wi-Fi. The host should establish only the control links and the chain links needed by the chosen plan; a full peer mesh becomes needlessly expensive as rooms grow.

### 5.3 Model descriptor is the source of truth

Create a local `models/registry.mjs` or equivalent, replacing the current single-entry `MODELS` object. A descriptor must contain at least:

```js
{
  id: "qwen3-0.6b",
  label: "Qwen3 0.6B · Q8_0",
  status: "experimental",              // unavailable | experimental | verified
  engineKind: "dense-gguf",            // cpu-smollm | dense-gguf | qwen35-gguf
  modelUrl: "...gguf",
  configUrl: "...config.json",
  tokenizerUrl: "...tokenizer.json",
  expectedFormat: "Q8_0",
  maxSeqDefault: 512,
  wireDtype: "f16",
  sourceRevision: "URL + content length + optional SHA-256",
  minRoomEnvelopeBytes: ...,
  capabilityRequirements: { webgpu: true, shaderF16: true },
}
```

Never hardcode layer count or bytes solely for a URL. Parse and validate the actual GGUF metadata at load time, then derive layer count, tensor spans, host-only bytes, per-layer bytes, and context cache requirements. The upstream catalogue is a good initial list of canonical URLs.[^upstream-models]

### 5.4 Engine adapter boundary

Add a locally owned `GpuEngineAdapter` rather than importing `DenseEngine` from `room/room.js` directly.

```text
Room runtime
    │ simple, stable async engine contract
    ▼
EngineFactory ── cpu-smollm ──► existing CpuEngine
    │
    ├────────── dense-gguf ──► GpuEngineAdapter ──► upstream DenseEngine
    │
    └────────── qwen35-gguf ─► Qwen35Adapter ─────► upstream Qwen35Engine
```

The adapter owns:

- WebGPU adapter/device acquisition and structured capability errors;
- model descriptor/config/tokenizer loading;
- `Range`/CORS/HTTP-206 validation and cache identity;
- mapping a room range to upstream engine options;
- normalized fields needed by the room (`layerCount`, `bytesLoaded`, `maxSeq`, EOS ids, cache/reset state, progress callbacks);
- conversion/validation of hidden-state typed arrays before the wire codec;
- optional batch APIs, exposed only when a model supports them;
- cleanup/reload behavior after failed allocations.

This boundary matters because upstream construction (`DenseEngine.create({ device, cfg, weights, ... })`) is not identical to the current CPU loader (`CpuEngine.load(dir, {...})`). The methods being async is now handled in the room, but construction, metadata, resource lifetime, and error reporting still need deliberate adaptation.

---

## 6. User experience specification

### 6.1 Room setup

1. **Create room**: choose display name; create a short, expiring, human-typable code and QR.
2. **Join room**: enter code or scan QR; show connection state and browser security requirements.
3. **Select model**: host selects a verified/experimental model. Changing model resets the room; participants explicitly acknowledge the new model download.
4. **Choose capacity budget**: each device selects a soft GPU budget, with a recommended safe setting and an “actual allocation may be lower” explanation.
5. **Profile**: run a small WebGPU workload and pairwise RTT/bandwidth test, then show a concise score—e.g. “4.2 ms/dense layer, 11 ms RTT to host.”
6. **Plan preview**: before weights load, show selected chain, excluded peers, predicted per-token lap, assigned range, bytes to download, and a human explanation.
7. **Start cluster**: all devices range-fetch/load their assigned tensors; progress reports show `PC: Layers 0–17, 412 MB` and `Phone: Layers 18–27, 227 MB`.
8. **Online chat**: every participant sees the same conversation and streamed token output. The host handles the computationally authoritative generation path.

### 6.2 Required plan/status UI

The room needs a readable visual rather than just a list of IDs:

```text
Qwen3 0.6B · Q8_0                         Room: lotus-cider
────────────────────────────────────────────────────────────────
PC (host)       WebGPU ✓  0.90 GB cap   Layers 0–17    410 MB loaded
Laptop          WebGPU ✓  0.80 GB cap   Layers 18–27   230 MB loaded

Pipeline: PC → Laptop → PC     predicted lap: 86 ms/token
Why: Laptop takes 10 layers; Phone was excluded because its extra hop costs
more than the compute it removes.

Status: Online · 14.2 tok/s · last lap 71 ms · prompt 41/512 tokens
```

During load, failure, or recovery, replace false certainty with precise states:

- `Downloading assigned tensors (63%)`
- `GPU allocation rejected; reducing usable budget from 1.0 GB to 0.7 GB`
- `Replanning after Laptop disconnected`
- `Replaying 58 tokens to rebuild caches`
- `Cannot recover: remaining room is 0.42 GB short`

### 6.3 Model selection rules

| User-facing state | Meaning | Initial use |
|---|---|---|
| **Verified** | Browser self-test, single-device golden test, two-device split test, and recovery test have passed on recorded hardware. | SmolLM CPU immediately; Qwen3 models only after their gates pass. |
| **Experimental** | Loader/engine can be tried, but a known cross-device proof is not yet recorded. | Qwen3 0.6B while being integrated. |
| **Unavailable on this device** | Missing WebGPU/feature or insufficient validated capacity. | Correctly disable rather than attempt load. |
| **Planned** | Descriptor is visible as a roadmap item but cannot be started. | Qwen3.8 27B before hybrid engine/scheduler gates close. |

---

## 7. Implementation plan

The phases below are dependency order, not a request to do all work at once. A phase is complete only when its exit gate passes; this prevents a visually impressive but numerically wrong model demo.

### Phase 0 — preserve the working CPU demo and establish baselines

**Goal:** make the existing SmolLM2 CPU room the stable reference while GPU work proceeds.

| Work item | Deliverable | Exit gate |
|---|---|---|
| Run and record current test suite | `npm test` report, browser/device note | All current 134 tests pass. |
| Rehearse two-device CPU demo | Room creation, join, forced split, mirrored reply, graceful exit/recovery | One recorded successful local run. |
| Freeze an oracle fixture | Fixed prompt, token IDs, expected greedy logits/argmax at selected positions | CPU split result equals single-device result. |
| Pin model assets | Manifest includes URL/version/byte lengths for Smol and future GGUFs | A later download cannot silently substitute another model. |

**Why it comes first:** the CPU path is independent of WebGPU and becomes the reference used to catch shader/loader regressions.

### Phase 1 — prove WebGPU before loading a real model

**Goal:** establish that the target browser/GPU can run the imported dense kernel stack correctly.

| Work item | Implementation detail | Exit gate |
|---|---|---|
| Run `gpu-test.html` on actual Chrome/Edge hardware | Acquire `navigator.gpu`, adapter, device; record limits/features | `kernelMicroTests` and `gpuSelfTest` both report `ok`. |
| Capture diagnostic report | Browser version, OS, GPU adapter name, `shader-f16`, limits, test output | A copyable support report appears in the UI. |
| Classify device capability | WebGPU absent, adapter unavailable, shader feature absent, or self-test mismatch all have distinct messages | Unsupported devices cannot enter GPU model plans. |
| Establish minimum allocation probe | Allocate/submit representative buffers incrementally with cleanup | UI produces a conservative initial soft-cap suggestion. |

**Do not skip this phase.** JavaScript imports loading under Node or a page opening successfully proves neither browser WebGPU availability nor numerical correctness.

### Phase 2 — create and test the dense GPU adapter

**Goal:** make the room runtime able to choose CPU or WebGPU engines through one contract.

1. Add `engine/factory.mjs` and `engine/gpu-adapter.mjs`.
2. Keep `CpuEngine` intact, including its present test API.
3. Implement descriptor validation and `navigator.gpu` device acquisition.
4. Normalize `DenseEngine.create()` into `load({ descriptor, layerRange, hasEmbed, hasHead, maxSeq, onProgress })`.
5. Normalize required runtime fields and `reset()` behavior.
6. Reject a range whose model metadata/config does not match the room’s chosen descriptor.
7. Make resource release/retry deterministic on allocation or fetch failure.
8. Change `room/room.js` to request an engine from the factory; do not import either implementation directly.

**Exit gate:** a synthetic local descriptor can load a GPU engine, call `embedRun`, `runHidden`, `headFromHidden`, reset it, and produce the same checked fixture output as the CPU oracle within the explicit quantization tolerance.

### Phase 3 — Qwen3 0.6B Q8 single-browser proof

**Goal:** prove a real Qwen3 model runs on WebGPU before involving WebRTC.

1. Add the Qwen3 0.6B Q8_0 descriptor using the official model/config/tokenizer URLs used upstream.[^upstream-models]
2. Implement GGUF metadata/range validation: HTTP success, range behavior, content size, tensor names, architecture, tokenizer/config compatibility, and Q8_0 support.
3. Load one browser with `[0, L)` plus host tensors.
4. Feed a fixed ChatML/Qwen-formatted prompt in deterministic greedy mode.
5. Compare token IDs/logit ordering against a pre-recorded trusted reference—not merely prose that “looks sensible.”
6. Display model source, GGUF size, parsed layer count, bytes downloaded, device limits, and actual load time.

**Exit gate:** repeated reloads on one real browser generate a stable fixed token sequence; GPU self-test remains green; cache hit avoids re-downloading the whole model; failures are actionable.

### Phase 4 — Qwen3 0.6B Q8 two-device distributed prototype

**Goal:** achieve the first requested WebGPU browser-to-browser demo.

1. Let the existing planner use the parsed dense model spec, not Smol-only assumptions.
2. Set two safe soft caps such that no one browser is assigned the full model but total capacity is sufficient.
3. Assign contiguous ranges, e.g. host `[0, 18)`, worker `[18, 28)` after actual planner output.
4. Use F16-packed hidden states on the wire; validate byte counts and finite values on every hop.
5. Compare distributed greedy output token IDs to the single-GPU Qwen run for the same prompt.
6. Broadcast plan/loading/generation events to both UIs.
7. Disconnect a worker after a short answer, re-plan, replay, and either continue identically or report genuine insufficient remaining capacity.

**Demo-definition exit gate:** a non-developer can open two browser devices, join by code, select Qwen3 0.6B Q8, see only assigned layers download, send one prompt from either device, see the same streamed answer on both, and observe the plan UI. This is the minimum viable WebGPU multi-device prototype.

### Phase 5 — make dense-model operation reliable, observable, and fast enough

**Goal:** turn the Qwen3 0.6B demo into a credible foundation rather than a one-shot experiment.

| Work item | Why it matters | Gate |
|---|---|---|
| Device profiler uses real loaded GPU layers | Synthetic throughput alone can mis-rank throttled or backgrounded tabs | Re-planning changes after calibration when measurements drift. |
| Per-hop telemetry | Separate GPU time, queue/readback, pack/unpack, and network time | UI and benchmark log explain each slow token. |
| Weight cache identity | Cache source URL + validator/length/model revision, not just a friendly name | Wrong or partial weights cannot be reused. |
| Two-browser test harness | Automate fixed split test plus exact activation/output checks where practical | CI/scheduled browser test catches protocol regression. |
| Network framing decision | Benchmark current local wire against sliced/striped upstream transport | Adopt slicing only with measured benefit and byte-exact tests. |
| Recovery matrix | Drop host/worker/bystander during idle, prefill, decode, and replay | Claims distinguish recoverable worker loss from host loss. |
| Resource safety | Cache cleanup, timeout/cancel, no duplicate load in foreground/background transitions | Repeated demo runs do not exhaust browser memory. |

### Phase 6 — promote models one dense rung at a time

The implementation order after Qwen3 0.6B is:

1. **SmolLM 135M WebGPU**: optional but valuable because it verifies the imported safetensors path; retain CPU Smol as oracle/fallback.
2. **Qwen3 1.7B Q8**: same dense code path; validate capacity planner and longer range fetches.
3. **Qwen3 4B Q8**: same dense code path; add three-device loading, cache, and capacity demonstrations.

For each rung, repeat the full promotion gate:

```text
GPU self-test → one-device deterministic golden → two-device split equivalence
→ range-only download evidence → real Wi-Fi run → worker-loss recovery
→ benchmark entry with exact browser/device/model/context/prompt
```

Do not mark 1.7B or 4B “supported” merely because their descriptors differ only by a URL. Larger vocabulary/head tensors, GPU limits, browser quotas, and model config quirks commonly appear only at load time.

### Phase 7 — import and adapt the Qwen 3.8 27B hybrid engine

**Goal:** make the 27B end state technically possible without compromising correctness.

1. Vendor the audited `engine/qwen35.js` and `engine/wgsl/qwen35.js` from one recorded upstream commit, with its MIT license/notice retained.
2. Add a `Qwen35Adapter` rather than embedding architecture-specific conditionals throughout `room/room.js`.
3. Parse/validate the target Q4_0 GGUF and expose block metadata, 64-layer topology, host-only tensors, attention layer cache requirements, and MTP tensors.
4. First implement **plain one-token decode** across one browser, then two browsers. No speculation until plain decode matches a trusted token reference.
5. Implement batched prefill across room links; verify that all layer caches/recurrent states are populated identically to sequential processing.
6. Implement MTP draft/verify only after plain/batched paths pass; test all rejection positions and state restore/rollback.
7. Add constrained device capabilities: feature requirements, minimum soft caps, supported context length, and a hard “not enough room” preflight rather than browser crashes.

The upstream engine’s Qwen 3.8 path includes MTP, snapshot/rollback state, and hybrid-specific GPU kernels; reuse is far safer than recreating that work under demo pressure.[^upstream-architecture][^upstream-kernels]

### Phase 8 — upgrade the scheduler for a hybrid model

**Goal:** preserve AI Swarm’s placement advantage once layers are not homogeneous.

For Qwen 3.8, replace the current scalar assumptions with model-derived per-layer arrays:

```js
layerProfile[layerIndex] = {
  family: "deltanet" | "attention",
  weightBytes,
  kvBytesAtContext,
  recurrentStateBytes,
  decodeWork,
  prefillWork,
};

deviceProfile[deviceId] = {
  denseMsPerWork,
  deltanetMsPerWork,
  attentionMsPerWork,
  q4Bandwidth,
  q8Bandwidth,
  usableBudgetBytes,
};
```

Then change all four required points together:

1. derive per-layer memory from actual model metadata and chosen context length;
2. profile device cost by layer family/shape, not one `msPerLayer` scalar;
3. make `layerCap`/feasibility sum concrete range bytes;
4. pass per-layer compute weights into allocation and include range-specific memory constraints.

At that point the problem no longer cleanly separates into an order-independent uniform allocation. Keep the current exact dense solver for dense models. For hybrid models, use an exact constrained dynamic program where feasible and a documented heuristic/branch-and-bound fallback for larger room sizes. Test it against brute force on small synthetic cases before trusting a performance claim.

### Phase 9 — demo hardening and deployment

**Goal:** make a room reliable enough to show repeatedly in front of people.

- Serve the app over a valid LAN HTTPS origin; self-signed certificates are acceptable for development but create avoidable demo friction.
- Support room codes/QR expiration and no-account ephemeral signaling.
- Make model assets cacheable and measure the cold-start versus warm-start time.
- Keep all tabs foreground, display wake-lock/battery/thermal warnings, and explain iOS/WebGPU limitations rather than silently falling back to CPU.
- Prepare a model-specific preflight page: WebGPU, `shader-f16`, allocation cap, device browser version, selected model size, room capacity, and connection quality.
- Record a reproducible benchmark report for every public number.
- Include a “clear local model cache” control with precise scope and confirmation.

---

## 8. Test and benchmark strategy

### Correctness hierarchy

The tests must progress from inexpensive/local to real/hardware-dependent:

| Level | Test | Failure it catches |
|---|---|---|
| 1 | Unit tests for wire framing, planner, QR, recovery decisions | Protocol and algorithm regressions. |
| 2 | GPU kernel micro-tests | Individual RMSNorm, quantized matvec, RoPE, attention, and packing errors. |
| 3 | Tiny synthetic GPU end-to-end self-test | Bind-group/layout/dispatch errors that a kernel-only test misses. |
| 4 | One-device real-model golden | Wrong GGUF mapping, tokenizer/config mismatch, wrong head, quantization error. |
| 5 | Split-vs-unsplit real-model golden | Boundary-layer/range/wire conversion errors. |
| 6 | Two-browser real Wi-Fi e2e | WebRTC ordering, UI synchronization, range download, device throttling. |
| 7 | Failure/recovery e2e | Missing cache replay, stale plan/range, broken continuation. |
| 8 | Performance benchmark | Regressions hidden by correct but much slower code. |

For deterministic correctness, set greedy sampling and record token IDs, not just rendered text. For quantized GPU paths, define tolerances at individual activation/logit checkpoints, then require identical chosen token IDs for the frozen short fixtures. Keep the CPU engine as an independent oracle wherever it can run the same model.

### Benchmark record format

Every reported number should include:

```text
model + exact GGUF URL/revision + quantization
prompt/template + input token count + generated token count + sampler
context limit + split ranges + host role
browser/OS/GPU adapter + WebGPU features/limits
network type + RTT/bandwidth + direct versus TURN
cold/warm cache + model load time + prefill time + decode tok/s
median/p95 token lap + per-stage GPU/network breakdown
correctness fixture result + source commit
```

The upstream benchmark log is exemplary in this respect and should be used as the format model, not copied as an AI Swarm result.[^upstream-bench]

---

## 9. Risks, assumptions, and non-goals

### Technical risks and mitigations

| Risk | Why it is real | Mitigation |
|---|---|---|
| Browser reports WebGPU but kernels are wrong or unsupported | Adapter existence is not numerical proof; shader capabilities differ by driver/browser | Gate every device with micro-tests + end-to-end self-test before room admission. |
| User budget exceeds actual allocatable memory | WebGPU budget is not directly controllable and the browser shares GPU memory | Use soft caps, allocation probing, safety margin, loader-authoritative replan. |
| A tiny slow/remote device harms every token | Pipeline decode is serial: compute and hops add | Preserve measured subset selection and peer dropping. |
| Qwen3.8 layers are heterogeneous | Uniform layer assumptions yield infeasible or slow plans | Implement range-specific cost/memory scheduler before claiming 27B placement. |
| Worker leaves during generation | KV/recurrent state is local to a range | Re-plan, retain same ranges where possible, reset/replay history, then retry. |
| Host leaves | Current host owns transcript/tokenizer/head/sampling | Explicitly report host loss for MVP; design session-state replication before claiming host-failover. |
| Network NAT blocks direct peers | WebRTC cannot always establish a direct path | STUN/TURN fallback with clear privacy/latency wording; do not hide relay use. |
| Public remote weights are mutable/unavailable | A URL alone is not a reproducible artifact | Store content length/version/optional SHA; permit a local static mirror for demos. |
| Mobile/browser thermal throttling | GPU throughput changes during a room | Periodic calibration, foreground/wake-lock UI, re-plan between turns. |
| Upstream source changes | Re-copying mixed local changes becomes unreviewable | Keep an attribution boundary and pin every imported source commit. |

### Privacy and trust boundary

- No inference backend should receive prompts or execute model layers.
- Every participating device receives the shared chat UI and is a trusted room member.
- Workers see activation tensors, timing, device names, budgets, and assignments. Activations are not a privacy primitive; an adversarial peer is out of scope for the prototype.[^upstream-security]
- Model weights are fetched from a public source or a self-hosted static source. “The model does not leave the room” should mean it is never uploaded to a third-party inference API and is only distributed as assigned public/local weight bytes—not that weights magically originate in the room.
- No account is needed for normal operation. A signaling code should be random, short-lived, and non-guessable enough for a demo room.

### Non-goals for the first functional prototype

- Training or fine-tuning across devices.
- Private computation against malicious room peers.
- Cross-model tensor parallelism or arbitrary layer interleaving.
- Guaranteed 27B operation on phones.
- Claiming that adding devices always makes generation faster; the cluster primarily increases capacity, and every extra stage adds a serial hop.
- Cloud-scale matchmaking, user accounts, persistent rooms, or a proprietary model-hosting service.

---

## 10. Recommended immediate sequence

This is the shortest path from today’s state to a compelling demo:

1. **Run the existing two-device SmolLM CPU demonstration** and save the result. It is the dependable fallback and establishes that room QR, planner, split, mirrored output, and recovery all work on the venue network.
2. **Run `gpu-test.html` on the two actual demo devices in Chrome/Edge.** Do not start Qwen work on an adapter that fails the self-test.
3. **Implement the engine factory and dense GPU adapter.** Connect no new model UI until the adapter can create/load/reset a dense engine correctly.
4. **Add Qwen3 0.6B Q8 as Experimental.** Prove it alone first with deterministic output and recorded load/adapter data.
5. **Run Qwen3 0.6B on two devices** with a forced-but-feasible split and show real range-only downloads / layer assignment / mirrored chat.
6. **Add a worker-loss recovery run** for the Qwen demo. If it is not stable in time, keep the feature behind an honest “CPU-demo validated; GPU recovery experimental” label instead of misrepresenting it.
7. **Promote 0.6B to Verified only after all gates pass**, then repeat for 1.7B and 4B.
8. **Begin Qwen3.8 only after the dense ladder is stable**: import the hybrid engine with proper attribution, add hybrid scheduling, prove plain decode, then prefill/speculation.

### Minimum success definition for the next demo

The next public demo is successful when all of the following are true in one room:

- Two people use ordinary browsers and a room code/QR; no native installation or account.
- Both have WebGPU self-test evidence.
- Qwen3 0.6B Q8 is selected and no browser downloads tensors outside its actual assignment plus host-required tensors.
- The plan UI shows soft budgets, measured performance, layer ranges, network chain, prediction, and any excluded peer with a reason.
- Every model-layer computation is performed by WebGPU on the participating browser that owns it.
- A prompt submitted on either screen causes one hidden-state lap through the devices and a synchronized streamed answer on both screens.
- A saved golden fixture proves the distributed run is numerically/token-equivalent to the one-device Qwen GPU baseline for the same deterministic prompt.

Once that exists, it is a real prototype of the stated product—not just a simulation or an architecture slide.

---

## 11. Decision log

| Decision | Recommendation | Reason |
|---|---|---|
| First GPU model | Qwen3 0.6B Q8_0 | It exercises real Qwen/GGUF/WebGPU while remaining a dense, 28-layer model. |
| Current fallback | Retain CPU SmolLM2 135M | It already works and acts as an independent correctness/recovery oracle. |
| Engine implementation | Reuse audited upstream engine code behind a local adapter | Avoid recreating proven WebGPU kernels; retain AI Swarm’s scheduler and UX ownership. |
| Model UI | Expose status tiers rather than fake availability | Users see the roadmap without hitting broken choices. |
| GPU capacity control | Soft cap + allocation probe + safety margin | Browser cannot reserve a literal VRAM amount. |
| Model distribution | Range-fetch/cached tensors per assigned contiguous range | Directly satisfies the partial-download product requirement. |
| P2P topology | Host control links + selected chain data links | Fits the actual inference graph and limits unnecessary peer links. |
| Qwen3.8 scheduling | Separate hybrid feasibility/cost implementation | Current uniform-layer solver is not sufficient. |
| Performance claims | Run AI Swarm’s own recorded benchmarks | Upstream numbers are a target/reference, not transferable proof. |

---

## Sources and audit references

[^local-readme]: [AI Swarm local README](README.md), especially Status, Quick Start, model-sharding, scheduler, and measured-run sections; checked 2026-09-09.

[^local-room]: [Current local room runtime](room/room.js), including `MODELS`, `_load()`, data-frame forwarding, planning, and recovery logic; checked 2026-09-09.

[^local-plan]: [Current local exact planner](scheduler/plan.js) and [cost model](scheduler/cost.js), including the explicit “NOT WIRED UP” hybrid-layer limitation; checked 2026-09-09.

[^local-upstream]: [Local upstream provenance record](UPSTREAM.md) and [third-party notices](THIRD_PARTY_NOTICES.md); checked 2026-09-09.

[^upstream-readme]: [SwarmLLM README, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/README.md).

[^upstream-architecture]: [SwarmLLM architecture, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/docs/architecture.md).

[^upstream-models]: [SwarmLLM model catalogue, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/room/models.js).

[^upstream-protocol]: [SwarmLLM room protocol, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/docs/protocol.md).

[^upstream-transport]: [SwarmLLM WebRTC transport implementation, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/room/transport.js).

[^upstream-dense]: [SwarmLLM dense WebGPU engine, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/engine/dense.js) and [GGUF loader](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/engine/gguf.js).

[^upstream-kernels]: [SwarmLLM WebGPU kernels document, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/docs/kernels.md).

[^upstream-bench]: [SwarmLLM benchmark log, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/docs/bench-log.md).

[^upstream-roadmap-27]: [SwarmLLM roadmap item 27: placement, chain order, host election, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/roadmap/27-placement-and-chain-order.md).

[^upstream-security]: [SwarmLLM security/threat model, audited commit `5781bc6`](https://github.com/Nehanth/swarmllm/blob/5781bc6fe45e3ad69c51eb368ea3b380f8f0e366/SECURITY.md).

[^qwen-06]: [Official Qwen3 0.6B GGUF model card](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), consulted 2026-09-09.

[^qwen-4b]: [Official Qwen3 4B GGUF model card](https://huggingface.co/Qwen/Qwen3-4B-GGUF), consulted 2026-09-09.

[^smollm-config]: [SmolLM2 135M Instruct configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/config.json), consulted 2026-09-09.
