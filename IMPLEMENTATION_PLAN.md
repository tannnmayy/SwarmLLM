# AI Swarm — Qwen-scale Distributed Inference Implementation Plan

**Status:** proposed implementation plan  
**Last reviewed:** 9 September 2026  
**Decision:** use [Nehanth/swarmllm](https://github.com/Nehanth/swarmllm) as the
MIT-licensed upstream engine baseline where it is materially ahead, and integrate
this repository's placement, profiling, and recovery work into that baseline.

---

## 1. Executive decision

AI Swarm is on the right product path, but it is currently a validated **CPU
prototype for one small dense model**, not a direct stepping stone to Qwen 3.8
27B. The peer-to-peer room, binary wire protocol, exact placement solver, live
profiling, and recovery behaviour are valuable assets and should be retained.
The current inference engine should not be incrementally stretched into a Qwen
3.8 engine.

The reason is architectural, not cosmetic:

- The current [`engine/cpu.mjs`](engine/cpu.mjs) loads f16 shard files into
  JavaScript `Float32Array`s and executes standard dense attention on the CPU.
  That is appropriate for SmolLM2 135M and cannot hold or efficiently execute a
  quantized 15 GB-class Qwen model.
- [`tools/fetch-model.mjs`](tools/fetch-model.mjs) only understands the
  SmolLM2 safetensors layout and the active room registry only exposes that one
  model.
- Qwen 3.8 is a hybrid model: its 64 blocks contain 48 Gated DeltaNet recurrent
  blocks and 16 full-attention blocks. It needs a different state model and
  different kernels from the current dense-attention loop.
- The linked project already contains the expensive, specialised work: a WebGPU
  engine, GGUF parsing, Q4/Q8 quantized-weight handling, GPU upload, DeltaNet
  kernels, batched prefill, and MTP speculative decoding. Its architecture and
  protocol are documented in its [architecture](https://raw.githubusercontent.com/Nehanth/swarmllm/main/docs/architecture.md),
  [protocol](https://raw.githubusercontent.com/Nehanth/swarmllm/main/docs/protocol.md),
  and [benchmark log](https://raw.githubusercontent.com/Nehanth/swarmllm/main/docs/bench-log.md).

**Recommended implementation strategy:** start a dedicated integration branch,
bring in the upstream browser engine and basic room runtime as a coherent
baseline, and port AI Swarm's scheduler/profiler/recovery policy into it through
explicit adapter interfaces. Do not copy isolated WGSL files into the present
CPU runtime: the loader, tensor layout, command submission, state handling,
wire protocol, and test harness form one system.

This is not a retreat from the original work. The scheduler remains the product
distinction: it should choose a useful subset of devices, exact layer cuts,
chain ordering, and host based on measured GPU performance and link cost, then
heal a room when those inputs change.

## 2. Definition of done

The target is not merely to display “Qwen” in the UI. A release is considered
successful only when all of the following are true.

### 2.1 Product contract

- A person creates a room from a browser page and another device joins by link
  or QR code; no native app, account, or inference server is required.
- Every participating device downloads and retains only its assigned model
  shard(s), plus required runtime state. Model files may be hosted statically;
  no prompt, activation, token stream, or inference request is sent to that
  host.
- The host owns text tokenization, embedding, final norm, LM head, sampling,
  transcript, and room authority. Workers own contiguous model layer ranges.
- A f16 packed hidden state travels directly through WebRTC from each stage to
  the next and returns to the host. For Qwen 3.8's hidden size of 5,120, that is
  10 KB per decoded token before frame overhead.
- The room can run a model that none of its individual devices could store.
  Adding a device is a capacity solution, not a promise of higher single-stream
  token throughput.
- A worker leaving during generation triggers a valid re-plan, re-deal, state
  rebuild, and continuation when the remaining pledges can still hold the model.

### 2.2 Engineering acceptance gates

| Gate | Required evidence |
|---|---|
| Qwen3 0.6B single-device | GPU golden tests agree with a trusted reference and produce a coherent chat response. |
| Qwen3 0.6B split | Two or more real browsers produce the same generated token stream as the single-device execution for deterministic sampling. |
| Qwen 3.8 baseline | One compatible desktop browser loads its Q4_0/Q8_0 model range, runs a prompt, and passes engine golden tests. |
| Qwen 3.8 split | A laptop plus one lower-memory device hold disjoint ranges and finish a real answer over WebRTC. |
| Planner | Measured GPU performance, actual GPU allocation capacity, network RTT/bandwidth, and variable layer costs determine the plan. |
| Recovery | A worker is disconnected during a Qwen answer; the room resumes deterministically if feasible, or states exactly why it cannot. |
| Performance | Decode, prefill, first-token time, rejoin time, and recovery time are recorded per tested hardware combination. |
| Safety | The UI states the room's trust model, handles WebGPU/device-loss failures, and never claims that intermediate activations are private from a malicious peer. |

### 2.3 Initial support boundary

The first reliable target should be **Chrome/Chromium with WebGPU** on desktop,
with compatible Android Chrome as a small-slice worker. Safari, Firefox,
cross-network rooms, public rooms, and large context windows are follow-on
compatibility targets. This keeps the first release testable rather than
promising every browser on every device.

## 3. What is reusable today

The following local work should be preserved and tested against the new engine.

| Existing asset | Why it remains valuable | Required change |
|---|---|---|
| [`scheduler/plan.js`](scheduler/plan.js) | Exact subset search, Held–Karp chain ordering, contiguous layer allocation, host election, and human-readable explanations. | Generalise its uniform layer-size and scalar `msPerLayer` assumptions. |
| [`scheduler/cost.js`](scheduler/cost.js) | Separates compute, hop, host overhead, memory, and solo-vs-split decisions. | Model decode, prefill, batch/speculation widths, heterogeneous blocks, and GPU residency. |
| [`scheduler/probe.js`](scheduler/probe.js) | The principle of measured performance, instability detection, and re-measurement is correct. | Replace CPU/JIT matvec profiling with GPU kernel and actual-engine profiling. |
| [`room/wire.js`](room/wire.js) | Binary framing, f16/f32 packing, SCTP-safe chunking, reassembly, and corruption checks already match the distributed activation problem. | Add batched activation frames, protocol versions, snapshots/rollback, and optional channel striping. |
| [`room/mesh.js`](room/mesh.js) | WebRTC introductions, data channels, RTT collection, and failure events are the right foundation. | Align its control/data semantics with the upstream room protocol and add TURN/relay support. |
| [`room/room.js`](room/room.js) | Room lifecycle, profiling, planning, assignment, streaming UI events, and recovery sequencing are proven for the small model. | Decouple it from `CpuEngine` and model-specific assumptions; let the engine advertise capabilities. |
| [`tests/`](tests) | The 133 passing tests are important behavioural specifications, especially split equality, planning invariants, recovery, and wire framing. | Retain them and add GPU/model/room-browser suites rather than replacing them. |

The present `CpuEngine` should be retained under a clearly named legacy/test path
until the GPU implementation has equivalent regression coverage. It can remain
useful for small-model logic tests and as a readable dense reference, but it is
not part of the production Qwen path.

## 4. Upstream adoption rules

### 4.1 Use upstream as a coherent baseline

The upstream repository's documented engine layout is the right import unit:

```text
engine/
  dense.js        dense Qwen3 / SmolLM engine
  qwen35.js       hybrid Qwen 3.8 engine, batched execution, MTP speculation
  gguf.js         GGUF header/tensor parsing and streamed quantized upload
  wgsl/           shared, cooperative-GEMV, and Qwen-specific WebGPU kernels
  tokenizer.js, sampling.js, quant.js, autotune.js, selftest.js
room runtime      signaling, peer links, assignments, generation loop
tests/            unit, GPU-golden, transport, and end-to-end tests
```

Importing only a shader or a parser will create incompatible copies of tensor
layout, quantization blocks, state ownership, and binary protocol logic. The
first integration should reproduce the upstream Qwen3 0.6B and Qwen 3.8 test
results unchanged before adding scheduler behaviour.

### 4.2 Preserve legal provenance

The upstream project is MIT-licensed. Any copied file or substantial adapted
portion must retain its copyright and MIT license notice. Before copying code:

1. Add `THIRD_PARTY_NOTICES.md` to this repository.
2. Record the upstream repository URL, commit SHA, import date, original file
   path, and whether the file was changed.
3. Keep upstream copyright/license text with the imported source, or preserve
   it in a clearly referenced root license notice as required by MIT.
4. Keep AI Swarm-authored scheduler and room-policy modules separately
   identifiable; do not relabel upstream work as local original work.
5. Review the licences of model weights, tokenizers, and any model-hosting
   dependencies separately. The project code licence does not grant model
   weight rights.

### 4.3 Prefer a one-way import plus periodic rebase

Do not make untracked manual copies and then lose the ability to update them.
The working convention should be:

- add `upstream-swarmllm` as a Git remote;
- record each imported upstream commit in `UPSTREAM.md`;
- perform changes on `codex/qwen-engine` or an equivalent dedicated branch;
- keep the scheduler integration in small commits which can be replayed after a
  future upstream update;
- run upstream and local test suites before and after every rebase.

If a direct fork is acceptable, make the upstream repository the new engine
base and migrate the AI Swarm scheduler into it. If the repository must remain
independent, use the same structure but retain an explicit upstream-import
history. A fork is lower risk because the engine changes quickly and its model,
kernel, and protocol components are tightly coupled.

## 5. Target architecture

```text
                          room host (browser)
 prompt -> tokenizer -> embedding -> local GPU layer range -> f16 activation
                                                        |
                                                        v
             WebRTC direct data channels: worker A -> worker B -> ...
                                                        |
                                                        v
 f16 activation <- final worker <- local GPU layer range <- each worker GPU
       |
       v
 final norm + LM head + sample + MTP draft -> streamed text to all room members

                    ^
                    |
       AI Swarm placement service
  device capability + actual GPU capacity + measured engine timings
  + direct-link RTT/bandwidth + model layer profiles
  -> selected subset + host + ordered chain + contiguous layer ranges
```

### 5.1 Engine boundary

All room code should depend on an engine capability interface rather than a
specific `CpuEngine` class. The exact function names can differ, but the
contract must cover these operations:

```js
load({ model, layerRange, hostRoles, contextLimit, onProgress })
runToken({ tokenId, position })                 // host embedding path
runHidden({ hidden, position })                 // a single decode activation
runBatch({ hiddenColumns, basePosition, mode }) // prefill or verify columns
finish({ hidden })                              // host final norm + LM head/sample
snapshot({ position })                          // required for speculative verification
rollback({ acceptedColumns })                   // required after a rejected draft
reset()
dispose()
capabilities()
```

`capabilities()` must report model architecture, hidden size, context limit,
layer count, per-layer memory metadata, supported batch widths, speculative
support, f16 support, and device-loss state. The room may then reject an
incompatible peer before it is assigned work.

### 5.2 Model descriptor

Replace the current fixed `modelSpec()` shape with a model descriptor generated
from the parsed GGUF metadata. It needs at least:

```js
{
  id, architecture, quantization, hiddenSize, vocabSize,
  layers: [
    {
      index, kind,                  // attention | gated-deltanet | other
      weightBytes, residentBytes,
      decodeCostClass, prefillCostClass,
      kvBytesPerToken, recurrentStateBytes
    }
  ],
  host: { embedBytes, finalNormBytes, lmHeadBytes, mtpBytes },
  wire: { activationBytes, encoding },
  maxContext, supportedBatchWidths
}
```

This is essential for Qwen 3.8. A uniform `layerBytes` and one `msPerLayer`
cannot fairly compare an attention block, a DeltaNet block, and the host's LM
head. It can result in a feasible-on-paper plan that overfills a phone or puts
the slowest stage on the critical path.

### 5.3 Placement model v2

The existing solver can remain exact while becoming model-aware:

- Profile each candidate device with representative **GPU kernels** and with a
  short assigned range after load. Store timing by layer cost class, batch width,
  and decode/prefill mode.
- Determine capacity by attempting bounded GPU buffer allocation/upload and
  reserving headroom. Browser-reported `maxBufferSize` is not free VRAM.
- Use a per-device/per-layer execution matrix `computeMs[device][layer]` and
  per-layer resident-memory requirements.
- Keep contiguous cuts, but change DP segment cost from `count * msPerLayer` to
  `sum(computeMs[device][layer])`. Reject any segment whose weights, KV/recurrent
  state, work buffers, and required host tensors exceed the measured pledge.
- Score decode and prefill independently. A plan may optimise first response
  time differently from long-answer decode; expose the selected policy in the
  UI.
- Include wire packet count, RTT, available throughput, WebRTC processing
  overhead, and batch/speculation depth in hop cost. A 10 KB Qwen activation
  needs SCTP-safe slicing; a 16-token prefill block needs a different transport
  estimate.
- Preserve the current subset search, Held–Karp ordering, solo check, and clear
  textual reasoning. Enforce a device-count limit that keeps planning bounded.

## 6. Implementation phases

Each phase has a narrow result and a hard gate. Do not begin the next one based
only on a UI demo.

### Phase 0 — Protect the current prototype and establish the upstream baseline

**Objective:** make engine integration reversible and reproducible.

Tasks:

1. Commit the currently passing local state and tag it, for example
   `cpu-prototype-133-tests`.
2. Add the upstream remote and capture its exact commit SHA in `UPSTREAM.md`.
3. Add third-party attribution and an import manifest as described above.
4. Create `codex/qwen-engine` from the clean current state, or fork upstream
   and create an integration branch there.
5. Compare the two repositories by responsibility, not file name: engine,
   tokenizer, GGUF/model loading, transport, room state machine, tests,
   scheduler, UI, and deployment.
6. Establish one test command that runs existing Node tests plus the imported
   unit tests; add CI before broad code migration.

**Gate:** the untouched CPU prototype is recoverable by tag, the imported
baseline identifies every third-party file, and both original test suites run.

### Phase 1 — Import the WebGPU engine without scheduler changes

**Objective:** reproduce an upstream-supported model on one device before
making it distributed or “smart.”

Tasks:

1. Import the upstream engine, GGUF parser, quantization/repacking code,
   tokenizer, sampling, and WebGPU self-test infrastructure as a consistent
   group.
2. Preserve the current CPU implementation as a legacy/reference engine rather
   than deleting it immediately.
3. Add WebGPU feature discovery:
   adapter availability, `shader-f16`, buffer limits, device-lost callback,
   timestamp support if available, and a clear unsupported-device screen.
4. Ensure model bytes are range-fetched, verified by length/hash where available,
   packed into GPU-friendly Q4/Q8 buffers, and cached locally. Never expand the
   entire model to JavaScript f32 memory.
5. Add a model registry that makes explicit which models/quantizations a browser
   can load. Remove the current single-model assumption from `room/room.js`.
6. Run the imported GPU golden tests on a known Chrome/WebGPU machine and save
   the baseline output and benchmark data.

**Gate:** a single browser runs the imported engine's supported dense test model
and all imported golden tests pass. No distributed code is changed yet.

### Phase 2 — Qwen3 0.6B as the integration model

**Objective:** prove the room can orchestrate a realistic dense Qwen model
before tackling the hybrid 27B model.

Qwen3 0.6B is the correct stepping stone because it is much smaller but reveals
the interface differences absent in SmolLM2: QK RMSNorm, GQA dimensions where
`heads * headDim` differs from hidden size, Qwen tokenizer/chat rules, GGUF
metadata, and quantized weights.

Tasks:

1. Use an upstream-supported Qwen3 0.6B GGUF artifact and record its URL,
   SHA/size, quantization, model licence, and tokenizer source.
2. Verify single-device prompt processing and greedy generation against the
   upstream golden/reference test.
3. Adapt the room load message to include model identity, architecture version,
   quantization, engine version, range, next peer, host role, and protocol
   version. Incompatible peers must fail loudly before loading weights.
4. Adapt the current activation transport to the engine's hidden state. Qwen3
   0.6B's 1,024 floats can use f32 when it remains one SCTP-safe frame; retain
   the local adaptive-precision logic and test both paths.
5. Run Qwen3 0.6B across two desktop browsers and then a desktop plus a phone.
   Test nonuniform cuts and the planner's decision to exclude a slower device.
6. Port the small-model split equality test to GPU execution. Permit only a
   documented numerical tolerance at tensor checkpoints; deterministic greedy
   output must match at the token level.

**Gate:** a 0.6B Qwen answer completes over real WebRTC with disjoint browser
GPU layer ranges and deterministic tokens match the single-device baseline.

### Phase 3 — Integrate AI Swarm's planner and profiling

**Objective:** replace simplistic/pledge-only allocation with the project’s
measured, exact placement policy without destabilising the imported runtime.

Tasks:

1. Add an engine profiling method which times real decode and prefill calls,
   excluding first-use shader compilation. Report medians and spread by layer
   class and batch width.
2. Replace CPU `measureMacsPerSec()` as the production decision input with a
   WebGPU probe plus post-load calibration against actual assigned layers.
3. Refactor `scheduler/cost.js` and `scheduler/plan.js` around the model
   descriptor described in section 5.2. Retain unit tests for the simple uniform
   case; add heterogeneous synthetic cases with a brute-force oracle.
4. Make each device advertise a user-controlled pledge and the verified amount
   that could be allocated. Schedule against the lower of the two, retaining
   safety headroom for browser and OS pressure.
5. Feed direct peer RTT and measured wire throughput into the planner. Unknown
   pairs must remain pessimistic, as they are today.
6. Explain every decision in the UI: chosen host, layer range, model memory,
   retained/dropped devices, expected decode and prefill cost, and confidence
   level of the measurements.
7. Re-plan on visibility, thermal/pressure, battery-policy, connection, or
   measured-performance changes, but debounce to avoid disrupting every answer.

**Gate:** generated heterogeneous-room tests prove feasibility, correct
contiguous coverage, capacity safety, and optimality against brute force for
small rooms. A live room demonstrably rejects a device that increases end-to-end
latency more than it helps.

### Phase 4 — Qwen 3.8 27B engine parity on one device

**Objective:** validate the hybrid Qwen engine in isolation before distributing
it.

Tasks:

1. Import and run the upstream Qwen 3.8 engine path as a unit: GGUF tensor map,
   Gated DeltaNet state, attention KV cache, MTP/draft head, final head, and
   sampling.
2. Add model-specific golden fixtures for one or more prompts. Golden tests must
   cover both recurrent and attention blocks, final logits, sampled greedy
   tokens, reset behaviour, and context extension.
3. Validate Q4_0/Q8_0 repacking and streaming upload separately. A wrong scale
   layout can produce plausible but incorrect language, so numeric tests must
   run before conversational demos.
4. Test GPU loss, OOM, failed range download, cache corruption, shader feature
   absence, and device state after a reset.
5. Benchmark plain decode, prompt prefill, model-load duration, browser memory,
   and cache-rejoin time on the intended host hardware.
6. Keep this phase single-browser. Do not debug a numerical kernel and WebRTC
   state propagation at the same time.

**Gate:** the selected host loads and generates deterministically with Qwen 3.8
27B using the imported engine's reference configuration and passes all its
GPU-golden tests.

### Phase 5 — Distributed Qwen 3.8 runtime

**Objective:** run Qwen 3.8 across real browsers with only assigned layers on
each device.

Tasks:

1. Extend `ai-load`/deal messages so workers range-fetch and upload only their
   assigned quantized tensors. The host loads embedding, final norm, LM head,
   MTP data, and its own range.
2. Implement single-token Qwen 3.8 decode across two devices first. The data
   path is host → workers in layer order → host, with f16 packed 5,120-wide
   activations.
3. Keep transport below the SCTP burst threshold. Reuse the current 4,600-byte
   slice discipline; benchmark on a real network because browser packet release
   behaviour changes latency substantially.
4. Scale to three and then more devices. Verify every device's actual layer
   range and cache footprint through diagnostic UI and logs.
5. Add batched prefill frames. Prompt processing must send ordered columns,
   preserve causal state across all workers, and avoid running the LM head for
   non-final prompt tokens.
6. Add MTP speculative decoding only after ordinary decode is stable. Draft
   proposals are generated on the host; workers verify batches; snapshots are
   taken per column; rejection invokes deterministic rollback.
7. Let the scheduler choose plain/depth-3/depth-5/depth-7 speculation based on
   measured compute and network lap cost, not a permanent default.

**Gate:** a laptop and phone/second computer run a real Qwen 3.8 answer using
disjoint layer ranges. Plain decode matches the single-device reference; the
speculative path matches plain decode for deterministic sampling.

### Phase 6 — Recovery, host continuity, and operational resilience

**Objective:** preserve the strongest current product behaviour at Qwen scale.

Tasks:

1. Generalise recovery to reset and replay both attention KV cache and DeltaNet
   recurrent state. Replay must fill a newly assigned range exactly as an
   uninterrupted run would.
2. On worker loss, atomically stop the active lap, mark the old plan stale,
   collect survivors and previously stood-down candidates, run placement v2,
   and deal the replacement ranges.
3. Reuse GPU weights on an unchanged range; reset only state. Re-download only
   newly assigned ranges. Report each source of recovery time to the user.
4. Add a recovery checkpoint test matrix: loss of first, middle, last, slowest,
   and spare worker during decode, prefill, and speculative verification.
5. Design host migration separately. The current host owns tokenizer, transcript,
   embedding/head weights, sampler state, and room authority. A real host-loss
   solution needs replicated encrypted transcript/control state, deterministic
   election, and a designated capable standby; it cannot be claimed from worker
   recovery alone.
6. Add WebRTC reconnect/TURN relay fallback for strict NATs. TURN relays
   encrypted transport but adds latency and should inform the planner.

**Gate:** forced worker loss while generating Qwen continues with identical
greedy output when capacity remains; host loss has either a validated takeover
path or an immediate, clear, non-hanging failure state.

### Phase 7 — Product hardening and release

**Objective:** make the demo honest, debuggable, and repeatable.

Tasks:

1. Build a topology view that shows host, active chain order, planned layer
   ranges, tensor state direction, network measurements, and standby devices.
2. Build a diagnostics bundle: browser/adapter, model/quantization/version,
   allocated memory, cache status, shader features, layer timing, RTT/bandwidth,
   selected plan, and last error. It must be exportable without prompt text by
   default.
3. Add model content integrity (at minimum URL, byte size, and hash verification
   when the hosting source supports it) and cache invalidation by model build.
4. Publish an explicit privacy/security page. WebRTC protects transport, but a
   participant can observe prompts/answers in the shared room and potentially
   infer text from activations. The upstream threat model explains this clearly:
   [Security](https://raw.githubusercontent.com/Nehanth/swarmllm/main/SECURITY.md).
5. Rehearse the intended hardware/venue setup: HTTPS certificate, WebGPU support,
   model cache warm-up, QR joining, sleep/lock handling, network failure, worker
   loss, and a recorded fallback demo.
6. Pin supported browsers and hardware expectations. Do not promise that every
   phone can contribute a useful Qwen 3.8 range.

**Gate:** a clean-machine rehearsal reaches a Qwen room, loads only assigned
shards, produces a response, exercises an intentional worker loss, and exports
a useful diagnostic report.

## 7. Required protocol evolution

The existing local protocol is sound for a single activation at a time. Qwen
prefill and speculation require a versioned extension rather than overloaded
ad-hoc messages.

| Operation | Required message behaviour |
|---|---|
| Join/capabilities | Engine/model/protocol versions, WebGPU feature flags, measured capacity, pledge, and profile quality. |
| Deal/load | Model URL/build/hash, quantization, layer range, host role, next peer, batch capability, and a monotonic plan epoch. |
| Decode | `hidden` / `hidden-ret` frames with position, plan epoch, message ID, encoding, and timeout correlation. |
| Prefill | `hidden-b` frames with base position and ordered column count; state must advance in causal order. |
| Speculative verify | Batch frame marked with snapshot semantics; every worker snapshots non-final columns consistently. |
| Rollback | Host sends accepted-column count; all workers restore the same snapshot before accepting new work. |
| Recovery | Stop current epoch, re-plan, deal replacement ranges, reset/replay, then resume under a new epoch. |
| Device loss | Distinguish expected leave, peer-connection failure, WebGPU device loss, range-load failure, and timeout. |

Every message must be rejected if its protocol/model/plan epoch does not match.
This prevents an old hidden state or delayed control message from advancing a
recurrent model state under a new plan.

## 8. Testing strategy

### 8.1 Keep current tests

Retain and run all 133 existing tests. They protect valuable guarantees:

- exact planning compared with independent/brute-force checks;
- f16/f32 wire framing, SCTP slice boundaries, reassembly, and malformed input;
- split-output equality for dense models;
- recovery’s replay and capacity-failure behaviour;
- QR and rendered-answer safety.

### 8.2 Add engine tests

| Test family | Examples |
|---|---|
| GGUF/parser | metadata fields, tensor offsets, range fetches, unsupported quantization, tokenizer extraction. |
| Quantization | Q4/Q8 unpack/repack known vectors, scale alignment, dequantized matvec tolerance. |
| WGSL kernels | RMSNorm, Q/K norms, RoPE, attention, DeltaNet recurrence, gating, matvec, MTP, batch widths. |
| Golden execution | Per-block activations/logits and whole-model greedy streams against a trusted reference. |
| Engine lifecycle | reset, snapshot, rollback, state replay, GPU-device loss, OOM, failed upload, cache rejoin. |

### 8.3 Add browser integration tests

Use an automated multi-browser harness wherever possible, plus scheduled physical
device runs.

- one browser, Qwen3 0.6B and Qwen 3.8;
- two browsers on one LAN, then physically separate devices on Wi-Fi;
- split placement with each valid contiguous cut;
- hidden-tab and thermal-throttle re-profile;
- worker loss during decode/prefill/speculation;
- plan epoch mismatch and delayed-wire-frame rejection;
- bandwidth/RTT/loss shaping with expected planner decisions;
- cache cold/warm paths and repeated join/leave cycles.

### 8.4 Performance reporting

Every performance statement must name model build, quantization, context/prompt
size, answer length, browser/OS/GPU, network type, number of devices, layer
assignment, scheduler policy, and whether it measures cold or warm cache.

Track at least:

- model range download and GPU-upload time;
- time to first token, prompt-prefill throughput, decode tok/s;
- per-stage GPU time and readback/upload overhead;
- activation frame count and hop latency;
- cache hit rate and allocated GPU memory;
- plan solve time and plan prediction error;
- worker-loss detection, reassignment, replay, and resume time.

## 9. Risks and non-goals

| Risk | Mitigation |
|---|---|
| A Qwen model fits in one strong device | The scheduler should run solo and explain why; swarming gains capacity, not automatic speed. |
| Browser-reported GPU limits are misleading | Allocate/upload conservatively, retain headroom, and measure the actual engine. |
| Phone browser is killed under memory pressure | Assign small ranges, retain warm standby options, expose incompatibility early, and test devices individually. |
| Numerical kernel bug emits plausible text | Require parsed-tensor, kernel, checkpoint, token-stream, and split tests before accepting a demo. |
| High RTT/loss makes a chain unusable | Use sliced frames, optional channel striping, batching/speculation, TURN-aware planning, and clear degraded-mode messaging. |
| Upstream changes break copied code | Pin imports by SHA, maintain attribution/import manifest, and update deliberately with tests. |
| A room member is untrusted | Treat the room as a shared conversation; never market activation transport as private computation. |
| Host disappears | Do not claim recovery until standby/migration is implemented and tested. |

Explicit non-goals for the first Qwen release:

- a public swarm of strangers;
- arbitrary model formats or architectures;
- Safari/Firefox parity before Chrome is stable;
- unlimited context;
- “faster because more devices” marketing;
- a guarantee that a phone will be a useful worker;
- strong privacy or malicious-worker verification.

## 10. Sequencing and effort

These are engineering ranges, not calendar promises. They assume one experienced
engineer, compatible test hardware, and active use of the upstream MIT engine.

| Milestone | Reuse upstream | Rebuild from the current repository |
|---|---:|---:|
| Establish imported engine baseline, licences, CI | 2–5 days | 2–5 days |
| Qwen3 0.6B browser demo with split layers | 1–3 weeks | 3–6 engineer-weeks |
| Model-aware scheduler integration | 1–3 weeks | same, after engine exists |
| Qwen 3.8 one-device parity | 1–3 weeks | several engineer-months |
| Qwen 3.8 multi-device, batching, recovery | 3–8 weeks | additional several engineer-months |
| Performance tuning to credible demo quality | ongoing | substantially longer |

The fastest high-confidence path is therefore:

1. import upstream engine and make its Qwen3 0.6B test pass locally;
2. prove a split 0.6B room;
3. integrate the AI Swarm placement policy;
4. enable upstream Qwen 3.8 single-device support;
5. distribute it, then add batching/speculation;
6. port and test recovery at Qwen scale;
7. harden, benchmark, and rehearse.

## 11. First implementation sprint

The next coding sprint should not try to “make 27B work.” Its output should be
a clean, testable engine baseline.

1. Tag the current 133-test CPU prototype and create the integration branch.
2. Add `UPSTREAM.md` and `THIRD_PARTY_NOTICES.md` templates.
3. Add the upstream remote and pin the exact source commit selected for import.
4. Import the upstream engine/model/test modules as a coherent set, preserving
   required MIT notices.
5. Add a single `npm` command which runs local tests and imported non-GPU tests.
6. Run the upstream self-test plus a Qwen3 0.6B single-device GPU golden test on
   the development machine.
7. Write a minimal engine adapter rather than modifying the scheduler yet.
8. Render a model-capability panel that says exactly why a browser can or cannot
   join.

**Sprint exit criterion:** a fresh Chrome/WebGPU browser can load the supported
Qwen3 0.6B quantized model and generate a reference-verified answer locally,
while the existing AI Swarm tests remain green. Only then should room protocol
and scheduler integration begin.

---

## Appendix: source-of-truth links

- [Upstream SwarmLLM repository](https://github.com/Nehanth/swarmllm)
- [Upstream architecture](https://raw.githubusercontent.com/Nehanth/swarmllm/main/docs/architecture.md)
- [Upstream room protocol](https://raw.githubusercontent.com/Nehanth/swarmllm/main/docs/protocol.md)
- [Upstream performance/transport benchmarks](https://raw.githubusercontent.com/Nehanth/swarmllm/main/docs/bench-log.md)
- [Upstream security/threat model](https://raw.githubusercontent.com/Nehanth/swarmllm/main/SECURITY.md)
- [Current project checklist](CHECKLIST.md)
- [Current project scheduler](scheduler/plan.js)
- [Current CPU engine](engine/cpu.mjs)
