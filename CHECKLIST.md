# AI Swarm — build checklist

Status as of 9 September 2026. Ticked items are verified by a test or a measured run,
not by "it looked like it worked".

Legend: **[x]** done and verified · **[~]** partially done · **[ ]** not started

---

## 1. Transport — the H0–6 checkpoint

- [x] HTTPS dev server with the LAN IP in the certificate SAN (`tools/serve.mjs`)
      *WebGPU and WebRTC need a secure context; plain `http://192.168.x.x` gives
      neither, and the failure looks exactly like "this phone has no WebGPU"*
- [x] Plain HTTP on localhost alongside it, so dev needs no cert warning
- [x] Cert regeneration script for a new venue IP (`tools/make-cert.sh`)
- [x] WebSocket signaling on the same origin at `/signal` (`tools/signal.mjs`)
      *No message type exists that could carry a prompt, activation or answer —
      the privacy claim is enforced by the protocol's shape, not by policy*
- [x] Room codes with an unambiguous alphabet (no O/0/I/1)
- [x] WebRTC mesh, negotiated ctrl + wire channels (`room/mesh.js`)
- [x] Glare avoided by rule: the newcomer offers, members only answer
- [x] Peer-to-peer RTT probe every 2 s
- [x] Binary framing sliced to 4600 B for Chrome's SCTP burst limit
- [x] Adaptive wire precision — f32 when it costs no extra slice, else f16
- [x] Reassembly: out-of-order, duplicate-safe, bounded memory
- [x] **30/30 wire tests**
- [x] Verified live: two peers, 50 round trips, bit-exact, 0.9 ms median
- [ ] TURN relay fallback for strict NATs *(STUN only today; ~1–2 h)*
- [ ] QR code join *(~0.5 h)*

## 2. Engine — the H6–14 checkpoint

- [x] Model fetch + reshape into per-layer shards (`tools/fetch-model.mjs`)
      *bf16 → f16 is lossless here: f16 has more mantissa (10 bits vs 7). 0 of
      134.5M values overflowed*
- [x] Byte-level BPE tokenizer, exact round trip on emoji/accents/digits
- [x] Sliceable CPU engine with the four-call contract (`engine/cpu.mjs`)
      `embedRun` · `runHidden` · `headFromHidden` · `reset`
- [x] A slice downloads only the layers it was dealt (54 MB for 8 layers)
- [x] Isomorphic: identical code in Node and the browser
- [x] CPU golden reference for the GPU port to match (`tools/reference.mjs`)
- [x] **11/11 split tests** — bit-identical logits across 2, 3 and 5 devices,
      with the real wire codec in the loop
- [x] **WebGPU kernels imported and verified on real hardware** — kernel micro-tests
      and a synthetic end-to-end forward pass at f32/Q8/Q4, bit-exact against CPU
      math (`gpu-test.html`, [report](docs/gpu-reports/2026-09-09-phase1-gpu-selftest.json))
- [x] **GPU engine factory + adapter** (`engine/factory.mjs`, `engine/gpu-adapter.mjs`) —
      loads a real GGUF over a real HTTP range request; whole-model and split across
      two engine instances both match an independent CPU reference to 1.1e-7 relative
      error (`gpu-adapter-test.html`, [report](docs/gpu-reports/2026-09-09-phase2-gpu-adapter-selftest.json)) · **34/34 tests**
- [ ] **Connect a real Qwen3 0.6B download to the room** — the model ladder registry
      (`models/registry.mjs`) has the descriptor, but `room.js` does not yet fetch a
      GGUF, derive a scheduler cost model from it, or offer model selection in the UI.
      *(this is the actual remaining gap, not "WebGPU" generally)*
- [ ] Batched prefill *(token-by-token today; ~3–4 h)*
- [ ] Larger models fetched and tested (Qwen3 0.6B / 1.7B) *(~1–2 h)*
- [x] Weight caching in the Cache API — byte-length checked against the manifest,
      quota refusals tolerated. Second join reports "all from cache, no download"

## 3. Room

- [x] Join, roster, live device list
- [x] Deal layers, load shards, report progress
- [x] Generation loop: embed → chain → head → sample
- [x] Answer mirrored to every screen
- [x] Anyone in the room can ask
- [x] Wake lock so a device in the chain cannot doze (`room/awake.js`)
- [x] Verified live: 60 tokens across two browsers, answer byte-identical to the
      single-device reference
- [x] ChatML prompting — the model is instruction-tuned, and fed raw text it
      continues sentences instead of answering. Templated, it replies and stops
- [x] Multi-turn: only the new turn is encoded, since the KV caches across the room
      already hold the rest. A follow-up costs ~26 positions, not a replay
- [x] Honest context limit — a question with no room to be answered is refused up
      front rather than cut off mid-sentence; usage is shown live
- [ ] Actionable errors and a diagnostic report *(~1 h)*

## 4. Scheduler — the H14–22 checkpoint · **the contribution**

- [x] Cost model with its assumptions stated (`scheduler/cost.js`)
      *Decode is a sum of stages plus hops, not a max*
- [x] Device profiling: ms/layer from a matvec of a layer's shape
- [x] Probe validated against the real engine — 0.94× (`tools/calibrate.mjs`)
- [x] Throttled-tab detection, re-measure on visibility, second calibration on
      real layers
- [x] RTT matrix gossiped so the host sees links it cannot measure itself
- [x] Exact contiguous layer allocation by DP, with balance tie-breaking
- [x] Exact chain ordering by Held-Karp
- [x] Subset search — drop a device that costs more in hops than it contributes
- [x] Solo check — refuse to split when one device can hold the model
- [x] Host election by measured speed, with hand-off to the elected device
- [x] Five strategies for the benchmark: solo / even / memory / compute / optimal
- [x] Plain-language reasoning for every decision
- [x] **29/29 planner tests**, including "optimal is never beaten by any
      baseline" over 1,483 generated rooms
- [x] Scenario analysis tool (`tools/scenario.mjs`)
- [ ] Battery and thermal signals **used in the cost model** *(read today, not
      weighted; ~1–2 h)*
- [x] Re-plan and re-deal while running — the recovery path does exactly this, and
      it recruits devices the planner had previously stood down

## 5. Recovery — the H22–30 checkpoint

- [x] Detect a peer leaving mid-generation, and fail the in-flight lap at once
      rather than waiting out its timeout
- [x] Re-plan over the survivors — including devices the planner had stood down,
      which get recruited back when they are suddenly needed
- [x] Re-deal the orphaned layer range
- [x] A device whose range is unchanged keeps its weights and clears only its
      cache — recovery costs seconds, not a re-download
- [x] Replay history so the new holder's KV cache is real, not empty
- [x] Resume from the same position, with the same answer
- [x] Honest failure: a room that can no longer hold the model says so instead of
      dealing an impossible plan
- [x] `dropWorker()` demo instrument — takes the same code path an unplanned
      disconnect takes, so the rehearsed version is the real one
- [x] **12/12 recovery tests** — the answer is byte-identical to the
      uninterrupted run after the middle device leaves, and after the last one does
- [x] **Gate met live:** dropped a worker 58 tokens into an answer; the room
      recruited the stood-down spare, replayed 58 tokens, recovered in **13.1 s**
      and finished the sentence it was in the middle of
- [ ] Spare layer copies for instant failover *(stretch; would cut the 13 s to
      near zero by keeping a warm replica)*
- [ ] Host loss *(unrecoverable by design today: the conversation, tokenizer and
      LM head all live on the host. Reported clearly rather than hung)*

## 6. Dashboard — the H30–36 checkpoint

- [x] Live device list with measured speed, pledge and layer range
- [x] Pledge slider — the live demo lever
- [x] Strategy switch
- [x] Five-way predicted comparison table
- [x] The planner's reasoning, in words
- [x] Predicted vs actual tokens/sec
- [ ] Topology graph showing the chain visually *(~1–2 h)*
- [x] "Kill a node" button for the demo — drives the real failure path, not a
      simulated one
- [ ] Three full rehearsals + a recorded backup *(non-negotiable)*

## 7. Project hygiene

- [x] MIT licence and `NOTICE.md` stating prior art and what is ours
- [x] README with honest status and findings
- [x] 168 tests across seven suites
- [ ] CI running the tests on push *(~0.5 h)*
- [ ] Round 1 pitch deck *(in progress)*

---

## Where this leaves us

| Against | Complete |
|---|---|
| What the pitch deck promises | **~85%** — 10 of 12 claims done and demonstrable |
| A finished product | ~50% |
| SwarmLLM parity (27B, custom WGSL) | ~25% — and not the goal |

**2–5 hours of work remain to deliver everything the deck claims**, against 36
hours available. Every headline claim in the deck is now built and demonstrated.

### Critical path

1. Battery/thermal into the cost model (1–2 h) — closes the last ⚠️
2. Topology graph showing the chain visually (1–2 h)
3. Three full rehearsals + a recorded backup (non-negotiable)

Everything after that is stretch: WebGPU, larger models, prefill batching, TURN.

### Known risks

| Risk | Mitigation |
|---|---|
| Venue Wi-Fi hands out a new IP; iOS refuses the cert | `npm run cert` is the first command on arrival |
| Venue Wi-Fi blocks peer-to-peer | Verified STUN works on our network; TURN fallback is unbuilt |
| A phone's screen locks and throttles the chain | Wake lock + re-measure on visibility; screens-on is a run-book item |
| The benchmark shows nothing on a homogeneous LAN | Demo scenario 4: throttle one device deliberately (3.57×) |
| Judges discount the project as derivative | `NOTICE.md`; the scheduler is ours and upstream has not built it |
