# AI Swarm — Implementation Plan (from here forward)

**Status:** supersedes the previous version of this file. That version proposed a
strategy ("fork upstream, rebase our scheduler into it") written before any GPU
code existed locally. It is obsolete: the actual path taken was a targeted,
attributed import into `engine/upstream/` (see `UPSTREAM.md`), and that path has
now produced a **working, live-verified, two-device Qwen3 0.6B WebGPU swarm**.
This document starts from that real, evidence-backed state and plans forward.

**Last updated:** 2026-09-10
**Read this first if you are picking this project up cold.** Section 2 is ground
truth — everything in it is backed by a test, a saved report, or a command you can
re-run. Section 3 is the set of contracts every phase below depends on; skim it
before writing code so you don't re-derive something that already exists.

---

## 1. How to use this document

Each phase below has the same six parts:

1. **Objective** — the one sentence this phase has to make true.
2. **Current state** — what exists today, with exact file/line references. If this
   section is wrong by the time you read it, trust the code and fix this document.
3. **Gap / risk analysis** — precisely what is untested or missing, and why it
   matters.
4. **Implementation steps** — ordered, concrete, with the exact files to touch.
5. **Verification procedure** — exact commands and exact pass/fail criteria. Not
   "test it works" — a specific number, a specific log line, a specific token
   sequence.
6. **Exit gate** — the single condition that must be true before the next phase
   starts. Do not start the next phase on partial evidence; that is exactly how
   the project's own blueprint (`SWARMLLM_RESEARCH_AND_EXECUTION_BLUEPRINT.md`)
   says this class of project goes wrong.

Phases are numbered in dependency order, not priority order — A is the
recommended next phase, but the numbering exists so later phases can reference
earlier ones unambiguously.

---

## 2. Verified state as of 2026-09-10 — ground truth

### 2.1 What is proven, with evidence

| Claim | Evidence | Where to re-check it |
|---|---|---|
| 177 automated tests pass (wire, split, plan, recovery, qr, markdown, gpu-adapter) | `npm test` output | Run `npm test`; expect `30/11/29/13/25/26/43` passed across the six suites, 0 failed in each |
| Imported WebGPU kernels (`DenseEngine`) compute correctly on real hardware | Kernel micro-tests + f32/Q8/Q4 end-to-end self-test, bit-exact (`maxDiff: 0`) | `docs/gpu-reports/2026-09-09-phase1-gpu-selftest.json`; reproduce via `gpu-test.html` |
| `engine/factory.mjs` + `engine/gpu-adapter.mjs` load a real GGUF over a real HTTP range request correctly, whole-model and split | Independent CPU reference match to `1.1e-7` relative error, both whole and split | `docs/gpu-reports/2026-09-09-phase2-gpu-adapter-selftest.json`; reproduce via `gpu-adapter-test.html` (needs `node tools/build-test-gguf.mjs` first) |
| The real Qwen3 0.6B Q8_0 GGUF is downloaded, byte-verified | SHA-256 of the local file matches Hugging Face's own `X-Linked-ETag` exactly: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | `models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`, 639,446,688 bytes. Re-verify: `sha256sum models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf` |
| The room is model-aware: `join()` derives a real scheduler spec from GGUF tensor bytes, not a manifest guess | `room/room.js`'s `_loadModelSpec()` (line ~94) | Read the function; it branches on `descriptor.engineKind` |
| A worker adopts the host's model on `deal`, not its own local default | `room/room.js`, the `"deal"` case in `_onMsg` (line ~318) | Read the function; `m.model !== this.model` triggers `_loadModelSpec()` again |
| Qwen3 0.6B runs **solo** (all 28 layers + head, one device) and produces correct, non-zero, sane logits | Direct diagnostic script; `nonZero: 151936`, real logit values | Reproducible via the script in §4 of the retired diagnostic session — see Appendix C for the exact script |
| Qwen3 0.6B runs **split** (14+14 layers) across two independent WebGPU device acquisitions, connected by real WebRTC in the actual room UI, with mirrored chat | Live two-tab browser test; both tabs showed identical conversation | Manual — see §Appendix D for the exact repro steps |
| Solo and split produce **token-identical** greedy output for the same prompt on the real model | `docs/gpu-reports/2026-09-10-phase3-qwen3-0.6b-split-golden.json` — token IDs `[151667,198,32313,11,279,1196,9733,429,279,12884,374,6303]` match exactly | Re-run the script in Appendix C with `secondEngine` both null and non-null |
| The output is qualitatively coherent, not garbage | Decoded: `"<think>\nOkay, the user mentioned that the sky is blue and the grass is green. I need to respond in a way that's helpful..."` | Same report as above |

### 2.2 Two real bugs found and fixed this session — know these before touching the adapter

These are not hypothetical risks from the blueprint's risk table; they were hit,
diagnosed, and fixed against the real model. Anyone changing `engine/gpu-adapter.mjs`
or `room/room.js`'s model-loading path should understand both.

**Bug 1 — silent zero output from an under-limited WebGPU device.**
`requestDevice()` **without** an explicit `requiredLimits` grants only WebGPU's
spec-default limits (Chrome/Dawn: `maxStorageBufferBindingSize` defaults to
128 MiB), regardless of what the adapter reports it can actually do. Qwen3
0.6B's tied embedding/LM-head tensor is ~148 MB as Q8_0 — over that default.
Binding an over-limit buffer as a storage buffer **fails WebGPU validation**,
and validation failures are **not thrown exceptions** — they fire
`"uncapturederror"` and the compute dispatch that referenced the bad bind group
silently becomes a no-op. Storage buffers are zero-initialized, so the visible
symptom was: model loads "successfully", generates real tokens through every
layer, and then the **LM head produces all-zero logits**, which `argmax`
resolves to index 0 for every position (a real, decodable token — this is why
the demo produced a wall of `!!!!!!!!` rather than an obvious crash).

Fixed in `engine/gpu-adapter.mjs`, `acquireDevice()`: `requestDevice()` now
passes `requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize }`
explicitly. An `uncapturederror` listener also now records
`device.__lastValidationError` so a future occurrence of this class of bug is
loud instead of silent. **If you ever see all-zero or all-identical-token
output from a real model that loaded without a thrown error, suspect this
class of bug first** — check `device.__lastValidationError`, and check
whether any single tensor buffer exceeds the granted (not just the adapter-
reported) limits.

**Bug 2 — EOS token never detected for GPU-loaded models.**
`engine/gpu-adapter.mjs`'s `fetchHeader()` always calls `parseGGUFHeader(buf, { skipTokenizer: true })` (deliberately, to avoid materializing a 150k-entry
token array just to read layer shapes). This causes `parseGGUFHeader` (vendored,
unmodified — see `engine/upstream/gguf.js`) to **drop every key whose name
starts with `"tokenizer."` from `meta`, including `tokenizer.ggml.eos_token_id`**,
not just the bulk arrays. So `cfg.eos` was always `null` for a `dense-gguf`
model, and the room's generation loop never stopped early — it ran to the
`maxTokens`/context ceiling every time.

Fixed in `room/room.js`'s `_loadModelSpec()`: the turn-closing token is now
derived from the **tokenizer's own special-tokens map** —
`this.tok.special.get("<|im_end|>")` — set as `this.eosId`, used by the
generation loop instead of `this.engine.cfg.eos`. This is model-agnostic (every
ChatML model, including SmolLM2, closes a turn with `<|im_end|>`) and does not
depend on GGUF metadata at all. **If you add a model that does not use ChatML,
this needs a per-descriptor override** — see Phase F/G below, which will hit
this the moment a model's chat template differs.

### 2.3 What is *not* yet proven — do not claim these

**Updated 2026-09-10, second pass: Phases A, B and D below are now closed.**
The three bullets they used to justify are struck through and kept for
context; two genuine gaps remain at the bottom.

- ~~**No GPU-worker recovery test has been run.**~~ **Closed (Phase A).** A
  live two-tab Qwen3 0.6B room survives a worker leaving mid-conversation:
  falls back to solo, reloads the model, replays history, and produces output
  token-identical to the Phase 3 golden reference. This also found and fixed
  a real bug the CPU recovery test could not have caught (a 20 s recovery
  stall when the peer leaves between laps rather than mid-lap, and a GPU
  buffer leak on every reload). See
  `docs/gpu-reports/2026-09-10-phase-a-gpu-recovery.json`.
- ~~**No cross-implementation reference check.**~~ **Closed (Phase B), with a
  caveat.** Greedy output matches an independent `transformers` reference
  (`Qwen/Qwen3-0.6B`, fp32) exactly — 36/36 prompt tokens, 12/12 generated
  tokens — for the fixed prompt used throughout this document. `qwen3-0.6b`
  is now `STATUS.VERIFIED`. The caveat: no C/C++ compiler and no
  `llama-cpp-python` wheel were available in this environment, so this
  checks against **unquantized (fp32)** weights, not a Q8_0-native tool
  (llama.cpp) — weaker evidence than Step B.1 of this document originally
  asked for, though it does meet §5.5's literal exit-gate wording (an exact
  match against a named, versioned reference implementation). Re-run against
  llama.cpp wherever a compiler is available, if a fully quantization-aware
  check is needed. See `docs/gpu-reports/2026-09-10-phase-b-reference-check.json`.
- The generation-length problem this section used to imply (60 tokens cutting
  a `<think>` block short) is also closed — see Phase D below, now done: the
  per-model cap is calibrated from a live measurement (275 tokens needed for
  a real prompt) rather than the old flat 60.
- **Only one physical GPU has run any of this** — an AMD GCN-5 adapter inside
  this development environment's browser pane. Nothing has been measured on
  the Chrome/Edge + real laptop/phone hardware the blueprint's demo scenario
  actually targets. Phase E below. **Still open** — not attempted this pass;
  no second physical device was available.
- **The tokenizer is not a byte-exact port of Qwen's real pre-tokenizer
  regex.** `tools/tokenizer.mjs` was written for SmolLM2's GPT-2-style
  byte-level BPE. It correctly loads Qwen3's real vocab, merges, and special
  tokens (verified — `<|im_start|>`/`<|im_end|>`/etc. all resolve correctly),
  but its pre-split regex is GPT-2's, not Qwen's
  (`(?i:'s|'t|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|...`). Phase B's reference
  check independently confirms the two regexes agree for the one plain-English,
  no-digits, no-unusual-whitespace prompt tested (36/36 prompt tokens matched
  Hugging Face's own tokenizer exactly). **Still open for the general case** —
  a single confirmed prompt is not a proof for all inputs. Do not trust this
  for prompts containing multi-digit numbers, tabs, or multiple consecutive
  newlines until a proper regex port is written.

---

## 3. Contracts to know before writing code

### 3.1 The four-call engine contract

Every engine — `CpuEngine`, `GpuEngineAdapter`, and any future
`Qwen35Adapter` — must expose:

```js
async embedRun(tokenId, pos)   // host   : embed a token, then run my layers
async runHidden(x, pos)        // worker : run my layers on an incoming hidden state
async headFromHidden(x)        // host   : final norm + LM head -> logits
reset()                        // all    : new conversation, caches back to position 0
```

Plus these fields, read directly by `room/room.js`:
`layerCount` (getter), `bytesLoaded`, `maxSeq`, `cache` (nullable), `cfg`
(only `cfg.eos` was historically read; **as of Bug 2's fix, `room.js` no
longer reads `cfg.eos` — it reads `this.eosId` instead**, so a new engine
kind does not need to populate `cfg.eos` correctly, but should still populate
`cfg` with whatever real config fields exist for debugging).

`GpuEngineAdapter` additionally exposes `dispose()` (calls
`this.device?.destroy?.()`). **`CpuEngine` does not have a `dispose()`
method.** Any code calling it must use `this.engine?.dispose?.()`, never
`this.engine.dispose()` unconditionally.

### 3.2 The model descriptor shape (`models/registry.mjs`)

```js
{
  id, label, status,              // STATUS.VERIFIED | EXPERIMENTAL | PLANNED
  engineKind,                     // "cpu-smollm" | "dense-gguf" | "qwen35-gguf"
  architecture,                   // GGUF general.architecture value, e.g. "qwen3"
  modelUrl, configUrl, tokenizerUrl,
  upstreamUrl,                    // canonical source, for provenance when modelUrl is a local mirror
  expectedFormat,                 // "Q8_0" | "Q4_0" | ...
  maxSeqDefault,
  wireDtype,
  sourceRevision,                 // human string: repo@commit, byte length, sha256
  minRoomEnvelopeBytes,
  capabilityRequirements: { webgpu, shaderF16 },
  why,                             // shown in the UI's model picker
  blockedOn,                       // shown when status is PLANNED
}
```

`dir` replaces `modelUrl`/`configUrl`/`tokenizerUrl` for `engineKind: "cpu-smollm"`.

### 3.3 The engine factory boundary

`room/room.js` never imports `CpuEngine` or `GpuEngineAdapter` directly. It
calls `createEngine(descriptor, opts)` from `engine/factory.mjs`, which
dispatches on `descriptor.engineKind`. `opts` includes `layerRange, hasEmbed,
hasHead, maxSeq, onProgress`, and for `dense-gguf`, optionally `device, caps`
(reuse an already-acquired device) and `preloaded: { header, cfg }` (reuse an
already-fetched GGUF header, avoiding a second range-fetch).

### 3.4 Model-aware room lifecycle

```
join()
  -> _loadModelSpec(getModel(this.model))     // spec, tokenizer, eosId, wireEnc
  -> mesh.connect()
  -> profile() (synthetic, pre-load estimate)

start() / _recover()
  -> solvePlan(spec, devices, rtt, {strategy})
  -> _deal(plan)
       -> send "deal" {model, range, next, host} to every worker
       -> _load(myRange, hasEmbed, hasHead)   // host's own range
       -> wait for all "ready"

"deal" handler (worker side)
  -> if m.model !== this.model: _loadModelSpec(getModel(m.model)) again, discard old engine
  -> if same range as before: engine.reset()  (cheap path)
  -> else: _load(m.range, false, false)
```

`_load()` calls `createEngine()`, passing `preloaded` when
`this._ggufHeader` is already cached from `_loadModelSpec()`.

**Known gap in this lifecycle, relevant to Phase A:** neither the `"deal"`
handler (`room/room.js` ~line 320) nor `_deal()` (~line 570) ever calls
`this.engine?.dispose?.()` before replacing `this.engine`. For `CpuEngine`
this is harmless (GC handles it). For `GpuEngineAdapter`, **this leaks the
old engine's WebGPU buffers** — weights, KV cache, everything — because
nothing ever calls `device.destroy()` on the superseded engine's device. On a
memory-constrained GPU (this project's actual dev hardware needed a split to
fit the full model), this will exhaust memory within a few re-plans/recoveries
if not fixed. This is the first concrete task in Phase A.

### 3.5 Environment gotchas discovered this session (read before testing)

- **Two-port dev server.** `tools/serve.mjs <PORT>` serves HTTPS on `PORT`
  and plain HTTP on `PORT-1`. `npm run serve` uses port 8443, so
  `http://localhost:8442` (no cert warning) is the one to use for testing.
  If the dev server process dies (it did once this session, silently, mid-test)
  every `fetch()` from an already-open tab will throw `Failed to fetch` / show
  as `ERR_CONNECTION_REFUSED` in the console — restart with `npm run serve`.
- **Same-document hash-only navigation does not reload the page.** Navigating
  a tab that is already on `room.html` to `room.html#r=X&model=Y` when it was
  previously on `room.html#r=A&model=B` is a **same-document navigation** in
  Chromium — the module script does **not** re-run, so `hash`/model-selection
  logic evaluated at load time keeps its stale value. **Always open a fresh
  tab** (`tabs_create`) for each simulated "device" in a manual multi-tab
  test; do not reuse a tab across room codes.
- **Screenshot-vs-real coordinate scaling.** The Browser pane's `computer`
  tool coordinates are sometimes reported in a different scale than the most
  recent screenshot when a tab was backgrounded/refronted; a stale coordinate
  can silently click nothing. Prefer `find()` → `ref`-based clicks, and take
  a fresh `screenshot` immediately before any coordinate-based click.
- **This dev machine's real GPU memory ceiling is well under WebGPU's
  reported limits.** The adapter reports `maxBufferSize`/
  `maxStorageBufferBindingSize` of ~2 GB, but a single 610 MB model plus
  KV cache is enough to be worth watching closely. Now that Bug 1 is fixed,
  solo load of the full 28-layer model at `maxSeq: 512` **does** succeed on
  this hardware — but do not assume equivalent headroom exists on other
  hardware without measuring it (Phase E).

---

## 4. Phase A — GPU worker recovery (recommended next phase)

### 4.1 Objective

A worker holding GPU-loaded layers disconnects mid-generation; the room
re-plans, re-deals, replays history into the new holder's KV cache, and
resumes — producing the **same answer** it would have without the failure —
exactly as the CPU path already does, proven by `tests/recovery.test.mjs`.

### 4.2 Current state

- `room/room.js`'s `_recover()` (search for `async _recover(why)`) is fully
  engine-agnostic: it calls `_devices()`, `solvePlan()`, and `_deal(p, { replay: true })`.
  Nothing in it is CPU-specific.
- `_deal()`'s reload branch calls `_load()`, which calls `createEngine()`,
  which already correctly routes to `GpuEngineAdapter.load()` for a
  `dense-gguf` model. **The reload mechanism itself needs no new code** —
  it was already made generic by the Phase 2 factory work.
- `_replay()` re-runs `_step()` for every historical token, which calls
  `engine.embedRun()`/`runHidden()`/`headFromHidden()` — again, already
  generic.
- **What has never executed once:** any of the above with a `GpuEngineAdapter`
  instance actually being disposed, reloaded, or replayed into. This entire
  phase is about proving (and where necessary, fixing) behavior that today
  is untested assumption, not measured fact.

### 4.3 Gap / risk analysis

| Risk | Why it's plausible | How Phase A addresses it |
|---|---|---|
| GPU buffer leak on every reload/recovery (see §3.4) | Confirmed by code inspection: `dispose()` is never called anywhere in `room/room.js` (verified via `grep dispose room/room.js` → no matches) | Step A.1 |
| `device.lost` firing *during* replay (not just during initial load) is unhandled mid-replay | `acquireDevice()`'s `device.lost` handler just records `device.__lost`; nothing in `_replay()`/`_step()` checks it or produces a clean error | Step A.2 |
| A dropped worker's **neighbor** (the device that was forwarding hidden states to it) needs to learn its `next` pointer changed | Already handled generically by `_deal()` resending fresh `{next}` to every chain member — but never tested with a GPU device sitting in the "next" role | Step A.4 (live test) |
| Replaying a long history into a GPU engine may be slow enough to matter for a demo | GPU dispatch has fixed per-call overhead (command encoder + submit + readback) that a CPU function call doesn't; replay does one call per historical token, sequentially | Step A.3 (measure, don't guess) |
| The host itself might be the device that leaves | Explicitly out of scope — the codebase and blueprint both say host loss is unrecoverable by design (transcript/tokenizer/head live only on host). Do not attempt to fix this in Phase A. | N/A — confirm the existing "host-lost" event still fires correctly and stop there |

### 4.4 Implementation steps

**Step A.1 — Fix the GPU buffer leak (do this first; everything else depends
on the room not silently exhausting memory across the tests you're about to
run).**

In `room/room.js`, both places that replace `this.engine` need to dispose the
outgoing one first:

1. In the `"deal"` handler (`_onMsg`, the `case "deal":` block), immediately
   before `this.engine = null;` (currently just before the range-changed
   branch that discards the old engine on a model change), add:
   ```js
   this.engine?.dispose?.();
   ```
2. In `_load()` itself (not `_deal()`) — the cleanest place, since **every**
   path that replaces `this.engine` goes through `_load()` except the
   "kept-range" fast path (which intentionally does *not* discard the engine —
   correct, leave that alone). At the top of `_load(range, hasEmbed, hasHead)`,
   before calling `createEngine(...)`:
   ```js
   this.engine?.dispose?.();
   ```
   This single addition covers the deal handler's reload branch, `_deal()`'s
   own host-range (re)load, and any future caller of `_load()` — one place,
   can't be wrong twice.
3. **Do not** add disposal to the "kept-range" branches (`this.engine.reset()`
   calls) — those intentionally keep the same engine/buffers because the
   layer range didn't change; disposing there would defeat the entire point
   of the "keep weights, only clear cache" optimization the recovery design
   already relies on (see the comment above `_deal()` in room.js).
4. Add a matching unit-level assertion: since this can't be tested in Node
   (no WebGPU), add a **spy-based** test instead — a fake engine object with
   a `dispose` method that increments a counter, injected via a test-only
   path, OR (simpler, recommended) skip a Node test for this specific
   mechanic and rely on the browser harness in Step A.4 to prove it via
   actual GPU memory behavior (if disposal didn't happen, the second load in
   a two-cycle recovery test would run out of memory and throw — that failure
   *is* the test).

**Step A.2 — Handle `device.lost` during replay/generation, not just during
load.**

Currently `acquireDevice()` records `device.__lost` but nothing reads it
outside of `GpuEngineAdapter.load()`'s own error path
(`explainDeviceLoss()`). If the device is lost *after* successful load —
e.g., during `_replay()` or a live `_step()` — the failure will surface as
whatever WebGPU error the in-flight `mapAsync`/`writeBuffer` call throws,
which per Bug 1's investigation can be a confusing Dawn-internal message.

1. In `engine/gpu-adapter.mjs`, export a small helper:
   ```js
   export function checkDeviceLost(device) {
     if (device.__lost) {
       throw new Error(`GPU device lost (${device.__lost.reason}): ${device.__lost.message || "no detail"} — this worker cannot continue; it must be re-planned around.`);
     }
   }
   ```
2. In `GpuEngineAdapter`'s `embedRun`/`runHidden`/`headFromHidden`, wrap the
   call to `this.dense.*` in a try/catch that calls `checkDeviceLost(this.device)`
   first if the underlying call throws, so a lost-device failure during
   generation produces the same clear message a lost-device failure during
   load does, rather than a raw Dawn exception.
3. **Do not** try to make the room auto-recover from a lost device
   transparently inside the engine — that is `room.js`'s job
   (`_stepSafe()` already catches any error from `_step()` and calls
   `_recover()`). The engine's job here is only to make the *error message*
   honest, not to implement recovery logic twice.

**Step A.3 — Measure replay cost on a real GPU engine before claiming a
number.**

The CPU recovery demo's headline number was "13.1 s to recover after 58
tokens." Do not carry that number over to GPU — measure fresh.

1. Add timing around `_replay()` in `room/room.js` (it likely already emits
   `"replaying"`/`"replayed"` events with a token count — check
   `_replay()`'s existing `_emit` calls and add wall-clock timing to the
   `"replayed"` payload if not already present).
2. In the live test (Step A.4), record: layers reloaded (byte count, from
   the `"loaded"` event), replay token count, and wall-clock time from
   "recovering" to "recovered". Put these in a new
   `docs/gpu-reports/2026-09-1X-phase-a-gpu-recovery.json` — same convention
   as the existing GPU reports.

**Step A.4 — Build the live, real-hardware test.**

This cannot be a Node test (no WebGPU). Two options, do both:

*Option 1 — Minimal, fast: two-tab test using the room UI directly.*

1. Start two fresh tabs (per §3.5, never reuse a tab across room codes).
   Host selects `qwen3-0.6b`, strategy `even`, creates a room.
2. Worker joins via the printed join link (carries `&model=qwen3-0.6b`
   automatically).
3. Host starts the swarm (14+14 split, as already proven working).
4. Host sends a prompt that will generate at least ~20 tokens (use a longer
   `maxTokens` — see Phase D — or a prompt that produces a short, complete
   `<|im_end|>`-terminated answer once Phase B's non-thinking mode is sorted;
   for Phase A alone, generating partway is enough).
5. **Mid-generation**, on the **host** tab, call the existing "kill a node"
   demo instrument: `room.dropWorker()` (exposed via the `#kill`/"Drop a
   device" button in `room.html`) targeting the worker.
6. Observe: `"chain-broken"` fires, `_recover()` runs, a new plan is solved.
   Since there is only one survivor (the host) and it already proved it can
   hold all 28 layers solo (§2.1), the expected plan is **solo** — this is
   the single cleanest recovery scenario to prove first, directly analogous
   to the CPU test's "a room down to one capable device falls back to
   running solo" case.
7. Confirm: the host's own range grows from `[0,14)` to `[0,28)` — i.e., it
   must **load the previously-worker-held layers `[14,28)`** in addition to
   keeping its own `[0,14)` (which, per Step A.1's design, it should keep
   without re-downloading — check the `"kept-range"` vs the reload event to
   confirm this optimization actually fires for the host's *unchanged*
   sub-range... **note:** the current recovery code deals whole ranges per
   device, not partial-range merges, so verify by reading the actual `"loaded"`
   event: does it report loading all 28 layers fresh, or does the log show
   evidence of partial reuse? Either is "recovery worked"; only the byte
   count differs. Record whichever actually happens — do not assume.
8. Confirm the answer **continues from the same partial text** shown before
   the kill (not restarted), and that the tokens generated after recovery are
   what a completely uninterrupted solo run would have produced — this is
   the real bar, matching the CPU test's "answer is IDENTICAL to the
   uninterrupted run" assertion. To check this rigorously: capture the exact
   token IDs generated before the kill (from `stats`/console) and separately
   run the *same prompt* solo, uninterrupted, for the same total token count
   (using the Appendix C script), then diff.

*Option 2 — Thorough, matches the CPU test's structure: three tabs.*

1. Three tabs, `strategy: "even"` or a manual pledge configuration that
   forces a genuine 3-way split (e.g., roughly 9/9/10 layers).
2. Kill the **middle** device mid-generation (matching
   `tests/recovery.test.mjs`'s `runWithFailure(1, "the middle device leaves")`
   case exactly) — this is the harder case because the chain topology itself
   must be re-ordered, not just shrunk.
3. Same verification as Option 1: continued answer, correct token sequence
   compared against an uninterrupted 2-device reference run (not solo, since
   two survivors remain).
4. Repeat killing the **last** device.

Do Option 1 first — it is the direct GPU analog of the "down to one capable
device" case already proven on CPU, and it is the cheapest to set up. Only
do Option 2 if Option 1 passes cleanly; a 3-way GPU split has not been
attempted at all yet and may surface its own new issues (e.g., a middle
device is both a `runHidden` source and target — first time that code path
runs on GPU).

### 4.5 Verification procedure

1. `npm test` — must still show `177/177` (no regression from Step A.1/A.2's
   changes; these are additive/defensive, should not change any existing
   assertion).
2. Run Option 1 above. Pass criteria, all required:
   - `"chain-broken"` and `"recovering"` events fire (check via the room's
     event log in the UI, or console).
   - A new plan is solved and logged (`"plan (...)"` log line).
   - The host successfully loads/keeps the full model range without a GPU
     device-lost error (i.e., Step A.1's fix actually prevented the leak from
     mattering — if you skip Step A.1 and this still passes, that alone is
     useful evidence the leak is not yet severe enough to matter at this
     scale, but implement the fix anyway since it is correct regardless).
   - `"recovered"` fires with a token/time count.
   - The full generated answer (pre-kill tokens + post-recovery tokens),
     decoded, reads as **one coherent continuation**, not a restart and not
     garbage.
   - The token-ID sequence matches an uninterrupted reference run for the
     same prompt and total token count (see Appendix C for how to generate
     that reference).
3. Save a dated report to `docs/gpu-reports/` following the existing JSON
   convention (see `docs/gpu-reports/README.md`), including: layers reloaded,
   bytes downloaded during recovery, replay token count, wall-clock recovery
   time, and the token-ID match result.

### 4.6 Exit gate

A live two-device Qwen3 0.6B room survives a worker disconnect mid-generation
and produces a token-identical continuation to an uninterrupted reference run,
with the GPU buffer disposal fix in place and verified not to regress
`npm test`. A dated report exists in `docs/gpu-reports/`.

---

## 5. Phase B — External reference cross-check (promote Qwen3 0.6B to Verified)

### 5.1 Objective

Prove the room's greedy-decoded output for a fixed prompt matches a
trusted, independent implementation's output for the identical model file,
quantization, and prompt — the one piece of evidence Phase 3's work does not
yet have (see §2.3).

### 5.2 Current state

No reference implementation has been run anywhere in this project. The
`Qwen3-0.6B-Q8_0.gguf` file already on disk (`models/qwen3-0.6b/`) is exactly
the artifact any reference tool would also load, so no new download is
needed for this phase.

### 5.3 Implementation steps

**Step B.1 — Get a reference inference tool running.** In order of
preference (try each in order; use the first that works in this
environment):

1. **`llama.cpp`'s `llama-cli`**, if it can be built or a prebuilt binary
   obtained. This is the most authoritative reference for a GGUF file
   specifically, since GGUF is llama.cpp's own format.
   ```bash
   git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
   cd /tmp/llama.cpp && cmake -B build && cmake --build build --config Release -j
   ./build/bin/llama-cli -m <path-to-gguf> -p "<exact prompt>" --temp 0 -n 12 --no-display-prompt
   ```
   `--temp 0` forces greedy decoding, matching this project's `argmax`
   sampler exactly. `-n 12` matches the golden test's 12-token generation
   length for a direct comparison.
2. **`llama-cpp-python`** (`pip install llama-cpp-python`), if a Python
   environment with pip/network access is available and building llama.cpp
   from source is impractical in this environment. Load with
   `Llama(model_path=..., verbose=False)`, call
   `llm.tokenize(prompt.encode())` then `llm(prompt, max_tokens=12, temperature=0)`,
   and request logprobs/token IDs, not just text, so you can diff token IDs
   directly rather than decoded strings (decoded-string comparison hides a
   token-boundary difference that happens to decode to the same characters).
3. **Hugging Face `transformers`**, only if neither of the above is
   available — this uses the *unquantized* weights (`Qwen/Qwen3-0.6B`, not
   the GGUF), so it validates the architecture/tokenizer but does **not**
   validate Q8_0 quantization correctness. Treat a pass here as weaker
   evidence than a llama.cpp match, and say so explicitly in the report.

**Step B.2 — Use the exact same prompt, byte for byte.** Use the ChatML
string already used throughout this session (see Appendix C), not a
"similar" prompt. Any difference in system-prompt text, whitespace, or
newline placement changes the tokenization and invalidates the comparison.

**Step B.3 — Compare token IDs, not decoded text.** Extract the room's own
output token IDs (Appendix C's script, `soloTokens`) and the reference
tool's output token IDs for the same prompt and length. A text-level match
can hide a tokenizer discrepancy that happens to produce visually identical
output; an ID-level match cannot.

**Step B.4 — If they diverge, root-cause before concluding "broken."**
Given §2.3's known tokenizer caveat, the first thing to check on any
divergence is whether the *input* token IDs (prompt encoding) already differ
between this project's `Tokenizer` and the reference tool — if so, the
divergence is in `tools/tokenizer.mjs`'s pre-tokenizer regex, not in the GPU
inference path already proven correct by Phase A/3's work. Print and diff
`tok.encode(prompt)` against the reference tool's prompt token IDs
*before* comparing generated tokens.

**Step B.5 — If the tokenizer is the culprit, fix it properly rather than
patching around it.** Port Qwen's actual pre-tokenizer regex (see §2.3 for
the exact pattern, taken directly from the real `tokenizer.json`'s
`pre_tokenizer.pretokenizers[0].pattern.Regex`) into a Qwen-specific
tokenizer path. Given `tools/tokenizer.mjs`'s `Tokenizer` class already
parametrizes almost everything from the JSON file except the hardcoded
`SPLIT` regex and the digit-isolation special-case in `_pieces()`, the
cleanest fix is to make the pre-tokenizer regex a per-instance property
(read from `json.pre_tokenizer` when present, falling back to the current
GPT-2 pattern when absent) rather than hardcoding one regex for both model
families.

### 5.4 Verification procedure

1. Reference tool's output token IDs for the fixed prompt, 12 tokens,
   `temperature=0`, recorded.
2. This project's room output token IDs for the identical prompt/length
   (Appendix C script), recorded.
3. Exact match required for a "Verified" claim. Partial match (e.g., first
   N tokens agree, then diverge) is still useful data — record where the
   divergence starts and what changes there (a `<think>` block boundary is
   a plausible divergence point if the reference tool applies a different
   default chat template than the raw ChatML string used here).
4. Update `models/registry.mjs`'s `qwen3-0.6b` entry: `status: STATUS.VERIFIED`
   only if the match is exact end-to-end. If partial, leave as
   `EXPERIMENTAL` and record the exact caveat in `why`.
5. Save `docs/gpu-reports/2026-09-1X-phase-b-reference-check.json` with both
   token sequences, the tool/version used, and the verdict.

### 5.5 Exit gate

Either (a) an exact token-ID match against a named, versioned reference
implementation, with the registry updated to `VERIFIED`, or (b) a documented,
root-caused divergence with a concrete fix identified (even if not yet
implemented) and the registry left honestly at `EXPERIMENTAL`.

---

## 6. Phase C — Per-hop telemetry

### 6.1 Objective

Every token's latency breaks down into GPU compute time, network hop time,
and pack/unpack time, visible in the UI and logged, so "why was that token
slow" has an answer.

### 6.2 Current state

`room.js`'s `_onFrame` already times the whole `runHidden` call per worker
(`const t0 = performance.now(); ... const ms = performance.now() - t0;`,
emitted as a `"stage"` event) and `_step()` times each hop
(`this.stats.hops.push(...)`). This is hop-level and hidden-layer-level, but
does **not** separate GPU dispatch time from JS-side pack/unpack
(`packWire`/`unpackWire`) time within a stage, nor does it separate GPU
compute from the network transit inside a hop.

### 6.3 Implementation steps

1. In `engine/gpu-adapter.mjs`'s `embedRun`/`runHidden`/`headFromHidden`,
   wrap the GPU submit+readback in its own timer, separate from the
   surrounding room-level timer, and expose it (e.g., return `{ result, gpuMs }`
   or accumulate on the engine instance as `this.lastGpuMs`).
2. In `room.js`'s `_onFrame`, split the existing per-stage timer into
   `packMs` (before dispatch), `gpuMs` (read from the engine), and `unpackMs`
   (after dispatch, before send).
3. Add these fields to the `"stage"` event payload; extend `room.html`'s
   live-measurement panel to show a breakdown (or at minimum, log it).
4. Update the benchmark record format (blueprint §8, "Benchmark record
   format") to include the new fields for any future recorded run.

### 6.4 Verification procedure

Run a live two-device Qwen3 0.6B split (already proven working), generate a
short answer, and confirm the emitted per-token breakdown sums to
approximately the total observed lap time (within measurement noise) —
`packMs + gpuMs + unpackMs + networkMs ≈ totalHopMs`. A breakdown that
doesn't roughly reconcile indicates a timing bug, not a real cost.

### 6.5 Exit gate

A recorded live run shows a per-stage timing breakdown that reconciles with
the observed total, for at least one real two-device Qwen3 0.6B answer.

---

## 7. Phase D — Generation length and conversation UX

### 7.1 Objective

A user can have an actual multi-turn conversation with Qwen3 0.6B, including
letting its `<think>` reasoning complete, without hitting an arbitrary demo
cap.

### 7.2 Current state

`room.js`'s `generate()` defaults `maxTokens = 60`. This session's live tests
repeatedly hit this cap mid-`<think>` block — not a bug, just a demo default
tuned for the original SmolLM2 CPU speed, not Qwen3's verbosity.

### 7.3 Implementation steps

1. Make `maxTokens` a per-model or per-descriptor default (e.g.,
   `descriptor.maxTokensDefault`), since a reasoning model genuinely needs
   more headroom than a small non-reasoning one. Thread it through
   `room.html`'s call to `room.generate()`.
2. Consider exposing a "thinking mode" toggle: Qwen3 supports disabling
   extended thinking via its chat template (`enable_thinking=False` in the
   HF chat template, or by prompting accordingly) — check whether a plain
   ChatML string without the thinking directive suffices, or whether the
   tokenizer/template needs an explicit flag. This directly affects how many
   tokens a "quick demo" answer needs.
3. Re-verify `context()`'s `maxSeq` accounting still leaves comfortable
   headroom at the new, larger `maxTokens` — `models/registry.mjs`'s
   `qwen3-0.6b.maxSeqDefault` is already `512` (restored in this session
   after Bug 1's fix made it safe); confirm a full-length conversation still
   fits.

### 7.4 Verification procedure

Run a live conversation that lets Qwen3 0.6B complete a full `<think>...</think>`
block and a final answer without truncation; confirm the "answer cut short"
event does **not** fire, and the conversation naturally stops at `<|im_end|>`
(verifying Bug 2's fix holds under real, longer generation, not just the
12-token golden test).

### 7.5 Exit gate

A recorded live run shows a complete, untruncated Qwen3 0.6B answer
(including a full think block) ending naturally at `<|im_end|>`.

---

## 8. Phase E — Multi-hardware validation

### 8.1 Objective

Everything proven so far on one AMD GCN-5 adapter inside this development
environment is re-confirmed on at least one *actual* target device: a real
Chrome or Edge browser on real demo hardware (a laptop, ideally the machine
intended for any live demo).

### 8.2 Implementation steps

1. Serve the app over the LAN HTTPS path (`npm run serve`, use the
   `https://<lan-ip>:8443` URL printed at startup — this is the path real
   phones/other laptops need, distinct from the localhost path used for all
   testing so far).
2. On the target hardware, run `gpu-test.html`, then `gpu-adapter-test.html`,
   then the live two-device Qwen3 0.6B split, in that order — do not skip to
   the live test on new hardware without the two self-tests passing first
   (this is the blueprint's own Phase 1 rule, and this session's experience
   shows exactly why: the self-tests would have caught the `requiredLimits`
   class of issue in isolation, faster, before ever touching the full model).
3. Save a new dated report to `docs/gpu-reports/` for each new device,
   following the existing convention — vendor/architecture, `shader-f16`,
   pass/fail, and (if run) the split-golden-token result for that hardware.

### 8.3 Verification procedure

Each new device gets its own report file. A device that fails the self-test
should be recorded as "Unavailable on this device" per the blueprint's
model-status tiers — do not attempt the live model test on hardware that
fails Phase 1's gate.

### 8.4 Exit gate

At least one device other than this development environment has a recorded
PASS on both self-test pages and, ideally, the live split test.

---

## 9. Phase F — Promote Qwen3 1.7B

### 9.1 Objective

Repeat Phase 3's entire gate (self-test → single-device golden → split →
recovery) for the next rung of the dense ladder.

### 9.2 Current state

`models/registry.mjs`'s `qwen3-1.7b` entry exists with real, cited URLs and
`status: STATUS.PLANNED`, explicitly `blockedOn: "Qwen3 0.6B must reach
Verified first"`. **Do not start this phase until Phase B's exit gate is
met** — this dependency is deliberate, not a formality: the tokenizer and
adapter code paths are shared, so an undiscovered bug at 0.6B scale would
just be re-discovered (more expensively, with a bigger download) at 1.7B
scale.

### 9.3 Implementation steps

1. Change `qwen3-1.7b.status` to `EXPERIMENTAL` only after Phase B closes.
2. Download and pin exactly as Phase 3 did for 0.6B: fetch the real
   `Qwen3-1.7B-Q8_0.gguf`, verify its SHA-256 against Hugging Face's
   `X-Linked-ETag` header (same `curl -sIL -H "Range: bytes=0-0"` trick used
   for 0.6B), update `sourceRevision` with the real, confirmed values — **not**
   the placeholder text currently in the registry.
3. Re-run the exact same probe sequence as Phase 3: parse the real header in
   Node, cross-check every field against the model's own `config.json`
   (hidden size, layer count, head counts, vocab — all independently
   fetchable), confirm `modelSpecFromGGUF()` produces sane numbers before
   ever touching a browser.
4. Run `gpu-adapter-test.html` unchanged (it uses the tiny synthetic
   fixture, not a real model, so no changes needed) as a pre-flight sanity
   check that nothing about the adapter itself regressed.
5. Live single-device solo test, live two-device split test (this model is
   explicitly called out in the blueprint as validating **three-device**
   loading too — attempt a 3-way split once 2-way passes), and a repeat of
   Phase A's recovery test at this scale.
6. Repeat Phase B's external reference cross-check at this scale — do not
   assume a reference match at 0.6B implies one at 1.7B; the same
   tokenizer/quantization code paths run, but a new model file could still
   reveal a new edge case (a different vocab padding boundary, a different
   quantization block-count remainder, etc.).

### 9.4 Verification procedure

Identical structure to Phase 3/A/B, at the new scale. Produces a full set of
dated reports in `docs/gpu-reports/` mirroring the 0.6B set.

### 9.5 Exit gate

Same six-part gate as the blueprint's own Phase 6 promotion checklist:
"GPU self-test → one-device deterministic golden → two-device split
equivalence → range-only download evidence → real Wi-Fi run →
worker-loss recovery → benchmark entry", all passed and recorded for
Qwen3 1.7B specifically.

---

## 10. Phase G — Promote Qwen3 4B (three-device)

Structurally identical to Phase F, one rung up, with `qwen3-4b`'s
`blockedOn` updated once Phase F closes. The blueprint calls out this rung
specifically for "three-device loading, cache, and capacity demonstrations"
— treat a genuine 3-device test (not just 2) as required for this phase's
exit gate, not optional, since capacity-through-more-devices is the actual
product claim being demonstrated at this scale.

Additional consideration specific to 4B: at ~4.6 GB total (per the
blueprint's planning envelope), verify actual per-device memory budgets
before assuming a 3-way even split is even the right strategy — this may be
the first model where the "optimal" strategy's subset-selection (excluding a
weak device entirely) actually differs visibly from "even split" in a live
demo, which is worth specifically showcasing per the scheduler's own
stated value proposition.

---

## 11. Phase H — Qwen 3.8 27B hybrid engine

This is the largest remaining body of work and the one place this plan
cannot be as concrete as Phases A–G, because the hybrid engine's code has
never been imported into this repository and its exact interface is only
known from upstream's documentation, not from local, tested code.

### 11.1 Objective

Make the 27B hybrid model technically loadable and correct, single-device
first, matching the blueprint's own Phase 7/8 exactly.

### 11.2 Current state

- `models/registry.mjs`'s `qwen3.8-27b` entry is `status: STATUS.PLANNED`,
  `modelUrl: null`, with `blockedOn` naming the exact two missing files:
  `engine/upstream/qwen35.js` and `engine/upstream/wgsl/qwen35.js`, plus the
  scheduler's documented uniform-layer limitation
  (`scheduler/plan.js`'s "NOT WIRED UP" comment on `allocate()`'s `weights`
  parameter).
- Neither file exists anywhere in this repository yet. This is a real gap,
  not a formality — do not attempt to write a hybrid Gated-DeltaNet engine
  from scratch; vendor it from upstream exactly as `dense.js`/`gguf.js` were.

### 11.3 Implementation steps (following the blueprint's own Phase 7/8 exactly)

1. Vendor `qwen35.js` and `wgsl/qwen35.js` from the same audited upstream
   commit already pinned in `UPSTREAM.md`, with the same three-line
   attribution header convention already used for every other file in
   `engine/upstream/`. Update `UPSTREAM.md` and `THIRD_PARTY_NOTICES.md`
   accordingly — this is a hard requirement per this repo's own working
   rules (`CLAUDE.md` / `UPSTREAM.md`), not optional bookkeeping.
2. Write `engine/qwen35-adapter.mjs` following the exact same pattern as
   `engine/gpu-adapter.mjs`: header probing, cfg mapping (this time using
   `qwen35LayerNames()`/`qwen35Weights()`/`qwen35ShardBytes()` — already
   present in the vendored `engine/upstream/gguf.js`, unused until now — see
   the "qwen3.5/3.8 (hybrid delta-net) shard loader" section of that file),
   capability-classified device acquisition (reuse `acquireDevice()`
   as-is), and the same `requiredLimits` discipline Bug 1 taught this
   project is not optional for any model with a large tied-embedding table.
3. Add `"qwen35-gguf"` as a real case in `engine/factory.mjs` (currently a
   deliberate, clearly-messaged throw — replace only once the adapter above
   exists and is tested).
4. Implement **plain one-token decode only** first, on one device, following
   the blueprint's explicit sequencing: "No speculation until plain decode
   matches a trusted token reference." Do not implement batched prefill or
   MTP in the same pass.
5. Build model-specific golden fixtures covering **both** block types —
   Gated DeltaNet recurrent blocks and full-attention blocks — since a
   hybrid model's correctness bar is that both kernel families are right,
   not just one. The vendored `qwen35.js`'s own test/self-test conventions
   (if any ship with it) should be checked first before writing new ones.
6. Only after plain decode passes its golden test: implement the scheduler
   changes Phase H.4 below describes, then batched prefill, then MTP — each
   gated on the previous stage's golden test passing, per the blueprint's
   explicit ordering in its own Phase 7/8.

### 11.4 Scheduler changes required (blueprint's own Phase 8, concretized)

`scheduler/plan.js`'s `allocate()` already accepts a `weights` parameter for
per-layer cost that nothing currently supplies (`planOptimal` calls it with
`weights` defaulted to `null`). Making hybrid scheduling real requires all
four of the following **together** — the blueprint is explicit that doing
only some of these produces "a plan that looks great and runs badly":

1. A per-layer model-derived array (`layerProfile[i] = { family, weightBytes,
   kvBytesAtContext, recurrentStateBytes, decodeWork, prefillWork }`),
   analogous to `modelSpecFromGGUF()` but per-layer instead of collapsed to
   one uniform `layerBytes`/`layerMACs` pair. This is a genuinely new
   function, not an extension of the existing one (the existing one's whole
   design assumes uniformity).
2. Device profiling by layer family, not one scalar `msPerLayer` — the
   `scheduler/probe.js` synthetic pre-load probe and the post-load
   `_calibrate()` in `room.js` both need a DeltaNet-shaped and an
   attention-shaped synthetic workload, not one matvec shape.
3. `layerCap()`/`memoryFor()` in `scheduler/cost.js` taking a per-layer byte
   array and summing the actual assigned range's bytes, not
   `count * layerBytes`.
4. `allocate()`'s DP already accepts a `weights` array structurally — thread
   real per-layer compute-cost weights into it from `planOptimal`, replacing
   the current `null` default.

Test this exactly as the blueprint prescribes: "Test it against brute force
on small synthetic cases before trusting a performance claim" — write a new
`tests/hybrid-plan.test.mjs` with small synthetic heterogeneous rooms (a
handful of devices, a handful of layers with mixed `family`), comparing the
DP's answer against an exhaustive brute-force search, mirroring
`tests/plan.test.mjs`'s existing pattern for the uniform case exactly.

### 11.5 Verification procedure

Follow the blueprint's own promotion gate, unchanged: "GPU self-test →
one-device deterministic golden → two-device split equivalence →
range-only download evidence → real Wi-Fi run → worker-loss recovery →
benchmark entry." Do not skip straight to a multi-device demo because the
26B download and single-device correctness proof are themselves substantial
work — treat single-device Qwen 3.8 correctness as its own complete phase
with its own exit gate before distributing it at all, exactly as §11.3 step
4-6 sequences it.

### 11.6 Exit gate

A single device loads and generates deterministically with Qwen 3.8 27B
using the vendored engine's own reference configuration, passing its GPU
golden tests, **before** any distributed or scheduler work in this phase is
considered started (per the blueprint's Phase 7 note: "Keep this phase
single-browser. Do not debug a numerical kernel and WebRTC state
propagation at the same time.").

---

## 12. Phase I — Product hardening for a real demo

Only after Phase H's single-device milestone (not necessarily the full
distributed hybrid rollout) does the blueprint's Phase 9 hardening work
become worth doing broadly — but several items are cheap and valuable
earlier, and can be pulled forward opportunistically:

**Can do any time (no dependency on later phases):**
- TURN relay fallback for strict NATs (`room/mesh.js` currently STUN-only).
- A diagnostics/support-report export: browser/adapter info, cache state,
  last error, selected plan — much of this already exists as scattered
  events; this phase is mostly about collecting them into one exportable
  blob, matching the blueprint's "a copyable support report" requirement
  already implemented for `gpu-test.html` specifically (generalize that
  pattern to the whole room).
- A "clear local model cache" control (`clearWeightCache()` already exists
  in `engine/cpu.mjs` for the CPU path; `engine/gpu-adapter.mjs`'s
  `aiswarm-gguf-v1` cache needs the equivalent function and a UI control for
  both).

**Depends on Phase E (multi-hardware) having actually happened:**
- Rehearsing the real venue/demo setup end to end (HTTPS cert acceptance
  flow on a phone, cold-start timing measurement) is meaningless without
  having already run on real target hardware at least once.

**Depends on whichever model rung is the actual demo target being Verified:**
- Recording public-facing benchmark numbers (blueprint §8's exact record
  format) — only for a model whose correctness is actually established
  (Phase B/F/G/H's exit gates), never for an Experimental model, per this
  project's own stated voice/accuracy rules (`CLAUDE.md`).

---

## 13. Master sequencing table

| Phase | Depends on | Produces | Status |
|---|---|---|---|
| A — GPU recovery | Phase 3 (done) | Recovery report, buffer-leak fix | **Done** (2026-09-10) |
| B — Reference cross-check | Phase 3 (done) | Verified/Experimental verdict for 0.6B | **Done** (2026-09-10) — VERIFIED, fp32 reference (caveat: no compiler for llama.cpp) |
| C — Telemetry | Phase 3 (done) | Per-hop timing breakdown | Not started — low priority |
| D — Generation length | Phase 3 (done) | Untruncated live conversations | **Done** (2026-09-10) |
| E — Multi-hardware | Phase A+B ideally closed first | Reports for real target hardware | **Partly done** — the maintainer ran Qwen3 0.6B across two real physical devices successfully. No metrics recorded, and nothing above 0.6B has run on a second device |
| F — Qwen3 1.7B | **Phase B must close (`blockedOn`)** | Full gate at 1.7B | **Done** (2026-09-10) — VERIFIED, full gate incl. an exact external reference match |
| G — Qwen3 4B | **Phase F must close (`blockedOn`)** | Full gate at 4B, 3-device proof | **Partial** — kernels proven correct at hidden 2560; the full model loses the GPU device on this machine. Blocked on a second physical GPU, not on code |
| H — Qwen 3.8 27B | Independent of F/G, but very large | Vendored hybrid engine, single-device correctness | Not started — largest remaining scope |
| I — Hardening | Partially independent; see §12 for exact sub-dependencies | Demo-readiness | Not started |

**What actually happened, in order:** A and B were done together (they touch
different files — A is `room.js`'s recovery path, B is tokenizer/reference
tooling), and D alongside them, exactly as this table originally recommended.
A second pass then closed A's remaining Option 2 case (three peers, kill the
middle device: 1 ms to detect, 7.8 s to recover, token-identical output) and
Step A.2's live device-loss case, which turned up on its own during the 4B work.
F went the whole distance and is Verified. G stopped where the hardware stopped.

**The one thing now gating three separate phases is hardware, not code.** E, the
rest of G, and any 3-device capacity demonstration all need real devices with
real GPUs. `tools/probe-gguf.mjs` exists so that the entire read-only half of a
promotion gate — provenance, header/config agreement, and the capacity envelope
for 1-, 2- and 3-way splits — can still be run in Node on any machine before
anyone plugs in a second laptop.

---

## Appendix A — Reproducing the Phase 1/2/3 evidence

```bash
npm test                                    # 177 tests, all suites
node tools/build-test-gguf.mjs              # regenerates the tiny adapter-test fixture
npm run serve                               # dev server: http://localhost:8442 (no cert warning)
```

Then in a real WebGPU browser: `gpu-test.html`, then `gpu-adapter-test.html`,
both have a "Copy results" button producing the exact JSON saved under
`docs/gpu-reports/`.

## Appendix B — Re-verifying the Qwen3 0.6B download's identity

```bash
curl -sIL -H "Range: bytes=0-0" \
  "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf" \
  | grep -i "x-linked"
sha256sum models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```
The `X-Linked-ETag` header must equal the local file's SHA-256 exactly.

## Appendix C — The solo-vs-split golden-test script

Run via the browser's `javascript_tool` (or paste into the devtools console)
on any page that can `import` this repo's modules (e.g., `room.html` or
`gpu-adapter-test.html`, served from the dev server, in a real WebGPU tab):

```js
const { createEngine } = await import("/engine/factory.mjs");
const { getModel } = await import("/models/registry.mjs");
const { acquireDevice } = await import("/engine/gpu-adapter.mjs");
const { Tokenizer } = await import("/tools/tokenizer.mjs");
const { argmax } = await import("/engine/cpu.mjs");

const descriptor = getModel("qwen3-0.6b");
const tok = await Tokenizer.load(descriptor.tokenizerUrl);
const prompt = "<|im_start|>system\nYou are a helpful AI assistant running across several devices at once.<|im_end|>\n<|im_start|>user\nThe sky is blue and the grass is green.<|im_end|>\n<|im_start|>assistant\n";
const ids = tok.encode(prompt);

async function runGreedy(engine, secondEngine, N) {
  const out = [];
  let pos = 0, logits = null;
  for (const id of ids) {
    let x = await engine.embedRun(id, pos);
    if (secondEngine) x = await secondEngine.runHidden(x, pos);
    logits = secondEngine ? await secondEngine.headFromHidden(x) : await engine.headFromHidden(x);
    pos++;
  }
  for (let i = 0; i < N; i++) {
    const next = argmax(logits);
    out.push(next);
    let x = await engine.embedRun(next, pos);
    if (secondEngine) x = await secondEngine.runHidden(x, pos);
    logits = secondEngine ? await secondEngine.headFromHidden(x) : await engine.headFromHidden(x);
    pos++;
  }
  return out;
}
// SOLO
const { device: d1, caps: c1 } = await acquireDevice({ requireShaderF16: false });
const whole = await createEngine(descriptor, { layerRange: [0, 28], hasEmbed: true, hasHead: true, maxSeq: 64, device: d1, caps: c1 });
const soloTokens = await runGreedy(whole, null, 12);
d1.destroy?.();
// SPLIT
const { device: d2, caps: c2 } = await acquireDevice({ requireShaderF16: false });
const host = await createEngine(descriptor, { layerRange: [0, 14], hasEmbed: true, hasHead: false, maxSeq: 64, device: d2, caps: c2 });
const worker = await createEngine(descriptor, { layerRange: [14, 28], hasEmbed: false, hasHead: true, maxSeq: 64, device: d2, caps: c2 });
const splitTokens = await runGreedy(host, worker, 12);
d2.destroy?.();

console.log({ soloTokens, splitTokens, match: JSON.stringify(soloTokens) === JSON.stringify(splitTokens), decoded: tok.decode(soloTokens) });
```

Adjust `N`, the layer split point, and the prompt as needed for later phases
(e.g., Phase F/G at a different layer count).

## Appendix D — Live two-tab manual test repro steps

1. `npm run serve`, open `http://localhost:8442/room.html` in a **fresh**
   tab (host). Select `Qwen3 0.6B · Q8_0` in the model dropdown, leave the
   room code blank, click Join.
2. Copy the printed join link (contains `&model=qwen3-0.6b`).
3. Open a **second, fresh** tab, navigate directly to that join link
   (do not reuse a tab that was ever on a different room code — see §3.5).
4. On the worker tab, click Join (if not already auto-connected).
5. On the host tab, set the strategy dropdown to `even split` (forces a
   genuine 2-device split rather than the planner correctly choosing solo,
   which it would otherwise do since one device can hold the whole model).
6. Click "Start the swarm". Expect layers `[0,14)` on host, `[14,28)` on
   worker, both loading from the local mirror
   (`models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`).
7. Send a prompt. Expect coherent, on-topic output (a `<think>...</think>`
   block is expected and correct — Qwen3 reasons by default).
8. Confirm the worker tab's conversation view shows the identical exchange
   (mirrored chat).
