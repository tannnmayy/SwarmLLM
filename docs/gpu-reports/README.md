# GPU self-test reports

Recorded runs of the two GPU self-test pages, kept as evidence rather than a claim.
Each file is the exact JSON the page itself produces (via its "Copy results" button) —
nothing here is hand-typed.

| File | Page | What it proves |
|---|---|---|
| `2026-09-09-phase1-gpu-selftest.json` | [`gpu-test.html`](../../gpu-test.html) | The imported `DenseEngine` computes correctly on this GPU: every WGSL kernel (rmsnorm, quantized matvec, RoPE, attention, SiLU) and a full synthetic end-to-end forward pass at f32/Q8/Q4, all bit-exact (`maxDiff: 0`) against a CPU reference. |
| `2026-09-09-phase2-gpu-adapter-selftest.json` | [`gpu-adapter-test.html`](../../gpu-adapter-test.html) | The **adapter/factory boundary** (`engine/factory.mjs`, `engine/gpu-adapter.mjs`) loads a real GGUF file over a real HTTP range request, maps its metadata correctly, and produces the room's four-call contract — whole-model and split across two independent engine instances (standing in for a two-device room) both agree with an independent CPU reference to `1.1e-7` relative error. |
| `2026-09-10-phase3-qwen3-0.6b-split-golden.json` | manual script (see `IMPLEMENTATION_PLAN.md` Appendix C) | The real Qwen3 0.6B Q8_0 GGUF, loaded solo (28 layers, 1 device) and split (14+14 across two independent device acquisitions), produces token-for-token identical greedy output — not just numerically close. |
| `2026-09-10-phase-a-gpu-recovery.json` | live two-tab `room.html` (see `IMPLEMENTATION_PLAN.md` Phase A) | A real two-device Qwen3 0.6B split survives the worker leaving mid-conversation: the room re-plans to solo, reloads the full model, replays history, and resumes — producing output token-identical to the Phase 3 golden reference. Also documents a real bug found by this test (a 20 s recovery stall when the peer leaves between laps rather than mid-lap) and its fix, verified by re-running the same scenario before and after. |
| `2026-09-10-phase-f-qwen3-1.7b.json` | `node tools/probe-gguf.mjs` + live `room.html` (see `IMPLEMENTATION_PLAN.md` Phase F) | The full promotion gate for **Qwen3 1.7B Q8_0**, start to finish: byte-verified download, a three-way header/config/registry cross-check, adapter self-test, one-device golden, split equivalence, a live two-device WebRTC room showing range-only downloads (worker pulled 714.2 MB, not 1743.8 MB), worker-loss recovery in 52.0 s, and an exact token-ID match against an independent `transformers` fp32 reference. Also records the first rung whose wire is **lossy f16**, making split-vs-solo equality an empirical result rather than a guarantee. |
| `2026-09-10-phase-g-qwen3-4b.json` | `node tools/probe-gguf.mjs` + a direct load/run script | **Qwen3 4B Q8_0, partial.** Byte-verified download and a clean 21/21 header probe, and — the useful part — proof that the engine's kernels are *correct at this model's dimensions*: half the model (18 of 36 layers, 2235 MB) loads and computes finite, non-zero hidden states. The full model allocates (4076 MB) and then **loses the GPU device on the first inference pass**. That failure is a capacity ceiling on this one adapter, not a correctness bug, and no split on a single machine avoids it. Left `EXPERIMENTAL`; the live gate needs a second physical GPU. |
| `2026-09-10-phase-b-reference-check.json` | `pip install torch transformers` + a one-off script (see `IMPLEMENTATION_PLAN.md` Phase B) | Cross-checks the room's real Q8_0/WebGPU output against an independent reference implementation (`Qwen/Qwen3-0.6B` on `transformers`, fp32, unquantized) for a fixed prompt: exact match on prompt tokenization (36/36 tokens) and greedy-decoded output (12/12 tokens). Promoted `qwen3-0.6b` to `STATUS.VERIFIED` on the strength of this. Caveat: no C/C++ compiler or `llama-cpp-python` wheel was available in this environment, so this checks against fp32 weights, not a Q8_0-native reference — see the report's `verdict` field. |

## Hardware/environment note

Both runs above were captured inside this development environment's browser pane —
a real Chromium build (`Claude/1.49585.0 Chrome/152.0.7977.76`) with a real WebGPU
adapter (`amd gcn-5`, `shader-f16`, 2 GB buffer limit), not a mock or a headless
stub. That satisfies the blueprint's Phase 1 requirement ("do not start Qwen work
on an adapter that fails the self-test") for *this* machine.

It is **not** the same as running on the actual demo hardware named in the
blueprint's recommended sequence (the venue laptop/phone in Chrome or Edge). Before
a live demo, re-run both pages — `gpu-test.html` first, then `gpu-adapter-test.html`
— on every device that will actually be in the room, and drop a new dated report
here. A pass on this machine does not transfer to a different GPU/driver/browser.

## Reproducing

```bash
npm run serve                        # starts the dev server (see package.json)
node tools/build-test-gguf.mjs       # regenerates tests/fixtures/tiny-qwen3-q8.gguf
```

Then open `http://localhost:8442/gpu-test.html` and `http://localhost:8442/gpu-adapter-test.html`
in the target browser, click through, and use "Copy results" to save a new dated
JSON file in this directory.
