# GPU self-test reports

Recorded runs of the two GPU self-test pages, kept as evidence rather than a claim.
Each file is the exact JSON the page itself produces (via its "Copy results" button) —
nothing here is hand-typed.

| File | Page | What it proves |
|---|---|---|
| `2026-09-09-phase1-gpu-selftest.json` | [`gpu-test.html`](../../gpu-test.html) | The imported `DenseEngine` computes correctly on this GPU: every WGSL kernel (rmsnorm, quantized matvec, RoPE, attention, SiLU) and a full synthetic end-to-end forward pass at f32/Q8/Q4, all bit-exact (`maxDiff: 0`) against a CPU reference. |
| `2026-09-09-phase2-gpu-adapter-selftest.json` | [`gpu-adapter-test.html`](../../gpu-adapter-test.html) | The **adapter/factory boundary** (`engine/factory.mjs`, `engine/gpu-adapter.mjs`) loads a real GGUF file over a real HTTP range request, maps its metadata correctly, and produces the room's four-call contract — whole-model and split across two independent engine instances (standing in for a two-device room) both agree with an independent CPU reference to `1.1e-7` relative error. |

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
