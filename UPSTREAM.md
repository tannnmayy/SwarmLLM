# Imported engine

## What is imported

`engine/upstream/` is a WebGPU inference engine imported from SwarmLLM, pinned to a
single commit and used **unmodified**. Keeping it byte-identical is deliberate: it
means upstream fixes can be re-imported by re-running the copy, and it keeps the
boundary between imported code and our own unambiguous.

| Pinned commit | `1c9763fcd7eb9c42865cd5937dceba678fe9d9a5` |
|---|---|
| Licence | MIT — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |
| Lines | ~2,570 across 12 files |

| File | Purpose |
|---|---|
| `engine.js` | public entry point, re-exports |
| `dense.js` | `DenseEngine` — dense Llama-architecture models (Qwen3, SmolLM), layer-shardable |
| `gguf.js` | GGUF parsing, quantisation repacking, streaming upload |
| `safetensors.js` | the safetensors path |
| `quant.js` · `sampling.js` · `tokenizer.js` · `autotune.js` | supporting modules |
| `selftest.js` | GPU self-test and kernel micro-tests |
| `wgsl/base.js` · `wgsl/coop.js` · `wgsl/gemm.js` | compute kernels |

The hybrid Gated-DeltaNet path (`qwen35.js`, `wgsl/qwen35.js`, ~1,570 lines) is
**not** imported. It is only needed for Qwen 3.8 27B, which is out of scope.

## Why this was worth importing

The imported engine already does two things our own CPU engine does not:

- **runs on the GPU**, where ours is JavaScript on the CPU at ~1.15 GMAC/s;
- **reads quantised GGUF** (Q4_0, Q4_1, Q5_K, Q6_K, Q8_0, F16, F32), where ours
  reads f16 safetensors and widens to f32 — four to eight times the memory.

It also already handles Qwen3's QK-norm (`attn_q_norm` / `attn_k_norm`), which our
engine does not, so Qwen3 0.6B / 1.7B / 4B become reachable without writing kernels.

## Why the two engines have the same shape

Both expose the same four calls, and that is not a coincidence — our engine was
written against the same published interface:

```js
embedRun(tokenId, pos)   // host   : embed a token, then run my layers
runHidden(x, pos)        // worker : run my layers on an incoming hidden state
headFromHidden(x)        // host   : final norm + LM head -> logits
reset()                  // all    : new conversation, caches back to position 0
```

Both are constructed with `{ layerRange, hasEmbed, hasHead, maxSeq }`. The room,
the scheduler and the recovery path therefore drive either one without changing.

## What remains ours

Everything outside `engine/upstream/`:

| | |
|---|---|
| `scheduler/` | cost model, exact placement solver, device profiler |
| `room/` | mesh, wire format, room orchestration, QR, Markdown, wake lock |
| `engine/cpu.mjs` | our CPU engine — kept as the reference the GPU path is checked against, and as the fallback for devices without WebGPU |
| `tools/`, `tests/` | tooling and the 133 gates |

## Re-importing

Bump the pinned commit, re-run the copy, re-run `npm test`, and update the SHA in
this file and in `THIRD_PARTY_NOTICES.md`. Because the imported files are unmodified,
this is a replace rather than a merge.
