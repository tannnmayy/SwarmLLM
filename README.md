<h1 align="center">AI Swarm</h1>

<p align="center"><b>Many devices, one model — and it keeps running even if one of them walks away.</b></p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#the-scheduler">The scheduler</a> ·
  <a href="#what-we-found">What we found</a> ·
  <a href="CHECKLIST.md">Checklist</a> ·
  <a href="NOTICE.md">Prior art</a>
</p>

<p align="center">
  <img alt="tests" src="https://img.shields.io/badge/tests-295%20passing-3ddc84">
  <img alt="runtime" src="https://img.shields.io/badge/runs%20on-WebRTC%20%2B%20any%20browser-3b5bff">
  <img alt="install" src="https://img.shields.io/badge/install-none-7c5cff">
  <img alt="licence" src="https://img.shields.io/badge/licence-MIT-16171c">
</p>

---

One large language model, cut into contiguous slices, one slice per device. Open a
link in a browser tab and your phone or laptop becomes one stage of a model far
bigger than it could ever hold. Tokens pass between devices peer-to-peer over
WebRTC. The answer appears on every screen.

**Team SE7EN** · Hack Summit 7.0 · AI & Automation track

## What this actually is

Existing swarms deal layers in proportion to the memory a device *pledges*, in
whatever order the devices happened to join. Neither number says how fast a device
actually computes, or how far it sits from its neighbours — so one slow phone, or one
distant peer, sets the pace for every token the room produces.

We measure both, and solve the placement exactly.

|  | exo | prima.cpp | SwarmLLM | **AI Swarm** |
|---|---|---|---|---|
| Install | Python per node | Native binary | None | **None** |
| Layers placed by | pledged memory | measured, ILP solver | pledged memory | **measured cost, solved exactly** |
| Chain order | memory ring | fixed ring | join order | **solved over measured RTT** |
| Drops a device that hurts | no | yes | no | **yes** |
| Refuses to split when it should | no | yes | no | **yes** |
| Node leaves mid-answer | repartition | not claimed | not claimed | **heals in 13 s, measured** |

## Status

Everything ticked is backed by a test in this repository or a recorded run, not by
"it looked like it worked". Full detail in [CHECKLIST.md](CHECKLIST.md).

| Stage | State |
|---|---|
| Device probe — WebGPU, memory, bandwidth, NAT type | working |
| HTTPS + signalling dev server | working |
| Wire format — adaptive f32/f16, SCTP-sized slicing | working · **30/30 tests** |
| WebRTC mesh, direct peer links | working · bit-exact tensor bounce |
| Sliceable inference engine | working · SmolLM2 135M |
| Split output identical to a single device | working · **11/11 tests** |
| End-to-end swarm over WebRTC | working · answer mirrored to every screen |
| Profiler + placement solver + chain order + host election | working · **29/29 tests** |
| Fault recovery — a device leaves mid-answer | working · **13/13 tests** · verified live |
| Weight caching | working · second join downloads nothing |
| ChatML prompting + multi-turn + honest context limit | working |
| WebGPU kernels (imported `DenseEngine`) | verified on real hardware — kernel + end-to-end self-test, bit-exact at f32/Q8/Q4 ([report](docs/gpu-reports/2026-09-09-phase1-gpu-selftest.json)) |
| GPU engine factory + adapter (`engine/factory.mjs`, `engine/gpu-adapter.mjs`) | verified on real hardware — loads a real GGUF over an HTTP range request, whole-model and split across two engine instances both match an independent CPU reference to 1.1e-7 relative error ([report](docs/gpu-reports/2026-09-09-phase2-gpu-adapter-selftest.json)) · **34/34 tests** |
| Model ladder registry (`models/registry.mjs`) — SmolLM/Qwen3 0.6B/1.7B/4B/Qwen3.8 | descriptors carry honest status tags (verified/experimental/planned) and both origins each model can be served from. Qwen3 0.6B and 1.7B are wired in and run in a real room; 4B is experimental (blocked on a second GPU) and Qwen3.8 27B is a roadmap entry with no engine vendored |
| Weight delivery — Hugging Face CDN, local mirror as LAN/offline fallback | working · **58/58 tests** · pinned upstream URLs re-checked live against the registry's recorded byte length and SHA-256 ([report](docs/delivery/2026-09-10-upstream-verification.json), 39 passed / 0 failed) |
| Deployable as a static site + a standalone signalling service | working · `npm run build` / `npm run signal`, or a Cloudflare Worker + Durable Object · **26/26 tests** on the static handler ([how](docs/DEPLOY.md)) |
| Two-device run over the deployed topology | working · separate origins for site and signalling, weights from HF · worker pulled **191.4 MB** of 610 MB, **144 tokens at 5.44 tok/s**, median lap **49.5 ms** · **23/23 tests** on the protocol regressions it found |

**Measured, on two browsers:** 60 tokens at **4.99 tok/s**, median network lap
**63.6 ms**, answer byte-identical to the single-device reference.

**Measured, recovery:** a worker dropped **58 tokens into an answer**. The room
re-planned, recruited a device it had earlier stood down, replayed 58 tokens and
**recovered in 13.1 s** — finishing the sentence it was in the middle of.

## Quick start

Requires Node 20+. No Python, no CUDA, no build step.

```bash
npm install
npm run model      # downloads SmolLM2-135M and reshapes it into per-layer shards (~257 MB)
npm run cert       # TLS cert for THIS machine's LAN IP — re-run whenever the IP changes
npm run serve
```

Then:

- **this machine** — <http://localhost:8442/room.html>
  (`http://localhost` is already a secure context, so no certificate warning)
- **other devices, same Wi-Fi** — `https://<lan-ip>:8443/room.html`, accepting the
  self-signed certificate once (Android: *Advanced → Proceed*; iOS: *Show Details →
  visit this website*)

Open the room on two or more devices with the same code, drag the **pledge** slider
down so the model no longer fits on one device, and press **Start the swarm**.

`npm run model` is only needed for SmolLM2, whose per-layer shards are built
locally. Every Qwen3 rung is range-fetched from Hugging Face at run time, or from
`models/` if you have mirrored it — the page probes and picks. To put it on the
internet rather than a LAN, see **[docs/DEPLOY.md](docs/DEPLOY.md)**: a static site
plus one small signalling process, with the weights coming off HF's CDN.

> **WebGPU and WebRTC both require a secure context.** Plain `http://192.168.x.x`
> gives you neither, and the failure looks exactly like *"this phone has no WebGPU"* —
> which is why the dev server serves TLS with the LAN IP in the certificate's SAN.
> iOS checks the SAN, not the common name, so `npm run cert` is the first command to
> run at a new venue.

### Pages

| Page | What it is for |
|---|---|
| `room.html` | The room. Join, pledge, plan, generate, watch it heal |
| `probe.html` | What can this device do? WebGPU adapter, usable memory, effective bandwidth, thermal/battery APIs, NAT type |
| `mesh-test.html` | Do two devices connect directly, and does a tensor survive the trip bit-exactly? |

## How it works

One device is the **host**. It owns the conversation, the tokenizer, the embedding
table, the LM head and the sampler. The others are **workers**, each holding a
contiguous range of transformer layers. Together they form a chain in layer order.

```
host      tokenize → embed → run my layers ─┐
                                            ▼
worker A  layers 0–13 ──► worker B  layers 14–23 ──► worker C  layers 24–29 ──┐
                                                                              │
host      final norm → LM head → sample → next token ◄────────────────────────┘
```

Per token, one hidden state walks the whole chain and comes back. For SmolLM2-135M
that vector is 576 floats — **2.3 KB on the wire**. That small payload is what makes
this work over ordinary Wi-Fi, and it is why layer-splitting is the right shape:
tensor parallelism would need two collective operations *per layer*, which is
hopeless on anything slower than a PCIe bus.

### The contract between devices

The entire interface a device must implement is four calls:

```js
embedRun(tokenId, pos)   // host   : embed a token, then run my layers
runHidden(x, pos)        // worker : run my layers on an incoming hidden state
headFromHidden(x)        // host   : final norm + LM head → logits
reset()                  // all    : new conversation, caches back to position 0
```

Everything else — the placement solver, the recovery path, the dashboard — is built
on top of those four. The imported WebGPU engine exposes the same four, so both are
reachable through one contract — see [UPSTREAM.md](UPSTREAM.md) for the one place the
two genuinely differ, and how it was reconciled.

### Model sharding

`tools/fetch-model.mjs` downloads the model and reshapes it into **one file per
layer**, so a device downloads only the layers it was dealt — a plain `GET`, with no
range-request bookkeeping and no partial-cache edge cases.

```
embed.bin       54.0 MB   host only
layer-00.bin     6.75 MB  ×30
final.bin        1.1 KB
                256.6 MB  total
```

A device dealt 8 layers pulls **54 MB**, once. After that it comes from the Cache
API, byte-length-checked against the manifest.

SmolLM2 ships bf16, which WebGPU cannot read. bf16 → f16 is lossless here, because
f16 actually has *more* mantissa (10 bits against 7) and only a narrower exponent
range, which weights sit far inside: **0 of 134.5M values overflowed**, 17 flushed to
zero.

## The scheduler

This is the contribution, and the only thing we ask to be judged on.

### The cost model

```
cost = Σ layers × ms/layer          compute at each stage
     + Σ hops (RTT/2 + bytes/bw)    one hop per link, plus the return to the host
     + host overhead                embed, final norm, LM head, sampling
```

subject to a per-device memory bound, layer contiguity, and chain order.

Two things fall out of this that a scheduler built on intuition gets wrong:

- **The LM head is ~8 layers of arithmetic** (49,152 × 576 against 3.54M MACs per
  layer) and it runs on the host alone. Host election is the single
  highest-leverage decision in the plan, ahead of any layer cut.
- **Every extra device costs a hop.** A phone that would hold one layer saves a few
  milliseconds of a fast device's compute and adds a full round trip. Including it is
  a net loss, and the solver has to be able to say so.

### The solver

The problem decomposes exactly, which is what lets it run without an ILP dependency:

- The chain is a **cycle** (host → workers → host), and a cycle's cost does not depend
  on where you start — so the tour is solved once per subset and reused for every host
  choice within it.
- With uniform per-layer cost — true of every dense model, including this one —
  the layer allocation depends only on *which* devices are present, not their order.
  So ordering and allocation separate, and each is solved exactly.

| Decision | Method | Cost |
|---|---|---|
| Which devices | exhaustive subset search | 2ⁿ, n ≤ 12 |
| What order | Held–Karp | O(2ⁿ·n²) |
| Which layers | DP over cut points | O(n·L²) |
| Which host | measured ms/layer, must fit the embedding table | O(n) |

**Solve time: ~0.1 ms** for a room of six. prima.cpp's Halda solves a comparable
problem with an ILP solver and reports 10–12 ms.

Ties are broken toward balance. With a linear objective `[1,1,28]` and `[10,10,10]`
cost exactly the same and a plain DP returns the first — but they are not equally
good, because the lopsided plan puts 28 layers of memory on one device and collapses
if that device turns out slightly slower than measured.

### Five strategies, one interface

So the comparison in the pitch is a measurement rather than an argument. `memory` is a
faithful reproduction of exo's documented ring memory-weighted partitioning, given
the same feasibility repair as everything else — it is the thing to beat, not a
strawman written to lose.

| Strategy | What it does |
|---|---|
| `solo` | everything on the fastest capable device — the honest ceiling |
| `even` | equal split, join order |
| `memory` | proportional to pledged memory, biggest first (**exo**) |
| `compute` | proportional to measured speed, network-blind |
| `optimal` | subset + chain order + layer cut + host, solved |

```bash
node tools/scenario.mjs        # what each strategy would do, across six room shapes
```

### Profiling

Twice, on purpose:

1. **At join** — a ~300 ms matvec of a layer's shape, converted to MACs/second so one
   probe serves any model. Validated against the real engine at **0.93–1.00×**
   across runs (`npm run calibrate`, which fails the build outside ±35%).
2. **After loading** — the real thing is timed on real layers and the room is told.

The second pass is not redundant. See the throttling finding below.

### Recovery

A device leaving mid-answer is normal behaviour, not an outage.

1. Fail the in-flight lap immediately, rather than waiting out its timeout
2. Re-plan over the survivors — **including devices the planner had stood down**,
   which get recruited back when they are suddenly needed
3. Re-deal the orphaned range. A device whose range did not change **keeps its
   weights** and clears only its cache
4. Replay the conversation, so the new holder's KV cache is real rather than empty
5. Resume from the same position

The property that matters is not "it does not crash" — it is that **the answer is
unchanged**. A room that survives a failure by quietly producing different text has
not recovered; it has started a different conversation without saying so. So the gate
is byte equality with an uninterrupted run, and that is what `tests/recovery.test.mjs`
asserts.

Replay is idempotent: re-running position *p* with the same token writes the same K
and V a device already held, so devices that kept their range are unharmed by being
replayed through.

## What we found

Four results worth stating before anyone asks.

### A swarm buys capacity, not speed

Single-stream decode is a **sum** of stages plus hops, not a max. One token is in
flight at a time and it walks the whole chain, so splitting a model over more devices
makes a single stream *slower*, never faster.

exo measures **49.3 / 44.4 / 39.7 tok/s** on 1 / 2 / 3 M4 Pros. prima.cpp's 70B win
(674 ms/token against 10,120 on one machine) comes from escaping memory exhaustion,
not from parallelism. SwarmLLM's own published benchmark shows **10.8 tok/s** on a
MacBook alone against **7.7 tok/s** with an iPhone added holding 2 of 64 layers — a
29% loss for adding a device.

The scheduler's job is therefore to know **when not to split**, and to minimise the
damage when the model genuinely does not fit. A planner built on the opposite belief
would happily add that phone.

### A hidden browser tab runs about 3× slower

Byte-identical matvec code measured **0.241 GMAC/s** inside a module at page load and
**0.746 GMAC/s** inline in the same hidden page. A device profiled while hidden
reports a speed it will not hold, and one wrong number there produces a confidently
wrong plan.

Three defences, since no single one is enough:

- the probe reports `hidden` and marks itself unstable rather than lying quietly
- visibility changes trigger a re-measure (debounced — the event fires in bursts)
- a second calibration against real layers after loading, which caught **14.0 → 4.4
  ms/layer** live, and `room/awake.js` takes a Screen Wake Lock so it stops happening

**Every device screen-on, tab in front** is a run-book item, not a hope.

### Lossy activations make the answer depend on where the model was cut

f16 halves the bytes on the wire, but that only buys anything if it removes an SCTP
slice — a hop's latency is set by send opportunities, not by size. At these hidden
sizes it does not:

| Model | dim | f16 | f32 | |
|---|---|---|---|---|
| SmolLM2 135M | 576 | 1 slice | 1 slice | f32 is free |
| Qwen3 0.6B | 1024 | 1 slice | 1 slice | f32 is free |
| Qwen3 1.7B | 2048 | 1 slice | 2 slices | f16 wins |

So `room/wire.js` sends f32 whenever it costs no extra slice, and the split answer is
then **bit-identical** to the single-device answer. Before this, the same prompt gave
different text depending on where the room happened to cut the model — which defeats
the point of a scheduler that re-plans.

### Chrome's SCTP releases about four packets per send opportunity

With a ~12 KB initial congestion window, so any send above roughly 4.6 KB costs an
*extra round trip on every hop* — around 200 ms per lap on a 100 ms link. Every frame
is sliced under that threshold. (This one is a measured fact about Chrome, published
by SwarmLLM; we re-measured it ourselves with `mesh-test.html`.)

## Repository layout

About 5,300 lines of runtime and tests, plus an 840-line deck generator. All of it ours.

```
scheduler/       the contribution
  cost.js        latency and memory model, with its assumptions stated out loud
  plan.js        five strategies; `optimal` solves subset + order + cut + host exactly
  probe.js       measures ms/layer, detects a throttled tab

room/
  wire.js        adaptive f32/f16 packing + SCTP-sized framing and reassembly
  mesh.js        WebRTC mesh, negotiated ctrl/wire channels, RTT gossip
  room.js        the room: profile, plan, deal, generate, heal
  awake.js       wake lock, so a device in the chain cannot doze
  config.js      deployment wiring: where signalling lives, where weights come from

engine/
  cpu.mjs          the sliceable CPU engine and the four-call contract; isomorphic
  factory.mjs      the one place room.js asks for an engine — cpu-smollm | dense-gguf | qwen35-gguf
  gpu-adapter.mjs  GGUF fetch/parse/validate, GGUF-meta -> DenseEngine cfg mapping, the
                   same four-call contract over the imported DenseEngine
  upstream/        vendored WebGPU engine (DenseEngine, GGUF loader, WGSL kernels) — see UPSTREAM.md

models/
  registry.mjs   the model ladder: one descriptor per model, with an honest status tag,
                 and the two origins it can be served from (pinned HF URL + local mirror)
  delivery.mjs   picks one of those origins per device, by probing; 206 + GGUF magic

tools/
  build-test-gguf.mjs  writes the tiny real Q8_0 GGUF fixture gpu-adapter-test.html loads
  serve.mjs      dev server: the static half and the signalling half on one listener
  static.mjs     static file serving with real range support, as a handler
  signal.mjs     introductions only — no message type can carry a prompt
  signal-server.mjs  the signalling half, standalone — the whole server side of a deployment
  build-static.mjs   walks the module graph into dist/; a CDN build, or a LAN bundle
  verify-delivery.mjs  live check that the pinned HF URLs are the bytes the registry claims
  fetch-model.mjs   download and reshape into per-layer shards
  tokenizer.mjs  byte-level BPE
  reference.mjs  CPU golden reference — what the GPU port must match
  calibrate.mjs  does the probe predict the engine? (gate: within ~35%)
  scenario.mjs   when is a swarm worth it?
  make-cert.sh   regenerate the TLS cert for this machine's LAN IP

tests/           wire · split · plan · recovery · qr · markdown · gpu-adapter
tests/fixtures/  the tiny real GGUF file gpu-adapter-test.html loads
docs/gpu-reports/  dated JSON from gpu-test.html / gpu-adapter-test.html runs, not hand-typed numbers
deck/            round 1 pitch deck generator
```

## Testing

```bash
npm test                       # all four suites
node tests/wire.test.mjs       # 30 — f16 normals and subnormals, slice boundaries,
                               #      out-of-order and duplicate slices, precision choice
node tests/split.test.mjs      # 11 — split output identical to one device, across
                               #      2, 3 and 5 devices, with the real wire codec in the loop
node tests/plan.test.mjs       # 29 — DP against an independent greedy optimum over 400
                               #      random rooms; Held–Karp against brute force over 120;
                               #      "optimal is never beaten" over 1,483 generated rooms
node tests/recovery.test.mjs   # 13 — the answer is byte-identical after a device leaves
node tests/gpu-adapter.test.mjs # 34 — GGUF-meta mapping, descriptor/range validation,
                                #      factory dispatch; the logic Node can run without a GPU
```

Where an exact algorithm exists, it is checked against an independent brute force.
Where a claim appears in the pitch, there is a test named after the claim.

The GPU path's *numerical* correctness cannot run in Node — there is no WebGPU there —
so it has its own pages, run manually in a real browser and recorded in
[`docs/gpu-reports/`](docs/gpu-reports/):

```bash
open http://localhost:8442/gpu-test.html          # imported kernels vs CPU math
open http://localhost:8442/gpu-adapter-test.html  # the adapter loading a real GGUF vs CPU math
```

## Honest limits

- **It will not beat a real GPU.** One machine that fits the model wins outright, and
  the planner says so and runs solo.
- **It scales in capacity, not speed.** Six devices run a model none of them could
  hold — at roughly the speed of the slowest useful subset, not six times anything.
- **The live room is CPU-only today**, around 5 tok/s. A WebGPU dense-model adapter
  (`engine/factory.mjs`, `engine/gpu-adapter.mjs`) now exists and is verified against
  a real GGUF file and an independent CPU reference (see `docs/gpu-reports/`), but
  no model selector or real Qwen download is wired into the room yet — that is the
  next phase, not a claim this README is making today.
- **Context is 512 positions.** Small and real. A question with no room to be answered
  is refused up front rather than cut off mid-sentence.
- **Losing the host is not recoverable.** The conversation, the tokenizer and the LM
  head all live there. Reported clearly rather than hung.
- **Strict NATs are unhandled.** STUN only; no TURN relay yet.
- **Phones are guests, not workhorses.** They join, hold a few layers, and get dropped
  when they would slow the room. That is the design, not a bug.

## Prior art

We did not invent browser mesh inference. [SwarmLLM](https://github.com/Nehanth/swarmllm)
shipped it and is our stated starting point; [exo](https://github.com/exo-explore/exo)
and [prima.cpp](https://arxiv.org/abs/2504.08791) shaped the placement work, and
prima.cpp published the device-dropping result first.

**We did not fork any of them.** Every file here was written by our team.
[NOTICE.md](NOTICE.md) sets out what we learned from whom, and draws the line around
what is ours.

## Licence

[MIT](LICENSE). Model weights and tokenizers come from their publishers under their
own licences and are not redistributed here.
