# AI Swarm

**Many devices, one model — and it keeps running even if one of them walks away.**

Team SE7EN · AI & Automation track

One large language model, cut into contiguous slices, one slice per device. Open a
link in a browser tab and your phone or laptop becomes one stage of a model far
bigger than it could ever hold. Tokens pass between devices peer-to-peer over
WebRTC; the answer appears on every screen.

## What we are actually building

Browser mesh inference already exists — see [NOTICE.md](NOTICE.md), which names our
prior art plainly. **We are building the scheduler that sits on top of it.**

Existing swarms deal layers in proportion to the memory a device *pledges*, in
whatever order the devices happened to join. Neither accounts for how fast a device
actually computes or how far it sits from its neighbours, so one slow phone or one
distant peer sets the pace for every token. We measure both and solve the placement
exactly.

| | Ours |
|---|---|
| Layer placement | measured compute + link cost, solved by dynamic programming |
| Chain order | minimises total lap latency over the RTT matrix |
| Re-plans while running | yes |
| Device leaves mid-answer | re-plan, replay, the stream continues |
| Install | none — it is a web page |

## Status

| Stage | State |
|---|---|
| Device probe (WebGPU, memory, bandwidth, NAT) | working |
| HTTPS + signaling dev server | working |
| Wire format: adaptive f32/f16, SCTP-sized slicing | working · 30/30 tests |
| WebRTC mesh, direct peer links | working · bit-exact tensor bounce |
| Inference engine, sliceable | working · SmolLM2 135M |
| Split output identical to one device | working · 11/11, wire codec in the loop |
| End-to-end swarm over WebRTC | working · answers stream to every screen |
| Profiler + DP partitioner + chain order + host election | working · 29/29 tests |
| Fault recovery (device leaves mid-answer) | working · 12/12 tests · verified live |
| WebGPU engine | next |

Measured across two browsers on one machine: 60 tokens, layers 0-12 and 13-29,
answer byte-identical to the single-device CPU reference.

## Running it

```bash
npm install
npm run cert     # regenerate the TLS cert for this machine's LAN IP -- RUN THIS AT THE VENUE
npm run serve
```

Then:

- **this machine** — `http://localhost:8442/probe.html` (localhost is already a
  secure context, so no cert warning)
- **other devices, same Wi-Fi** — `https://<lan-ip>:8443/probe.html`, and accept the
  self-signed cert once (Android: *Advanced → Proceed*; iOS: *Show Details → visit
  this website*)

WebGPU and WebRTC both require a secure context. Plain `http://192.168.x.x` gives
you neither, and the failure looks exactly like "this phone has no WebGPU" — which
is why the dev server serves TLS with the LAN IP in the certificate's SAN.

### Pages

| Page | What it is for |
|---|---|
| `probe.html` | What can this device do? WebGPU adapter, usable GPU memory, effective bandwidth, thermal/battery APIs, NAT type |
| `mesh-test.html` | Do two devices connect directly, and does a tensor survive the trip bit-exactly? |

### Tests

```bash
node tests/wire.test.mjs
```

## Layout

```
scheduler/    cost.js   the latency and memory model, with its assumptions stated
              plan.js   five strategies; the optimal one solves subset + order +
                        layer cut + host election exactly            <- the contribution
              probe.js  measures ms/layer, detects a throttled tab
room/         wire.js   adaptive f32/f16 + SCTP-sized framing
              mesh.js   WebRTC mesh, negotiated channels, RTT gossip
              room.js   the room: plan, deal, load, generate
              awake.js  wake lock, so a device in the chain cannot doze
engine/       cpu.mjs   the sliceable engine: embedRun / runHidden / headFromHidden
tools/        serve.mjs (HTTPS + signaling) · fetch-model · reference · calibrate ·
              scenario (when is a swarm worth it?) · tokenizer · make-cert
tests/        wire · split · plan
```

## What we found

Three results worth stating before anyone asks:

- **A swarm buys capacity, not speed.** Single-stream decode is a sum of stages plus
  hops, so splitting a model over more devices makes one stream slower, never faster.
  exo measures 49.3 / 44.4 / 39.7 tok/s on 1 / 2 / 3 M4 Pros. The scheduler's job is
  to know when *not* to split, and to minimise the damage when the model genuinely
  does not fit.
- **A hidden browser tab runs about 3x slower.** Byte-identical code measured 0.24
  GMAC/s in a background tab against 0.75 inline in the same page. A device profiled
  while hidden reports a speed it will not hold, and one wrong number produces a
  confidently wrong plan. Hence: a `hidden` flag, a re-measure on visibility, a second
  calibration against real layers, and a wake lock.
- **Lossy activations make the answer depend on where the model was cut.** f16 only
  pays for itself when it removes an SCTP slice, which at these hidden sizes it does
  not. `room/wire.js` sends f32 when it is free, and the split answer is then
  bit-identical to the single-device answer.

`node tools/scenario.mjs` prints the whole picture for a given room.

## Licence

MIT. See [NOTICE.md](NOTICE.md) for prior art and attribution.
