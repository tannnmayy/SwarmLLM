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
| Wire format: f16 packing, SCTP-sized slicing | working · 22/22 tests |
| WebRTC mesh, direct peer links | working · bit-exact tensor bounce |
| Inference engine (single device) | next |
| Two-node split | next |
| Profiler + DP partitioner | next |
| Fault recovery | next |
| Dashboard | next |

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
room/         wire.js   f16 packing + SCTP-sized framing
              mesh.js   WebRTC mesh, negotiated data channels, RTT probe
scheduler/    profiling, the DP partitioner, chain order, recovery   (ours - the contribution)
engine/       the WebGPU inference engine
tools/        serve.mjs (HTTPS + signaling), signal.mjs, make-cert.sh
tests/        correctness gates
```

## Licence

MIT. See [NOTICE.md](NOTICE.md) for prior art and attribution.
