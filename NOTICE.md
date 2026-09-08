# Prior art and what is ours

## The honest statement

AI Swarm did not invent browser-based mesh LLM inference. **SwarmLLM**
(<https://github.com/Nehanth/swarmllm>, MIT, © 2026 Nehanth Narendrula) shipped it
first and is our stated starting point, exactly as our pitch deck says.

We read that project's public documentation and source to understand the problem —
its room protocol, its wire format, its engine structure and, most usefully, its
published research notes. **We did not fork it and we did not copy its code.**
Every file in this repository was written by our team.

If any file here ever does carry code from SwarmLLM, it must say so at the top of
that file and retain the MIT copyright notice. As of this writing, none do.

## What we learned from them, and used

These are facts and designs, published openly. Using them is what public research
is for; claiming we discovered them would not be.

| What | Where it came from | What we did with it |
|---|---|---|
| Chrome's SCTP data channel releases ~4 packets per send opportunity, ~12 KB initial congestion window, so sends above ~4.6 KB cost an extra round trip per hop | SwarmLLM `docs/bench-log.md`, measured | Our `room/wire.js` slices every send under that threshold. We re-measured it ourselves (`mesh-test.html`) |
| Layer-split (pipeline) parallelism beats tensor parallelism on any network slower than a PCIe bus | SwarmLLM `docs/research/network-scheduler.md` §5.5 | Adopted as a design decision, not re-litigated |
| Splitting layers proportional to *pledged memory*, with an arbitrary chain order, leaves measurable performance on the table | SwarmLLM `room.js:797`, `room.js:828-838` | This is the gap our scheduler targets — see below |

## What is ours

The contribution we claim, and the only thing we ask to be judged on:

- **`scheduler/`** — capability profiling, a peer-to-peer RTT matrix, and an exact
  dynamic-programming solver for contiguous layer placement, chain order and host
  election, against a measured cost model rather than pledged memory.
- **Fault recovery** — re-planning and resuming a generation when a device leaves
  mid-answer.
- **`room/`**, **`engine/`**, **`tools/`** — our own transport, mesh, inference
  engine and dev server, written to serve the above.

SwarmLLM has *designed* comparable placement and recovery work in its public roadmap
(items [27](https://github.com/Nehanth/swarmllm/blob/main/roadmap/27-placement-and-chain-order.md)
and [03](https://github.com/Nehanth/swarmllm/blob/main/roadmap/03-swarm-recovery.md));
at the time we started, neither was implemented. We are not claiming to have invented
the idea of scheduling a swarm. We are claiming to have built and measured one.

## Third-party dependencies

| Package | License | Used for |
|---|---|---|
| `ws` | MIT | WebSocket server for signaling (dev/host side only) |

Model weights and tokenizers come from their respective publishers under their own
licenses and are not redistributed here.
