# Deploying AI Swarm

The deployed system is two pieces, and only one of them is a server.

```
  static site                          signalling service
  ───────────                          ──────────────────
  index.html, room.html                one small always-on process
  room/ engine/ scheduler/ models/     brokers WebRTC introductions
  ~0.4 MB, any static host             no disk, no database, no state

                    weights
                    ───────
                    huggingface.co, range-fetched by the browser
                    per device, only the layers that device was dealt
```

Nothing runs on a server except introductions. Prompts, activations and answers
travel peer to peer over WebRTC data channels; weights come off Hugging Face's CDN.
That is why the hosting bill for the middle box is a websocket and the hosting bill
for the weights is zero.

---

## Recommended: all on Cloudflare, free tier

One vendor, one dashboard, no credit card, and the only piece that has to stay
running is a Worker that sleeps when nobody is in a room.

| piece | where | cost |
|---|---|---|
| static site | Cloudflare Pages | free |
| signalling | Cloudflare Worker + Durable Object | free tier: 100,000 requests/day, 13,000 GB-s/day |
| TURN (optional) | Cloudflare Realtime TURN | free tier: 1,000 GB, and a room moves KB per token |
| weights | huggingface.co | free, and not your bandwidth |

Durable Objects are what make this work rather than a plain Worker: a Worker has no
identity and no memory, so two people typing the same room code would land on
different isolates and never see each other. One Durable Object per room code is
what makes "the same room" mean something. The free plan carries them with the
SQLite storage backend, which is why `wrangler.toml` declares
`new_sqlite_classes` — this object stores nothing, but the backend still has to be
named.

The Worker uses the WebSocket **Hibernation** API (`state.acceptWebSocket`, not
`ws.accept()`), so an idle room is evicted from memory with its sockets still open
and costs no duration. A room sitting there waiting for someone to type would
otherwise burn the daily GB-s allowance on nothing.

### Step by step

Everything below is run from a checkout. Steps marked **(you)** need your own
account and cannot be done for you.

**1. Deploy the signalling Worker (you)**

```bash
cd deploy/cloudflare-signal
npm install
npx wrangler login          # opens a browser; authorises this machine
npx wrangler deploy
```

`wrangler deploy` prints the URL, e.g. `https://ai-swarm-signal.<subdomain>.workers.dev`.
Check it:

```bash
curl https://ai-swarm-signal.<subdomain>.workers.dev/healthz
```

**2. Build the site against that Worker**

```bash
cd ../..
npm run build -- --signal wss://ai-swarm-signal.<subdomain>.workers.dev/signal
```

`--ice-url` is derived from `--signal` unless you pass it, so this also points the
page at the Worker's `/ice`.

**3. Publish the site (you)**

```bash
npx wrangler pages deploy dist --project-name ai-swarm
```

First run asks to create the project and pick a production branch. It prints
`https://ai-swarm.pages.dev`.

**4. Lock the Worker to that origin (you, optional but do it)**

```bash
cd deploy/cloudflare-signal
npx wrangler secret put SIGNAL_ALLOW_ORIGIN     # paste: https://ai-swarm.pages.dev
```

This stops another site using your Worker as free infrastructure. It is **not**
authentication — the room code still is the only thing gating a room. Note that
setting it also blocks a local page from using the deployed Worker, so add
`http://localhost:8442` to the comma-separated list if you want to keep testing
that way.

**5. Turn on TURN, if you want two devices on different networks (you)**

Cloudflare dashboard → **Realtime** → **TURN** → create a key. Then:

```bash
npx wrangler secret put TURN_KEY_ID
npx wrangler secret put TURN_KEY_API_TOKEN
```

Confirm:

```bash
curl https://ai-swarm-signal.<subdomain>.workers.dev/ice
# {"iceServers":[...turn...],"turn":true}
```

Cloudflare does not issue long-lived TURN credentials, which is why `/ice` exists
rather than a static list in `swarm-config.json`: the Worker mints a short-lived
pair per request. Leave the secrets unset and `/ice` returns STUN only — a
supported configuration, not an error.

### If you would rather not port anything

`tools/signal-server.mjs` is the same service as a plain Node process and runs
unchanged on Fly, Railway, Render or any host that keeps a process alive and
terminates TLS in front of it. Then build with `--signal wss://<that host>/signal`.
Vercel is not an option for this half — its functions do not hold a long-lived
WebSocket — though it is fine for the static half.

---

## 1. Build the static site

```bash
npm run build -- --signal wss://your-signal-host/signal
```

Writes `dist/` — HTML and ES modules, no bundler, no transpile step. The file list
is walked from the HTML entry points through the import graph, so `tests/`,
`tools/` and the multi-GB `models/` directory stay out without a deny list to
maintain. A bare specifier or a static `node:` import anywhere in that graph fails
the build rather than shipping a page that 404s on someone else's device.

Deploy `dist/` to GitHub Pages, Netlify, Cloudflare Pages, S3 — anything that serves
files over HTTPS. HTTPS is not optional: WebGPU and WebRTC both require a secure
context.

`dist/swarm-config.json` is the only file that differs between deployments:

```json
{
  "delivery": "upstream",
  "models": [],
  "signalUrl": "wss://your-signal-host/signal"
}
```

Re-pointing a built site at a different signalling service is editing that file. No
rebuild.

## 2. Deploy the signalling service

```bash
npm run signal          # PORT=8080 by default
```

`tools/signal-server.mjs` is the whole thing. It listens on `$PORT`, serves
`/healthz`, and brokers `/signal`. TLS is terminated by the platform in front of it
(Fly, Railway, Render, Cloud Run all do this), so it speaks plain HTTP and the
browser still reaches it as `wss://`.

| env | default | what it does |
|---|---|---|
| `PORT` | `8080` | listen port |
| `HOST` | `0.0.0.0` | bind address |
| `SIGNAL_ALLOW_ORIGIN` | unset | comma-separated origin allowlist. Unset means any page may connect, which is what a local checkout and a `#signal=` override both need. Set it in a real deployment so another site cannot use this as free infrastructure. It is **not** authentication — the room code still is. |

Health check:

```bash
curl https://your-signal-host/healthz
# {"ok":true,"service":"ai-swarm-signal","uptimeSeconds":41,"rooms":1,"peers":2}
```

Counts only. A health endpoint that listed live rooms would hand out join codes to
anyone who curled it.

## 3. What a visitor's browser then does

1. Loads ~0.4 MB of HTML and modules from the CDN.
2. Reads `swarm-config.json` and opens a websocket to the signalling service.
3. Resolves where weights come from (below) and range-fetches a 16 MiB GGUF header.
4. Range-fetches only the layers the scheduler dealt this device.

Weight bytes never pass through your infrastructure.

---

## Where weights come from

Each model in `models/registry.mjs` declares up to two origins, and
`models/delivery.mjs` picks one **per device, at load time**:

| origin | what it is | when it wins |
|---|---|---|
| `mirror` | this origin's own `/models/` directory | present, and faster — a LAN room should pull off the host it is already talking to, and it works with no internet at all |
| `upstream` | the pinned artifact on huggingface.co | everything else, and always on a CDN deployment |

`swarm-config.json`'s `delivery` field chooses the policy:

- `"auto"` — probe the mirror, fall back to upstream. One build works in both
  places. This is the default when there is no config at all, which is the dev
  server case.
- `"upstream"` — go straight to Hugging Face. What `npm run build` writes, because a
  CDN build has no `/models/` directory and probing for one can only ever 404.
- `"mirror"` — offline. What `npm run build:lan` writes.

The probe is four bytes: `Range: bytes=0-3`, and the answer must be a **206** whose
body is the ASCII magic `GGUF`. Both halves matter. The 206 proves the origin honours
Range at all — one that ignores it would hand a device the whole multi-GB file for
every tensor it wanted. The magic bytes catch what a status code cannot: a static
host with an SPA fallback answers `/models/anything.gguf` with 200 and an HTML page,
which looks exactly like a healthy mirror until the GGUF parser hits it several
hundred megabytes later.

Per-visit overrides, for testing: `#src=upstream`, `#src=mirror`,
`#signal=wss://host/signal`.

### Why the URLs are pinned to a commit

Every upstream URL names a commit, not `main`:

```
https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc.../Qwen3-0.6B-Q8_0.gguf
```

Two devices in one room may legitimately resolve to different origins — one on the
LAN mirror, one on the CDN. That is only safe while both origins are the same bytes.
`main` is a mutable ref, and a room where one device loaded the mirror's tokenizer
and another loaded a newer upstream one would put mismatched token IDs on the wire
with nothing raising an error.

`config.json` and `tokenizer.json` come from the **base** repo (`Qwen/Qwen3-0.6B`),
not the `-GGUF` repo, which returns 404 for both. They are pinned to that repo's own
revision.

### Verifying it

```bash
npm run verify:delivery              # every model, ~15 s
npm run verify:delivery -- --full    # streams and hashes the whole file
```

Checks, live, that each pinned URL resolves; that the byte length and SHA-256 match
what `models/registry.mjs` records; that a cross-origin range request returns 206
with GGUF magic; that CORS permits it; and that the local mirror is the same file as
upstream. Last recorded run:
[`docs/delivery/2026-09-10-upstream-verification.json`](delivery/2026-09-10-upstream-verification.json)
— 39 passed, 0 failed.

Two traps that tool exists to keep checking:

- **`X-Linked-ETag` is the SHA-256, and it only exists on the 302 *before* the
  redirect to the CDN.** Follow the redirect and the `ETag` you get is
  `X-Xet-Hash`, a Xet content hash — a completely different number. Compare that to
  a sha256 and every model looks corrupted. Fetch with `redirect: "manual"`.
- **`Range: bytes=N-M` is a CORS-safelisted request header**, so a cross-origin
  range request skips preflight entirely. A suffix range (`bytes=-4`) or a
  multi-range is not safelisted and would put an `OPTIONS` in front of every model
  load. (Hugging Face answers preflight correctly anyway — `allow-headers: range` —
  but relying on that is a choice, not an accident.)

---

## LAN and offline

A venue with bad wifi should not have every device pulling gigabytes from the
internet.

```bash
npm run build:lan -- --models qwen3-0.6b
```

Bundles `models/qwen3-0.6b/` into `dist/` and writes `delivery: "mirror"` plus the
list of what it actually carries. That list matters: the registry declares a mirror
path for every model in the ladder because it describes the project rather than one
deployment of it, so without it a bundle carrying one model would still offer the
other three and 404 on whichever one somebody picked.

The whole ladder is 6.7 GB; one rung is usually what you want. `--delivery auto` on
a bundled build gives you the mirror when it is there and Hugging Face when it is
not.

`tools/fetch-model.mjs`'s raw `_source.safetensors` download is excluded — it is the
input to a build step, not an output, and it more than doubles a SmolLM2 bundle.

### The dev server is not a deployment

`npm run serve` (`tools/serve.mjs`) puts both halves on one listener with a
self-signed cert, because on a LAN there is no CDN and a page on `https://` can only
open `wss://` on its own origin. That is the demo path. In a deployment the two
halves live apart and neither of them is that file.

---

## Verifying a live deployment from two devices

Do these in order. Each one fails differently, and the difference is the diagnosis.

1. **Open the site on device A.** The join card should say
   `Weights: Hugging Face CDN` and name your Worker under `Signalling:`. If it says
   `same origin`, the build did not get `--signal`.
2. **Press Join on A.** The log should say `joined room XXXX`. If the room code
   never appears, signalling is unreachable — check `SIGNAL_ALLOW_ORIGIN` first,
   since a mismatch there fails exactly this way.
3. **Open the same link with `#r=XXXX` on device B and Join.** Both devices should
   list each other with an RTT. No RTT means the WebRTC connection never came up:
   this is the TURN case, and the fix is step 5 above, not anything in the room.
4. **Lower the pledge on both** until the model no longer fits on one device, then
   press *Start the swarm* on either. A real split is the point; leave the pledge
   high and the planner will correctly decide one device is enough.
5. **Ask something.** The answer should stream onto both screens at once.

Measured on the deployed topology (static site and signalling on separate origins,
weights from huggingface.co, two browser contexts sharing one GPU): Qwen3 0.6B split
14+14 layers, worker pulled **191.4 MB** and host **412.8 MB** — each device
downloads only its own share, not the whole 610 MB — then **144 tokens at 5.44
tok/s**, median network lap **49.5 ms**.

Two things that first run turned up, both now fixed, both worth knowing about
because they only appear once weights come from a CDN:

- The host used to give each worker a flat **120 s** to finish loading. That is
  generous against a local mirror and far too short against a CDN — the worker
  needed 138 s and the host 193 s — so the host gave up before either finished and
  the room never started. The wait is now a watchdog on **silence** (90 s with no
  progress message) rather than a deadline on the whole download.
- A device with no WebGPU used to join, get dealt a layer range, and only then
  discover it could not load anything, costing the room the full timeout. Devices
  now advertise whether they can run the model and the planner skips the ones that
  cannot; a worker that fails anyway answers `cannot-load` and the host re-plans
  around it immediately.

---

## Known limits

- **STUN only unless you configure TURN.** Out of the box `room/mesh.js` lists
  public STUN servers and nothing else. Two devices on the same network connect;
  two devices behind symmetric NATs on different networks will not. Step 5 above
  fixes this; without it, "two separate networks" is a coin flip that depends on
  both routers.
- **Every device in the chain runs the model.** There is no arrangement where a
  device merely relays, so a phone whose browser has no WebGPU cannot take part in
  a Qwen3 room — it will say so rather than break the room, but it cannot help. For
  a first live test, use two devices you know have WebGPU.
- **The `<think>` block is not suppressed yet.** Qwen3 reasons before it answers,
  and in the run above that was about 110 of the 144 tokens. Correct, and slow to
  watch.
- **The room code is the only access control.** Four characters from a 32-symbol
  alphabet. Anyone who has it can join a room and read the conversation.
  `SIGNAL_ALLOW_ORIGIN` restricts which *pages* may connect, not which people.
- **`smollm2-135m` cannot be served from a CDN build.** Its per-layer f16 shards are
  built locally by `tools/fetch-model.mjs` and published nowhere upstream, so a
  build without `models/` genuinely cannot offer it. The model picker says so and
  falls through to the first model the deployment can actually run.
- **Graceful shutdown is untested on Windows.** `tools/signal-server.mjs` drains on
  `SIGTERM`, but Windows does not deliver the signal — `process.kill` there maps to
  `TerminateProcess` and the handler never runs. The deploy target is Linux, where
  it does.
