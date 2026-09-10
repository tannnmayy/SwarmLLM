// AI Swarm signalling, as a Cloudflare Worker + Durable Object.
//
// The same service as tools/signal-server.mjs, in the shape Cloudflare's free plan
// can run: a Worker routes, and one Durable Object per room code holds that room's
// sockets. Durable Objects are the reason this works at all — a plain Worker has no
// identity and no memory, so two people hitting the same room code would land on
// different isolates and never see each other. Routing by room code to a named DO
// is what makes "the same room" mean something.
//
// Free plan, checked 2026-09: Durable Objects are available with the SQLite storage
// backend, 100,000 requests/day and 13,000 GB-s/day of duration. This object stores
// nothing — the room is a Map that empties itself as people leave — so the SQLite
// backend is declared in wrangler.toml purely because it is the only one the free
// plan offers.
//
// Hibernation matters here. `state.acceptWebSocket()` (rather than `ws.accept()`)
// lets the runtime evict this object from memory between messages while keeping the
// sockets open, so an idle room costs no duration. A room that sat on a live isolate
// waiting for someone to type would burn the daily GB-s allowance on nothing.
//
// What it can see is unchanged from the Node version, and that is the whole privacy
// claim: room codes, peer names, and WebRTC SDP/ICE. Never a prompt, an activation,
// a weight or an answer — there is no message type that could carry one.

const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";   // no O/0/I/1: these get read aloud
const MAX_ROOM = 12;

const json = (obj, init = {}) => new Response(JSON.stringify(obj), {
  ...init,
  headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(), ...(init.headers || {}) },
});

// The pages are on a different origin than this Worker by construction, so every
// plain HTTP response here is a cross-origin one.
const cors = () => ({
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, OPTIONS",
  "access-control-allow-headers": "content-type",
  "access-control-max-age": "86400",
});

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors() });

    if (url.pathname === "/healthz" || url.pathname === "/health") {
      // Deliberately no room list and no counts across rooms. Each Durable Object
      // knows only its own room, and asking every one of them would mean keeping a
      // registry of live room codes — which is exactly the thing that must not
      // exist, because a health endpoint that listed rooms would hand out join
      // codes to anyone who curled it.
      return json({ ok: true, service: "ai-swarm-signal", runtime: "cloudflare-worker" });
    }

    // Short-lived TURN credentials, minted per request. Cloudflare's TURN does not
    // issue long-lived ones, which is why this endpoint exists rather than a static
    // list in swarm-config.json. Unconfigured, it returns STUN only — a deployment
    // without TURN is a supported configuration, not an error.
    if (url.pathname === "/ice") return iceServers(env);

    if (url.pathname === "/signal") {
      if (request.headers.get("upgrade") !== "websocket") {
        return new Response("expected a websocket upgrade", { status: 426, headers: cors() });
      }
      const allow = (env.SIGNAL_ALLOW_ORIGIN || "").split(",").map((s) => s.trim()).filter(Boolean);
      if (allow.length && !allow.includes(request.headers.get("origin") || "")) {
        return new Response("origin not allowed", { status: 403, headers: cors() });
      }

      // The room code has to be known before the socket is accepted, because it
      // decides WHICH Durable Object handles it. The Node server reads it from the
      // first message instead; here it rides the query string, and room/mesh.js
      // appends it. A socket that arrives without one is answered rather than
      // dropped, so an older client gets a real error instead of a silent failure.
      const code = (url.searchParams.get("room") || "").toUpperCase().slice(0, 8);
      if (!code) return new Response("missing ?room=CODE", { status: 400, headers: cors() });

      const id = env.ROOM.idFromName(code);
      return env.ROOM.get(id).fetch(request);
    }

    return new Response(
      "AI Swarm signalling service.\n\n" +
      "This host brokers WebRTC introductions and nothing else. It never sees a prompt,\n" +
      "an activation, a weight or an answer.\n\n" +
      "  websocket : /signal?room=CODE\n" +
      "  ice       : /ice\n" +
      "  health    : /healthz\n\n" +
      "The application itself is a static site; point its swarm-config.json at this host.\n",
      { status: url.pathname === "/" ? 200 : 404, headers: { "content-type": "text/plain; charset=utf-8", ...cors() } },
    );
  },
};

async function iceServers(env) {
  const stun = [{ urls: "stun:stun.cloudflare.com:3478" }, { urls: "stun:stun.l.google.com:19302" }];
  if (!env.TURN_KEY_ID || !env.TURN_KEY_API_TOKEN) {
    return json({ iceServers: stun, turn: false, why: "TURN_KEY_ID / TURN_KEY_API_TOKEN are not set" });
  }
  try {
    const res = await fetch(
      `https://rtc.live.cloudflare.com/v1/turn/keys/${env.TURN_KEY_ID}/credentials/generate-ice-servers`,
      {
        method: "POST",
        headers: { "authorization": `Bearer ${env.TURN_KEY_API_TOKEN}`, "content-type": "application/json" },
        // Long enough to cover a join and a conversation, short enough that a
        // credential scraped off the wire is not worth much.
        body: JSON.stringify({ ttl: 3600 }),
      },
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json();
    const list = body.iceServers ? (Array.isArray(body.iceServers) ? body.iceServers : [body.iceServers]) : [];
    if (!list.length) throw new Error("no iceServers in Cloudflare's response");
    return json({ iceServers: [...list, ...stun], turn: true });
  } catch (e) {
    // STUN alone still connects the common case. Degrading is strictly better than
    // refusing to hand out any ICE configuration at all.
    return json({ iceServers: stun, turn: false, why: String(e.message || e) });
  }
}

export class Room {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.seq = 0;
    // Peer state lives in the socket's own attachment rather than in a field on
    // this object, because hibernation evicts the object and its fields but keeps
    // the sockets. Anything remembered here would be gone on the next message.
  }

  newId() {
    return `p${(++this.seq).toString(36)}${Math.random().toString(36).slice(2, 6)}`;
  }

  sockets() {
    return this.state.getWebSockets();
  }

  attach(ws) {
    try { return ws.deserializeAttachment() || null; } catch { return null; }
  }

  roster(exclude = null) {
    const out = [];
    for (const ws of this.sockets()) {
      const a = this.attach(ws);
      if (!a || a.id === exclude) continue;
      out.push({ id: a.id, name: a.name, meta: a.meta });
    }
    return out;
  }

  send(ws, msg) {
    try { ws.send(JSON.stringify(msg)); } catch { /* closing */ }
  }

  broadcast(msg, exclude = null) {
    for (const ws of this.sockets()) {
      const a = this.attach(ws);
      if (!a || a.id === exclude) continue;
      this.send(ws, msg);
    }
  }

  async fetch(request) {
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    // acceptWebSocket, not server.accept(): this is what makes the object
    // hibernatable. See the note at the top of this file.
    this.state.acceptWebSocket(server);
    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(ws, raw) {
    let m;
    try { m = JSON.parse(raw); } catch { return; }
    const me = this.attach(ws);

    if (m.t === "join") {
      if (me) return;                                    // one join per socket
      const code = String(m.room || "").toUpperCase().slice(0, 8);
      if (!code) return this.send(ws, { t: "error", why: "no room code" });
      if (this.sockets().length > MAX_ROOM) return this.send(ws, { t: "error", why: "room full" });

      const id = this.newId();
      const info = { id, name: String(m.name || "device").slice(0, 40), meta: m.meta || {} };
      // Tell the newcomer who is already here BEFORE anyone hears about them, so
      // the glare rule holds: the newcomer offers to everyone present, and everyone
      // present only ever answers. Reversing these two lines would let two peers
      // offer each other at once.
      this.send(ws, { t: "welcome", id, room: code, peers: this.roster() });
      ws.serializeAttachment(info);
      this.broadcast({ t: "peer-join", id, name: info.name, meta: info.meta }, id);
      return;
    }

    if (!me) return;                                     // everything below needs a join

    switch (m.t) {
      case "signal": {
        // Relay SDP offers/answers and ICE candidates verbatim to one named peer.
        for (const other of this.sockets()) {
          const a = this.attach(other);
          if (a?.id === m.to) { this.send(other, { t: "signal", from: me.id, data: m.data }); break; }
        }
        break;
      }
      case "meta": {
        const next = { ...me, meta: m.meta || {} };
        ws.serializeAttachment(next);
        this.broadcast({ t: "peer-meta", id: me.id, meta: next.meta }, me.id);
        break;
      }
      case "ping":
        this.send(ws, { t: "pong", ts: m.ts });
        break;
    }
  }

  async webSocketClose(ws) { this.gone(ws); }
  async webSocketError(ws) { this.gone(ws); }

  gone(ws) {
    const me = this.attach(ws);
    if (!me) return;
    // The socket is still in getWebSockets() while this handler runs, so exclude it
    // explicitly rather than relying on it having been removed already.
    this.broadcast({ t: "peer-left", id: me.id }, me.id);
  }
}
