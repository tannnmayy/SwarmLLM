// Signaling: brokers introductions between browsers, nothing else.
//
// This server sees room codes, peer names and WebRTC SDP/ICE. It never sees a
// prompt, an activation, a weight or an answer -- those go peer-to-peer over the
// data channels the SDP sets up. That property is the whole privacy claim, so it
// is enforced here by simply having no message type that could carry them.
//
// Attached to the same HTTPS listener as the static files (see tools/serve.mjs) so
// a page on https:// talks to wss:// on its own origin: one port, no mixed content.

import { WebSocketServer } from "ws";

// no O/0/I/1 -- these get read aloud and typed in by hand under demo pressure
const ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
const rooms = new Map(); // code -> Map<id, {ws, name, meta}>

let seq = 0;
const newId = () => `p${(++seq).toString(36)}${Math.random().toString(36).slice(2, 6)}`;

export function newRoomCode() {
  let c = "";
  for (let i = 0; i < 4; i++) c += ALPHABET[Math.floor(Math.random() * ALPHABET.length)];
  return rooms.has(c) ? newRoomCode() : c;
}

function send(ws, msg) {
  if (ws.readyState === 1) ws.send(JSON.stringify(msg));
}

function roster(room) {
  return [...room.entries()].map(([id, p]) => ({ id, name: p.name, meta: p.meta }));
}

export function attachSignaling(server, { log = true } = {}) {
  const wss = new WebSocketServer({ server, path: "/signal" });

  wss.on("connection", (ws) => {
    let id = null, code = null;
    ws.isAlive = true;
    ws.on("pong", () => { ws.isAlive = true; });

    ws.on("message", (raw) => {
      let m;
      try { m = JSON.parse(raw); } catch { return; }

      if (m.t === "join") {
        if (id) return;                                  // one join per socket
        code = String(m.room || "").toUpperCase().slice(0, 8);
        if (!code) return send(ws, { t: "error", why: "no room code" });
        if (!rooms.has(code)) rooms.set(code, new Map());
        const room = rooms.get(code);
        if (room.size >= 12) return send(ws, { t: "error", why: "room full" });

        id = newId();
        const me = { ws, name: String(m.name || "device").slice(0, 40), meta: m.meta || {} };
        // tell the newcomer who is already here, before anyone hears about them
        send(ws, { t: "welcome", id, room: code, peers: roster(room) });
        for (const [, p] of room) send(p.ws, { t: "peer-join", id, name: me.name, meta: me.meta });
        room.set(id, me);
        if (log) console.log(`  + ${me.name} -> ${code} (${room.size} in room)`);
        return;
      }

      if (!id) return;                                   // everything below needs a join
      const room = rooms.get(code);
      if (!room) return;

      switch (m.t) {
        // relay SDP offers/answers and ICE candidates, verbatim, to one named peer
        case "signal": {
          const dst = room.get(m.to);
          if (dst) send(dst.ws, { t: "signal", from: id, data: m.data });
          break;
        }
        // a device revising what it can contribute (memory, measured speed, battery)
        case "meta": {
          const me = room.get(id);
          if (!me) break;
          me.meta = m.meta || {};
          for (const [pid, p] of room) if (pid !== id) send(p.ws, { t: "peer-meta", id, meta: me.meta });
          break;
        }
        case "ping":
          send(ws, { t: "pong", ts: m.ts });
          break;
      }
    });

    ws.on("close", () => {
      if (!id || !code) return;
      const room = rooms.get(code);
      if (!room) return;
      const gone = room.get(id);
      room.delete(id);
      for (const [, p] of room) send(p.ws, { t: "peer-left", id });
      if (log) console.log(`  - ${gone?.name || id} left ${code} (${room.size} left)`);
      if (room.size === 0) rooms.delete(code);
    });
  });

  // drop sockets that stopped answering: a closed laptop lid does not send a FIN
  const beat = setInterval(() => {
    for (const ws of wss.clients) {
      if (!ws.isAlive) { ws.terminate(); continue; }
      ws.isAlive = false;
      ws.ping();
    }
  }, 15000);
  wss.on("close", () => clearInterval(beat));

  return wss;
}
