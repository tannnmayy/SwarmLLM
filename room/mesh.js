// WebRTC mesh: every device holds a direct connection to every other device.
//
// The signaling server introduces peers and then stops mattering -- activations,
// prompts and answers only ever travel over these data channels.
//
// Two negotiated channels per link:
//   ctrl (id 1)  JSON control messages and the RTT probe
//   wire (id 77) binary activation frames, sliced by room/wire.js
// Negotiated channels are created identically on both ends, so neither side waits
// for an ondatachannel event and a link is usable the moment ICE completes.
//
// Glare is avoided by rule rather than by rollback: the *newcomer* offers to
// everyone already in the room, and existing members only ever answer. Two peers
// can therefore never offer each other simultaneously.

import { encodeFrame, decodeSlice, makeReassembler } from "./wire.js";

const ICE = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
];

const CTRL_ID = 1;
const WIRE_ID = 77;

export class Mesh {
  constructor({ room, name = "device", meta = {}, url = null } = {}) {
    this.room = String(room || "").toUpperCase();
    this.name = name;
    this.meta = meta;
    this.id = null;
    this.peers = new Map();          // id -> { id, name, meta, pc, ctrl, wire, rs, rtt, msgId, ready }
    this.url = url || (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/signal";
    this.ws = null;
    this._h = new Map();
    this._beat = null;
    this._closed = false;
  }

  on(evt, fn) {
    if (!this._h.has(evt)) this._h.set(evt, new Set());
    this._h.get(evt).add(fn);
    return this;
  }
  _emit(evt, ...args) {
    for (const fn of this._h.get(evt) || []) {
      try { fn(...args); } catch (e) { console.error("[mesh] handler for " + evt, e); }
    }
  }

  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      this.ws = ws;
      const fail = (e) => reject(new Error("signaling unreachable: " + this.url + " (" + e + ")"));
      ws.onerror = () => fail("error");
      ws.onopen = () => ws.send(JSON.stringify({ t: "join", room: this.room, name: this.name, meta: this.meta }));
      ws.onclose = () => {
        if (!this._closed) this._emit("signal-lost");
      };
      ws.onmessage = async (ev) => {
        let m;
        try { m = JSON.parse(ev.data); } catch { return; }
        switch (m.t) {
          case "welcome":
            this.id = m.id;
            this.room = m.room;
            this._emit("open", { id: m.id, room: m.room, peers: m.peers });
            // I am the newcomer: I offer to everyone already here.
            for (const p of m.peers) this._link(p, true);
            this._startBeat();
            resolve({ id: m.id, room: m.room, peers: m.peers });
            break;
          case "peer-join":
            // They are the newcomer: they will offer, I only answer.
            this._link({ id: m.id, name: m.name, meta: m.meta }, false);
            this._emit("roster", this.roster());
            break;
          case "signal":
            await this._onSignal(m.from, m.data);
            break;
          case "peer-meta": {
            const p = this.peers.get(m.id);
            if (p) { p.meta = m.meta; this._emit("roster", this.roster()); }
            break;
          }
          case "peer-left":
            this._drop(m.id, "left the room");
            break;
          case "error":
            reject(new Error(m.why));
            break;
        }
      };
    });
  }

  roster() {
    return [...this.peers.values()].map((p) => ({
      id: p.id, name: p.name, meta: p.meta, rtt: p.rtt, ready: p.ready,
    }));
  }

  _sig(to, data) {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ t: "signal", to, data }));
  }

  _peer(info) {
    let p = this.peers.get(info.id);
    if (!p) {
      p = {
        id: info.id, name: info.name, meta: info.meta || {},
        pc: null, ctrl: null, wire: null,
        rs: makeReassembler(), rtt: null, msgId: 1, ready: false,
        pending: [],                                  // remote ICE before setRemoteDescription
      };
      this.peers.set(info.id, p);
    }
    return p;
  }

  _link(info, initiator) {
    const p = this._peer(info);
    if (p.pc) return p;

    const pc = new RTCPeerConnection({ iceServers: ICE });
    p.pc = pc;

    pc.onicecandidate = (e) => {
      if (e.candidate) this._sig(p.id, { ice: e.candidate });
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed" || pc.connectionState === "closed") {
        this._drop(p.id, "connection " + pc.connectionState);
      }
    };

    // Both sides create the same negotiated channels, so this is symmetric.
    p.ctrl = pc.createDataChannel("ctrl", { negotiated: true, id: CTRL_ID, ordered: true });
    p.wire = pc.createDataChannel("wire", { negotiated: true, id: WIRE_ID, ordered: true });
    p.wire.binaryType = "arraybuffer";

    p.ctrl.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); } catch { return; }
      if (m.t === "__ping") return p.ctrl.readyState === "open" && p.ctrl.send(JSON.stringify({ t: "__pong", ts: m.ts }));
      if (m.t === "__pong") { p.rtt = Math.round(performance.now() - m.ts); return; }
      this._emit("msg", p.id, m);
    };
    p.wire.onmessage = (ev) => {
      const frame = decodeSlice(p.rs, ev.data);
      if (frame) this._emit("frame", p.id, frame);
    };
    p.ctrl.onopen = () => {
      if (p.ready) return;
      p.ready = true;
      this._emit("peer", p);
      this._emit("roster", this.roster());
    };

    if (initiator) {
      (async () => {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        this._sig(p.id, { sdp: pc.localDescription });
      })().catch((e) => console.error("[mesh] offer to " + p.id, e));
    }
    return p;
  }

  async _onSignal(from, data) {
    const p = this.peers.get(from) || this._link({ id: from, name: "device" }, false);
    if (!p.pc) return;
    if (data.sdp) {
      await p.pc.setRemoteDescription(data.sdp);
      if (data.sdp.type === "offer") {
        await p.pc.setLocalDescription(await p.pc.createAnswer());
        this._sig(from, { sdp: p.pc.localDescription });
      }
      // candidates that arrived before we had a remote description
      for (const c of p.pending.splice(0)) {
        await p.pc.addIceCandidate(c).catch(() => {});
      }
    } else if (data.ice) {
      if (p.pc.remoteDescription) await p.pc.addIceCandidate(data.ice).catch(() => {});
      else p.pending.push(data.ice);
    }
  }

  _drop(id, why) {
    const p = this.peers.get(id);
    if (!p) return;
    this.peers.delete(id);
    try { p.pc?.close(); } catch {}
    this._emit("left", id, why, p);
    this._emit("roster", this.roster());
  }

  // RTT every 2 s. The scheduler needs a peer-to-peer latency matrix to order the
  // chain, and this is where the numbers come from.
  _startBeat() {
    clearInterval(this._beat);
    this._beat = setInterval(() => {
      for (const p of this.peers.values()) {
        if (p.ctrl?.readyState === "open") p.ctrl.send(JSON.stringify({ t: "__ping", ts: performance.now() }));
      }
    }, 2000);
  }

  send(id, obj) {
    const p = this.peers.get(id);
    if (p?.ctrl?.readyState !== "open") return false;
    p.ctrl.send(JSON.stringify(obj));
    return true;
  }

  broadcast(obj) {
    let n = 0;
    for (const id of this.peers.keys()) if (this.send(id, obj)) n++;
    return n;
  }

  // msg: { t, pos, n?, flags?, data: Uint16Array }
  sendFrame(id, msg) {
    const p = this.peers.get(id);
    if (p?.wire?.readyState !== "open") return false;
    for (const slice of encodeFrame(msg, p.msgId++)) p.wire.send(slice);
    return true;
  }

  // Tell the room what this device is now worth (memory freed, speed measured, battery).
  setMeta(meta) {
    this.meta = meta;
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ t: "meta", meta }));
  }

  close() {
    this._closed = true;
    clearInterval(this._beat);
    for (const p of this.peers.values()) { try { p.pc?.close(); } catch {} }
    this.peers.clear();
    try { this.ws?.close(); } catch {}
  }
}
