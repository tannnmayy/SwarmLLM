// The room: turns a set of connected browsers into one model.
//
// One device is the host. It owns the conversation, the tokenizer, the embedding
// table, the LM head and the sampler. The others are workers holding a contiguous
// range of layers. Together they form a chain in layer order, and the last worker
// sends the hidden state back to the host.
//
//   host    embed(token) -> run my layers ─┐
//                                          ▼
//   worker A  layers a..b ──► worker B  layers b..c ──► ... ──┐
//                                                             │
//   host    final norm -> LM head -> sample -> next token ◄────┘
//
// Layer dealing here is deliberately naive -- an even split in join order. That is
// the *baseline* the scheduler in scheduler/ has to beat, and keeping it intact
// means the comparison is honest rather than a strawman we wrote to lose.

import { Mesh } from "./mesh.js";
import { packF16, unpackF16, looksBad } from "./wire.js";
import { CpuEngine, argmax } from "../engine/cpu.mjs";
import { Tokenizer } from "../tools/tokenizer.mjs";

export const MODELS = {
  "smollm2-135m": { label: "SmolLM2 135M", dir: "/models/smollm2-135m", layers: 30 },
};

export class Room {
  constructor({ code, name, model = "smollm2-135m" } = {}) {
    this.model = model;
    this.mesh = new Mesh({ room: code, name, meta: {} });
    this.engine = null;
    this.tok = null;
    this.isHost = false;
    this.hostId = null;
    this.chain = [];              // ordered worker ids, host first conceptually
    this.range = null;            // my [lo, hi)
    this.next = null;             // who I forward to, or null = back to the host
    this.pos = 0;
    this.busy = false;
    this.waiting = new Map();     // pos -> resolve, host side
    this._h = new Map();
    this.stats = { tokens: 0, ms: 0, hops: [] };
    this._wire();
  }

  on(evt, fn) {
    if (!this._h.has(evt)) this._h.set(evt, new Set());
    this._h.get(evt).add(fn);
    return this;
  }
  _emit(evt, ...a) {
    for (const fn of this._h.get(evt) || []) { try { fn(...a); } catch (e) { console.error(evt, e); } }
  }

  get peers() { return this.mesh.roster(); }
  get id() { return this.mesh.id; }
  get code() { return this.mesh.room; }

  async join() {
    const r = await this.mesh.connect();
    this.tok = await Tokenizer.load(MODELS[this.model].dir + "/tokenizer.json");
    this._emit("joined", r);
    this._emit("roster", this.peers);
    return r;
  }

  _wire() {
    this.mesh.on("peer", () => this._emit("roster", this.peers));
    this.mesh.on("roster", () => this._emit("roster", this.peers));
    this.mesh.on("left", (id, why) => {
      this._emit("roster", this.peers);
      this._emit("peer-left", id, why);
      // A device in the chain leaving mid-answer is the case the scheduler's recovery
      // path exists for. Until that lands, fail loudly rather than hanging forever.
      if (this.busy && (this.chain.includes(id) || id === this.hostId)) {
        for (const [, resolve] of this.waiting) resolve(null);
        this.waiting.clear();
        this._emit("chain-broken", id);
      }
    });

    this.mesh.on("msg", (from, m) => this._onMsg(from, m));
    this.mesh.on("frame", (from, f) => this._onFrame(from, f));
  }

  // ---------------------------------------------------------------- control
  async _onMsg(from, m) {
    switch (m.t) {
      case "deal":
        this.hostId = m.host;
        this.range = m.range;
        this.next = m.next;
        this.isHost = false;
        await this._load(m.range, false, false);
        this.mesh.send(this.hostId, { t: "ready", range: this.range });
        break;

      case "progress":
        this._emit("progress", from, m.pct);
        break;

      case "ready":
        this._readyCount = (this._readyCount || 0) + 1;
        this._emit("worker-ready", from, m.range);
        if (this._readyCount >= this.chain.length) {
          this.mesh.broadcast({ t: "ready-all" });
          this._emit("ready");
        }
        break;

      case "ready-all":
        this._emit("ready");
        break;

      case "ask":
        if (this.isHost) this.generate(m.text, from);
        break;

      case "gen-start":
        this._emit("gen-start", m.text, m.by);
        break;
      case "token":
        this._emit("token", m.text);
        break;
      case "gen-done":
        this._emit("gen-done", m.stats);
        break;

      case "reset":
        this.engine?.reset();
        this.pos = 0;
        this._emit("reset");
        break;
    }
  }

  // ---------------------------------------------------------------- data
  _onFrame(from, f) {
    if (!this.engine) return;

    if (f.t === "hidden") {
      // I am a worker: run my layers, pass it on.
      const x = unpackF16(f.data);
      if (looksBad(x)) { this._emit("error", "non-finite hidden state from " + from); return; }
      const t0 = performance.now();
      const out = this.engine.runHidden(x, f.pos);
      const ms = performance.now() - t0;
      this._emit("stage", { pos: f.pos, ms, layers: this.engine.layerCount });
      const data = packF16(out);
      if (this.next) this.mesh.sendFrame(this.next, { t: "hidden", pos: f.pos, data });
      else this.mesh.sendFrame(this.hostId, { t: "hidden-ret", pos: f.pos, data });
      return;
    }

    if (f.t === "hidden-ret") {
      const resolve = this.waiting.get(f.pos);
      if (resolve) { this.waiting.delete(f.pos); resolve(unpackF16(f.data)); }
    }
  }

  // ---------------------------------------------------------------- loading
  async _load(range, hasEmbed, hasHead) {
    this._emit("loading", { range, pct: 0 });
    const dir = MODELS[this.model].dir;
    let lastSent = 0;
    this.engine = await CpuEngine.load(dir, {
      layerRange: range,
      hasEmbed,
      hasHead,
      onProgress: (frac) => {
        const pct = Math.round(frac * 100);
        this._emit("loading", { range, pct });
        if (!this.isHost && this.hostId && pct - lastSent >= 10) {
          lastSent = pct;
          this.mesh.send(this.hostId, { t: "progress", pct });
        }
      },
    });
    this._emit("loaded", { range, mb: this.engine.bytesLoaded / 2 ** 20 });
    return this.engine;
  }

  // ---------------------------------------------------------------- start
  // Baseline placement: even split over everyone, in join order. Replaced by
  // scheduler/plan.js once profiling lands -- and kept, so the two can be compared.
  planEven(deviceIds, layers) {
    const n = deviceIds.length;
    const base = Math.floor(layers / n);
    const extra = layers % n;
    const ranges = [];
    let at = 0;
    for (let i = 0; i < n; i++) {
      const take = base + (i < extra ? 1 : 0);
      ranges.push([at, at + take]);
      at += take;
    }
    return ranges;
  }

  async start(plan = null) {
    this.isHost = true;
    this.hostId = this.id;
    this._readyCount = 0;

    const workers = this.peers.filter((p) => p.ready).map((p) => p.id);
    const order = [this.id, ...workers];
    const L = MODELS[this.model].layers;
    const ranges = plan || this.planEven(order, L);

    this.chain = workers;
    this._emit("plan", order.map((id, i) => ({ id, range: ranges[i], self: id === this.id })));

    // Tell each worker its range and who it forwards to. The last one returns to me.
    for (let i = 1; i < order.length; i++) {
      this.mesh.send(order[i], {
        t: "deal",
        model: this.model,
        range: ranges[i],
        next: i + 1 < order.length ? order[i + 1] : null,
        host: this.id,
      });
    }

    this.range = ranges[0];
    this.next = workers[0] || null;
    await this._load(this.range, true, true);

    if (!workers.length) { this._emit("ready"); return; }
    this._emit("waiting-for-workers", workers.length);
  }

  // ---------------------------------------------------------------- generate
  async _step(tokenId, pos) {
    let x = this.engine.embedRun(tokenId, pos);
    if (this.next) {
      const t0 = performance.now();
      x = await new Promise((resolve) => {
        this.waiting.set(pos, resolve);
        this.mesh.sendFrame(this.next, { t: "hidden", pos, data: packF16(x) });
        setTimeout(() => {
          if (this.waiting.has(pos)) { this.waiting.delete(pos); resolve(null); }
        }, 20000);
      });
      if (!x) throw new Error("the chain did not answer in time");
      this.stats.hops.push(performance.now() - t0);
    }
    return this.engine.headFromHidden(x);
  }

  async generate(text, askedBy = null, { maxTokens = 60 } = {}) {
    if (this.busy) return;
    if (!this.isHost) { this.mesh.send(this.hostId, { t: "ask", text }); return; }
    this.busy = true;
    this.stats = { tokens: 0, ms: 0, hops: [] };

    const by = askedBy ? (this.peers.find((p) => p.id === askedBy)?.name || "someone") : "you";
    this.mesh.broadcast({ t: "gen-start", text, by });
    this._emit("gen-start", text, by);

    const t0 = performance.now();
    try {
      const ids = this.tok.encode(text);
      let logits = null;
      for (const id of ids) logits = await this._step(id, this.pos++);

      for (let n = 0; n < maxTokens; n++) {
        const next = argmax(logits);
        if (next === this.engine.cfg.eos) break;
        const piece = this.tok.decode([next]);
        this.mesh.broadcast({ t: "token", text: piece });
        this._emit("token", piece);
        this.stats.tokens++;
        logits = await this._step(next, this.pos++);
        if (this.pos >= this.engine.maxSeq - 2) break;
      }
    } catch (e) {
      this._emit("error", e.message);
    }

    this.stats.ms = performance.now() - t0;
    const s = {
      tokens: this.stats.tokens,
      ms: Math.round(this.stats.ms),
      tps: +(this.stats.tokens / (this.stats.ms / 1000)).toFixed(2),
      medianHopMs: this.stats.hops.length
        ? +this.stats.hops.slice().sort((a, b) => a - b)[this.stats.hops.length >> 1].toFixed(1)
        : null,
    };
    this.mesh.broadcast({ t: "gen-done", stats: s });
    this._emit("gen-done", s);
    this.busy = false;
  }

  resetConversation() {
    this.engine?.reset();
    this.pos = 0;
    this.mesh.broadcast({ t: "reset" });
    this._emit("reset");
  }
}
