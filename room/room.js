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
// Placement comes from scheduler/plan.js, which decides four things together: which
// devices are worth including at all, in what order, holding which layers, and which
// one should be host. The naive strategies it competes against are kept and reachable
// (`strategy: "even" | "memory" | ...`), so the comparison stays a measurement rather
// than a strawman written to lose.
//
// Profiling happens twice, on purpose. A synthetic probe at join gives the planner
// something to work with before anyone holds weights; once layers are loaded, the
// real thing is timed and the room is told. That second pass is not redundant -- a
// backgrounded tab measures ~3x slow, and an uncorrected number there would let one
// throttled phone quietly set the pace for every token.

import { Mesh } from "./mesh.js";
import { packF16, unpackF16, looksBad } from "./wire.js";
import { CpuEngine, argmax } from "../engine/cpu.mjs";
import { Tokenizer } from "../tools/tokenizer.mjs";
import { modelSpec, plan as solvePlan, compareAll } from "../scheduler/plan.js";
import { rttLookup } from "../scheduler/cost.js";
import { profile, watchPressure, defaultPledgeBytes } from "../scheduler/probe.js";
import { keepAwake } from "./awake.js";

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
    this.spec = null;             // model cost model, set on join
    this.myProfile = null;        // this device's measured speed and pledge
    this.pledgeBytes = null;      // what the user chose to contribute
    this.rttMatrix = new Map();   // id -> { id: ms }, gossiped so the host sees it all
    this.lastPlan = null;
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

  async join({ pledgeBytes = null } = {}) {
    const dir = MODELS[this.model].dir;
    const manifest = await (await fetch(dir + "/manifest.json")).json();
    this.spec = modelSpec(manifest, { precision: "f32", maxSeq: 512 });

    const r = await this.mesh.connect();
    this.tok = await Tokenizer.load(dir + "/tokenizer.json");
    this._emit("joined", r);

    // Measure this device before telling anyone what it is worth. ~300 ms.
    this.pledgeBytes = pledgeBytes ?? defaultPledgeBytes();
    this._emit("profiling");
    this.myProfile = await profile(this.spec, { pledgeBytes: this.pledgeBytes });
    this._announce();
    this._emit("profiled", this.myProfile);

    // A phone that starts throttling should be re-planned around, not left to set
    // the pace for the whole room.
    this._unwatch = watchPressure((state) => {
      this._emit("pressure", state);
      if (state === "serious" || state === "critical") {
        this.myProfile.pressure = state;
        this._announce();
      }
    });

    // Gossip RTTs: each device only measures its own links, and the planner needs
    // the whole matrix to order the chain.
    this._rttTimer = setInterval(() => this._announce(), 4000);

    // A hidden tab is throttled hard -- on a phone with the screen off, by more than
    // an order of magnitude. Any speed measured while hidden is a lie, so re-measure
    // on the way back to visible and tell the room the device changed.
    this._onVis = async () => {
      if (document.visibilityState !== "visible" || this.busy) return;
      const now = Date.now();
      if (now - (this._lastProbe || 0) < 5000) return;      // debounce: it fires in bursts
      this._lastProbe = now;
      if (this.engine) this._calibrate();
      else {
        this.myProfile = await profile(this.spec, { pledgeBytes: this.pledgeBytes });
        this._announce();
        this._emit("profiled", this.myProfile);
      }
    };
    document.addEventListener("visibilitychange", this._onVis);

    this._emit("roster", this.peers);
    return r;
  }

  _announce() {
    if (!this.myProfile) return;
    const rtts = {};
    for (const p of this.mesh.roster()) if (p.rtt != null) rtts[p.id] = p.rtt;
    this.rttMatrix.set(this.id, rtts);
    this.mesh.setMeta({
      msPerLayer: this.myProfile.msPerLayer,
      budgetBytes: this.pledgeBytes,
      stable: this.myProfile.stable,
      pressure: this.myProfile.pressure,
      battery: this.myProfile.battery,
      rtts,
    });
  }

  setPledge(bytes) {
    this.pledgeBytes = bytes;
    this._announce();
    this._emit("roster", this.peers);
  }

  // Everything the planner needs, assembled from the roster.
  _devices() {
    const out = [{
      id: this.id,
      name: this.mesh.name,
      msPerLayer: this.myProfile.msPerLayer,
      budgetBytes: this.pledgeBytes,
    }];
    for (const p of this.peers) {
      if (!p.ready || !p.meta?.msPerLayer) continue;
      out.push({
        id: p.id,
        name: p.name,
        msPerLayer: p.meta.msPerLayer,
        budgetBytes: p.meta.budgetBytes,
      });
    }
    return out;
  }

  _rtt() {
    const m = {};
    for (const [from, row] of this.rttMatrix) m[from] = { ...row };
    for (const p of this.peers) if (p.meta?.rtts) m[p.id] = { ...(m[p.id] || {}), ...p.meta.rtts };
    // Unmeasured pairs fall back to the worst link we have actually seen, not to
    // zero: assuming a free link to a device nobody has pinged is how you get a
    // plan that looks great and runs badly.
    const seen = Object.values(m).flatMap((r) => Object.values(r)).filter(Number.isFinite);
    const fallback = seen.length ? Math.max(...seen) : 25;
    return rttLookup(m, { fallback });
  }

  // What every strategy would do with this room, right now. The benchmark table.
  compare() {
    if (!this.spec || !this.myProfile) return null;
    return compareAll(this.spec, this._devices(), this._rtt());
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

      case "become-host":
        // The previous device's planner elected me. Re-solve locally rather than
        // trusting a plan computed elsewhere: by now I may know links it did not.
        this._emit("became-host");
        await this.start({ strategy: m.strategy || "optimal" });
        break;

      case "stand-by":
        this._emit("stand-by", m.why);
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
    // From here this device is in the chain, so it must not be allowed to doze.
    keepAwake().catch(() => {});
    this._calibrate();
    return this.engine;
  }

  // Second-phase profiling: now that real layers are loaded, time the real thing.
  //
  // The pre-load probe has to be synthetic -- placement is decided before anyone has
  // any weights -- but a synthetic number can be wrong for reasons that matter. A
  // backgrounded tab or a phone with its screen off is throttled hard by the browser,
  // and a device that measured fast at join can be several times slower by the time it
  // is holding layers. Left uncorrected, that device silently sets the pace for every
  // token in the room.
  _calibrate() {
    if (!this.engine || !this.engine.layerCount) return;
    const n = this.engine.layerCount;
    const x = new Float32Array(this.spec.hidden);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.31) * 0.7;

    const runs = [];
    for (let i = 0; i < 5; i++) {
      const t0 = performance.now();
      this.engine.runHidden(x, i);
      runs.push((performance.now() - t0) / n);
    }
    this.engine.reset();                       // undo the cache these probes wrote
    runs.sort((a, b) => a - b);
    const measured = runs[runs.length >> 1];

    const before = this.myProfile.msPerLayer;
    this.myProfile.msPerLayer = measured;
    this.myProfile.calibrated = true;
    this._announce();

    const drift = measured / before;
    this._emit("calibrated", { before, after: measured, drift });
    if (drift > 1.5 || drift < 0.67) {
      this._emit("recalibrated", { before, after: measured, drift });
    }
  }

  // ---------------------------------------------------------------- start
  //
  // Solve, then act on the answer -- including when the answer is "someone else
  // should be host". Whoever pressed Start is not necessarily the right device to
  // run the LM head, and that decision is worth more than any layer cut: the head
  // is ~8 layers of arithmetic and it is serial on the host.
  async start({ strategy = "optimal" } = {}) {
    if (!this.myProfile) { this._emit("error", "still measuring this device"); return; }

    const devices = this._devices();
    const p = solvePlan(this.spec, devices, this._rtt(), { strategy });
    if (!p) {
      this._emit("error", `no feasible plan: the room pledges too little memory for ${MODELS[this.model].label}`);
      return;
    }
    this.lastPlan = p;
    this._emit("planned", p);

    if (p.host !== this.id) {
      // The planner elected someone else. Hand over rather than overrule it.
      const name = this.peers.find((x) => x.id === p.host)?.name || p.host;
      this._emit("handoff", p.host, name);
      this.mesh.send(p.host, { t: "become-host", strategy });
      return;
    }
    await this._deal(p);
  }

  async _deal(p) {
    this.isHost = true;
    this.hostId = this.id;
    this._readyCount = 0;

    const workers = p.chain.slice(1);
    this.chain = workers;
    this._emit("plan", p.chain.map((id, i) => ({ id, range: p.ranges[i], self: id === this.id })));

    // Each worker learns its range and who it forwards to; the last returns to me.
    for (let i = 1; i < p.chain.length; i++) {
      this.mesh.send(p.chain[i], {
        t: "deal",
        model: this.model,
        range: p.ranges[i],
        next: i + 1 < p.chain.length ? p.chain[i + 1] : null,
        host: this.id,
      });
    }
    // Devices the planner left out are told why, rather than left wondering.
    for (const id of p.dropped) {
      this.mesh.send(id, { t: "stand-by", why: p.why.find((w) => w.includes(this.peers.find((x) => x.id === id)?.name || " ")) || "not needed for this plan" });
    }

    this.range = p.ranges[0];
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
