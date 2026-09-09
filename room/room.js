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
import { packWire, unpackWire, looksBad, chooseEncoding } from "./wire.js";
import { argmax } from "../engine/cpu.mjs";
import { createEngine } from "../engine/factory.mjs";
import { probeHeader, modelSpecFromGGUF } from "../engine/gpu-adapter.mjs";
import { Tokenizer } from "../tools/tokenizer.mjs";
import { modelSpec, plan as solvePlan, compareAll } from "../scheduler/plan.js";
import { rttLookup } from "../scheduler/cost.js";
import { profile, watchPressure, defaultPledgeBytes } from "../scheduler/probe.js";
import { keepAwake } from "./awake.js";
import { MODELS, getModel } from "../models/registry.mjs";

// The system turn. Short on purpose: every token here is a token of context the
// conversation does not get, and the window is only 512 positions wide.
const SYSTEM = "You are a helpful AI assistant running across several devices at once.";

// Re-exported for callers that already do `import { MODELS } from "./room.js"`
// (room.html). The room itself now asks the registry, not this file, for a
// model's shape — see models/registry.mjs.
export { MODELS };

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
    this.history = [];            // every token fed so far; what recovery replays
    this.turns = [];              // the conversation, as chat turns
    this.recovering = false;
    this.strategy = "optimal";
    // Ids the mesh has told us left the chain, checked by _step() before it sends a
    // frame to `this.next` -- see the comment where it is read for why this exists.
    this._deadPeers = new Set();
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

  // Everything that depends on WHICH model this room runs: the scheduler spec
  // (real GGUF tensor bytes for a GPU model, the manifest for the CPU one), the
  // tokenizer, and the wire encoding those bytes imply. Used both at join and
  // when a "deal" message tells a worker the host is running a different model
  // than the one it assumed (see the "deal" case in _onMsg) — one code path for
  // both, so there is only one place this can be wrong.
  async _loadModelSpec(descriptor) {
    const maxSeq = descriptor.maxSeqDefault || 512;
    if (descriptor.engineKind === "cpu-smollm") {
      const manifest = await (await fetch(descriptor.dir + "/manifest.json")).json();
      this.spec = modelSpec(manifest, { precision: "f32", maxSeq });
      this.tok = await Tokenizer.load(descriptor.dir + "/tokenizer.json");
      this._ggufHeader = null; this._ggufCfg = null;
    } else if (descriptor.engineKind === "dense-gguf") {
      const { header, cfg } = await probeHeader(descriptor);
      // Cached so _load() does not pay a second header round trip for the model
      // it just resolved this very spec from.
      this._ggufHeader = header; this._ggufCfg = cfg;
      this.spec = modelSpecFromGGUF(header, cfg, descriptor, { maxSeq });
      this.tok = await Tokenizer.load(descriptor.tokenizerUrl);
    } else {
      throw new Error(`${descriptor.label}: ${descriptor.blockedOn || "not available on this build yet"}`);
    }
    // f32 on the wire when it costs no extra SCTP slice, so a split answer is
    // bit-identical to the single-device answer, for models whose hidden size
    // allows it; wider models (Qwen3) fall to f16. See room/wire.js.
    this.wireEnc = chooseEncoding(this.spec.hidden);

    // The turn-closing token, from the tokenizer's own special tokens rather than
    // the engine's cfg: a GGUF header is always fetched with skipTokenizer:true
    // (see engine/gpu-adapter.mjs's fetchHeader), which drops every
    // "tokenizer.ggml.*" key including eos_token_id, so engine.cfg.eos is never
    // populated for a dense-gguf model. Every ChatML model closes a turn with
    // <|im_end|>, so this is also the model-agnostic answer CpuEngine's manifest
    // "eos" field was really encoding all along.
    this.eosId = this.tok.special?.get("<|im_end|>") ?? null;
  }

  async join({ pledgeBytes = null } = {}) {
    await this._loadModelSpec(getModel(this.model));

    const r = await this.mesh.connect();
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
      if (this.engine) await this._calibrate();
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

  // Can this room hold the model yet, and if not, how far short is it?
  //
  // "No feasible plan" is a true answer and a useless one. A person needs to know
  // whether to lend more of their own device or fetch another one, so this returns
  // the shortfall in bytes as well as the verdict.
  capacity() {
    if (!this.spec || !this.myProfile) return null;
    const devices = this._devices();
    const have = devices.reduce((s, d) => s + d.budgetBytes, 0);
    // What the model costs to hold: every layer, plus the embedding table that the
    // host carries on top of its own share.
    const need = this.spec.layers * (this.spec.layerBytes + this.spec.kvBytesPerLayer)
               + this.spec.embedBytes + this.spec.scratchBytes;
    // Feasibility is the planner's to decide, not arithmetic's: it knows the host
    // carries the embedding table and that each device has its own ceiling.
    let plan = null;
    try { plan = solvePlan(this.spec, devices, this._rtt(), { strategy: this.strategy }); } catch {}
    return {
      needBytes: need,
      haveBytes: have,
      shortBytes: Math.max(0, need - have),
      ready: !!plan,
      devices: devices.length,
      usable: plan ? plan.chain.length : 0,
      frac: Math.min(1, have / need),
    };
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

      if (!this.isHost) {
        // Losing the host is not recoverable from here: the conversation, the
        // tokenizer state and the LM head all live there. Say so plainly.
        if (id === this.hostId) this._emit("host-lost", id);
        return;
      }
      if (!this.chain.includes(id)) return;          // a bystander left; nothing to do

      this._emit("chain-broken", id);
      // Fail any lap still in flight straight away rather than waiting out its
      // timeout -- _stepSafe turns that failure into a recovery. This only covers
      // a lap that happens to be sitting in `this.waiting` at this exact instant,
      // though: measured live (see IMPLEMENTATION_PLAN.md Phase A), a departure
      // that lands between two laps -- generation is busy, but no frame is in
      // flight right now -- fell through this fast path entirely, and the NEXT
      // _step() call sent its frame to a `this.next` that was already known dead,
      // paying the full 20 s per-lap timeout before _stepSafe ever saw a failure.
      // Recording the id here closes that gap; see _step()'s check below.
      this._deadPeers.add(id);
      for (const [, resolve] of this.waiting) resolve(null);
      this.waiting.clear();

      // If nothing is generating, there is no lap to fail, so re-plan now rather
      // than letting the next question discover the chain is broken.
      if (!this.busy && !this.recovering && this.engine) {
        this._recover("a device left while the room was idle");
      }
    });

    this.mesh.on("msg", (from, m) => this._onMsg(from, m));
    this.mesh.on("frame", (from, f) => this._onFrame(from, f));
  }

  // ---------------------------------------------------------------- control
  async _onMsg(from, m) {
    switch (m.t) {
      case "deal": {
        // The deal names the model the HOST is actually running. A worker that
        // joined before knowing that (or that was constructed with a stale
        // default) must adopt it here rather than silently loading its own
        // assumption — every device in the chain has to agree on one model, or
        // the wire's hidden-state width and the tokenizer's vocabulary are both
        // wrong without anything raising an error until logits come out as noise.
        const modelChanged = m.model !== this.model;
        if (modelChanged) {
          let descriptor;
          try { descriptor = getModel(m.model); }
          catch (e) { this._emit("error", `host dealt an unknown model "${m.model}": ${e.message}`); break; }

          this._emit("model-changed", { from: this.model, to: m.model });
          this.model = m.model;
          try {
            await this._loadModelSpec(descriptor);
          } catch (e) {
            this._emit("error", `could not load model "${m.model}": ${e.message}`);
            break;
          }
          // A model swap invalidates anything held under the old model's shape —
          // never treat this as "kept range" even if the byte-range numbers
          // happen to coincide. Dispose here, not just discard: a GpuEngineAdapter
          // holds real WebGPU buffers, and nothing else will ever call dispose()
          // on this particular instance once the reference is gone.
          this.engine?.dispose?.();
          this.engine = null;
          this.range = null;
        }

        // A re-deal after a failure usually leaves most devices holding exactly what
        // they held before. Re-downloading those layers would turn a two-second
        // recovery into a thirty-second one, so keep the weights and clear only the
        // cache — the host is about to replay the conversation into it anyway.
        const same = !modelChanged && this.engine && this.range &&
          this.range[0] === m.range[0] && this.range[1] === m.range[1];
        this.hostId = m.host;
        this.range = m.range;
        this.next = m.next;
        this.isHost = false;
        if (same) {
          this.engine.reset();
          this._emit("kept-range", this.range);
        } else {
          await this._load(m.range, false, false);
        }
        this.mesh.send(this.hostId, { t: "ready", range: this.range });
        break;
      }

      case "progress":
        this._emit("progress", from, m.pct);
        break;

      case "ready":
        this._readyCount = (this._readyCount || 0) + 1;
        this._emit("worker-ready", from, m.range);
        if (this._pending && this._readyCount >= this._pending.need) {
          clearTimeout(this._pending.timer);
          const done = this._pending.resolve;
          this._pending = null;
          done();
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

      case "please-leave":
        // Demo instrument: the host asks a device to walk out, so a failure can be
        // shown on cue instead of hoping somebody's laptop misbehaves on stage.
        this._emit("asked-to-leave");
        this.mesh.close();
        break;

      case "reset":
        this.engine?.reset();
        this.pos = 0;
        this._emit("reset");
        break;
    }
  }

  // ---------------------------------------------------------------- data
  async _onFrame(from, f) {
    if (!this.engine) return;

    if (f.t === "hidden") {
      // I am a worker: run my layers, pass it on.
      const x = unpackWire(f.data, f.enc);
      if (looksBad(x)) { this._emit("error", "non-finite hidden state from " + from); return; }
      const t0 = performance.now();
      const out = await this.engine.runHidden(x, f.pos);
      const ms = performance.now() - t0;
      this._emit("stage", { pos: f.pos, ms, layers: this.engine.layerCount });
      const data = packWire(out, this.wireEnc);
      if (this.next) this.mesh.sendFrame(this.next, { t: "hidden", pos: f.pos, data });
      else this.mesh.sendFrame(this.hostId, { t: "hidden-ret", pos: f.pos, data });
      return;
    }

    if (f.t === "hidden-ret") {
      const resolve = this.waiting.get(f.pos);
      if (resolve) { this.waiting.delete(f.pos); resolve(unpackWire(f.data, f.enc)); }
    }
  }

  // ---------------------------------------------------------------- loading
  async _load(range, hasEmbed, hasHead) {
    // Every path that replaces this.engine with a freshly loaded one goes through
    // here (the deal handler's reload branch, _deal()'s own host-range (re)load,
    // and any future caller) except the "kept-range" fast path, which intentionally
    // keeps the same engine/buffers because the layer range did not change — do
    // not dispose there, it would defeat the point of that optimization. A
    // GpuEngineAdapter's dispose() destroys its WebGPU device; CpuEngine has no
    // dispose() at all, hence the optional chain.
    this.engine?.dispose?.();
    this._emit("loading", { range, pct: 0 });
    const descriptor = getModel(this.model);
    let lastSent = 0;
    this.engine = await createEngine(descriptor, {
      layerRange: range,
      hasEmbed,
      hasHead,
      // Reuse the header this device already fetched in join() (or in the "deal"
      // handler, on a model change) instead of range-fetching it a second time —
      // it describes the same file either way.
      preloaded: this._ggufHeader ? { header: this._ggufHeader, cfg: this._ggufCfg } : undefined,
      onProgress: (frac) => {
        const pct = Math.round(frac * 100);
        this._emit("loading", { range, pct });
        if (!this.isHost && this.hostId && pct - lastSent >= 10) {
          lastSent = pct;
          this.mesh.send(this.hostId, { t: "progress", pct });
        }
      },
    });
    this._emit("loaded", { range, mb: this.engine.bytesLoaded / 2 ** 20, cache: this.engine.cache || null });
    // From here this device is in the chain, so it must not be allowed to doze.
    keepAwake().catch(() => {});
    await this._calibrate();
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
  async _calibrate() {
    if (!this.engine || !this.engine.layerCount) return;
    const n = this.engine.layerCount;
    const x = new Float32Array(this.spec.hidden);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.31) * 0.7;

    const runs = [];
    for (let i = 0; i < 5; i++) {
      const t0 = performance.now();
      await this.engine.runHidden(x, i);
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
    this.strategy = strategy;

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

  // Wait until `n` workers report ready. Resolves immediately if they already have --
  // a worker that kept its range replies almost instantly, and can beat us here.
  _waitForWorkers(n, timeoutMs = 120000) {
    if (n <= 0 || (this._readyCount || 0) >= n) return Promise.resolve();
    return new Promise((resolve, reject) => {
      this._pending = {
        need: n, resolve,
        timer: setTimeout(() => {
          this._pending = null;
          reject(new Error(`${n - (this._readyCount || 0)} device(s) never finished loading`));
        }, timeoutMs),
      };
    });
  }

  // Deal a plan and bring the room online. Used both for the first start and for
  // every recovery, so there is one code path that can be wrong rather than two.
  async _deal(p, { replay = false } = {}) {
    this.isHost = true;
    this.hostId = this.id;
    this._readyCount = 0;

    const workers = p.chain.slice(1);
    this.chain = workers;
    this._emit("plan", p.chain.map((id, i) => ({ id, range: p.ranges[i], self: id === this.id })));

    // Arm the wait before dealing: a worker that keeps its range can reply "ready"
    // before this function gets another turn on the event loop.
    const ready = this._waitForWorkers(workers.length);

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
      this.mesh.send(id, { t: "stand-by", why: p.why.find((w) => w.includes(this.peers.find((x) => x.id === id)?.name || " ")) || "not needed for this plan" });
    }

    const mine = p.ranges[0];
    const same = this.engine && this.range && this.range[0] === mine[0] && this.range[1] === mine[1];
    this.range = mine;
    this.next = workers[0] || null;
    if (same) this.engine.reset(); else await this._load(mine, true, true);

    if (workers.length) this._emit("waiting-for-workers", workers.length);
    await ready;

    if (replay) await this._replay();
    this.mesh.broadcast({ t: "ready-all" });
    this._emit("ready");
  }

  // Re-run the conversation so far, so a device holding a moved layer range ends up
  // with a real KV cache rather than an empty one.
  //
  // Deliberately does NOT broadcast a reset first. Every worker already cleared its
  // cache when it handled `deal`, and control messages travel on a different data
  // channel from activations -- so a reset sent here could arrive *after* the first
  // replayed frame and wipe the very state it was meant to prepare.
  //
  // Replay is idempotent: re-running position p with the same token writes the same
  // K and V a device already held, so devices that kept their range are unharmed.
  async _replay() {
    const hist = this.history.slice();
    if (!hist.length) return null;
    this.history = [];
    this.pos = 0;
    this._emit("replaying", { total: hist.length });
    const t0 = performance.now();

    let logits = null;
    for (let i = 0; i < hist.length; i++) {
      logits = await this._step(hist[i], this.pos);
      this.history.push(hist[i]);
      this.pos++;
      if (i % 4 === 3 || i === hist.length - 1) {
        this._emit("replay-progress", { done: i + 1, total: hist.length });
      }
    }
    this._emit("replayed", { total: hist.length, ms: Math.round(performance.now() - t0) });
    return logits;
  }

  // ---------------------------------------------------------------- recovery
  //
  // A device leaving mid-answer is normal behaviour, not an outage. Re-plan over who
  // is left, move the orphaned layers, rebuild the lost cache by replaying what has
  // been said, and carry on from the same position.
  //
  // The property that matters is not "it does not crash" -- it is that the answer is
  // unchanged. A room that survives a failure by quietly producing different text has
  // not recovered; it has started a different conversation without saying so.
  async _recover(why) {
    if (this.recovering) return false;
    this.recovering = true;
    const t0 = performance.now();
    this._emit("recovering", { why, tokens: this.history.length });

    try {
      const devices = this._devices();
      const p = solvePlan(this.spec, devices, this._rtt(), { strategy: this.strategy || "optimal" });

      if (!p) {
        this._emit("recover-failed", "the devices left cannot hold the model between them");
        return false;
      }
      if (p.host !== this.id) {
        // The planner would rather someone else hosted. Mid-answer we decline: the
        // conversation history lives here, and handing over would lose it. Note it
        // and re-plan properly at the end of the answer.
        this._emit("host-suboptimal", p.host);
      }
      this.lastPlan = p;
      this._emit("planned", p);

      await this._deal(p, { replay: true });

      this._emit("recovered", {
        ms: Math.round(performance.now() - t0),
        devices: p.chain.length,
        tokens: this.history.length,
      });
      return true;
    } catch (e) {
      this._emit("recover-failed", e.message);
      return false;
    } finally {
      this.recovering = false;
    }
  }

  // ---------------------------------------------------------------- generate
  async _step(tokenId, pos) {
    let x = await this.engine.embedRun(tokenId, pos);
    if (this.next) {
      // this.next is only repointed once a re-deal actually runs (inside
      // _recover()), so a lap that starts after a "left" event for this exact
      // peer -- but before recovery has had a chance to act -- would otherwise
      // send into the void and only fail via the 20 s timeout below. Failing it
      // here instead is what makes recovery start in milliseconds, not seconds,
      // for that case (see the "left" handler's comment on _deadPeers).
      if (this._deadPeers.has(this.next)) throw new Error(`${this.next} already left the chain`);
      const t0 = performance.now();
      x = await new Promise((resolve) => {
        this.waiting.set(pos, resolve);
        this.mesh.sendFrame(this.next, { t: "hidden", pos, data: packWire(x, this.wireEnc) });
        setTimeout(() => {
          if (this.waiting.has(pos)) { this.waiting.delete(pos); resolve(null); }
        }, 20000);
      });
      if (!x) throw new Error("the chain did not answer in time");
      this.stats.hops.push(performance.now() - t0);
    }
    return await this.engine.headFromHidden(x);
  }

  // Advance the conversation by one token, surviving a device that leaves while the
  // lap is in flight. History is appended only after the step succeeds, so a recovery
  // replays exactly what was actually processed -- no more, no less.
  async _feed(tokenId) {
    const pos = this.pos;
    const logits = await this._stepSafe(tokenId, pos);
    this.history.push(tokenId);
    this.pos = pos + 1;
    return logits;
  }

  async _stepSafe(tokenId, pos) {
    try {
      return await this._step(tokenId, pos);
    } catch (e) {
      if (this.recovering) throw e;               // already recovering; do not nest
      const healed = await this._recover(e.message);
      if (!healed) throw e;
      // _replay left this.pos exactly where it was, so the same position is retried
      // against the new chain.
      return await this._step(tokenId, pos);
    }
  }

  // ChatML, and only for the turn being added. `SYSTEM` is prepended once.
  _turnPrompt(text) {
    const first = this.turns.length === 0;
    const sys = first ? `<|im_start|>system
${SYSTEM}<|im_end|>
` : "";
    return `${sys}<|im_start|>user
${text}<|im_end|>
<|im_start|>assistant
`;
  }

  // How much of the context window is spoken for. The limit is real and small
  // (512 positions), so it is reported rather than discovered when the room stops
  // making sense.
  context() {
    const limit = (this.engine?.maxSeq || 512) - 4;
    return { used: this.pos, limit, frac: this.pos / limit, turns: this.turns.length };
  }

  // maxTokens defaults from the model descriptor, not a flat 60: a reasoning
  // model's <think> block can legitimately run long, and 60 was tuned for
  // SmolLM2's CPU speed, not for a model that reasons before it answers. An
  // explicit override still wins, for callers that want one.
  async generate(text, askedBy = null, { maxTokens = null } = {}) {
    if (this.busy) return;
    if (!this.isHost) { this.mesh.send(this.hostId, { t: "ask", text }); return; }
    const cap = maxTokens ?? getModel(this.model).maxTokensDefault ?? 60;

    // Refuse a question there is no room to answer, rather than starting one and
    // stopping mid-sentence. Sliding the window would mean re-prefilling the whole
    // conversation across the room; at this context size, saying so is honester.
    const need = this.tok.encode(this._turnPrompt(text)).length + 16;
    const ctx = this.context();
    if (ctx.used + need > ctx.limit) {
      this._emit("context-full", { ...ctx, need });
      return;
    }

    this.busy = true;
    this.stats = { tokens: 0, ms: 0, hops: [] };

    const by = askedBy ? (this.peers.find((p) => p.id === askedBy)?.name || "someone") : "you";
    this.mesh.broadcast({ t: "gen-start", text, by });
    this._emit("gen-start", text, by);

    const t0 = performance.now();
    try {
      // The model is instruction-tuned on ChatML, so it has to be spoken to in
      // ChatML. Fed raw text it does what a base model does -- continues the
      // sentence -- which is why an untemplated demo reads like autocomplete
      // rather than an assistant answering.
      //
      // Only THIS turn is encoded. Everything before it is already in the KV
      // caches spread across the room, so a follow-up question costs one short
      // prefill rather than replaying the conversation.
      const ids = this.tok.encode(this._turnPrompt(text));
      this.turns.push({ role: "user", content: text });

      let logits = null;
      for (const id of ids) logits = await this._feed(id);

      let answer = "";
      for (let n = 0; n < cap; n++) {
        const next = argmax(logits);
        if (next === this.eosId) break;                    // <|im_end|>
        const piece = this.tok.decode([next]);
        answer += piece;
        this.mesh.broadcast({ t: "token", text: piece });
        this._emit("token", piece);
        this.stats.tokens++;
        logits = await this._feed(next);
        if (this.pos >= this.engine.maxSeq - 4) { this._emit("truncated"); break; }
      }
      this.turns.push({ role: "assistant", content: answer });

      // Close the assistant turn in the cache. Without this the next question
      // would run straight on from the answer instead of starting a new turn,
      // and the model would keep writing the reply it had just finished.
      for (const id of this.tok.encode("<|im_end|>\n")) {
        if (this.pos >= this.engine.maxSeq - 2) break;
        await this._feed(id);
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
    s.context = this.context();
    this.mesh.broadcast({ t: "gen-done", stats: s });
    this._emit("gen-done", s);
    this.busy = false;
  }

  // Drop a device from the chain on purpose. This is the "kill a node" demo, and it
  // exercises exactly the same path an unplanned disconnect takes -- there is no
  // separate, gentler code path for the rehearsed version.
  dropWorker(id = null) {
    if (!this.isHost || !this.chain.length) return null;
    const victim = id || this.chain[this.chain.length - 1];
    const name = this.peers.find((p) => p.id === victim)?.name || victim;
    this._emit("dropping", victim, name);
    this.mesh.send(victim, { t: "please-leave" });
    return name;
  }

  resetConversation() {
    this.engine?.reset();
    this.pos = 0;
    this.history = [];
    this.turns = [];
    this.mesh.broadcast({ t: "reset" });
    this._emit("reset");
  }
}
