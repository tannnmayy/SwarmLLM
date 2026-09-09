// The H22-30 gate: a device leaves mid-answer and the answer still finishes.
//
//   node tests/recovery.test.mjs
//
// Proved here in software before the network is involved, because the property that
// matters is not "it does not crash" — it is that the answer is UNCHANGED. A room
// that survives a failure by quietly producing different text has not recovered, it
// has started a different conversation and hidden it.
//
// The recovery sequence, and why each step is there:
//
//   1. re-plan over the survivors            the room is now a different room
//   2. re-deal the orphaned layers           somebody has to hold 15-29 now
//   3. reset every cache and replay history  the new holder's KV cache is empty, and
//                                            it can only be filled by running the
//                                            real tokens through in order
//   4. resume                                from the position we stopped at
//
// Step 3 is the expensive one and the reason a real room shows "reconnecting" rather
// than resuming instantly. Replaying is idempotent: re-running position p with the
// same token writes the same K and V a device already had, so devices that kept their
// range are unharmed by it.

import { CpuEngine, argmax } from "../engine/cpu.mjs";
import { Tokenizer } from "../tools/tokenizer.mjs";
import { chooseEncoding, packWire, unpackWire } from "../room/wire.js";
import { modelSpec, plan as solvePlan } from "../scheduler/plan.js";
import { rttLookup } from "../scheduler/cost.js";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIR = join(ROOT, "models", process.env.MODEL || "smollm2-135m");

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};

const manifest = JSON.parse(await readFile(join(DIR, "manifest.json"), "utf8"));
const spec = modelSpec(manifest, { precision: "f32", maxSeq: 512 });
const tok = await Tokenizer.load(join(DIR, "tokenizer.json"));
const enc = chooseEncoding(spec.hidden);
const overWire = (x) => unpackWire(packWire(x, enc), enc);

const PROMPT = "Once upon a time";
const ids = tok.encode(PROMPT);
const TOTAL = 12;          // tokens to generate
const KILL_AT = 5;         // how many are out before a device disappears

// ---------------------------------------------------------------- a swarm
// A minimal stand-in for the room: engines, a chain, and the four calls. Everything
// the real room does over WebRTC, done in-process so the algorithm can be tested
// without a browser.
class Swarm {
  static async deal(ranges) {
    const s = new Swarm();
    s.slices = [];
    s.loaded = 0;
    for (let i = 0; i < ranges.length; i++) {
      s.slices.push(await CpuEngine.load(DIR, {
        layerRange: ranges[i], hasEmbed: i === 0, hasHead: i === 0,
      }));
      s.loaded += ranges[i][1] - ranges[i][0];
    }
    s.pos = 0;
    s.history = [];
    return s;
  }

  step(tokenId, pos) {
    let x = this.slices[0].embedRun(tokenId, pos);
    for (let i = 1; i < this.slices.length; i++) x = this.slices[i].runHidden(overWire(x), pos);
    return this.slices[0].headFromHidden(overWire(x));
  }

  feed(tokenId) {
    const logits = this.step(tokenId, this.pos);
    this.history.push(tokenId);
    this.pos++;
    return logits;
  }

  // Reset every cache and re-run the conversation so far. The new holder of a moved
  // range has an empty KV cache, and this is the only thing that fills it.
  replay() {
    for (const e of this.slices) e.reset();
    const hist = this.history.slice();
    this.history = [];
    this.pos = 0;
    let logits = null;
    for (const id of hist) logits = this.feed(id);
    return logits;
  }
}

// ---------------------------------------------------------------- reference
console.log("\nuninterrupted reference (3 devices, nothing fails)");
const refRanges = [[0, 10], [10, 20], [20, 30]];
const ref = await Swarm.deal(refRanges);
let logits = null;
for (const id of ids) logits = ref.feed(id);
const refOut = [];
for (let n = 0; n < TOTAL; n++) {
  const next = argmax(logits);
  refOut.push(next);
  logits = ref.feed(next);
}
console.log(`  -> ${JSON.stringify(PROMPT + tok.decode(refOut))}`);

// ---------------------------------------------------------------- recovery
async function runWithFailure(killIndex, label) {
  console.log(`\n${label}`);
  const devices = [
    { id: "a", name: "A", msPerLayer: 4, budgetBytes: 3 * 2 ** 30 },
    { id: "b", name: "B", msPerLayer: 4, budgetBytes: 3 * 2 ** 30 },
    { id: "c", name: "C", msPerLayer: 4, budgetBytes: 3 * 2 ** 30 },
  ];
  const swarm = await Swarm.deal(refRanges);

  let lg = null;
  for (const id of ids) lg = swarm.feed(id);
  const out = [];
  for (let n = 0; n < KILL_AT; n++) {
    const next = argmax(lg);
    out.push(next);
    lg = swarm.feed(next);
  }

  // ---- a device walks away, mid-answer
  const lost = devices[killIndex];
  const survivors = devices.filter((d) => d.id !== lost.id);
  const survivorSlices = swarm.slices.filter((_, i) => i !== killIndex);
  const heldBefore = swarm.slices[killIndex].layerCount;

  // 1. re-plan over what is left
  const p = solvePlan(spec, survivors, rttLookup({}, { fallback: 8 }), { strategy: "optimal" });
  ok(`${label}: a plan exists over the survivors`, !!p);
  if (!p) return;
  ok(`${label}: the plan still covers all ${spec.layers} layers`,
     p.counts.reduce((a, b) => a + b, 0) === spec.layers, JSON.stringify(p.ranges));

  // 2. re-deal. A device whose range is unchanged keeps its weights; only the
  //    orphaned range is actually downloaded again.
  const newRanges = p.chain.map((id) => p.ranges[p.chain.indexOf(id)]);
  const oldRangeOf = new Map(survivors.map((d, i) => [d.id, survivorSlices[i].hi !== undefined ? [survivorSlices[i].lo, survivorSlices[i].hi] : null]));
  let reloaded = 0, kept = 0;
  const rebuilt = [];
  for (let i = 0; i < p.chain.length; i++) {
    const id = p.chain[i], want = newRanges[i];
    const had = oldRangeOf.get(id);
    const idx = survivors.findIndex((d) => d.id === id);
    if (had && had[0] === want[0] && had[1] === want[1]) {
      kept++;
      rebuilt.push(survivorSlices[idx]);
    } else {
      reloaded += want[1] - want[0];
      rebuilt.push(await CpuEngine.load(DIR, {
        layerRange: want, hasEmbed: i === 0, hasHead: i === 0,
      }));
    }
  }
  ok(`${label}: the orphaned ${heldBefore} layers are re-dealt`, reloaded > 0,
     `${reloaded} layers reloaded, ${kept} device(s) kept their range`);

  swarm.slices = rebuilt;

  // 3. replay everything said so far, so the new holder's cache is real
  const before = swarm.history.length;
  lg = swarm.replay();
  ok(`${label}: history replayed in full`, swarm.history.length === before && swarm.pos === before,
     `${swarm.history.length} of ${before}`);

  // 4. resume
  for (let n = KILL_AT; n < TOTAL; n++) {
    const next = argmax(lg);
    out.push(next);
    lg = swarm.feed(next);
  }

  console.log(`  -> ${JSON.stringify(PROMPT + tok.decode(out))}`);
  const same = out.length === refOut.length && out.every((v, i) => v === refOut[i]);
  ok(`${label}: the answer is IDENTICAL to the uninterrupted run`, same,
     same ? "" : `got ${JSON.stringify(tok.decode(out))}\n         want ${JSON.stringify(tok.decode(refOut))}`);
  return same;
}

await runWithFailure(1, "the middle device leaves");
await runWithFailure(2, "the last device leaves");

// ---------------------------------------------------------------- limits
console.log("\nwhat recovery cannot do");
{
  // A room that no longer has the memory to hold the model cannot be recovered by
  // re-planning. It has to say so rather than deal an impossible plan.
  const tiny = [
    { id: "a", name: "A", msPerLayer: 4, budgetBytes: 0.2 * 2 ** 30 },
    { id: "b", name: "B", msPerLayer: 4, budgetBytes: 0.2 * 2 ** 30 },
  ];
  const p = solvePlan(spec, tiny, rttLookup({}, { fallback: 8 }), { strategy: "optimal" });
  ok("a room too small after the loss reports no plan, rather than dealing a bad one", p === null);

  const one = [{ id: "a", name: "A", msPerLayer: 4, budgetBytes: 4 * 2 ** 30 }];
  const p2 = solvePlan(spec, one, rttLookup({}, { fallback: 8 }), { strategy: "optimal" });
  ok("a room down to one capable device falls back to running solo",
     !!p2 && p2.chain.length === 1);
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
