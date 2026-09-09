// The H6-14 gate, proved in software before the network is involved:
// a model split across N slices must produce exactly what one whole model produces.
//
//   node tests/split.test.mjs
//
// If this ever fails, no amount of transport or scheduling work matters, so it runs
// before them. "Exactly" means bit-identical logits, not close: every slice boundary
// is just a place where a hidden state is handed along, and handing it along cannot
// change arithmetic. Any drift here is a real bug in the layer loop.

import { CpuEngine, argmax } from "../engine/cpu.mjs";
import { Tokenizer } from "../tools/tokenizer.mjs";
import { chooseEncoding, packWire, unpackWire } from "../room/wire.js";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIR = join(ROOT, "models", process.env.MODEL || "smollm2-135m");

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "  " + extra : "")); }
};

const tok = await Tokenizer.load(join(DIR, "tokenizer.json"));
const prompt = "Once upon a time";
const ids = tok.encode(prompt);
const NEW = 8;

// ---------------------------------------------------------------- whole
console.log("\nwhole model (1 device)");
const whole = await CpuEngine.load(DIR, { layerRange: [0, 30], hasEmbed: true, hasHead: true });
console.log(`  loaded ${(whole.bytesLoaded / 2 ** 20).toFixed(1)} MB, layers 0-29`);

async function runWhole() {
  whole.reset();
  let pos = 0, logits = null;
  for (const id of ids) logits = await whole.headFromHidden(await whole.embedRun(id, pos++));
  const out = [];
  for (let n = 0; n < NEW; n++) {
    const next = argmax(logits);
    out.push(next);
    logits = await whole.headFromHidden(await whole.embedRun(next, pos++));
  }
  return { out, logits };
}
const ref = await runWhole();
console.log(`  -> ${JSON.stringify(prompt + tok.decode(ref.out))}`);

// ---------------------------------------------------------------- split
// The host holds the first range plus the embedding table and the LM head; every
// other slice is a plain worker. This mirrors the real room exactly.
async function runSplit(cuts) {
  const ranges = [];
  for (let i = 0; i < cuts.length - 1; i++) ranges.push([cuts[i], cuts[i + 1]]);
  const slices = [];
  for (let i = 0; i < ranges.length; i++) {
    slices.push(await CpuEngine.load(DIR, {
      layerRange: ranges[i],
      hasEmbed: i === 0,
      hasHead: i === 0,
    }));
  }

  // Send every hidden state through the real wire codec between slices, exactly as
  // the network would. Without this the test proves something weaker than it looks:
  // f16 on the wire perturbs the activation, so the answer starts depending on where
  // the model happened to be cut, and a room that re-plans mid-conversation would
  // change its own output. chooseEncoding is what keeps that from happening.
  const enc = chooseEncoding(whole.cfg.hiddenSize);
  const overWire = (x) => unpackWire(packWire(x, enc), enc);

  const step = async (tokenId, pos) => {
    // host embeds and runs its own range, then the hidden state walks the chain
    let x = await slices[0].embedRun(tokenId, pos);
    for (let i = 1; i < slices.length; i++) x = await slices[i].runHidden(overWire(x), pos);
    return await slices[0].headFromHidden(overWire(x));   // and comes back to the host
  };

  let pos = 0, logits = null;
  for (const id of ids) logits = await step(id, pos++);
  const out = [];
  for (let n = 0; n < NEW; n++) {
    const next = argmax(logits);
    out.push(next);
    logits = await step(next, pos++);
  }
  return { out, logits, ranges, slices };
}

console.log(`
wire: dim ${whole.cfg.hiddenSize} -> ${chooseEncoding(whole.cfg.hiddenSize)} (hidden states pass through the real codec below)`);

const layouts = [
  { name: "2 devices, even", cuts: [0, 15, 30] },
  { name: "3 devices, uneven", cuts: [0, 4, 21, 30] },
  { name: "5 devices, one holds a single layer", cuts: [0, 1, 9, 17, 24, 30] },
];

for (const layout of layouts) {
  console.log(`\n${layout.name}`);
  const got = await runSplit(layout.cuts);
  const shape = got.ranges.map((r, i) =>
    `${r[1] - r[0]}L/${(got.slices[i].bytesLoaded / 2 ** 20).toFixed(0)}MB`).join("  ");
  console.log(`  ${shape}`);

  const sameTokens = got.out.length === ref.out.length && got.out.every((v, i) => v === ref.out[i]);
  ok("token stream identical to the whole model", sameTokens,
     sameTokens ? "" : `got ${JSON.stringify(tok.decode(got.out))} want ${JSON.stringify(tok.decode(ref.out))}`);

  let worst = 0;
  for (let i = 0; i < ref.logits.length; i++) worst = Math.max(worst, Math.abs(got.logits[i] - ref.logits[i]));
  ok("final logits bit-identical", worst === 0, worst ? "worst delta " + worst.toExponential(3) : "");

  // a worker must never need the embedding table or the LM head
  const workersClean = got.slices.slice(1).every((s) => !s.embed && !s.lmHead);
  ok("workers hold no embedding table and no LM head", workersClean);
}

// ---------------------------------------------------------------- download claim
console.log("\ndownload footprint");
{
  const eight = await CpuEngine.load(DIR, { layerRange: [10, 18] });
  const mb = eight.bytesLoaded / 2 ** 20;
  ok(`a worker dealt 8 layers loads only those (${mb.toFixed(1)} MB)`, mb > 50 && mb < 60);
  ok("that is far less than the whole model", mb < whole.bytesLoaded / 2 ** 20 / 3,
     `${mb.toFixed(1)} MB vs ${(whole.bytesLoaded / 2 ** 20).toFixed(1)} MB whole`);
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
