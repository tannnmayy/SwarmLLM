// When is a swarm actually worth it?
//
//   node tools/scenario.mjs            all scenarios
//   node tools/scenario.mjs pledged    just one
//
// This is the benchmark table from the pitch, run offline against measured-shaped
// numbers so the demo configuration can be chosen deliberately rather than discovered
// on stage. It answers one question honestly: a swarm costs latency, so what has to
// be true before it pays for itself?

import { modelSpec, compareAll, plan, layerCap, canHost } from "../scheduler/plan.js";
import { rttLookup } from "../scheduler/cost.js";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const GB = 2 ** 30, MB = 2 ** 20;

// Real manifest for the model we actually ship.
const smol = modelSpec(
  JSON.parse(await readFile(join(ROOT, "models/smollm2-135m/manifest.json"), "utf8")),
  { precision: "f32", maxSeq: 512 }
);

// Synthetic specs for models we have not downloaded, so placement can be reasoned
// about before committing an hour to a download. Configs are the published ones.
const synth = (label, c) => modelSpec({ label, config: { ...c, tiedEmbeddings: true } },
  { precision: "f32", maxSeq: 512 });

const qwen06 = synth("Qwen3 0.6B", {
  hiddenSize: 1024, layers: 28, heads: 16, kvHeads: 8, headDim: 128,
  intermediate: 3072, vocab: 151936, rmsEps: 1e-6, ropeTheta: 1e6,
});
const qwen17 = synth("Qwen3 1.7B", {
  hiddenSize: 2048, layers: 28, heads: 16, kvHeads: 8, headDim: 128,
  intermediate: 6144, vocab: 151936, rmsEps: 1e-6, ropeTheta: 1e6,
});

const dev = (id, name, msPerLayer, budgetGB) => ({ id, name, msPerLayer, budgetBytes: budgetGB * GB });
const flat = (ms) => rttLookup({}, { fallback: ms });

// The devices actually in the room, with ms/layer scaled from the measured two-device
// run (30 layers + an 8-layer head in ~200 ms on the main PC). Phones are estimated
// until probe.html reports real numbers; the shape is what matters here.
const REAL = (budgets = {}) => [
  dev("pc1", "Tanmay PC", 5.0, budgets.pc1 ?? 6),
  dev("pc2", "PC 2", 6.5, budgets.pc2 ?? 6),
  dev("and1", "Android 1", 20, budgets.and1 ?? 2),
  dev("and2", "Android 2", 24, budgets.and2 ?? 2),
  dev("ios", "iPhone 14+", 12, budgets.ios ?? 2.5),
];

function table(spec, devices, rtt, note) {
  const rows = compareAll(spec, devices, rtt);
  const fits = devices.filter((d) => layerCap(spec, d, true) >= spec.layers);
  const modelMB = (spec.layers * spec.layerBytes + spec.embedBytes) / MB;

  console.log(`\n  model: ${spec.label} — ${modelMB.toFixed(0)} MB in memory, ${spec.layers} layers`);
  console.log(`  devices that could run it alone: ${fits.length ? fits.map((d) => d.name).join(", ") : "NONE — the swarm is mandatory"}`);
  if (note) console.log(`  ${note}`);
  console.log("\n    strategy   devices   tok/s    per token   host");
  for (const r of rows) {
    if (!r.feasible) { console.log(`    ${r.strategy.padEnd(9)}   ${"—".padStart(5)}   infeasible (does not fit)`); continue; }
    const mark = r.strategy === "optimal" ? " <-" : "";
    console.log(`    ${r.strategy.padEnd(9)}  ${String(r.devices).padStart(5)}  ${r.tokensPerSec.toFixed(2).padStart(6)}  ${r.totalMs.toFixed(0).padStart(8)} ms   ${r.host}${mark}`);
  }
  const opt = rows.find((r) => r.strategy === "optimal");
  const mem = rows.find((r) => r.strategy === "memory");
  if (opt?.feasible && mem?.feasible) {
    console.log(`\n    optimal vs exo-style memory-weighted: ${(mem.totalMs / opt.totalMs).toFixed(2)}x`);
  } else if (opt?.feasible && !mem?.feasible) {
    console.log(`\n    optimal finds a plan where memory-weighted has none.`);
  }
  if (opt?.plan) for (const w of opt.plan.why) console.log("    · " + w);
  return rows;
}

const SCENARIOS = {
  // ---------------------------------------------------------------------------
  "small": () => {
    console.log("\n=== 1. Small model, generous devices ================================");
    console.log("  The null result, stated up front: when the model fits comfortably on");
    console.log("  one device, every split is a loss and the planner should say so.");
    table(smol, REAL(), flat(9));
  },

  // ---------------------------------------------------------------------------
  "pledged": () => {
    console.log("\n=== 2. Same model, but people pledge a slice of their device ========");
    console.log("  Nobody hands a demo their whole laptop. A pledge is what a real user");
    console.log("  gives, and it is what makes a swarm necessary rather than optional.");
    const budgets = { pc1: 0.32, pc2: 0.30, and1: 0.25, and2: 0.25, ios: 0.28 };
    table(smol, REAL(budgets), flat(9),
      "each device pledges 250-320 MB — the model needs 513 MB");
  },

  // ---------------------------------------------------------------------------
  "bigger": () => {
    console.log("\n=== 3. A model that genuinely does not fit ==========================");
    console.log("  This is the demo the deck actually describes: no single device can");
    console.log("  hold the model, so the swarm is the only way to run it at all.");
    table(qwen06, REAL(), flat(9));
    table(qwen17, REAL(), flat(9));
  },

  // ---------------------------------------------------------------------------
  "slow": () => {
    console.log("\n=== 4. One deliberately slow device =================================");
    console.log("  The benchmark from slide 7: same devices, same model, one of them");
    console.log("  throttled. Memory-weighted cannot see it; we measure it.");
    const room = REAL({ pc1: 0.32, pc2: 0.30, and1: 0.25, and2: 0.25, ios: 0.28 });
    room[3].msPerLayer = 95;           // Android 2 on battery saver / thermally throttled
    room[3].name = "Android 2 (throttled)";
    table(smol, room, flat(9), "Android 2 throttled to 95 ms/layer");
  },

  // ---------------------------------------------------------------------------
  "twosite": () => {
    console.log("\n=== 5. One device on mobile data ====================================");
    console.log("  Chain order is worth nothing on a flat LAN and a great deal the");
    console.log("  moment one link is slow. Put a phone on cellular and watch.");
    const room = REAL({ pc1: 0.32, pc2: 0.30, and1: 0.25, and2: 0.25, ios: 0.28 });
    const far = "and2";
    const rtt = (a, b) => (a === b ? 0 : (a === far || b === far) ? 130 : 8);
    table(smol, room, rtt, "Android 2 on mobile data: 130 ms RTT to everyone else");
  },

  // ---------------------------------------------------------------------------
  "crossover": () => {
    console.log("\n=== 6. Where is the crossover? ======================================");
    console.log("  Sweep the pledge each device makes. Below the line the model does not");
    console.log("  fit on one device and the swarm is mandatory; above it, splitting is");
    console.log("  a pure loss. The scheduler's job is to sit on the right side of it.");
    console.log("\n    pledge/device   fits alone?   best plan            tok/s");
    for (const gb of [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]) {
      const room = REAL({ pc1: gb, pc2: gb, and1: gb, and2: gb, ios: gb });
      const alone = room.some((d) => layerCap(smol, d, true) >= smol.layers);
      const p = plan(smol, room, flat(9), { strategy: "optimal" });
      const desc = !p ? "infeasible" : p.chain.length === 1 ? "solo" : `${p.chain.length} devices`;
      console.log(`    ${(gb * 1024).toFixed(0).padStart(9)} MB   ${(alone ? "yes" : "no").padStart(9)}   ${desc.padEnd(18)}  ${p ? p.predicted.tokensPerSec.toFixed(2) : "—"}`);
    }
    console.log("\n  The model needs 513 MB in memory. Under ~530 MB per device the swarm");
    console.log("  is the only way to run it; over that, one device wins outright.");
  },
};

const want = process.argv[2];
console.log("AI Swarm — placement scenarios");
for (const [name, run] of Object.entries(SCENARIOS)) {
  if (want && want !== name) continue;
  run();
}
console.log();
