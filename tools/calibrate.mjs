// Does the probe predict what the engine actually does?
//
//   node tools/calibrate.mjs
//
// The planner's entire output is a function of ms-per-layer. If the probe is off by
// 4x, every drop/keep decision is made against a fiction. So this measures both and
// prints the ratio, and it is a gate: a probe that cannot predict the engine within a
// reasonable factor is not a probe, it is a random number.

import { CpuEngine } from "../engine/cpu.mjs";
import { modelSpec } from "../scheduler/plan.js";
import { measureMacsPerSec } from "../scheduler/probe.js";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIR = join(ROOT, "models/smollm2-135m");
const manifest = JSON.parse(await readFile(join(DIR, "manifest.json"), "utf8"));
const spec = modelSpec(manifest, { precision: "f32", maxSeq: 512 });

console.log(`\n  model MACs per layer: ${(spec.layerMACs / 1e6).toFixed(2)}M`);

// ---- what the probe thinks
const p = await measureMacsPerSec({ budgetMs: 400 });
const probeMsPerLayer = (spec.layerMACs / p.macsPerSec) * 1000;
console.log(`\n  probe:  ${(p.macsPerSec / 1e9).toFixed(3)} GMAC/s -> ${probeMsPerLayer.toFixed(2)} ms/layer  (spread ${p.spread.toFixed(2)})`);

// ---- what the engine actually does
const N = 8;
const eng = await CpuEngine.load(DIR, { layerRange: [0, N] });
const x = new Float32Array(spec.hidden);
for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.31) * 0.7;

for (let i = 0; i < 3; i++) eng.runHidden(x, i);      // warm up
const runs = [];
for (let i = 0; i < 12; i++) {
  const t0 = performance.now();
  eng.runHidden(x, 3 + i);
  runs.push((performance.now() - t0) / N);
}
runs.sort((a, b) => a - b);
const actual = runs[runs.length >> 1];
const engineMacsPerSec = spec.layerMACs / (actual / 1000);

console.log(`  engine: ${(engineMacsPerSec / 1e9).toFixed(3)} GMAC/s -> ${actual.toFixed(2)} ms/layer  (median of 12)`);

const ratio = probeMsPerLayer / actual;
console.log(`\n  probe / engine = ${ratio.toFixed(2)}x`);
if (ratio > 1.35) console.log("  -> probe is PESSIMISTIC: it will over-value splitting and recruit devices that do not help.");
else if (ratio < 0.74) console.log("  -> probe is OPTIMISTIC: it will under-value splitting and keep devices that hurt.");
else console.log("  -> within tolerance.");

console.log(`\n  suggested calibration constant: ${(1 / ratio).toFixed(3)}\n`);
process.exit(ratio > 1.35 || ratio < 0.74 ? 1 : 0);
