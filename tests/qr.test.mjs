// Does our QR encoder produce codes a real scanner can read?
//
//   node tests/qr.test.mjs
//
// Structural checks catch some mistakes, but "the finder patterns are in the right
// place" is not the same as "a phone can read this", and the only way to know the
// difference is to decode it back. This writes each generated matrix to a file that
// tools/qr-verify.py decodes with OpenCV's detector; run `npm run test:qr` for both
// halves together.

import { encode, svg } from "../room/qr.js";
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const OUT = join(ROOT, "tests", ".qr");
mkdirSync(OUT, { recursive: true });

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "  " + extra : "")); }
};

const CASES = [
  ["short code", "SWRM"],
  ["lan join url", "https://192.168.1.169:8443/room.html#r=HMH6"],
  ["localhost url", "http://localhost:8442/room.html#r=ABCD"],
  ["long domain", "https://ai-swarm-demo.example.org/room.html#r=X7K2&name=Tanmay%20PC"],
  ["punctuation", "https://10.0.0.42:8443/room.html#r=9Q4Z"],
];

console.log("\nstructure");
const manifest = [];
for (const [name, text] of CASES) {
  const { modules, size, version } = encode(text);
  const dim = 17 + version * 4;
  ok(`${name}: version ${version}, ${size}x${size}`, size === dim && modules.length === size);

  // finder patterns: a 7x7 ring with a 3x3 core, in three corners
  const finder = (br, bc) =>
    modules[br + 0][bc + 0] === 1 && modules[br + 3][bc + 3] === 1 &&
    modules[br + 1][bc + 1] === 0 && modules[br + 6][bc + 6] === 1;
  ok(`${name}: three finder patterns`, finder(0, 0) && finder(0, size - 7) && finder(size - 7, 0));

  // timing patterns alternate
  let timing = true;
  for (let i = 8; i < size - 8; i++) {
    if (modules[6][i] !== (i % 2 ? 0 : 1)) timing = false;
    if (modules[i][6] !== (i % 2 ? 0 : 1)) timing = false;
  }
  ok(`${name}: timing patterns alternate`, timing);
  ok(`${name}: dark module present`, modules[size - 8][8] === 1);

  // PBM is the simplest format OpenCV reads: a text bitmap, 1 = black
  const scale = 8, margin = 4, W = (size + margin * 2) * scale;
  const rows = [];
  for (let y = 0; y < W; y++) {
    const r = Math.floor(y / scale) - margin;
    let line = "";
    for (let x = 0; x < W; x++) {
      const c = Math.floor(x / scale) - margin;
      const on = r >= 0 && c >= 0 && r < size && c < size && modules[r][c];
      line += on ? "1 " : "0 ";
    }
    rows.push(line.trim());
  }
  const file = name.replace(/[^a-z0-9]+/gi, "_") + ".pbm";
  writeFileSync(join(OUT, file), `P1\n${W} ${W}\n${rows.join("\n")}\n`);
  manifest.push({ file, text, version });
}

writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));

console.log("\nsvg output");
{
  const s = svg("https://192.168.1.169:8443/room.html#r=TEST");
  ok("renders an svg", s.startsWith("<svg") && s.endsWith("</svg>"));
  ok("has a light background and a dark path", s.includes("<rect") && s.includes("<path"));
  ok("no external references", !/https?:\/\/(?!www\.w3\.org)/.test(s.replace(/#r=[^"]*/g, "")));
}

console.log("\nlimits");
{
  let threw = false;
  try { encode("x".repeat(300)); } catch { threw = true; }
  ok("refuses data too long for version 10, rather than emitting a broken code", threw);
  ok("a 216-byte payload still encodes", (() => {
    try { return !!encode("y".repeat(210)); } catch { return false; }
  })());
}

console.log(`\n${pass} passed, ${fail} failed`);
console.log(`wrote ${manifest.length} bitmaps to tests/.qr — decode them with tools/qr-verify.py\n`);
process.exit(fail ? 1 : 0);
