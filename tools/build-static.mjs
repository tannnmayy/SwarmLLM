// Build the static half of a deployment into dist/.
//
//   node tools/build-static.mjs                                  # CDN: weights from HF
//   node tools/build-static.mjs --with-models --models qwen3-0.6b # LAN/offline bundle
//   node tools/build-static.mjs --signal wss://swarm-signal.fly.dev/signal
//   node tools/build-static.mjs --out build --delivery auto
//
// What "static" means here is literal: dist/ is HTML, ES modules and one small JSON
// config, served by anything that serves files. There is no bundler, no transpile
// step and no server-side rendering, because there is nothing to bundle — the room
// runs as native modules in the browser, and the only large thing it needs (weights)
// is fetched by range from huggingface.co at run time rather than shipped.
//
// The file list is not hand-maintained. It is walked from the HTML entry points
// through the ES module graph, because a missed module is a deployment that 404s on
// a device that is not the one it was built on. Anything the graph reaches is copied;
// anything it does not is left behind, which is how tests/, tools/ and the multi-GB
// models/ directory stay out of the CDN build without a deny list to keep in sync.

import { mkdir, copyFile, readFile, writeFile, rm, readdir, stat } from "node:fs/promises";
import { dirname, join, relative, resolve, posix } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));

const argv = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = argv.indexOf("--" + name);
  return i >= 0 && argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[i + 1] : fallback;
};
const has = (name) => argv.includes("--" + name);

const OUT = resolve(ROOT, flag("out", "dist"));
const WITH_MODELS = has("with-models");
// Which models a bundled build carries. The whole ladder is 6.7 GB and a venue
// almost never wants all of it; naming one or two keeps a LAN bundle something you
// can actually put on a laptop and walk into a room with.
const ONLY_MODELS = flag("models", null)?.split(",").map((s) => s.trim()).filter(Boolean) || null;
const SIGNAL_URL = flag("signal", null);
// Where the page asks for ICE servers. Defaults to /ice on the same host as
// signalling, because that is where both the Cloudflare Worker and any sensible
// alternative put it — a deployment that has a signalling service has somewhere to
// mint TURN credentials, and one that does not has no TURN either.
const ICE_URL = flag("ice-url", null) ??
  (SIGNAL_URL ? SIGNAL_URL.replace(/^ws/, "http").replace(/\/signal$/, "/ice") : null);
// A CDN build has no /models/ directory, so probing for a mirror on every model load
// is a request that can only ever 404. Say so in the config and skip it. A bundled
// build is the reverse: the mirror is right there, and the internet may not be.
const DELIVERY = flag("delivery", WITH_MODELS ? "mirror" : "upstream");

// Pages a browser can be pointed at. Everything else arrives by being imported.
const ENTRY_HTML = [
  "index.html",
  "room.html",
  "probe.html",
  "gpu-test.html",
  "gpu-adapter-test.html",
  "mesh-test.html",
];
// Reached at run time but not through an import statement, so the graph walk cannot
// see them.
const EXTRA = [
  "favicon.svg",               // <link rel="icon">, not an import
  "README.md",                 // index.html links to it
  "THIRD_PARTY_NOTICES.md",    // vendored code is attributed in the deployed build too
  "LICENSE",
];

// ---------------------------------------------------------------- module graph
// Deliberately a scanner, not a parser. Every import in this codebase is a static
// relative specifier or a dynamic import of a string literal; anything else is
// rejected loudly below rather than silently missed.
const IMPORT_RE = /(?:^|[\s;{}(])(?:import|export)\s[^'"]*?from\s*["']([^"']+)["']|(?:^|[\s;{}(=])import\s*\(\s*["']([^"']+)["']\s*\)/g;

function specifiersIn(source) {
  const out = [];
  // Static vs dynamic matters for exactly one case: `await import("node:fs/promises")`
  // is the established dual-target idiom here (engine/cpu.mjs, tools/tokenizer.mjs
  // both read a file in Node and fetch a URL in the browser, behind a
  // `typeof window === "undefined"` guard). A browser never evaluates that branch,
  // so it is correct. The same specifier as a *static* import would be loaded
  // unconditionally at parse time and would break the page.
  for (const m of source.matchAll(IMPORT_RE)) {
    out.push({ spec: m[1] || m[2], dynamic: m[2] != null });
  }
  return out;
}

const seen = new Set();
const problems = [];

async function walk(relPath) {
  const rel = posix.normalize(relPath.split("\\").join("/"));
  if (seen.has(rel)) return;
  seen.add(rel);

  const abs = join(ROOT, rel);
  let source;
  try {
    source = await readFile(abs, "utf8");
  } catch {
    problems.push(`missing file referenced by the module graph: ${rel}`);
    return;
  }
  if (!/\.(m?js|html)$/.test(rel)) return;

  for (const { spec, dynamic } of specifiersIn(source)) {
    if (/^(https?:)?\/\//.test(spec)) continue;                    // absolute URL, fetched at run time
    if (!spec.startsWith(".") && !spec.startsWith("/")) {
      if (spec.startsWith("node:")) {
        if (!dynamic) problems.push(`${rel} statically imports "${spec}" — a Node builtin cannot load in the browser`);
      } else {
        problems.push(`${rel} imports the bare specifier "${spec}" — a static build has no resolver for it`);
      }
      continue;
    }
    const target = spec.startsWith("/")
      ? spec.slice(1)
      : posix.normalize(posix.join(posix.dirname(rel), spec));
    await walk(target);
  }
}

// ---------------------------------------------------------------- copy
async function copyInto(rel) {
  const from = join(ROOT, rel);
  const to = join(OUT, rel);
  await mkdir(dirname(to), { recursive: true });
  await copyFile(from, to);
  return (await stat(from)).size;
}

async function copyTree(rel, { include = null } = {}) {
  let bytes = 0, files = 0;
  const dirs = [];
  const walkDir = async (d, depth) => {
    for (const ent of await readdir(join(ROOT, d), { withFileTypes: true })) {
      const child = posix.join(d, ent.name);
      // At the top of models/, each directory is one model. A LAN bundle usually
      // wants one of them, not all 6.7 GB of the ladder.
      if (depth === 0 && ent.isDirectory() && include && !include.includes(ent.name)) continue;
      // tools/fetch-model.mjs keeps the raw download it sharded from, as
      // models/<id>/_source.safetensors. It is 269 MB, no engine reads it, and it is
      // the input to a build step rather than an output of one — shipping it would
      // more than double a SmolLM2 bundle for nothing.
      if (ent.isFile() && ent.name.startsWith("_")) continue;
      if (ent.isDirectory()) { if (depth === 0) dirs.push(ent.name); await walkDir(child, depth + 1); }
      else { bytes += await copyInto(child); files++; }
    }
  };
  await walkDir(rel, 0);
  return { bytes, files, dirs };
}

// ---------------------------------------------------------------- run
console.log(`\n  building ${WITH_MODELS ? "LAN/offline" : "CDN"} static site -> ${relative(ROOT, OUT) || OUT}\n`);

for (const html of ENTRY_HTML) await walk(html);
for (const extra of EXTRA) seen.add(extra);

if (problems.length) {
  console.error("  the module graph will not load in a browser:\n");
  for (const p of problems) console.error("    " + p);
  console.error("");
  process.exit(1);
}

await rm(OUT, { recursive: true, force: true });
await mkdir(OUT, { recursive: true });

let bytes = 0, files = 0;
for (const rel of [...seen].sort()) {
  try { bytes += await copyInto(rel); files++; }
  catch (e) { console.error(`  could not copy ${rel}: ${e.message}`); process.exit(1); }
}

let bundled = [];
if (WITH_MODELS) {
  const include = ONLY_MODELS;
  console.log(`  copying models/${include ? " (" + include.join(", ") + ")" : " — the whole ladder, this is the multi-GB part"}`);
  const m = await copyTree("models", { include });
  bytes += m.bytes; files += m.files;
  bundled = m.dirs;
  if (include) {
    const missing = include.filter((id) => !bundled.includes(id));
    if (missing.length) {
      console.error(`\n  --models named ${missing.join(", ")}, which is not in models/ — nothing to bundle.\n`);
      process.exit(1);
    }
  }
}

// ---------------------------------------------------------------- config
// The one file that makes a build a deployment. Everything else in dist/ is
// identical between a CDN build and a LAN bundle; this says where signalling lives
// and where weights come from. Editing it re-points a built site with no rebuild.
const config = {
  _comment: "AI Swarm deployment config. See room/config.js for how this is read.",
  delivery: DELIVERY,
  // Exactly what this build put under /models/. The registry declares a mirror path
  // for every model in the ladder, so without this a bundle carrying one model would
  // still offer the other three and 404 on whichever one somebody picked.
  ...(WITH_MODELS ? { models: bundled } : { models: [] }),
  ...(SIGNAL_URL ? { signalUrl: SIGNAL_URL } : {}),
  ...(ICE_URL ? { iceServersUrl: ICE_URL } : {}),
  builtAt: new Date().toISOString(),
};
await writeFile(join(OUT, "swarm-config.json"), JSON.stringify(config, null, 2) + "\n");

const MB = (b) => (b / 2 ** 20).toFixed(1) + " MB";
console.log(`  ${files} files, ${MB(bytes)}`);
console.log(`  delivery: ${DELIVERY}`);
console.log(`  signal:   ${SIGNAL_URL || "(same origin — set --signal for a CDN deployment)"}`);
console.log(`  ice:      ${ICE_URL || "(none — public STUN only; two devices on different networks may not connect)"}`);
if (!SIGNAL_URL && !WITH_MODELS) {
  console.log("\n  note: no --signal given. A CDN build has no websocket on its own origin,");
  console.log("        so joining will fail until swarm-config.json names one.");
}
console.log("");
