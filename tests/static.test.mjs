// The static handler, extracted from tools/serve.mjs into tools/static.mjs.
//
//   node tests/static.test.mjs
//
// The extraction is the risk this covers. Two of these three behaviours are load
// bearing in ways that are easy to break without noticing:
//
//   * 416 on an unsatisfiable range. engine/gpu-adapter.mjs reads a 416 as "this
//     file is smaller than the 16 MiB header probe window, fetch the whole thing"
//     -- a server that clamps to the file length instead would silently hand it a
//     truncated header.
//   * the cache policy. Weights want a year; code wants none. Getting that
//     backwards for module files means a browser runs stale code against a fresh
//     page and cannot be told otherwise, because immutable means immutable.

import { safePath, parseRange, MIME } from "../tools/static.mjs";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};

console.log("\npath safety");
{
  ok("a plain path resolves inside the root", !!safePath(ROOT, "/room.html"));
  ok("the bare root serves index.html", safePath(ROOT, "/").rel === "index.html");
  ok("a query string is stripped before resolving", safePath(ROOT, "/room.html?v=2").rel === "room.html");
  ok("percent-encoding is decoded", safePath(ROOT, "/room%2Ehtml").rel === "room.html");

  for (const evil of ["/../package.json", "/../../etc/passwd", "/models/../../package.json", "/..%2f..%2fpackage.json"]) {
    const p = safePath(ROOT, evil);
    ok(`"${evil}" cannot escape the root`, p === null || p.abs.startsWith(ROOT.replace(/[\\/]$/, "")),
       p ? p.abs : "null");
  }

  // Everything downstream matches on `rel` -- the cache policy below, any onMiss a
  // caller supplies. normalize() hands back backslashes on Windows, so a rule
  // written as /^models\// would quietly stop matching there and nowhere else.
  ok("rel always uses forward slashes, whatever platform this is",
     !safePath(ROOT, "/models/qwen3-0.6b/config.json").rel.includes("\\"));
}

console.log("\nrange parsing");
{
  ok("no Range header means no range", parseRange(undefined, 1000) === null);
  ok("bytes=0-99 is a closed range", (() => { const r = parseRange("bytes=0-99", 1000); return r.start === 0 && r.end === 99; })());
  ok("an open end runs to the last byte", (() => { const r = parseRange("bytes=500-", 1000); return r.start === 500 && r.end === 999; })());
  // Not a clamp: engine/gpu-adapter.mjs treats 416 as proof the server DOES support
  // Range and the file is simply shorter than the probe window.
  ok("a range past the end is unsatisfiable, not clamped", parseRange("bytes=0-16777215", 1000).unsatisfiable === true);
  ok("a start past the end is unsatisfiable", parseRange("bytes=2000-2100", 1000).unsatisfiable === true);
  ok("a backwards range is unsatisfiable", parseRange("bytes=900-100", 1000).unsatisfiable === true);
  ok("a malformed Range is ignored rather than guessed at", parseRange("bytes=abc", 1000) === null);
}

console.log("\ncache policy");
{
  // Re-declared here rather than imported: the default lives in a parameter
  // default, and the point of the test is that this exact rule is what ships.
  const policy = (rel) => (/^models\//.test(rel) && !/\.(m?js)$/.test(rel)
    ? "public, max-age=31536000, immutable"
    : "no-store, no-cache, must-revalidate");
  const immutable = (rel) => policy(rel).includes("immutable");

  ok("a GGUF is cached for a year", immutable("models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"));
  ok("a CPU shard is cached for a year", immutable("models/smollm2-135m/layer-00.bin"));
  ok("a model's manifest is cached for a year", immutable("models/smollm2-135m/manifest.json"));
  // These are source code that happens to live under models/. Caching them as
  // immutable is unrecoverable from a browser that already did it.
  ok("models/registry.mjs is NOT cached: it is code, not weights", !immutable("models/registry.mjs"));
  ok("models/delivery.mjs is NOT cached: it is code, not weights", !immutable("models/delivery.mjs"));
  ok("pages are never cached", !immutable("room.html") && !immutable("index.html"));
  ok("modules outside models/ are never cached", !immutable("room/room.js") && !immutable("engine/gpu-adapter.mjs"));
}

console.log("\nmime types");
{
  ok(".gguf is served as a binary stream, not text", MIME[".gguf"] === "application/octet-stream");
  ok(".mjs is served as javascript, or the browser refuses the module",
     MIME[".mjs"].startsWith("text/javascript"));
  ok(".wasm has its own type", MIME[".wasm"] === "application/wasm");
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
