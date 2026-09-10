// Can a browser actually get these weights from huggingface.co, and are they the
// same bytes the local mirror has?
//
//   node tools/verify-delivery.mjs                  # every model with an upstream
//   node tools/verify-delivery.mjs qwen3-0.6b
//   node tools/verify-delivery.mjs --json           # machine-readable, for docs/
//   node tools/verify-delivery.mjs --full           # stream and hash the whole file
//
// This is the network half of the promotion gate that tools/probe-gguf.mjs is the
// disk half of. probe-gguf answers "is the file on this machine what the registry
// claims"; this answers "is the file on the internet the same file, and can a page
// on some other origin range-fetch it".
//
// It is a separate script and not part of `npm test` on purpose: it needs the
// internet, and a test suite that fails on a train is a test suite people stop
// running. tests/delivery.test.mjs covers the resolution logic offline.
//
// Two traps this exists to keep checking:
//
//   1. `X-Linked-ETag` -- the file's real SHA-256 -- is only on the 302 from
//      huggingface.co. Follow the redirect and the `ETag` you get from the CDN is
//      `X-Xet-Hash`, a Xet content hash that is a completely different number.
//      Compare that to a sha256 and every model looks corrupted. Hence
//      redirect: "manual" below.
//   2. The registry pins each URL to a commit rather than `main`. `main` is a
//      mutable ref: the 0.6B tokenizer.json at the pinned base revision and at
//      `main` happen to be identical today, and nothing guarantees they stay that
//      way. A room where one device loaded a mirror tokenizer and another loaded a
//      newer upstream one would produce mismatched token IDs on the wire, which is
//      not a failure that announces itself.

import { readFile, stat, open } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { MODELS, ORIGIN, sourcesFor } from "../models/registry.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const argv = process.argv.slice(2);
const JSON_OUT = argv.includes("--json");
const FULL = argv.includes("--full");
const only = argv.filter((a) => !a.startsWith("--"));

let pass = 0, fail = 0, skip = 0;
const results = [];
const say = (...a) => { if (!JSON_OUT) console.log(...a); };
const ok = (model, name, cond, detail = "") => {
  results.push({ model, check: name, pass: !!cond, detail });
  if (cond) { pass++; say("  ok   " + name + (detail ? "   " + detail : "")); }
  else { fail++; say("  FAIL " + name + (detail ? "\n         " + detail : "")); }
};
const note = (model, name, detail) => {
  results.push({ model, check: name, pass: null, detail });
  skip++; say("  ..   " + name + (detail ? "   " + detail : ""));
};
const MB = (b) => (b / 2 ** 20).toFixed(1) + " MB";

// The pre-redirect metadata huggingface.co attaches to a resolve URL. Everything
// interesting is here and nowhere else.
async function hfHead(url) {
  const r = await fetch(url, { method: "HEAD", redirect: "manual" });
  return {
    status: r.status,
    commit: r.headers.get("x-repo-commit"),
    size: Number(r.headers.get("x-linked-size") || r.headers.get("content-length") || 0),
    sha256: (r.headers.get("x-linked-etag") || "").replace(/"/g, "") || null,
    xetHash: r.headers.get("x-xet-hash"),
    location: r.headers.get("location"),
  };
}

async function rangeBytes(url, start, end) {
  const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (r.status !== 206) throw new Error(`expected 206, got ${r.status}`);
  return { buf: Buffer.from(await r.arrayBuffer()), contentRange: r.headers.get("content-range") };
}

// Exactly the request a page on another origin makes before a Range fetch, if the
// browser decides one is needed. A single `bytes=N-M` is CORS-safelisted so this
// preflight is normally skipped entirely -- but if a future change ever emits a
// multi-range or suffix range, the safelist stops applying and this becomes the
// request that must succeed. Checking it now is what makes that a caught
// regression rather than a mystery in someone's browser.
async function preflight(url, origin = "https://swarm.example") {
  const r = await fetch(url, {
    method: "OPTIONS",
    headers: {
      Origin: origin,
      "Access-Control-Request-Method": "GET",
      "Access-Control-Request-Headers": "range",
    },
  });
  return {
    status: r.status,
    allowOrigin: r.headers.get("access-control-allow-origin"),
    allowHeaders: r.headers.get("access-control-allow-headers"),
    exposeHeaders: r.headers.get("access-control-expose-headers"),
  };
}

async function sha256Local(path) {
  const h = createHash("sha256");
  const fh = await open(path, "r");
  for await (const chunk of fh.createReadStream()) h.update(chunk);
  await fh.close();
  return h.digest("hex");
}

async function sha256Remote(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const h = createHash("sha256");
  for await (const chunk of r.body) h.update(chunk);
  return h.digest("hex");
}

const ids = (only.length ? only : Object.keys(MODELS))
  .filter((id) => MODELS[id] && sourcesFor(MODELS[id], ORIGIN.UPSTREAM));

if (!ids.length) {
  console.error("no models with an upstream origin to verify" +
    (only.length ? ` (asked for: ${only.join(", ")})` : ""));
  process.exit(2);
}

for (const id of ids) {
  const d = MODELS[id];
  const up = sourcesFor(d, ORIGIN.UPSTREAM);
  const mirror = sourcesFor(d, ORIGIN.MIRROR);
  const p = d.provenance;
  say(`\n${d.label}  (${id})`);
  say(`  ${up.model}\n`);

  // ------------------------------------------------------------ the weights
  const head = await hfHead(up.model).catch((e) => ({ status: 0, error: e.message }));
  ok(id, "the pinned upstream URL resolves", head.status === 302 || head.status === 200,
     `HTTP ${head.status}${head.error ? " " + head.error : ""}`);
  if (head.status !== 302 && head.status !== 200) continue;

  ok(id, "upstream serves the revision the registry pins",
     !head.commit || head.commit === p.revision,
     `HF says ${head.commit}, registry pins ${p.revision}`);
  ok(id, "upstream byte length matches the registry", head.size === p.bytes,
     `HF says ${head.size}, registry says ${p.bytes}`);
  ok(id, "upstream SHA-256 (X-Linked-ETag) matches the registry", head.sha256 === p.sha256,
     `HF says ${head.sha256}, registry says ${p.sha256}`);
  if (head.xetHash) {
    note(id, "X-Xet-Hash differs from the SHA-256, as expected", head.xetHash +
      " -- this is what a redirect-following client sees as ETag; it is not a sha256");
  }

  // ------------------------------------------------------------ browser reachability
  const first = await rangeBytes(up.model, 0, 3).catch((e) => ({ error: e.message }));
  ok(id, "a cross-origin range request returns 206 with the GGUF magic bytes",
     !first.error && first.buf?.toString("latin1") === "GGUF",
     first.error || `content-range: ${first.contentRange}`);

  const pf = await preflight(up.model).catch((e) => ({ status: 0, error: e.message }));
  ok(id, "a CORS preflight for a Range request is answered",
     pf.status >= 200 && pf.status < 300 && /range/i.test(pf.allowHeaders || ""),
     `HTTP ${pf.status}, allow-headers: ${pf.allowHeaders || "(none)"}`);
  ok(id, "content-range is readable cross-origin (exposed or wildcarded)",
     /\*|content-range/i.test(pf.exposeHeaders || ""),
     `expose-headers: ${pf.exposeHeaders || "(none)"}`);

  // ------------------------------------------------------------ mirror equality
  // The claim the whole fallback rests on: whichever origin a device happens to
  // pick, it gets the same model. Spot ranges rather than a full download by
  // default -- four 64 KB windows at fixed positions across a 1.8 GB file is a
  // strong check that costs a second, and --full is there when "strong" is not
  // enough.
  const mirrorPath = mirror?.model ? join(ROOT, mirror.model.replace(/^\//, "")) : null;
  const local = mirrorPath ? await stat(mirrorPath).catch(() => null) : null;
  if (!local) {
    note(id, "local mirror not present, skipping mirror-vs-upstream comparison",
         mirrorPath || "no mirror declared");
  } else {
    ok(id, "local mirror is the same byte length as upstream", local.size === head.size,
       `mirror ${local.size}, upstream ${head.size}`);

    if (FULL) {
      const [a, b] = await Promise.all([sha256Local(mirrorPath), sha256Remote(up.model)]);
      ok(id, "local mirror and upstream hash identically (full download)", a === b,
         `mirror ${a}, upstream ${b}`);
    } else {
      const W = 64 * 1024;
      const offsets = [0, Math.floor(local.size * 0.25), Math.floor(local.size * 0.5), local.size - W];
      const fh = await open(mirrorPath, "r");
      let same = true, detail = "";
      for (const off of offsets) {
        const buf = Buffer.alloc(W);
        await fh.read(buf, 0, W, off);
        const remote = await rangeBytes(up.model, off, off + W - 1);
        if (Buffer.compare(buf, remote.buf) !== 0) { same = false; detail = `differ at byte ${off}`; break; }
      }
      await fh.close();
      ok(id, `local mirror matches upstream at ${offsets.length} x 64 KB spot ranges`, same,
         detail || `offsets ${offsets.join(", ")} (run --full to hash the whole ${MB(local.size)})`);
    }
  }

  // ------------------------------------------------------------ config + tokenizer
  // These come from the base repo, not the *-GGUF repo, at their own pinned
  // revision. They are small, so compare them exactly rather than by spot range.
  for (const [kind, url, mirrorUrl] of [
    ["config.json", up.config, mirror?.config],
    ["tokenizer.json", up.tokenizer, mirror?.tokenizer],
  ]) {
    if (!url) { note(id, `${kind} has no upstream URL`, ""); continue; }
    const r = await fetch(url).catch((e) => ({ ok: false, status: 0, error: e.message }));
    if (!r.ok) { ok(id, `upstream ${kind} resolves`, false, `HTTP ${r.status} ${r.error || ""}`); continue; }
    const remote = Buffer.from(await r.arrayBuffer());
    ok(id, `upstream ${kind} resolves`, true, `${remote.length} bytes`);
    if (!mirrorUrl) continue;
    const localBuf = await readFile(join(ROOT, mirrorUrl.replace(/^\//, ""))).catch(() => null);
    if (!localBuf) { note(id, `local ${kind} not present`, mirrorUrl); continue; }
    ok(id, `local ${kind} is byte-identical to upstream`, Buffer.compare(localBuf, remote) === 0,
       `mirror sha256:${createHash("sha256").update(localBuf).digest("hex")}, ` +
       `upstream sha256:${createHash("sha256").update(remote).digest("hex")}`);
  }
}

if (JSON_OUT) {
  console.log(JSON.stringify({
    tool: "tools/verify-delivery.mjs",
    at: new Date().toISOString(),
    full: FULL,
    summary: { pass, fail, notes: skip },
    results,
  }, null, 2));
} else {
  console.log(`\n${pass} passed, ${fail} failed, ${skip} notes\n`);
}
process.exit(fail ? 1 : 0);
