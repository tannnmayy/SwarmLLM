// Static file serving, as a handler rather than a server.
//
// Extracted from tools/serve.mjs so the two halves of the deployment can be built
// out of the same code: the dev server mounts this on two listeners next to the
// signalling websocket, and nothing else needs it at all, because a static
// deployment replaces this file with a CDN. That is the point of the split — the
// production static host is Pages/Netlify/Cloudflare, not Node, so this stays a
// development convenience and never becomes a thing that has to scale.
//
// It has to be a genuinely correct range server regardless, because it is the LAN
// path a real two-device run uses: a device only downloads the layers it was dealt,
// which is a range request per tensor.

import { stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { join, normalize, extname, resolve } from "node:path";

export const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".md": "text/markdown; charset=utf-8",
  ".gguf": "application/octet-stream",
  ".safetensors": "application/octet-stream",
  ".bin": "application/octet-stream",
};

// Resolve a request path to a file inside root, or null if it escapes.
//
// Two independent guards, because either alone has a hole: normalize() collapses
// `..` segments, and the resolved-prefix check catches anything normalize did not
// (a symlinked path, an encoded separator on Windows). The trailing separator in
// the comparison matters — without it, a sibling directory whose name merely starts
// with the root's name would pass.
export function safePath(root, urlPath) {
  const url = decodeURIComponent((urlPath || "/").split("?")[0]);
  let rel = normalize(url).replace(/^(\.\.[/\\])+/, "").replace(/^[/\\]+/, "");
  if (rel === "") rel = "index.html";
  const abs = resolve(join(root, rel));
  const base = resolve(root);
  if (abs !== base && !abs.startsWith(base + (process.platform === "win32" ? "\\" : "/"))) return null;
  // `rel` goes back out with forward slashes whatever platform this is running on.
  // normalize() hands back backslashes on Windows, and anything matching on the
  // path — the cache policy below, a caller's onMiss — would then quietly stop
  // matching there and nowhere else.
  return { abs, rel: rel.split("\\").join("/") };
}

// Parse one `bytes=a-b` header against a known file size.
// Returns null for "no range asked", {unsatisfiable:true} for a range outside the
// file (which must be a 416, not a clamp — engine/gpu-adapter.mjs relies on 416
// meaning "the whole file is smaller than the probe window").
export function parseRange(header, size) {
  if (!header) return null;
  const m = /bytes=(\d*)-(\d*)/.exec(header);
  if (!m) return null;
  const start = m[1] ? Number(m[1]) : 0;
  const end = m[2] ? Number(m[2]) : size - 1;
  if (start >= size || end >= size || start > end) return { unsatisfiable: true };
  return { start, end };
}

export function createStaticHandler(root, {
  // Development default: never cache. A stale kernel or a stale module costs an
  // hour of confusion, and on a LAN the re-fetch is free.
  //
  // Weights are the exception and always have been: a GGUF is content-addressed by
  // (url, byte range) in the browser's Cache API (engine/gpu-adapter.mjs), it is
  // pinned to a commit, and it is measured in gigabytes. Telling the browser not to
  // cache it would make every reload a fresh multi-GB download.
  //
  // "under models/" is NOT the rule, though it is the obvious one to reach for:
  // models/registry.mjs and models/delivery.mjs live there and are source code. A
  // year-long immutable cache on those is the stale-module failure this policy
  // exists to avoid, and it is not hypothetical — it happened during this file's
  // own testing, with a browser cheerfully running a build-old module against a
  // fresh page.
  cacheControl = (rel) => (/^models\//.test(rel) && !/\.(m?js)$/.test(rel)
    ? "public, max-age=31536000, immutable"
    : "no-store, no-cache, must-revalidate"),
  onMiss = null,
} = {}) {
  return async function handle(req, res) {
    const p = safePath(root, req.url);
    if (!p) { res.writeHead(403).end("forbidden"); return true; }

    let info;
    try {
      info = await stat(p.abs);
      if (info.isDirectory()) throw new Error("dir");
    } catch {
      if (onMiss && (await onMiss(req, res, p))) return true;
      res.writeHead(404, { "content-type": "text/plain" }).end("not found: " + p.rel);
      return true;
    }

    const head = {
      "content-type": MIME[extname(p.abs).toLowerCase()] || "application/octet-stream",
      "cache-control": cacheControl(p.rel),
    };
    // Deliberately NOT setting COOP/COEP. They only buy SharedArrayBuffer, which the
    // WebGPU path does not need, and `require-corp` blocks cross-origin weight
    // downloads (Hugging Face) unless every response carries CORP — which is now the
    // primary delivery path, not a fallback. Not worth the risk.

    const range = parseRange(req.headers.range, info.size);
    if (range?.unsatisfiable) {
      res.writeHead(416, { "content-range": `bytes */${info.size}` }).end();
      return true;
    }
    if (range) {
      res.writeHead(206, {
        ...head,
        "content-range": `bytes ${range.start}-${range.end}/${info.size}`,
        "accept-ranges": "bytes",
        "content-length": range.end - range.start + 1,
      });
      if (req.method === "HEAD") return res.end(), true;
      createReadStream(p.abs, { start: range.start, end: range.end }).pipe(res);
      return true;
    }

    res.writeHead(200, { ...head, "accept-ranges": "bytes", "content-length": info.size });
    if (req.method === "HEAD") return res.end(), true;
    createReadStream(p.abs).pipe(res);
    return true;
  };
}
