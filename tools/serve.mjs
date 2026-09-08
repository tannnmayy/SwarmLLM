// AI Swarm dev server: HTTPS static host over the LAN.
//
// WebGPU and WebRTC both require a secure context, so plain http://192.168.x.x
// makes every phone report "no WebGPU" even when the hardware is fine. We serve
// HTTPS with a self-signed cert whose SAN covers the LAN IP (iOS checks SAN, not CN).
//
//   node tools/serve.mjs [port]
//
// Regenerate the cert when the LAN IP changes: see tools/make-cert.sh

import { createServer } from "node:https";
import { createServer as createHttp } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { networkInterfaces } from "node:os";
import { join, normalize, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { attachSignaling } from "./signal.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const PORT = Number(process.argv[2]) || 8443;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".safetensors": "application/octet-stream",
  ".bin": "application/octet-stream",
};

function lanIPs() {
  const out = [];
  for (const list of Object.values(networkInterfaces())) {
    for (const ni of list || []) {
      if (ni.family === "IPv4" && !ni.internal) out.push(ni.address);
    }
  }
  return out;
}

const [key, cert] = await Promise.all([
  readFile(join(ROOT, "certs/key.pem")),
  readFile(join(ROOT, "certs/cert.pem")),
]).catch(() => {
  console.error("No cert found. Run: bash tools/make-cert.sh");
  process.exit(1);
});

const handler = async (req, res) => {
  // strip query, decode, and refuse anything that escapes ROOT
  const url = decodeURIComponent((req.url || "/").split("?")[0]);
  let rel = normalize(url).replace(/^(\.\.[/\\])+/, "").replace(/^[/\\]+/, "");
  if (rel === "") rel = "index.html";
  const path = join(ROOT, rel);
  if (!path.startsWith(ROOT)) {
    res.writeHead(403).end("forbidden");
    return;
  }

  let info;
  try {
    info = await stat(path);
    if (info.isDirectory()) throw new Error("dir");
  } catch {
    res.writeHead(404, { "content-type": "text/plain" }).end("not found: " + rel);
    return;
  }

  const head = {
    "content-type": MIME[extname(path).toLowerCase()] || "application/octet-stream",
    // never cache during a hackathon: a stale kernel costs an hour of confusion
    "cache-control": "no-store, no-cache, must-revalidate",
  };
  // Deliberately NOT setting COOP/COEP. They only buy SharedArrayBuffer, which the
  // WebGPU path does not need, and `require-corp` blocks cross-origin weight
  // downloads (Hugging Face) unless every response carries CORP. Not worth the risk.

  // range support so a phone can resume a partial weight download
  const range = req.headers.range;
  if (range) {
    const m = /bytes=(\d*)-(\d*)/.exec(range);
    if (m) {
      const start = m[1] ? Number(m[1]) : 0;
      const end = m[2] ? Number(m[2]) : info.size - 1;
      if (start >= info.size || end >= info.size || start > end) {
        res.writeHead(416, { "content-range": `bytes */${info.size}` }).end();
        return;
      }
      res.writeHead(206, {
        ...head,
        "content-range": `bytes ${start}-${end}/${info.size}`,
        "accept-ranges": "bytes",
        "content-length": end - start + 1,
      });
      createReadStream(path, { start, end }).pipe(res);
      return;
    }
  }

  res.writeHead(200, { ...head, "accept-ranges": "bytes", "content-length": info.size });
  if (req.method === "HEAD") return res.end();
  createReadStream(path).pipe(res);
};

// Two listeners on the same files:
//   HTTP  on localhost  - http://localhost is already a secure context, so WebGPU and
//                         WebRTC work here with no cert warning. Use this for dev.
//   HTTPS on the LAN    - phones need a real secure context, which over the LAN means TLS.
const HTTP_PORT = PORT - 1;
const plain = createHttp(handler);
const secure = createServer({ key, cert }, handler);

// Signaling rides on the same listeners, at /signal. A page served over https
// can only open wss:// on its own origin, so sharing the port removes the whole
// mixed-content class of failure.
attachSignaling(plain, { log: false });
attachSignaling(secure);

plain.listen(HTTP_PORT, "127.0.0.1", () => {});
secure.listen(PORT, "0.0.0.0", () => {
  console.log("\n  AI Swarm dev server\n");
  console.log(`  this machine   http://localhost:${HTTP_PORT}/probe.html   (no warning)`);
  for (const ip of lanIPs()) {
    console.log(`  other devices  https://${ip}:${PORT}/probe.html`);
  }
  console.log("\n  The LAN cert is self-signed, so every phone shows a warning once:");
  console.log("    Chrome/Android : Advanced -> Proceed to ...");
  console.log("    iOS Safari     : Show Details -> visit this website -> Visit");
  console.log("  Accept it, and WebGPU + WebRTC become available.\n");
});
