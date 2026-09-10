// AI Swarm dev server: the static half and the signalling half on one listener.
//
// WebGPU and WebRTC both require a secure context, so plain http://192.168.x.x
// makes every phone report "no WebGPU" even when the hardware is fine. We serve
// HTTPS with a self-signed cert whose SAN covers the LAN IP (iOS checks SAN, not CN).
//
//   node tools/serve.mjs [port]
//
// Regenerate the cert when the LAN IP changes: see tools/make-cert.sh
//
// This is development and LAN demos only. In a deployment the two halves it
// combines live apart and neither of them is this file:
//
//   static  ->  tools/build-static.mjs writes dist/, which goes on a CDN
//   signal  ->  tools/signal-server.mjs, one small always-on process
//
// Keeping them together here is deliberate, not a leftover: on a LAN there is no
// CDN, the weights should come off this machine rather than the internet, and a
// page on https:// can only open wss:// on its own origin — sharing one listener
// removes the whole mixed-content class of failure from the demo path.

import { createServer } from "node:https";
import { createServer as createHttp } from "node:http";
import { readFile } from "node:fs/promises";
import { networkInterfaces } from "node:os";
import { fileURLToPath } from "node:url";
import { attachSignaling } from "./signal.mjs";
import { createStaticHandler } from "./static.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const PORT = Number(process.argv[2]) || 8443;

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
  readFile(new URL("../certs/key.pem", import.meta.url)),
  readFile(new URL("../certs/cert.pem", import.meta.url)),
]).catch(() => {
  console.error("No cert found. Run: bash tools/make-cert.sh");
  process.exit(1);
});

const handler = createStaticHandler(ROOT);

// Two listeners on the same files:
//   HTTP  on localhost  - http://localhost is already a secure context, so WebGPU and
//                         WebRTC work here with no cert warning. Use this for dev.
//   HTTPS on the LAN    - phones need a real secure context, which over the LAN means TLS.
const HTTP_PORT = PORT - 1;
const plain = createHttp(handler);
const secure = createServer({ key, cert }, handler);

// Signalling rides on both listeners and shares one room table, which is what lets
// a laptop on http://localhost and a phone on https://<lan-ip> join the same room.
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
  console.log("  Weights: whichever origin answers first — the mirror under models/ if");
  console.log("  it is present, else huggingface.co. Force one with #src=mirror / #src=upstream.\n");
});
