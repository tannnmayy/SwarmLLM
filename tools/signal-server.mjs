// The signalling service, standalone.
//
//   node tools/signal-server.mjs            # PORT=8080 by default
//   PORT=3000 node tools/signal-server.mjs
//
// This is the entire server half of a deployment. Everything else — the pages, the
// modules, the weights — is static files on a CDN and a range-fetch to
// huggingface.co. What is left here is introductions: it brokers SDP and ICE
// between browsers and then stops mattering, because activations, prompts and
// answers only ever travel over the WebRTC data channels the SDP sets up.
//
// It holds no model, no weights, no conversation and no user data. Its memory is a
// Map of room code -> live sockets, and that Map empties itself as people leave. It
// is intentionally the cheapest thing in the system to run: no disk, no database,
// and it can be restarted under load with no worse consequence than everyone
// currently mid-join retrying.
//
// TLS is terminated by the platform in front of it (Fly, Railway, Render and
// Cloud Run all do this), so this listens on plain HTTP and the browser still
// reaches it as wss://. Running it behind something that does NOT terminate TLS
// means the page must also be http://, which rules out WebGPU and WebRTC — see
// tools/serve.mjs, which is the dev path that solves the same problem with a cert.

import { createServer } from "node:http";
import { attachSignaling, signalStats } from "./signal.mjs";

const PORT = Number(process.env.PORT) || 8080;
const HOST = process.env.HOST || "0.0.0.0";
// Optional origin allowlist, comma separated. Off by default: with it unset any
// page may open a socket, which is what a local checkout and a `#signal=` override
// both need. Set it in a real deployment to keep someone else's page from using
// this service as free infrastructure. It is not authentication — the room code
// still is — it just stops the most casual case.
const ALLOW = (process.env.SIGNAL_ALLOW_ORIGIN || "").split(",").map((s) => s.trim()).filter(Boolean);

const started = Date.now();

const server = createServer((req, res) => {
  const url = (req.url || "/").split("?")[0];

  // Same-origin is not a given here: the pages are on a different host than this
  // service. Only the health endpoint is a plain HTTP resource anyone would fetch
  // cross-origin; the websocket handshake does its own origin handling.
  res.setHeader("access-control-allow-origin", "*");

  if (url === "/healthz" || url === "/health") {
    const body = JSON.stringify({
      ok: true,
      service: "ai-swarm-signal",
      uptimeSeconds: Math.round((Date.now() - started) / 1000),
      ...signalStats(),
    });
    res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
    res.end(body);
    return;
  }

  // Anything else gets a plain explanation rather than a 404, because the first
  // thing anyone does with a deployed signalling URL is open it in a browser.
  res.writeHead(url === "/" ? 200 : 404, { "content-type": "text/plain; charset=utf-8" });
  res.end(
    "AI Swarm signalling service.\n\n" +
    "This host brokers WebRTC introductions and nothing else. It never sees a prompt,\n" +
    "an activation, a weight or an answer.\n\n" +
    "  websocket : /signal\n" +
    "  health    : /healthz\n\n" +
    "The application itself is a static site; point its swarm-config.json at this host.\n"
  );
});

const wss = attachSignaling(server, { log: true, allowOrigins: ALLOW });

server.listen(PORT, HOST, () => {
  console.log(`\n  AI Swarm signalling service`);
  console.log(`  listening on http://${HOST}:${PORT}`);
  console.log(`  websocket    /signal`);
  console.log(`  health       /healthz`);
  if (ALLOW.length) console.log(`  origins      ${ALLOW.join(", ")}`);
  else console.log(`  origins      any (set SIGNAL_ALLOW_ORIGIN to restrict)`);
  console.log("");
});

// A platform restart sends SIGTERM and then waits. Closing the listener first stops
// new joins; the open sockets are told to close so browsers see a clean shutdown
// and can retry, rather than sitting on a connection that will never answer again.
for (const sig of ["SIGTERM", "SIGINT"]) {
  process.on(sig, () => {
    console.log(`\n  ${sig}: draining`);
    server.close(() => process.exit(0));
    for (const ws of wss.clients) { try { ws.close(1001, "server restarting"); } catch { /* already gone */ } }
    setTimeout(() => process.exit(0), 3000).unref();
  });
}
