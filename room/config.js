// Runtime configuration: the two things that differ between "running the dev
// server on my laptop" and "deployed as a static site".
//
//   signalUrl   where the signalling websocket lives. On the dev server it rides
//               the same listener as the pages (wss://<this host>/signal), which is
//               why there has never been a mixed-content problem. On a static
//               deployment the pages are on a CDN and signalling is a separate
//               small service on another host, so it has to be nameable.
//   delivery    which origin weights come from: "auto" (probe), "mirror", or
//               "upstream". See models/delivery.mjs.
//
// Nothing else belongs here. This is deployment wiring, not product settings — the
// room code, the model and the context window all travel in the URL hash or the
// deal, where they can differ per room rather than per deployment.
//
// Resolution order, first hit wins:
//   1. the URL hash          #signal=wss://host/signal  #src=upstream
//      Per-visit override, for testing a deployment's real signalling service from
//      a local page, or forcing the CDN path on a machine that has a mirror.
//   2. window.SWARM_CONFIG   inline <script> in the served HTML
//      For a host that templates its config in and would rather not pay a fetch.
//   3. ./swarm-config.json   fetched next to the page, optional
//      What tools/build-static.mjs writes. Re-pointing a built site at a different
//      signalling service is editing one small file, not a rebuild.
//   4. defaults              same-origin /signal, delivery "auto"
//      Exactly the current dev-server behaviour, so a checkout with no config at
//      all still runs the way it always has.

import { PREFERENCE } from "../models/delivery.mjs";

// `models`: which model ids this deployment actually bundled under /models/, or
// null for "no claim, find out by probing". The registry declares a mirror path for
// every model because it describes the project, not one deployment of it — so a
// build that shipped one model says which one, and the picker stops offering the
// others as though they were there.
// `iceServers` / `iceServersUrl`: what WebRTC should try in order to get a direct
// path between two devices. Absent, room/mesh.js uses public STUN, which is enough
// on one network and usually enough across two home networks, but not behind a
// symmetric NAT. A deployment that wants those cases to work supplies TURN.
//
// Two shapes, because TURN providers come in two shapes:
//   iceServers      a literal RTCIceServer[] — a provider with static credentials
//   iceServersUrl   an endpoint that mints short-lived credentials on demand.
//                   Cloudflare's TURN works this way and does not allow long-lived
//                   ones; the signalling Worker exposes /ice for exactly this.
// If both are given, the fetched list wins and the literal one is the fallback for
// when that fetch fails — which must not be fatal, because STUN alone still
// connects the common case.
const DEFAULTS = {
  signalUrl: null, delivery: PREFERENCE.AUTO, models: null,
  iceServers: null, iceServersUrl: null, deployment: "dev",
};

let cached = null;

function sameOriginSignalUrl() {
  if (typeof location === "undefined") return null;
  return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/signal";
}

// A page served over https can only open wss://. Getting this wrong produces a
// browser security error with no useful text, and it is an easy mistake to make
// when copying a signalling URL out of a local dev run into a deployed config.
function checkScheme(url) {
  if (!url || typeof location === "undefined") return null;
  if (location.protocol === "https:" && url.startsWith("ws://")) {
    return `signalling URL ${url} is insecure (ws://) but this page is https — the browser will block it. Use wss://`;
  }
  if (!/^wss?:\/\//.test(url)) return `signalling URL ${url} is not a websocket URL (expected ws:// or wss://)`;
  return null;
}

function fromHash() {
  if (typeof location === "undefined") return {};
  const h = new URLSearchParams(location.hash.slice(1));
  const out = {};
  if (h.get("signal")) out.signalUrl = h.get("signal");
  const src = h.get("src");
  if (src && Object.values(PREFERENCE).includes(src)) out.delivery = src;
  return out;
}

async function fromFile() {
  if (typeof fetch === "undefined") return {};
  try {
    // Relative, not absolute: a deployment under a sub-path (project pages) has to
    // find its own config, not the origin root's.
    const res = await fetch(new URL("swarm-config.json", document.baseURI).href, { cache: "no-cache" });
    if (!res.ok) return {};
    const j = await res.json();
    return typeof j === "object" && j ? j : {};
  } catch {
    return {};                       // no config file is the normal dev case
  }
}

// Resolved once per page load. Every caller awaits the same promise.
export function loadConfig() {
  if (cached) return cached;
  cached = (async () => {
    const file = await fromFile();
    const inline = (typeof window !== "undefined" && window.SWARM_CONFIG) || {};
    const hash = fromHash();
    const cfg = { ...DEFAULTS, ...file, ...inline, ...hash };

    cfg.signalUrl = cfg.signalUrl || sameOriginSignalUrl();
    cfg.warning = checkScheme(cfg.signalUrl);
    if (!Object.values(PREFERENCE).includes(cfg.delivery)) cfg.delivery = PREFERENCE.AUTO;
    if (!Array.isArray(cfg.models)) cfg.models = null;

    // Short-lived TURN credentials, if this deployment mints them. Deliberately
    // non-fatal: a room where TURN is unreachable still connects every peer that
    // has a direct path, which is the common case. Failing the join instead would
    // trade a rare problem for a universal one.
    if (cfg.iceServersUrl) {
      try {
        const res = await fetch(cfg.iceServersUrl, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const j = await res.json();
        const list = Array.isArray(j) ? j : j.iceServers;
        if (Array.isArray(list) && list.length) cfg.iceServers = list;
        else throw new Error("no iceServers in the response");
      } catch (e) {
        cfg.iceWarning = `could not fetch TURN credentials from ${cfg.iceServersUrl} (${e.message}) — ` +
          "continuing with STUN only, so two devices on different networks may not connect";
      }
    }
    if (!Array.isArray(cfg.iceServers) || !cfg.iceServers.length) cfg.iceServers = null;
    // Where each field came from, so the room log can say why it is talking to the
    // host it is talking to. Deployment misconfiguration is otherwise invisible
    // until a join silently fails.
    cfg.from = {
      signalUrl: hash.signalUrl ? "url hash" : inline.signalUrl ? "window.SWARM_CONFIG"
                 : file.signalUrl ? "swarm-config.json" : "same origin",
      delivery: hash.delivery ? "url hash" : inline.delivery ? "window.SWARM_CONFIG"
                 : file.delivery ? "swarm-config.json" : "default",
    };
    return cfg;
  })();
  return cached;
}

export function resetConfig() { cached = null; }
