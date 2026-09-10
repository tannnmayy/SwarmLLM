// Where this device gets its weight bytes from, decided per device at load time.
//
// The registry declares up to two origins per model (models/registry.mjs, ORIGIN):
// a local mirror under /models/, and the pinned artifact on huggingface.co. This
// module picks one and hands back a descriptor with the flat `modelUrl` /
// `configUrl` / `tokenizerUrl` / `dir` fields the rest of the codebase already
// reads, so nothing downstream — engine/gpu-adapter.mjs, engine/factory.mjs,
// tools/probe-gguf.mjs — has to learn about origins at all.
//
// Why this is a per-device decision and not a build-time constant: the same build
// is meant to run in two places at once. A laptop on the venue LAN should pull
// 1.8 GB off the host it is already talking to; a stranger on a static deployment
// has no such host and pulls from HF's CDN. Both are the same bytes — the registry
// records a sha256 for each file and tools/verify-delivery.mjs re-checks the live
// upstream against it — so which origin a device used changes its download time,
// never its output. Two devices in one room may legitimately disagree.
//
// The probe is four bytes. `Range: bytes=0-3` on the GGUF, and the answer has to be
// a 206 whose body is the ASCII magic `GGUF`. Both halves matter:
//   * 206, not 200, proves the origin honours Range at all. An origin that ignores
//     it would hand a device the whole multi-GB file for every tensor it wanted.
//   * the magic bytes reject the failure that a status code alone cannot see — a
//     static host with an SPA fallback answers /models/anything.gguf with 200 and
//     an HTML page, which would otherwise look exactly like a working mirror until
//     the GGUF parser hit it several hundred megabytes later.

import { ORIGIN, sourcesFor, deliveryOrigins } from "./registry.mjs";

// A bundled build carries some models under /models/ and not others (see
// tools/build-static.mjs --models). The registry declares a mirror path for every
// GGUF model regardless, because the registry describes the project rather than one
// deployment of it — so a deployment that knows what it shipped says so in
// swarm-config.json, and a mirror path for anything else is treated as absent.
// Under "auto" the probe would discover this anyway; this is what makes the answer
// right under a forced preference, where nothing is probed.
function carriesMirror(descriptor, carries) {
  return !Array.isArray(carries) || carries.includes(descriptor.id);
}

export { ORIGIN };

// What a caller may ask for. "auto" is the default and the interesting one; the
// other two are escape hatches for a deployment that already knows the answer and
// would rather not spend a round trip finding it out again (see tools/build-static.mjs,
// which writes one into the generated swarm-config.json).
export const PREFERENCE = { AUTO: "auto", MIRROR: ORIGIN.MIRROR, UPSTREAM: ORIGIN.UPSTREAM };

const GGUF_MAGIC = "GGUF";

// Resolutions are memoised per (model id, preference). A room re-resolves the same
// model on every deal and every re-plan, and re-probing each time would put a
// pointless request in front of every recovery.
const memo = new Map();

export function clearDeliveryCache() { memo.clear(); }

// Thrown when no origin can serve this model to this page. Distinct from a network
// error on purpose: the UI needs to say "this deployment does not carry SmolLM2"
// rather than "fetch failed", which is a different problem with a different fix.
export class NoDeliveryOriginError extends Error {
  constructor(descriptor, tried) {
    const detail = descriptor.mirrorOnlyReason
      ? ` — ${descriptor.mirrorOnlyReason}`
      : "";
    super(`${descriptor.label} cannot be delivered to this page${detail}. Tried: ${
      tried.map((t) => `${t.origin} (${t.why})`).join("; ") || "no origin declared"}`);
    this.name = "NoDeliveryOriginError";
    this.descriptor = descriptor;
    this.tried = tried;
  }
}

// One origin, one four-byte question: can this page range-fetch a GGUF from here?
export async function probeGGUF(url, { fetchImpl = fetch, timeoutMs = 8000 } = {}) {
  const t0 = Date.now();
  const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctl && setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, {
      headers: { Range: "bytes=0-3" },
      signal: ctl?.signal,
      // Cross-origin, this must stay a plain CORS request. `Range` with a single
      // `bytes=N-M` is a CORS-safelisted request header, so it does not trigger a
      // preflight; credentials would take it out of that easy path for nothing.
      credentials: "omit",
    });
    if (timer) clearTimeout(timer);
    if (res.status !== 206) {
      return { ok: false, status: res.status, ms: Date.now() - t0,
               why: res.status === 200 ? "answered 200, so it ignores Range" : `HTTP ${res.status}` };
    }
    const buf = await res.arrayBuffer();
    const magic = new TextDecoder().decode(new Uint8Array(buf).slice(0, 4));
    if (magic !== GGUF_MAGIC) {
      return { ok: false, status: res.status, ms: Date.now() - t0,
               why: `first four bytes were ${JSON.stringify(magic)}, not ${GGUF_MAGIC}` };
    }
    return { ok: true, status: res.status, ms: Date.now() - t0, why: "206 + GGUF magic" };
  } catch (e) {
    if (timer) clearTimeout(timer);
    return { ok: false, status: 0, ms: Date.now() - t0, why: String(e?.message || e) };
  }
}

// The CPU model has no single file to sniff — it is a directory of per-layer shards
// described by a manifest. Presence of a parseable manifest is the same question in
// that shape, and it catches the same SPA-fallback trap (an HTML page is not JSON).
export async function probeManifest(url, { fetchImpl = fetch, timeoutMs = 8000 } = {}) {
  const t0 = Date.now();
  const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctl && setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, { signal: ctl?.signal, credentials: "omit" });
    if (timer) clearTimeout(timer);
    if (!res.ok) return { ok: false, status: res.status, ms: Date.now() - t0, why: `HTTP ${res.status}` };
    const j = await res.json();
    if (!j || !j.shards) return { ok: false, status: res.status, ms: Date.now() - t0, why: "no shard index in the manifest" };
    return { ok: true, status: res.status, ms: Date.now() - t0, why: "manifest parsed" };
  } catch (e) {
    if (timer) clearTimeout(timer);
    return { ok: false, status: 0, ms: Date.now() - t0, why: String(e?.message || e) };
  }
}

// Flatten one origin's source record onto the descriptor, in the shape every
// existing consumer already expects.
function applySource(descriptor, origin, src) {
  return {
    ...descriptor,
    // GGUF models
    modelUrl: src.model || null,
    configUrl: src.config || null,
    tokenizerUrl: src.tokenizer || null,
    // CPU shard model
    dir: src.dir || null,
    delivery: { origin, sources: src },
  };
}

function probeFor(descriptor, src, opts) {
  if (descriptor.engineKind === "cpu-smollm") {
    return probeManifest(src.manifest || `${src.dir}/manifest.json`, opts);
  }
  return probeGGUF(src.model, opts);
}

// Resolve one descriptor to a concrete set of URLs.
//
//   prefer: "auto"      probe each declared origin in order, take the first that
//                       answers correctly. This is what makes one build work both
//                       on the LAN dev server and on a static deployment.
//           "mirror"    use the mirror without probing; fail if it is not declared.
//           "upstream"  use huggingface.co without probing.
//
// A forced preference skips the probe entirely: a deployment that already knows it
// carries no /models/ directory has nothing to learn from asking, and the round
// trip would sit in front of every model load.
export async function resolveDelivery(descriptor, opts = {}) {
  const { prefer = PREFERENCE.AUTO, fetchImpl = fetch, timeoutMs = 8000, cache = true, carries = null } = opts;
  if (!descriptor?.sources) {
    throw new NoDeliveryOriginError(descriptor || { label: "unknown model" }, []);
  }

  const key = `${descriptor.id}::${prefer}::${Array.isArray(carries) ? carries.join(",") : "*"}`;
  if (cache && memo.has(key)) return memo.get(key);

  const hasMirror = carriesMirror(descriptor, carries);

  const run = (async () => {
    const tried = [];

    // Forced: take it or fail, no probe.
    if (prefer !== PREFERENCE.AUTO) {
      const src = prefer === ORIGIN.MIRROR && !hasMirror ? null : sourcesFor(descriptor, prefer);
      if (!src) {
        tried.push({ origin: prefer, why: prefer === ORIGIN.MIRROR && !hasMirror
          ? "this deployment did not bundle it" : "not declared for this model" });
        throw new NoDeliveryOriginError(descriptor, tried);
      }
      return applySource(descriptor, prefer, src);
    }

    const origins = deliveryOrigins(descriptor).filter((o) => o !== ORIGIN.MIRROR || hasMirror);
    if (!origins.length) {
      tried.push({ origin: ORIGIN.MIRROR, why: "this deployment did not bundle it" });
      throw new NoDeliveryOriginError(descriptor, tried);
    }
    for (let i = 0; i < origins.length; i++) {
      const origin = origins[i];
      const src = sourcesFor(descriptor, origin);
      // The last remaining candidate is taken on faith rather than probed. There is
      // nothing to fall back to, so a probe could only turn a real, specific load
      // error ("HTTP 503 range-fetching blk.12.attn_q.weight") into a vaguer one.
      if (i === origins.length - 1) {
        return applySource(descriptor, origin, { ...src, __unprobed: true });
      }
      const r = await probeFor(descriptor, src, { fetchImpl, timeoutMs });
      tried.push({ origin, ...r });
      if (r.ok) {
        const resolved = applySource(descriptor, origin, src);
        resolved.delivery.probed = r;
        resolved.delivery.tried = tried;
        return resolved;
      }
    }
    throw new NoDeliveryOriginError(descriptor, tried);
  })();

  if (cache) {
    memo.set(key, run);
    // A failed resolution must not be cached: the mirror may simply not have been
    // deployed yet, and a retry after a refresh should get a real answer.
    run.catch(() => memo.delete(key));
  }
  return run;
}

// Cheap, synchronous "could this page ever offer this model?" for the model picker,
// which has to render before any probe has run. It answers the one question the
// picker can answer without a network round trip: a mirror-only model on a
// deployment configured for upstream delivery is never going to work.
export function deliverableUnder(descriptor, prefer, carries = null) {
  if (!descriptor?.sources) return { ok: false, why: descriptor?.blockedOn || "no source declared" };
  const hasMirror = carriesMirror(descriptor, carries);
  if (prefer === PREFERENCE.AUTO) {
    // Under auto a probe still has to run, so the only thing knowable synchronously
    // is whether every origin is already ruled out.
    return hasMirror || sourcesFor(descriptor, ORIGIN.UPSTREAM)
      ? { ok: true, why: null }
      : { ok: false, why: "this deployment did not bundle it, and it is published nowhere upstream" };
  }
  if (prefer === ORIGIN.MIRROR && !hasMirror) {
    return { ok: false, why: "this deployment did not bundle it" };
  }
  return sourcesFor(descriptor, prefer)
    ? { ok: true, why: null }
    : { ok: false, why: descriptor.mirrorOnlyReason
        ? `this deployment serves weights from ${prefer}, and ${descriptor.mirrorOnlyReason}`
        : `this deployment serves weights from ${prefer}, which does not carry this model` };
}

// Short human label for the UI, e.g. "Hugging Face CDN" / "this host".
export function originLabel(origin) {
  return origin === ORIGIN.UPSTREAM ? "Hugging Face CDN" : origin === ORIGIN.MIRROR ? "this host" : "unknown";
}
