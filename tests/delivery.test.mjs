// Weight delivery: which origin a device pulls from, and why.
//
//   node tests/delivery.test.mjs
//
// Offline by design. Every fetch here is a stub, because the interesting cases are
// the failures — a mirror that is not deployed, a static host that answers 200 with
// an HTML page for a URL that should be a GGUF — and those are precisely the ones
// you cannot arrange against a real network on demand. The live upstream check is
// tools/verify-delivery.mjs, run separately.
//
// The registry invariants at the bottom are the other half: the resolution logic
// can be perfect and still send devices to two different files if a URL and the
// provenance record it is supposed to have come from drift apart.

import {
  resolveDelivery, probeGGUF, probeManifest, deliverableUnder, originLabel,
  clearDeliveryCache, NoDeliveryOriginError, PREFERENCE, ORIGIN,
} from "../models/delivery.mjs";
import { MODELS, sourcesFor, deliveryOrigins } from "../models/registry.mjs";

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};
const throws = async (name, fn, matches) => {
  try { await fn(); ok(name, false, "did not throw"); }
  catch (e) { ok(name, !matches || matches.test(e.message), e.message); }
};

const bodyOf = (s) => {
  const u = new TextEncoder().encode(s);
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength);
};

// A stub origin, described by what it does to a request rather than by a canned
// response, so one factory covers "serves ranges", "404s", "ignores Range" and
// "answers everything with the index page".
function stubFetch(routes) {
  const calls = [];
  const f = async (url, opts) => {
    calls.push({ url: String(url), range: opts?.headers?.Range || null });
    for (const [prefix, behaviour] of routes) {
      if (String(url).startsWith(prefix)) return behaviour(String(url), opts);
    }
    return { status: 404, ok: false, arrayBuffer: async () => bodyOf(""), json: async () => { throw new Error("not json"); } };
  };
  f.calls = calls;
  return f;
}

const servesGGUF = () => ({ status: 206, ok: true, arrayBuffer: async () => bodyOf("GGUF") });
const ignoresRange = () => ({ status: 200, ok: true, arrayBuffer: async () => bodyOf("GGUF") });
const spaFallback = () => ({ status: 206, ok: true, arrayBuffer: async () => bodyOf("<!doctype html>") });
const notFound = () => ({ status: 404, ok: false, arrayBuffer: async () => bodyOf("") });

const GGUF_MODEL = MODELS["qwen3-0.6b"];
const CPU_MODEL = MODELS["smollm2-135m"];

// ---------------------------------------------------------------- the probe
console.log("\nprobe: what counts as a working origin");
{
  ok("206 whose first four bytes are GGUF is accepted",
     (await probeGGUF("/m.gguf", { fetchImpl: stubFetch([["/", servesGGUF]]) })).ok);

  const ignored = await probeGGUF("/m.gguf", { fetchImpl: stubFetch([["/", ignoresRange]]) });
  ok("200 is rejected: an origin that ignores Range would send the whole file per tensor",
     !ignored.ok && /ignores Range/.test(ignored.why), ignored.why);

  // The failure a status code cannot see. A static host with an SPA fallback will
  // happily answer /models/anything.gguf with its index page, and without the magic
  // check that looks exactly like a healthy mirror until the GGUF parser chokes on
  // it several hundred megabytes later.
  const spa = await probeGGUF("/m.gguf", { fetchImpl: stubFetch([["/", spaFallback]]) });
  ok("206 whose body is HTML is rejected (SPA fallback, not a GGUF)",
     !spa.ok && /not GGUF/.test(spa.why), spa.why);

  const gone = await probeGGUF("/m.gguf", { fetchImpl: stubFetch([["/nope", servesGGUF]]) });
  ok("404 is rejected", !gone.ok && gone.status === 404);

  const threw = await probeGGUF("/m.gguf", { fetchImpl: async () => { throw new Error("offline"); } });
  ok("a network error is a failed probe, not an exception", !threw.ok && /offline/.test(threw.why));

  // Not a style preference: `Range: bytes=N-M` is a CORS-safelisted request header,
  // so a cross-origin probe skips preflight entirely. A suffix range (`bytes=-4`) or
  // a multi-range is not safelisted and would add an OPTIONS round trip in front of
  // every model load, on an origin that is under no obligation to answer it.
  const spy = stubFetch([["/", servesGGUF]]);
  await probeGGUF("/m.gguf", { fetchImpl: spy });
  ok("the probe emits a single closed byte range, which is CORS-safelisted",
     /^bytes=\d+-\d+$/.test(spy.calls[0].range), spy.calls[0].range);

  const manifest = await probeManifest("/models/x/manifest.json", {
    fetchImpl: stubFetch([["/", () => ({ status: 200, ok: true, json: async () => ({ shards: {} }) })]]),
  });
  ok("the CPU model's probe accepts a manifest with a shard index", manifest.ok);
  const notManifest = await probeManifest("/models/x/manifest.json", {
    fetchImpl: stubFetch([["/", () => ({ status: 200, ok: true, json: async () => ({}) })]]),
  });
  ok("a JSON document with no shard index is rejected", !notManifest.ok);
}

// ---------------------------------------------------------------- resolution
console.log("\nresolution: auto");
{
  clearDeliveryCache();
  const f = stubFetch([["/models/", servesGGUF]]);
  const r = await resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.AUTO, fetchImpl: f, cache: false });
  ok("a working mirror wins: same origin beats a CDN across the internet",
     r.delivery.origin === ORIGIN.MIRROR);
  ok("the resolved descriptor carries flat modelUrl/configUrl/tokenizerUrl",
     r.modelUrl.startsWith("/models/") && r.configUrl.endsWith("config.json") && r.tokenizerUrl.endsWith("tokenizer.json"));
  ok("only the mirror was probed; upstream was not touched",
     f.calls.length === 1 && f.calls[0].url.startsWith("/models/"), JSON.stringify(f.calls));
}
{
  clearDeliveryCache();
  const f = stubFetch([["/models/", notFound]]);
  const r = await resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.AUTO, fetchImpl: f, cache: false });
  ok("no mirror deployed falls through to huggingface.co",
     r.delivery.origin === ORIGIN.UPSTREAM);
  ok("the upstream URL is the pinned huggingface.co one",
     /^https:\/\/huggingface\.co\/Qwen\/Qwen3-0\.6B-GGUF\/resolve\/[0-9a-f]{40}\//.test(r.modelUrl), r.modelUrl);
  // The last candidate is taken on faith: there is nothing left to fall back to, so
  // probing it could only replace a specific load error with a vaguer one.
  ok("the last remaining origin is not probed", f.calls.length === 1);
}
{
  clearDeliveryCache();
  const f = stubFetch([["/models/", spaFallback]]);
  const r = await resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.AUTO, fetchImpl: f, cache: false });
  ok("a host that answers every path with its index page does not masquerade as a mirror",
     r.delivery.origin === ORIGIN.UPSTREAM);
}

console.log("\nresolution: forced");
{
  clearDeliveryCache();
  const f = stubFetch([["/", servesGGUF]]);
  const up = await resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.UPSTREAM, fetchImpl: f, cache: false });
  ok("prefer:upstream takes huggingface.co with no probe at all",
     up.delivery.origin === ORIGIN.UPSTREAM && f.calls.length === 0);

  const mi = await resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.MIRROR, fetchImpl: f, cache: false });
  ok("prefer:mirror takes the local mirror with no probe at all",
     mi.delivery.origin === ORIGIN.MIRROR && f.calls.length === 0);
}
{
  clearDeliveryCache();
  const cpu = await resolveDelivery(CPU_MODEL, { prefer: PREFERENCE.MIRROR, fetchImpl: stubFetch([]), cache: false });
  ok("the CPU model resolves to a shard directory, not a single file",
     cpu.dir === "/models/smollm2-135m" && cpu.modelUrl === null);

  // The case a CDN deployment actually hits. SmolLM2's per-layer f16 shards are
  // built locally by tools/fetch-model.mjs and published nowhere, so a build that
  // does not carry models/ genuinely cannot offer this model — and has to say so
  // rather than 404 halfway through a join.
  await throws("a mirror-only model cannot be forced to upstream",
    () => resolveDelivery(CPU_MODEL, { prefer: PREFERENCE.UPSTREAM, fetchImpl: stubFetch([]), cache: false }),
    /cannot be delivered/);

  const err = await resolveDelivery(CPU_MODEL, { prefer: PREFERENCE.UPSTREAM, fetchImpl: stubFetch([]), cache: false })
    .catch((e) => e);
  ok("the error names the real reason, not just a missing URL",
     err instanceof NoDeliveryOriginError && /published anywhere upstream/.test(err.message), err.message);

  const d = deliverableUnder(CPU_MODEL, PREFERENCE.UPSTREAM);
  ok("deliverableUnder answers the same question synchronously, for the model picker",
     !d.ok && /published anywhere upstream/.test(d.why), d.why);
  ok("deliverableUnder is permissive under auto, where a probe still has to run",
     deliverableUnder(CPU_MODEL, PREFERENCE.AUTO).ok);
  ok("a roadmap entry with no sources is never deliverable",
     !deliverableUnder(MODELS["qwen3.8-27b"], PREFERENCE.AUTO).ok);
}
{
  clearDeliveryCache();
  await throws("a model with no sources at all is refused rather than fetched",
    () => resolveDelivery(MODELS["qwen3.8-27b"], { fetchImpl: stubFetch([]), cache: false }),
    /cannot be delivered/);
}

console.log("\nresolution: what a bundle actually carries");
{
  // tools/build-static.mjs --models qwen3-0.6b ships one model. The registry still
  // declares a mirror path for all of them, because it describes the project rather
  // than one deployment of it — so without the bundle saying what it shipped, a
  // LAN build offers three models it does not have and 404s on whichever one gets
  // picked. Observed, then fixed: the picker listed all four.
  clearDeliveryCache();
  const carries = ["smollm2-135m"];
  await throws("a forced-mirror deployment refuses a model it did not bundle",
    () => resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.MIRROR, carries, fetchImpl: stubFetch([]), cache: false }),
    /did not bundle it/);

  ok("the picker says so synchronously too",
     !deliverableUnder(GGUF_MODEL, PREFERENCE.MIRROR, carries).ok);
  ok("a model the bundle does carry is still fine",
     deliverableUnder(CPU_MODEL, PREFERENCE.MIRROR, carries).ok);

  // Under auto the mirror is skipped for an unbundled model rather than probed --
  // the deployment already said it is not there, and a 404 per model load is a
  // request that can only ever fail.
  clearDeliveryCache();
  const f = stubFetch([["/models/", servesGGUF]]);
  const r = await resolveDelivery(GGUF_MODEL, { prefer: PREFERENCE.AUTO, carries, fetchImpl: f, cache: false });
  ok("under auto, an unbundled model goes straight to upstream without a doomed probe",
     r.delivery.origin === ORIGIN.UPSTREAM && f.calls.length === 0, JSON.stringify(f.calls));

  // The offline case with no way out: mirror-only, and this bundle did not ship it.
  await throws("a mirror-only model the bundle skipped has nowhere left to go",
    () => resolveDelivery(CPU_MODEL, { prefer: PREFERENCE.AUTO, carries: ["qwen3-0.6b"], fetchImpl: stubFetch([]), cache: false }),
    /did not bundle it/);
  ok("and the picker rules it out without a probe",
     !deliverableUnder(CPU_MODEL, PREFERENCE.AUTO, ["qwen3-0.6b"]).ok);

  ok("no claim (null) means no restriction, which is the dev-server case",
     deliverableUnder(GGUF_MODEL, PREFERENCE.MIRROR, null).ok);
}

console.log("\nresolution: caching");
{
  clearDeliveryCache();
  const f = stubFetch([["/models/", servesGGUF]]);
  await resolveDelivery(GGUF_MODEL, { fetchImpl: f });
  await resolveDelivery(GGUF_MODEL, { fetchImpl: f });
  await resolveDelivery(GGUF_MODEL, { fetchImpl: f });
  // A room re-resolves the same model on every deal and every re-plan. Re-probing
  // each time would put a pointless request in front of every recovery.
  ok("a repeated resolution is memoised, not re-probed", f.calls.length === 1, `${f.calls.length} calls`);

  clearDeliveryCache();
  const g = stubFetch([["/models/", notFound]]);
  await resolveDelivery(CPU_MODEL, { prefer: PREFERENCE.AUTO, fetchImpl: g }).catch(() => {});
  const h = stubFetch([["/models/", () => ({ status: 200, ok: true, json: async () => ({ shards: {} }) })]]);
  const retry = await resolveDelivery(CPU_MODEL, { prefer: PREFERENCE.AUTO, fetchImpl: h }).catch((e) => e);
  // A mirror that was not deployed yet when the page loaded must not be remembered
  // as permanently absent.
  ok("a failed resolution is not cached, so a retry can succeed",
     retry?.delivery?.origin === ORIGIN.MIRROR, String(retry?.message || retry));
  clearDeliveryCache();
}

// ---------------------------------------------------------------- registry invariants
console.log("\nregistry: the two origins have to be the same file");
{
  for (const [id, m] of Object.entries(MODELS)) {
    const up = sourcesFor(m, ORIGIN.UPSTREAM);
    if (!up) continue;
    const p = m.provenance;
    ok(`${id}: the upstream URL pins the revision its provenance records`,
       up.model.includes(`/resolve/${p.revision}/`), up.model);
    ok(`${id}: the upstream URL names the file its provenance records`,
       up.model.endsWith("/" + p.file), up.model);
    ok(`${id}: config.json and tokenizer.json come from the base repo, not the GGUF repo`,
       up.config.includes(`/${p.baseRepo}/`) && up.tokenizer.includes(`/${p.baseRepo}/`),
       `${up.config} / ${up.tokenizer}`);
    ok(`${id}: the base repo URLs are pinned too`,
       up.config.includes(`/resolve/${p.baseRevision}/`) && up.tokenizer.includes(`/resolve/${p.baseRevision}/`));
    // `main` is a mutable ref. Two devices in one room may resolve to different
    // origins, and if one of them followed a moving ref they could load different
    // tokenizers — mismatched token IDs on the wire, with nothing raising an error.
    ok(`${id}: no URL points at a mutable ref`,
       !Object.values(up).some((u) => /\/resolve\/(main|master)\//.test(u)));
    ok(`${id}: sourceRevision is generated from provenance and still parses`,
       m.sourceRevision.includes(p.revision) &&
       new RegExp(`${p.bytes}\\s*bytes`).test(m.sourceRevision) &&
       m.sourceRevision.includes("sha256:" + p.sha256), m.sourceRevision);

    const mir = sourcesFor(m, ORIGIN.MIRROR);
    ok(`${id}: the mirror is a same-origin path under /models/`,
       mir.model.startsWith("/models/") && mir.model.endsWith("/" + p.file), mir.model);
  }

  ok("the CPU model declares a mirror and nothing else",
     deliveryOrigins(CPU_MODEL).join() === ORIGIN.MIRROR);
  ok("the mirror-only model says why it is mirror-only", !!CPU_MODEL.mirrorOnlyReason);
  ok("every GGUF model prefers the mirror and falls back to upstream",
     ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"].every(
       (id) => deliveryOrigins(MODELS[id]).join() === `${ORIGIN.MIRROR},${ORIGIN.UPSTREAM}`));
  ok("origins have human labels for the UI",
     originLabel(ORIGIN.UPSTREAM) === "Hugging Face CDN" && originLabel(ORIGIN.MIRROR) === "this host");
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
