// Pre-flight for promoting a model rung: does this GGUF actually contain what the
// registry claims, and what will it cost to hold?
//
//   node tools/probe-gguf.mjs qwen3-1.7b
//
// Runs the whole read-only half of a promotion gate in Node, before a browser is
// ever opened -- which is the point. A wrong head count or a mis-parsed vocab
// size produces plausible-looking but wrong language rather than a crash, and
// finding that in a two-device WebRTC session is enormously more expensive than
// finding it here.
//
// Three independent sources have to agree:
//   1. the GGUF file's own header (what the engine will actually load)
//   2. the model's published config.json (what the authors say the model is)
//   3. models/registry.mjs (what this project claims about it)
// Any disagreement is a hard failure -- see the exit code.

import { readFile, stat } from "node:fs/promises";
import { open } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { parseGGUFHeader } from "../engine/upstream/gguf.js";
import { cfgFromGGUFMeta, modelSpecFromGGUF, validateDescriptor } from "../engine/gpu-adapter.mjs";
import { chooseEncoding } from "../room/wire.js";
import { getModel, sourcesFor, ORIGIN } from "../models/registry.mjs";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const HEADER_PROBE_BYTES = 16 * 2 ** 20;   // same window engine/gpu-adapter.mjs uses

const id = process.argv[2];
if (!id) {
  console.error("usage: node tools/probe-gguf.mjs <model-id>   (e.g. qwen3-1.7b)");
  process.exit(2);
}
const withHash = process.argv.includes("--hash");

let pass = 0, fail = 0;
const ok = (name, cond, detail = "") => {
  if (cond) { pass++; console.log("  ok   " + name + (detail ? "   " + detail : "")); }
  else { fail++; console.log("  FAIL " + name + (detail ? "\n         " + detail : "")); }
};
const MB = (b) => (b / 2 ** 20).toFixed(1) + " MB";

const descriptor = getModel(id);
console.log(`\n${descriptor.label}  (${descriptor.id})`);
console.log(`status: ${descriptor.status}   engineKind: ${descriptor.engineKind}\n`);

// This tool reads from disk, so it is a check on the local mirror specifically --
// the upstream origin is checked over the network by tools/verify-delivery.mjs
// instead. A model with no mirror declared has nothing here to read.
const mirror = sourcesFor(descriptor, ORIGIN.MIRROR);
if (!mirror?.model) {
  console.error(`${descriptor.id} declares no local mirror` +
    (descriptor.sources ? " -- run tools/verify-delivery.mjs to check its upstream origin instead."
                        : " -- it is a roadmap entry, not a loadable model."));
  process.exit(2);
}
const ggufPath = join(ROOT, mirror.model.replace(/^\//, ""));

// ---------------------------------------------------------------- file identity
console.log("file identity");
const info = await stat(ggufPath).catch(() => null);
ok("the GGUF exists on disk", !!info, ggufPath);
if (!info) process.exit(1);
console.log(`  ..   ${info.size} bytes (${MB(info.size)})`);

// sourceRevision is free-form prose, but it is supposed to carry the real byte
// length and hash. If it names a length, it must be this file's length.
const claimedLen = /(\d{6,})\s*bytes/.exec(descriptor.sourceRevision || "");
if (claimedLen) {
  ok("registry sourceRevision's byte length matches the file",
     Number(claimedLen[1]) === info.size, `registry says ${claimedLen[1]}, file is ${info.size}`);
}
const claimedHash = /sha256:([0-9a-f]{64})/.exec(descriptor.sourceRevision || "");
if (claimedHash && withHash) {
  const h = createHash("sha256");
  const fh = await open(ggufPath, "r");
  for await (const chunk of fh.createReadStream()) h.update(chunk);
  await fh.close();
  const got = h.digest("hex");
  ok("registry sourceRevision's sha256 matches the file", got === claimedHash[1], `file is sha256:${got}`);
} else if (claimedHash) {
  console.log("  ..   sha256 in registry; re-run with --hash to verify it (reads the whole file)");
}

// ---------------------------------------------------------------- header
console.log("\nGGUF header -> cfg");
const fh = await open(ggufPath, "r");
const buf = Buffer.alloc(Math.min(HEADER_PROBE_BYTES, info.size));
await fh.read(buf, 0, buf.length, 0);
await fh.close();
const header = parseGGUFHeader(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), { skipTokenizer: true });
const cfg = cfgFromGGUFMeta(header, { archHint: descriptor.architecture });
console.log(`  ..   architecture ${cfg.architecture}, ${Object.keys(header.tensors).length} tensors`);

try {
  validateDescriptor(descriptor, header, cfg);
  ok("descriptor validates against the file (architecture + quantization)", true);
} catch (e) {
  ok("descriptor validates against the file (architecture + quantization)", false, e.message);
}

// ---------------------------------------------------------------- vs config.json
console.log("\nGGUF header vs the model's own config.json");
const cfgPath = mirror.config ? join(ROOT, mirror.config.replace(/^\//, "")) : null;
const published = cfgPath ? await readFile(cfgPath, "utf8").then(JSON.parse).catch(() => null) : null;

if (!published) {
  console.log("  ..   no local config.json to compare against (skipped)");
} else {
  // Every field the engine actually depends on. A mismatch here is the failure
  // mode that produces fluent nonsense rather than an error.
  // Integer fields must match exactly. The two float fields must not: GGUF stores
  // them as f32 while config.json is JSON (f64), so a config.json 1e-6 arrives
  // from the GGUF as 9.999999974752427e-7 -- the same number to f32 precision, and
  // comparing those with === reports a failure that is really a rounding artifact.
  const pairs = [
    ["hidden_size", cfg.hidden_size, published.hidden_size, "int"],
    ["num_hidden_layers", cfg.num_hidden_layers, published.num_hidden_layers, "int"],
    ["intermediate_size", cfg.intermediate_size, published.intermediate_size, "int"],
    ["num_attention_heads", cfg.num_attention_heads, published.num_attention_heads, "int"],
    ["num_key_value_heads", cfg.num_key_value_heads, published.num_key_value_heads, "int"],
    ["head_dim", cfg.head_dim, published.head_dim, "int"],
    ["vocab_size", cfg.vocab_size, published.vocab_size, "int"],
    ["rms_norm_eps", cfg.rms_norm_eps, published.rms_norm_eps, "f32"],
    ["rope_theta", cfg.rope_theta, published.rope_theta, "f32"],
  ];
  const f32Equal = (a, b) => Math.fround(a) === Math.fround(b) ||
    Math.abs(a - b) <= 1e-6 * Math.max(Math.abs(a), Math.abs(b));
  for (const [name, got, want, kind] of pairs) {
    if (want === undefined) { console.log(`  ..   ${name}: ${got} (config.json does not declare it)`); continue; }
    const same = kind === "f32" ? f32Equal(got, want) : got === want;
    const note = kind === "f32" && same && got !== want ? " (equal to f32 precision)" : "";
    ok(`${name} agrees${note}`, same, `GGUF says ${got}, config.json says ${want}`);
  }
  const tied = !header.tensors["output.weight"];
  ok("tied embeddings agree", tied === !!published.tie_word_embeddings,
     `GGUF ${tied ? "has no" : "has an"} output.weight, config.json says tie_word_embeddings=${published.tie_word_embeddings}`);
}

// ---------------------------------------------------------------- scheduler spec
console.log("\nmodelSpecFromGGUF (what the scheduler will plan against)");
const maxSeq = descriptor.maxSeqDefault || 512;
const spec = modelSpecFromGGUF(header, cfg, descriptor, { maxSeq });
const sane = (n) => Number.isFinite(n) && n > 0;
ok("layerBytes is a real, positive size", sane(spec.layerBytes), MB(spec.layerBytes) + " per layer");
ok("embedBytes is a real, positive size", sane(spec.embedBytes), MB(spec.embedBytes));
ok("kvBytesPerLayer scales with maxSeq", sane(spec.kvBytesPerLayer), MB(spec.kvBytesPerLayer) + ` per layer at maxSeq=${maxSeq}`);
ok("layerMACs / headMACs are positive", sane(spec.layerMACs) && sane(spec.headMACs));
ok("spec.layers matches the header's block_count", spec.layers === cfg.num_hidden_layers);

// The sum has to land back on the file's own size, or something is being missed.
const weightsAccounted = spec.layerBytes * spec.layers + spec.embedBytes;
const covered = weightsAccounted / info.size;
ok("per-layer + embedding bytes account for most of the file",
   covered > 0.95 && covered <= 1.0,
   `${MB(weightsAccounted)} of ${MB(info.size)} = ${(covered * 100).toFixed(1)}% (norms/rope are the remainder)`);

// ---------------------------------------------------------------- what it costs to run
console.log("\ncapacity envelope");
const wire = chooseEncoding(spec.hidden);
console.log(`  ..   hidden ${spec.hidden} -> wire encoding ${wire}` +
  (wire === "f16" ? "  (LOSSY: split output is not guaranteed bit-identical to solo)"
                  : "  (lossless: split output should be bit-identical to solo)"));
if (descriptor.wireDtype && descriptor.wireDtype !== wire) {
  ok("registry wireDtype matches what chooseEncoding actually picks", false,
     `registry says ${descriptor.wireDtype}, chooseEncoding(${spec.hidden}) returns ${wire}`);
} else if (descriptor.wireDtype) {
  ok("registry wireDtype matches what chooseEncoding actually picks", true);
}

const soloWeights = spec.layerBytes * spec.layers + spec.embedBytes;
const soloKV = spec.kvBytesPerLayer * spec.layers;
const solo = soloWeights + soloKV + spec.scratchBytes;
console.log(`  ..   solo (all ${spec.layers} layers + embed + KV + scratch): ${MB(solo)}`);
for (const n of [2, 3]) {
  const per = Math.ceil(spec.layers / n);
  const host = per * (spec.layerBytes + spec.kvBytesPerLayer) + spec.embedBytes + spec.scratchBytes;
  const worker = per * (spec.layerBytes + spec.kvBytesPerLayer) + spec.scratchBytes;
  console.log(`  ..   ${n}-way even split: host ~${MB(host)} (holds the embedding table), worker ~${MB(worker)}`);
}
if (descriptor.minRoomEnvelopeBytes) {
  ok("registry minRoomEnvelopeBytes is at least the real solo cost",
     descriptor.minRoomEnvelopeBytes >= solo,
     `registry says ${MB(descriptor.minRoomEnvelopeBytes)}, real solo cost is ${MB(solo)}`);
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
