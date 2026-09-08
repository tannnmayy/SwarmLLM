// Fetch a model and reshape it into per-layer shards this engine can load.
//
//   node tools/fetch-model.mjs [smollm2-135m]
//
// Why shard by layer: a device should download only the layers it was dealt, and
// nothing else. One file per layer makes that a plain GET, with no range-request
// bookkeeping and no partial-cache edge cases.
//
// Why f16 on disk: SmolLM2 ships bf16, which WebGPU cannot read. bf16 -> f16 is
// exact for every value inside f16's range, because f16 actually has *more*
// mantissa (10 bits vs 7); only the exponent range narrows, and weights live far
// inside it. Half the bytes of f32 with no loss that matters.

import { mkdir, writeFile, open, stat } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));

const MODELS = {
  "smollm2-135m": {
    repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
    file: "model.safetensors",
    label: "SmolLM2 135M Instruct",
  },
};

const key = process.argv[2] || "smollm2-135m";
const M = MODELS[key];
if (!M) {
  console.error("unknown model: " + key + " (have: " + Object.keys(MODELS).join(", ") + ")");
  process.exit(1);
}

const OUT = join(ROOT, "models", key);
const hf = (f) => `https://huggingface.co/${M.repo}/resolve/main/${f}`;

await mkdir(OUT, { recursive: true });

// ---------------------------------------------------------------- download
const raw = join(OUT, "_source.safetensors");
async function download(url, dest) {
  try {
    const s = await stat(dest);
    if (s.size > 1e6) { console.log(`  have ${dest} (${(s.size / 2 ** 20).toFixed(0)} MB), skipping download`); return; }
  } catch {}
  process.stdout.write("  downloading " + url + "\n  ");
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
  const total = Number(res.headers.get("content-length") || 0);
  const fh = await open(dest, "w");
  let got = 0, lastPct = -1;
  for await (const chunk of res.body) {
    await fh.write(chunk);
    got += chunk.length;
    const pct = total ? Math.floor(got / total * 100) : -1;
    if (pct !== lastPct && pct % 5 === 0) { process.stdout.write(pct + "% "); lastPct = pct; }
  }
  await fh.close();
  process.stdout.write("done\n");
}

console.log(`\n${M.label}\n`);
await download(hf("config.json"), join(OUT, "config.json"));
await download(hf("tokenizer.json"), join(OUT, "tokenizer.json"));
await download(hf(M.file), raw);

const cfg = JSON.parse(await (await import("node:fs/promises")).readFile(join(OUT, "config.json"), "utf8"));

// ---------------------------------------------------------------- safetensors
const fh = await open(raw, "r");
const lenBuf = Buffer.alloc(8);
await fh.read(lenBuf, 0, 8, 0);
const headerLen = Number(lenBuf.readBigUInt64LE(0));
const headerBuf = Buffer.alloc(headerLen);
await fh.read(headerBuf, 0, headerLen, 8);
const header = JSON.parse(headerBuf.toString("utf8"));
const DATA0 = 8 + headerLen;

const names = Object.keys(header).filter((k) => k !== "__metadata__");
console.log(`  ${names.length} tensors, header ${headerLen} B`);
const dtypes = new Set(names.map((n) => header[n].dtype));
console.log(`  dtypes: ${[...dtypes].join(", ")}`);

async function readTensor(name) {
  const t = header[name];
  if (!t) throw new Error("missing tensor: " + name);
  const [s, e] = t.data_offsets;
  const buf = Buffer.alloc(e - s);
  await fh.read(buf, 0, e - s, DATA0 + s);
  return { buf, shape: t.shape, dtype: t.dtype };
}

// bf16 is the top 16 bits of an f32, so widening is a shift. Then narrow to f16.
const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
function bf16ToF32(h) { u32buf[0] = h << 16; return f32buf[0]; }

function f32ToF16(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let man = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    man |= 0x800000;
    const shift = 14 - e;
    let h = man >>> shift;
    if ((man >>> (shift - 1)) & 1) h += 1;
    return sign | h;
  }
  let h = (e << 10) | (man >>> 13);
  if (man & 0x1000) h += 1;
  return sign | h;
}

let clipped = 0, flushed = 0, total = 0;
function toF16(t) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  const out = new Uint16Array(n);
  if (t.dtype === "BF16") {
    const src = new Uint16Array(t.buf.buffer, t.buf.byteOffset, n);
    for (let i = 0; i < n; i++) {
      const v = bf16ToF32(src[i]);
      const h = f32ToF16(v);
      if (!Number.isFinite(v)) { /* keep */ }
      else if ((h & 0x7c00) === 0x7c00) clipped++;
      else if (h === 0 && v !== 0) flushed++;
      out[i] = h;
    }
  } else if (t.dtype === "F32") {
    const src = new Float32Array(t.buf.buffer, t.buf.byteOffset, n);
    for (let i = 0; i < n; i++) {
      const h = f32ToF16(src[i]);
      if ((h & 0x7c00) === 0x7c00 && Number.isFinite(src[i])) clipped++;
      else if (h === 0 && src[i] !== 0) flushed++;
      out[i] = h;
    }
  } else if (t.dtype === "F16") {
    out.set(new Uint16Array(t.buf.buffer, t.buf.byteOffset, n));
  } else {
    throw new Error("unhandled dtype " + t.dtype);
  }
  total += n;
  return out;
}

// ---------------------------------------------------------------- shard
// Every layer holds the same tensors in the same order, so the engine can slice a
// single ArrayBuffer by offset without consulting a per-layer index.
const LAYER_TENSORS = [
  ["input_layernorm.weight", "inNorm"],
  ["self_attn.q_proj.weight", "wq"],
  ["self_attn.k_proj.weight", "wk"],
  ["self_attn.v_proj.weight", "wv"],
  ["self_attn.o_proj.weight", "wo"],
  ["post_attention_layernorm.weight", "postNorm"],
  ["mlp.gate_proj.weight", "wGate"],
  ["mlp.up_proj.weight", "wUp"],
  ["mlp.down_proj.weight", "wDown"],
];

async function writeShard(file, parts) {
  let bytes = 0;
  for (const p of parts) bytes += p.data.byteLength;
  const out = Buffer.alloc(bytes);
  const index = [];
  let off = 0;
  for (const p of parts) {
    Buffer.from(p.data.buffer, p.data.byteOffset, p.data.byteLength).copy(out, off);
    index.push({ name: p.name, shape: p.shape, offset: off, length: p.data.length });
    off += p.data.byteLength;
  }
  await writeFile(join(OUT, file), out);
  return { file, bytes, index };
}

const manifest = {
  key,
  label: M.label,
  source: `${M.repo}/${M.file}`,
  dtype: "f16",
  config: {
    hiddenSize: cfg.hidden_size,
    layers: cfg.num_hidden_layers,
    heads: cfg.num_attention_heads,
    kvHeads: cfg.num_key_value_heads,
    headDim: cfg.hidden_size / cfg.num_attention_heads,
    intermediate: cfg.intermediate_size,
    vocab: cfg.vocab_size,
    rmsEps: cfg.rms_norm_eps,
    ropeTheta: cfg.rope_theta,
    tiedEmbeddings: !!cfg.tie_word_embeddings,
    bos: cfg.bos_token_id,
    eos: cfg.eos_token_id,
    maxPos: cfg.max_position_embeddings,
  },
  shards: {},
};

console.log("\n  sharding");
{
  const t = await readTensor("model.embed_tokens.weight");
  manifest.shards.embed = await writeShard("embed.bin", [{ name: "embed", shape: t.shape, data: toF16(t) }]);
  console.log(`    embed.bin  ${(manifest.shards.embed.bytes / 2 ** 20).toFixed(1)} MB  [${t.shape}]`);
}

manifest.shards.layers = [];
for (let i = 0; i < cfg.num_hidden_layers; i++) {
  const parts = [];
  for (const [suffix, short] of LAYER_TENSORS) {
    const t = await readTensor(`model.layers.${i}.${suffix}`);
    parts.push({ name: short, shape: t.shape, data: toF16(t) });
  }
  const s = await writeShard(`layer-${String(i).padStart(2, "0")}.bin`, parts);
  manifest.shards.layers.push(s);
  if (i === 0) console.log(`    layer-00.bin  ${(s.bytes / 2 ** 20).toFixed(2)} MB  (x${cfg.num_hidden_layers})`);
}

{
  const t = await readTensor("model.norm.weight");
  const parts = [{ name: "finalNorm", shape: t.shape, data: toF16(t) }];
  if (!cfg.tie_word_embeddings && header["lm_head.weight"]) {
    const h = await readTensor("lm_head.weight");
    parts.push({ name: "lmHead", shape: h.shape, data: toF16(h) });
  }
  manifest.shards.final = await writeShard("final.bin", parts);
  console.log(`    final.bin  ${(manifest.shards.final.bytes / 1024).toFixed(1)} KB`);
}

await fh.close();

const layerBytes = manifest.shards.layers[0].bytes;
manifest.bytes = {
  embed: manifest.shards.embed.bytes,
  perLayer: layerBytes,
  final: manifest.shards.final.bytes,
  total: manifest.shards.embed.bytes + layerBytes * cfg.num_hidden_layers + manifest.shards.final.bytes,
};
await writeFile(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));

console.log(`
  converted ${(total / 1e6).toFixed(1)}M values to f16
  ${clipped} overflowed to Inf, ${flushed} flushed to zero${clipped ? "   <-- INVESTIGATE" : ""}

  total  ${(manifest.bytes.total / 2 ** 20).toFixed(1)} MB
  embed  ${(manifest.bytes.embed / 2 ** 20).toFixed(1)} MB   (host only)
  layer  ${(layerBytes / 2 ** 20).toFixed(2)} MB x ${cfg.num_hidden_layers}

  a device dealt 8 layers downloads ${(layerBytes * 8 / 2 ** 20).toFixed(1)} MB
  models/${key}/manifest.json written
`);
