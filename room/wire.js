// Wire format for activations moving between peers.
//
// Two jobs: choose a precision for the hidden state, and cut every send into slices
// small enough to clear Chrome's SCTP stack in one go.
//
// Precision is chosen per model rather than fixed at f16 -- see chooseEncoding below.
// f16 is lossy, and lossy activations make the answer depend on where the model was
// split, which quietly destroys the property this project most wants to guarantee.
//
// The slice size is not arbitrary. Chrome's dcSCTP releases at most ~4 packets per
// send opportunity and opens with a ~12 KB congestion window, so a single message
// larger than roughly 4-5 KB costs an *extra round trip* on every hop -- on a 100 ms
// link that is ~200 ms per lap, which dwarfs anything the kernels can win back.
// Slicing under that threshold keeps a hop at one one-way trip.
//
// Exact by construction: only the packaging changes, never the values.

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

export function f32ToF16(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let man = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);   // Inf / NaN
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;                          // overflow -> Inf
  if (e <= 0) {
    if (e < -10) return sign;                                   // underflow -> signed zero
    man |= 0x800000;
    const shift = 14 - e;
    let h = man >>> shift;
    if ((man >>> (shift - 1)) & 1) h += 1;                      // round to nearest
    return sign | h;
  }
  let h = (e << 10) | (man >>> 13);
  if (man & 0x1000) h += 1;                                     // guard bit; carry into exp is correct
  return sign | h;
}

export function f16ToF32(h) {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const man = h & 0x3ff;
  if (exp === 0) {
    if (man === 0) { u32buf[0] = sign; return f32buf[0]; }
    let e = -1, m = man;
    do { e++; m <<= 1; } while (!(m & 0x400));                  // normalize the subnormal
    u32buf[0] = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    return f32buf[0];
  }
  if (exp === 0x1f) { u32buf[0] = sign | 0x7f800000 | (man << 13); return f32buf[0]; }
  u32buf[0] = sign | ((exp - 15 + 127) << 23) | (man << 13);
  return f32buf[0];
}

export function packF16(f) {
  const out = new Uint16Array(f.length);
  for (let i = 0; i < f.length; i++) out[i] = f32ToF16(f[i]);
  return out;
}

export function unpackF16(u) {
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = f16ToF32(u[i]);
  return out;
}

// A NaN loose in a hidden state poisons every downstream layer and shows up as
// garbage text three hops later, by which point it is very hard to attribute.
//
// Scan every element. A strided spot-check is the traditional trick here, but at
// 576-5120 floats a full scan costs single-digit microseconds against a network hop
// measured in milliseconds -- and a stride silently misses exactly the single-element
// corruption that is hardest to debug. Not a trade worth making.
export function looksBad(a) {
  if (!a.length) return true;
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return true;
  return false;
}

// ---------------------------------------------------------------- framing

export const SLICE_BYTES = 4600;          // ~4 SCTP packets of 1150 B payload
const HDR = 24;
const MAGIC = 0x4153;                     // "AS"
export const KINDS = ["hidden", "hidden-b", "hidden-ret", "hidden-ret-b"];
const FLAG_F32 = 2;                       // flags bit 1: payload is f32, not f16

// Pick the wire precision for a given hidden size.
//
// f16 halves the bytes, but that only buys anything if it removes a slice -- latency
// on a hop is set by how many SCTP send opportunities the frame needs, not by its
// size. And f16 is lossy: it perturbs the hidden state, so a room that splits the
// model at a different layer produces a different token stream from the same prompt.
// That silently breaks the one property worth guaranteeing, that a split answer
// equals the single-device answer.
//
// So: send f32 whenever it costs no extra slice, and only pay the precision when the
// bytes genuinely buy a round trip back.
//
//   dim 576  (SmolLM2 135M)   2304 B f32 -> 1 slice, same as f16. Free.
//   dim 1024 (Qwen3 0.6B)     4096 B f32 -> 1 slice, same as f16. Free.
//   dim 2048 (Qwen3 1.7B)     8192 B f32 -> 2 slices vs 1. f16 wins.
export function chooseEncoding(dim) {
  const per = SLICE_BYTES - HDR;
  return Math.ceil((dim * 4) / per) === Math.ceil((dim * 2) / per) ? "f32" : "f16";
}

// Pack a hidden state for the wire at the chosen precision.
export function packWire(f, enc) {
  return enc === "f32" ? f : packF16(f);
}

export function unpackWire(data, enc) {
  return enc === "f32" ? (data instanceof Float32Array ? data : new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4)) : unpackF16(data);
}

// msg: { t, pos, n?, flags?, data: Uint16Array (f16) | Float32Array (f32) }
// The payload type decides the encoding; the receiver is told via a header flag, so
// mixed-precision peers interoperate without negotiation.
// Returns an array of ArrayBuffers, each small enough to leave in one send.
export function encodeFrame(msg, msgId) {
  const kind = KINDS.indexOf(msg.t);
  if (kind < 0) throw new Error("not a wire kind: " + msg.t);
  const isF32 = msg.data instanceof Float32Array;
  const flags = (msg.flags || 0) | (isF32 ? FLAG_F32 : 0);
  const src = msg.data;
  const bytes = new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
  const per = SLICE_BYTES - HDR;
  const nSlices = Math.max(1, Math.ceil(bytes.length / per));
  const out = [];
  for (let k = 0, off = 0; k < nSlices; k++) {
    const len = Math.min(per, bytes.length - off);
    const buf = new ArrayBuffer(HDR + len);
    const dv = new DataView(buf);
    dv.setUint16(0, MAGIC);
    dv.setUint8(2, kind);
    dv.setUint8(3, flags);
    dv.setUint32(4, msgId >>> 0);
    dv.setUint32(8, msg.pos >>> 0);
    dv.setUint16(12, msg.n || 1);
    dv.setUint16(14, k);
    dv.setUint16(16, nSlices);
    dv.setUint16(18, 0);
    dv.setUint32(20, bytes.length);
    new Uint8Array(buf, HDR).set(bytes.subarray(off, off + len));
    off += len;
    out.push(buf);
  }
  return out;
}

export function makeReassembler() {
  // `done` remembers recently completed message ids so a retransmitted slice cannot
  // start a second reassembly of a message we already delivered. Ordered+reliable
  // channels do not duplicate, but the striping path deliberately is not, and a
  // double-delivered activation would advance a recurrent state twice.
  return { rx: new Map(), done: new Set() };
}

const DONE_MEMORY = 256;

// Feed every received ArrayBuffer here; returns a complete message or null.
export function decodeSlice(rs, buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < HDR) return null;
  const dv = new DataView(buf);
  if (dv.getUint16(0) !== MAGIC) return null;
  const kind = dv.getUint8(2), flags = dv.getUint8(3);
  const id = dv.getUint32(4), pos = dv.getUint32(8), n = dv.getUint16(12);
  const k = dv.getUint16(14), nSlices = dv.getUint16(16), total = dv.getUint32(20);

  if (rs.done.has(id)) return null;                      // late slice of a delivered message

  let r = rs.rx.get(id);
  if (!r) {
    r = { seen: new Uint8Array(nSlices), got: 0, need: nSlices, buf: new Uint8Array(total), t: performance.now() };
    rs.rx.set(id, r);
  }
  if (r.seen[k]) return null;                            // duplicate slice
  r.seen[k] = 1;
  r.got++;
  r.buf.set(new Uint8Array(buf, HDR), k * (SLICE_BYTES - HDR));
  if (r.got < r.need) {
    // a lost slice must not pin memory forever
    if (rs.rx.size > 64) {
      const now = performance.now();
      for (const [i, v] of rs.rx) if (now - v.t > 30000) rs.rx.delete(i);
    }
    return null;
  }
  rs.rx.delete(id);
  rs.done.add(id);
  if (rs.done.size > DONE_MEMORY) rs.done.delete(rs.done.values().next().value);
  const f32 = !!(flags & FLAG_F32);
  const data = f32
    ? new Float32Array(r.buf.buffer, 0, total >> 2)
    : new Uint16Array(r.buf.buffer, 0, total >> 1);
  return { t: KINDS[kind], pos, n, flags: flags & ~FLAG_F32, enc: f32 ? "f32" : "f16", data };
}
