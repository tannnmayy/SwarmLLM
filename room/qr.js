// A QR encoder, in about 200 lines.
//
// Scanning a code is the difference between "everyone in the room can join" and
// "everyone in the room can join if they can type an IP address and accept a
// certificate warning". It is the single largest piece of friction between a person
// and a working swarm, so it is worth writing rather than importing.
//
// No CDN, no dependency: a venue's Wi-Fi may have no route to the internet at all,
// and a join code that needs a download to render is a join code that fails exactly
// when it matters.
//
// Byte mode, error correction level M (~15% recoverable), versions 1-10, which is
// 216 data codewords -- comfortably more than any join URL. Verified by generating
// codes and decoding them back with OpenCV's detector; see tests/qr.test.mjs.

// ---------------------------------------------------------------- GF(256)
// Reed-Solomon works over a field of 256 elements, built with the primitive
// polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D) that the QR spec mandates.
const EXP = new Uint8Array(512);
const LOG = new Uint8Array(256);
{
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP[i] = x;
    LOG[x] = i;
    x <<= 1;
    if (x & 0x100) x ^= 0x11d;
  }
  for (let i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
}
const mul = (a, b) => (a === 0 || b === 0 ? 0 : EXP[LOG[a] + LOG[b]]);

// The generator polynomial for n error-correction codewords is
// (x - a^0)(x - a^1)...(x - a^(n-1)), expanded in GF(256).
function generator(n) {
  let g = [1];
  for (let i = 0; i < n; i++) {
    const next = new Array(g.length + 1).fill(0);
    for (let j = 0; j < g.length; j++) {
      // g is stored leading-coefficient first, so multiplying by (x + a^i) shifts
      // each term up by one and adds a^i times itself in place. Swapping these two
      // lines builds the polynomial backwards -- (x+a^0)(x+a^1) comes out [2,3,1]
      // instead of [1,3,2] -- and every error-correction codeword is then wrong,
      // which a decoder rejects outright.
      next[j] ^= g[j];
      next[j + 1] ^= mul(g[j], EXP[i]);
    }
    g = next;
  }
  return g;
}

function ecc(data, n) {
  const g = generator(n);
  const rem = new Uint8Array(data.length + n);
  rem.set(data);
  for (let i = 0; i < data.length; i++) {
    const factor = rem[i];
    if (!factor) continue;
    for (let j = 0; j < g.length; j++) rem[i + j] ^= mul(g[j], factor);
  }
  return rem.slice(data.length);
}

// ---------------------------------------------------------------- tables (level M)
// [ec codewords per block, group1 blocks, group1 data cw, group2 blocks, group2 data cw]
const SPEC = {
  1:  [10, 1, 16, 0, 0],
  2:  [16, 1, 28, 0, 0],
  3:  [26, 1, 44, 0, 0],
  4:  [18, 2, 32, 0, 0],
  5:  [24, 2, 43, 0, 0],
  6:  [16, 4, 27, 0, 0],
  7:  [18, 4, 31, 0, 0],
  8:  [22, 2, 38, 2, 39],
  9:  [22, 3, 36, 2, 37],
  10: [26, 4, 43, 1, 44],
};
const ALIGN = {
  1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30],
  6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
};
// 15-bit format strings for level M, masks 0-7 (BCH-encoded and XOR-masked already)
const FORMAT = [
  0x5412, 0x5125, 0x5e7c, 0x5b4b, 0x45f9, 0x40ce, 0x4f97, 0x4aa0,
];
// 18-bit version information, versions 7-10
const VERSION_INFO = { 7: 0x07c94, 8: 0x085bc, 9: 0x09a99, 10: 0x0a4d3 };

const dataCapacity = (v) => {
  const [, g1, d1, g2, d2] = SPEC[v];
  return g1 * d1 + g2 * d2;
};

// ---------------------------------------------------------------- encode
export function encode(text) {
  const bytes = new TextEncoder().encode(text);

  // Smallest version that fits. The 4-bit mode indicator plus the character count
  // (8 bits below version 10, 16 bits at 10 and above) ride with the data.
  let version = 0;
  for (let v = 1; v <= 10; v++) {
    const countBits = v < 10 ? 8 : 16;
    if (Math.ceil((4 + countBits + bytes.length * 8) / 8) <= dataCapacity(v)) { version = v; break; }
  }
  if (!version) throw new Error(`too long for a version-10 QR: ${bytes.length} bytes`);

  const [ecPerBlock, g1, d1, g2, d2] = SPEC[version];
  const capacity = dataCapacity(version);

  // ---- bit stream
  const bits = [];
  const push = (val, len) => { for (let i = len - 1; i >= 0; i--) bits.push((val >> i) & 1); };
  push(0b0100, 4);                                    // byte mode
  push(bytes.length, version < 10 ? 8 : 16);
  for (const b of bytes) push(b, 8);
  for (let i = 0; i < 4 && bits.length < capacity * 8; i++) bits.push(0);   // terminator
  while (bits.length % 8) bits.push(0);
  const codewords = [];
  for (let i = 0; i < bits.length; i += 8) {
    codewords.push(bits.slice(i, i + 8).reduce((a, b) => (a << 1) | b, 0));
  }
  // pad alternately with the two bytes the spec names
  for (let i = 0; codewords.length < capacity; i++) codewords.push(i % 2 ? 0x11 : 0xec);

  // ---- split into blocks, compute ECC per block
  const blocks = [], eccs = [];
  let at = 0;
  for (let i = 0; i < g1; i++) { blocks.push(codewords.slice(at, at + d1)); at += d1; }
  for (let i = 0; i < g2; i++) { blocks.push(codewords.slice(at, at + d2)); at += d2; }
  for (const b of blocks) eccs.push(ecc(Uint8Array.from(b), ecPerBlock));

  // ---- interleave: one codeword from each block in turn, data then ECC
  const out = [];
  const maxData = Math.max(d1, d2);
  for (let i = 0; i < maxData; i++) for (const b of blocks) if (i < b.length) out.push(b[i]);
  for (let i = 0; i < ecPerBlock; i++) for (const e of eccs) out.push(e[i]);

  return { modules: place(version, out), version, size: 17 + version * 4 };
}

// ---------------------------------------------------------------- matrix
function place(version, codewords) {
  const size = 17 + version * 4;
  const m = Array.from({ length: size }, () => new Int8Array(size).fill(-1));   // -1 = free
  const set = (r, c, v) => { if (r >= 0 && c >= 0 && r < size && c < size) m[r][c] = v; };

  // finder patterns, with their separators
  for (const [br, bc] of [[0, 0], [0, size - 7], [size - 7, 0]]) {
    for (let r = -1; r <= 7; r++) for (let c = -1; c <= 7; c++) {
      const inner = r >= 2 && r <= 4 && c >= 2 && c <= 4;
      const ring = r === 0 || r === 6 || c === 0 || c === 6;
      const on = (r >= 0 && r <= 6 && c >= 0 && c <= 6) && (ring || inner);
      set(br + r, bc + c, on ? 1 : 0);
    }
  }
  // timing patterns
  for (let i = 8; i < size - 8; i++) { m[6][i] = i % 2 ? 0 : 1; m[i][6] = i % 2 ? 0 : 1; }
  // alignment patterns, skipping the three that would collide with finders
  const ap = ALIGN[version];
  for (const r of ap) for (const c of ap) {
    if ((r === 6 && c === 6) || (r === 6 && c === size - 7) || (r === size - 7 && c === 6)) continue;
    for (let dr = -2; dr <= 2; dr++) for (let dc = -2; dc <= 2; dc++) {
      const edge = Math.max(Math.abs(dr), Math.abs(dc));
      set(r + dr, c + dc, edge === 1 ? 0 : 1);
    }
  }
  m[size - 8][8] = 1;                                  // the always-dark module

  // reserve format areas so data placement skips them
  for (let i = 0; i < 9; i++) { if (m[8][i] === -1) m[8][i] = 0; if (m[i][8] === -1) m[i][8] = 0; }
  for (let i = 0; i < 8; i++) { if (m[8][size - 1 - i] === -1) m[8][size - 1 - i] = 0; if (m[size - 1 - i][8] === -1) m[size - 1 - i][8] = 0; }
  if (version >= 7) {
    for (let i = 0; i < 18; i++) {
      const r = Math.floor(i / 3), c = i % 3;
      if (m[size - 11 + c][r] === -1) m[size - 11 + c][r] = 0;
      if (m[r][size - 11 + c] === -1) m[r][size - 11 + c] = 0;
    }
  }

  // ---- data, snaking up and down in two-column strips, right to left
  const free = Array.from({ length: size }, () => new Uint8Array(size));
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) free[r][c] = m[r][c] === -1 ? 1 : 0;

  let bit = 0;
  const total = codewords.length * 8;
  for (let col = size - 1; col > 0; col -= 2) {
    if (col === 6) col--;                              // the vertical timing column
    for (let i = 0; i < size; i++) {
      const up = ((size - 1 - col) >> 1) % 2 === 0;
      const row = up ? size - 1 - i : i;
      for (const c of [col, col - 1]) {
        if (!free[row][c]) continue;
        const v = bit < total ? (codewords[bit >> 3] >> (7 - (bit & 7))) & 1 : 0;
        m[row][c] = v;
        bit++;
      }
    }
  }

  // ---- choose the mask that scores best, then stamp the format bits
  let best = null, bestScore = Infinity;
  for (let mask = 0; mask < 8; mask++) {
    const cand = m.map((row) => Int8Array.from(row));
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
      if (free[r][c] && maskAt(mask, r, c)) cand[r][c] ^= 1;
    }
    stampFormat(cand, size, mask, version);
    const sc = penalty(cand, size);
    if (sc < bestScore) { bestScore = sc; best = cand; }
  }
  return best;
}

function maskAt(mask, r, c) {
  switch (mask) {
    case 0: return (r + c) % 2 === 0;
    case 1: return r % 2 === 0;
    case 2: return c % 3 === 0;
    case 3: return (r + c) % 3 === 0;
    case 4: return (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0;
    case 5: return ((r * c) % 2) + ((r * c) % 3) === 0;
    case 6: return (((r * c) % 2) + ((r * c) % 3)) % 2 === 0;
    default: return (((r + c) % 2) + ((r * c) % 3)) % 2 === 0;
  }
}

function stampFormat(m, size, mask, version) {
  const f = FORMAT[mask];
  for (let i = 0; i < 15; i++) {
    // MSB first: the first module in the sequence carries bit 14. (Version
    // information, below, goes the other way round — LSB first. They are genuinely
    // different, and getting this backwards produces a symbol that looks perfect
    // and decodes to nothing.)
    const b = (f >> (14 - i)) & 1;
    // the two copies of the format string, per the spec's placement
    if (i < 6) m[8][i] = b;
    else if (i === 6) m[8][7] = b;
    else if (i === 7) m[8][8] = b;
    else if (i === 8) m[7][8] = b;
    else m[14 - i][8] = b;

    // Second copy: seven modules climbing column 8 from the bottom, then eight
    // running right along row 8. The split is 7/8, not 8/7 — the module at
    // (size-8, 8) is the always-dark one, not a format bit, and writing a bit there
    // costs you the last column of the horizontal run.
    if (i < 7) m[size - 1 - i][8] = b;
    else m[8][size - 15 + i] = b;
  }
  m[size - 8][8] = 1;
  if (version >= 7) {
    const v = VERSION_INFO[version];
    for (let i = 0; i < 18; i++) {
      const b = (v >> i) & 1;
      const r = Math.floor(i / 3), c = i % 3;
      m[size - 11 + c][r] = b;
      m[r][size - 11 + c] = b;
    }
  }
}

// The spec's four penalty rules, which together pick the mask that scanners find
// easiest: long same-colour runs, 2x2 blocks, finder-like sequences, and imbalance.
function penalty(m, size) {
  let score = 0;
  for (let r = 0; r < size; r++) {
    for (const line of [0, 1]) {
      let run = 1;
      for (let i = 1; i < size; i++) {
        const a = line ? m[i - 1][r] : m[r][i - 1];
        const b = line ? m[i][r] : m[r][i];
        if (a === b) { run++; if (run === 5) score += 3; else if (run > 5) score++; }
        else run = 1;
      }
    }
  }
  for (let r = 0; r < size - 1; r++) for (let c = 0; c < size - 1; c++) {
    const v = m[r][c];
    if (v === m[r][c + 1] && v === m[r + 1][c] && v === m[r + 1][c + 1]) score += 3;
  }
  const pat = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0];
  for (let r = 0; r < size; r++) for (let c = 0; c + 11 <= size; c++) {
    let h = true, v = true;
    for (let i = 0; i < 11; i++) {
      if (m[r][c + i] !== pat[i]) h = false;
      if (m[c + i][r] !== pat[i]) v = false;
    }
    if (h) score += 40;
    if (v) score += 40;
  }
  let dark = 0;
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (m[r][c]) dark++;
  score += Math.floor(Math.abs((dark * 100) / (size * size) - 50) / 5) * 10;
  return score;
}

// ---------------------------------------------------------------- render
// An <svg> string. Sharp at any size, prints, and costs nothing to draw.
export function svg(text, { scale = 6, margin = 4, dark = "#000", light = "#fff" } = {}) {
  const { modules, size } = encode(text);
  const dim = (size + margin * 2) * scale;
  let path = "";
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    if (modules[r][c]) path += `M${(c + margin) * scale} ${(r + margin) * scale}h${scale}v${scale}h-${scale}z`;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dim} ${dim}" width="${dim}" height="${dim}" shape-rendering="crispEdges">` +
         `<rect width="${dim}" height="${dim}" fill="${light}"/><path d="${path}" fill="${dark}"/></svg>`;
}
