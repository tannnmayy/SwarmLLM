// Device profiling: what is this machine actually worth to the swarm?
//
// The planner is only as good as its inputs, and the input that matters is
// ms-per-layer. Guessing it from a device name, a core count or a memory size does
// not work -- a throttled laptop and a fast phone are indistinguishable that way,
// and thermal state changes the answer minute to minute on exactly the devices the
// pitch cares about.
//
// So we measure. The probe runs the same shape of arithmetic a layer runs (a matvec
// over a weight matrix that does not fit in cache), converts to MACs/second, and
// scales by the model's known MACs per layer. That makes the number portable across
// models: probe once, plan for anything.
//
// Budget: browsers deliberately hide real memory to resist fingerprinting, so there
// is no honest way to ask "how much can I spare". The user pledges instead, which is
// also the socially correct answer -- nobody should hand a web page their whole
// laptop. `navigator.deviceMemory` (Chrome only, coarse) seeds a sane default.

const MB = 2 ** 20, GB = 2 ** 30;

// One matvec of roughly a projection's size: 1536 x 576 is the FF shape in
// SmolLM2 and a representative mix elsewhere. Big enough to leave L2 and hit the
// memory path the real thing hits.
const ROWS = 1536, COLS = 576;
const MACS_PER_CALL = ROWS * COLS;

function matvec(W, x, out, rows, cols) {
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) acc += W[base + c] * x[c];
    out[r] = acc;
  }
}

// Measure sustained MACs/second. Returns the median of several windows rather than
// the mean: one scheduler hiccup or a GC pause should not define a device's speed for
// the rest of the room's life.
export async function measureMacsPerSec({ budgetMs = 300, windows = 5 } = {}) {
  const W = new Float32Array(ROWS * COLS);
  const x = new Float32Array(COLS);
  const out = new Float32Array(ROWS);
  for (let i = 0; i < W.length; i++) W[i] = (i % 251) * 0.004 - 0.5;
  for (let i = 0; i < COLS; i++) x[i] = (i % 97) * 0.01;

  // Warm up until the optimising JIT has actually taken over.
  //
  // A fixed iteration count is not enough. Measured in a browser, twelve warmup calls
  // left the probe reporting 0.25 GMAC/s where the real engine sustained 0.80 -- a
  // 3.2x pessimism, entirely from running in a lower JIT tier. The same code in Node
  // was accurate, which is what makes this the kind of bug that survives testing and
  // then ruins the first plan of a live demo.
  //
  // Warm by time, not by count, and keep going until the rate stops improving.
  {
    const t0 = performance.now();
    let last = 0, stable = 0;
    while (performance.now() - t0 < 120 && stable < 2) {
      const s = performance.now();
      let n = 0;
      while (performance.now() - s < 20) { matvec(W, x, out, ROWS, COLS); n++; }
      const rate = n / (performance.now() - s);
      if (last && rate < last * 1.15) stable++; else stable = 0;
      last = Math.max(last, rate);
    }
  }

  // Measure back to back, with NO await inside the timed region.
  //
  // Yielding between windows seems polite -- it lets a phone's UI breathe -- but it
  // destroys the measurement. In a hidden tab Chrome throttles timers and runs the
  // continuation at low priority, and the probe came back 3.1x pessimistic against
  // byte-identical code measured inline in the same page (0.24 vs 0.75 GMAC/s).
  // A wrong number here silently produces a wrong plan, which is worse than a
  // few hundred milliseconds of jank once at join.
  const rates = [];
  const per = Math.max(20, Math.floor(budgetMs / windows));
  for (let w = 0; w < windows; w++) {
    let calls = 0;
    const t0 = performance.now();
    let elapsed = 0;
    do {
      matvec(W, x, out, ROWS, COLS);
      calls++;
      elapsed = performance.now() - t0;
    } while (elapsed < per);
    rates.push((calls * MACS_PER_CALL) / (elapsed / 1000));
  }
  rates.sort((a, b) => a - b);
  // guard against the optimiser eliminating the loop
  if (!Number.isFinite(out[0])) throw new Error("probe produced no result");
  return {
    macsPerSec: rates[rates.length >> 1],
    spread: rates[rates.length - 1] / rates[0],
    samples: rates,
    // A number measured while hidden cannot be trusted whatever we do, so say so and
    // let the caller re-measure when the tab comes back.
    hidden: typeof document !== "undefined" && document.visibilityState === "hidden",
  };
}

// Sane default pledge. Browsers round deviceMemory down to a power of two and cap it
// at 8, so this is a hint, never a measurement.
export function defaultPledgeBytes() {
  const hinted = navigator.deviceMemory ? navigator.deviceMemory * GB : null;
  const phone = /Android|iPhone|iPad/i.test(navigator.userAgent);
  if (hinted) return Math.max(256 * MB, Math.min(hinted * 0.25, phone ? 1.5 * GB : 4 * GB));
  return phone ? 512 * MB : 2 * GB;
}

// The full profile a device advertises to the room.
export async function profile(spec, { pledgeBytes = null, budgetMs = 300 } = {}) {
  const t0 = performance.now();
  const { macsPerSec, spread, samples, hidden } = await measureMacsPerSec({ budgetMs });
  const msPerLayer = (spec.layerMACs / macsPerSec) * 1000;

  // Thermal and battery state, where the platform will tell us. Chrome and Android
  // will; iOS Safari will not, which is a fact about the demo, not a bug.
  let pressure = null, battery = null;
  try {
    if (typeof navigator.getBattery === "function") {
      const b = await navigator.getBattery();
      battery = { level: b.level, charging: b.charging };
    }
  } catch { /* not available */ }

  return {
    msPerLayer,
    macsPerSec,
    budgetBytes: pledgeBytes ?? defaultPledgeBytes(),
    spread,                       // >1.5 means the device is unstable: throttling, or busy
    stable: spread < 1.5 && !hidden,
    hidden,                       // measured in a background tab: re-measure when visible
    cores: navigator.hardwareConcurrency || null,
    battery,
    pressure,
    probeMs: performance.now() - t0,
    samples: samples.length,
  };
}

// Watch for thermal pressure and tell the room when it changes, so a phone that
// starts throttling gets re-planned around instead of quietly setting the pace.
// Chromium only; returns a no-op teardown elsewhere.
export function watchPressure(onChange) {
  if (typeof PressureObserver === "undefined") return () => {};
  try {
    const obs = new PressureObserver((records) => {
      const last = records[records.length - 1];
      if (last) onChange(last.state);          // nominal | fair | serious | critical
    });
    obs.observe("cpu", { sampleInterval: 2000 });
    return () => obs.disconnect();
  } catch {
    return () => {};
  }
}
