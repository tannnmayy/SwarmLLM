// Room protocol: the decisions the host makes about other devices.
//
//   node tests/room-protocol.test.mjs
//
// Both groups here are regressions from a real two-device run over the deployed
// topology (static site on one origin, signalling on another, weights from
// huggingface.co). Neither could have been caught on the LAN path, which is the
// point of writing them down:
//
//   1. The load watchdog. The host used to give a worker a flat 120 s to finish
//      loading. Against a local mirror a worker's share arrives in seconds; against
//      a CDN the measured figures were 138 s for the worker's 191 MB and 193 s for
//      the host's 413 MB. The host gave up before either finished and the room never
//      started -- every time, on every connection that is not a LAN.
//
//   2. The capability gate. A device with no WebGPU used to profile, join, get
//      dealt a layer range, and only then discover it could not load anything. On a
//      public link that is the most likely first visit there is.
//
// Room is browser code, so the few globals its constructor touches are shimmed
// below. The Room instance itself is real -- these exercise the actual methods, not
// a re-implementation of them.

globalThis.location = { protocol: "https:", host: "test.invalid" };

const { Room } = await import("../room/room.js");
const { capabilityGap, MODELS } = await import("../models/registry.mjs");

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// A Room with its mesh neutered: no sockets, no WebRTC, and every outbound message
// recorded so a test can assert on what the host actually told a worker.
function makeRoom({ peers = [] } = {}) {
  const room = new Room({ code: "TEST", name: "me", model: "qwen3-0.6b" });
  room.sent = [];
  room.mesh.send = (id, obj) => { room.sent.push({ id, ...obj }); return true; };
  room.mesh.broadcast = (obj) => { room.sent.push({ id: "*", ...obj }); return peers.length; };
  room.myProfile = { msPerLayer: 5, stable: true };
  room.pledgeBytes = 2 ** 30;
  for (const p of peers) {
    room.mesh.peers.set(p.id, { id: p.id, name: p.name || p.id, meta: p.meta || {}, ready: p.ready !== false, rtt: 5 });
  }
  return room;
}

// ---------------------------------------------------------------- watchdog
console.log("\nload watchdog: silence, not elapsed time");
{
  const room = makeRoom();
  // 300 ms of silence instead of 90 s, so the test runs in the time a test should.
  let settled = null;
  room._readyCount = 0;
  const waiting = room._waitForWorkers(1, 300).then(() => { settled = "resolved"; }, (e) => { settled = e.message; });

  // A worker that keeps reporting progress must never be timed out, however long it
  // takes. Six kicks across ~900 ms, all inside a 300 ms idle window individually,
  // and well past it in total.
  for (let i = 0; i < 6; i++) {
    await sleep(150);
    await room._onMsg("worker1", { t: "progress", pct: (i + 1) * 15 });
  }
  ok("a worker still reporting progress is not timed out, however long it takes",
     settled === null, String(settled));

  await room._onMsg("worker1", { t: "ready", range: [14, 28] });
  await waiting;
  ok("the wait resolves when the worker finally reports ready", settled === "resolved", String(settled));
}
{
  const room = makeRoom();
  room._readyCount = 0;
  let settled = null;
  const waiting = room._waitForWorkers(1, 250).then(() => { settled = "resolved"; }, (e) => { settled = e.message; });
  await sleep(120);
  await room._onMsg("worker1", { t: "progress", pct: 30 });   // alive, then stops
  await sleep(500);
  await waiting.catch(() => {});
  ok("a worker that goes silent mid-download IS timed out", typeof settled === "string" && settled !== "resolved", String(settled));
  ok("the failure says it stopped responding, not that it was slow",
     /stopped responding/.test(settled || ""), String(settled));
}
{
  const room = makeRoom();
  room._readyCount = 2;
  let resolved = false;
  await room._waitForWorkers(2, 50).then(() => { resolved = true; });
  // A worker that kept its range replies almost instantly and can beat the host here.
  ok("workers that already reported ready resolve the wait immediately", resolved);
}
{
  const room = makeRoom();
  room._readyCount = 0;
  let settled = null;
  const waiting = room._waitForWorkers(2, 300).then(() => { settled = "resolved"; }, (e) => { settled = e.message; });
  await sleep(150);
  await room._onMsg("fast", { t: "ready", range: [0, 14] });   // one lands early
  await sleep(200);                                            // past the original window
  ok("one worker landing rearms the wait for the others",
     settled === null, "a three-device room must not time out on its slowest member");
  await room._onMsg("slow", { t: "ready", range: [14, 28] });
  await waiting;
  ok("the wait resolves once every worker has landed", settled === "resolved", String(settled));
}

// ---------------------------------------------------------------- capability gate
console.log("\ncapability gate: do not deal layers to a device that cannot run them");
{
  const gpu = MODELS["qwen3-0.6b"];
  const cpu = MODELS["smollm2-135m"];
  ok("a GPU model is refused on a device with no WebGPU",
     capabilityGap(gpu, { webgpu: false }) === "this device has no WebGPU");
  ok("a GPU model is refused on a GPU without shader-f16",
     /shader-f16/.test(capabilityGap(gpu, { webgpu: true, shaderF16: false }) || ""));
  ok("a GPU model is allowed on a capable device",
     capabilityGap(gpu, { webgpu: true, shaderF16: true }) === null);
  // undefined is "not probed yet", which must not disqualify a device before its
  // adapter request has resolved.
  ok("an unprobed device is not disqualified", capabilityGap(gpu, {}) === null);
  ok("the CPU model needs nothing and runs anywhere",
     capabilityGap(cpu, { webgpu: false }) === null);
}
{
  const room = makeRoom({ peers: [
    { id: "good", meta: { msPerLayer: 4, budgetBytes: 2 ** 30, canRun: true } },
    { id: "nogpu", meta: { msPerLayer: 9, budgetBytes: 2 ** 30, canRun: false, cannotRunWhy: "this device has no WebGPU" } },
    { id: "unknown", meta: { msPerLayer: 6, budgetBytes: 2 ** 30 } },
  ] });
  const ids = room._devices().map((d) => d.id);
  ok("a peer that says it cannot run the model is not a planning candidate", !ids.includes("nogpu"), ids.join(","));
  ok("a peer that can is", ids.includes("good"), ids.join(","));
  // An older client, or one whose meta has not arrived yet, says nothing about
  // canRun. Excluding it would be worse than the bug this fixes.
  ok("a peer that has not said either way is still a candidate", ids.includes("unknown"), ids.join(","));

  // A device that tried and failed anyway -- a lost GPU device, an out-of-memory, a
  // weight fetch that would not complete. It stays connected and in the roster, so
  // it has to be remembered as unusable or the next solve deals it the same range.
  room.isHost = true;
  await room._onMsg("good", { t: "cannot-load", why: "GPU ran out of memory" });
  ok("a worker that reports cannot-load is dropped from later plans",
     !room._devices().map((d) => d.id).includes("good"), room._devices().map((d) => d.id).join(","));
}
{
  const room = makeRoom({ peers: [{ id: "other", meta: { msPerLayer: 4, budgetBytes: 2 ** 30, canRun: true } }] });
  room.cannotRun = "this device has no WebGPU";
  const errors = [];
  room.on("error", (e) => errors.push(e));
  await room.start();
  // The host holds the embedding table and the LM head, so there is no arrangement
  // where it merely coordinates. Saying "no feasible plan: too little memory" here
  // would send someone hunting entirely the wrong problem.
  ok("the host refuses to start a model it cannot run", errors.length === 1, JSON.stringify(errors));
  ok("and says why, rather than blaming memory",
     /no WebGPU/.test(errors[0] || "") && !/memory/.test(errors[0] || ""), errors[0]);
  ok("nothing was dealt", room.sent.length === 0, JSON.stringify(room.sent));
}
{
  const room = makeRoom();
  room.cannotRun = "this device has no WebGPU";
  room.model = "qwen3-0.6b";
  // As a real worker is after join: it resolved the model spec before anyone dealt
  // it anything. Without this the deal below looks like a context-window change and
  // takes the reload path instead, which is a different test.
  room.spec = { maxSeq: 2048, hidden: 1024, layers: 28 };
  const errors = [];
  room.on("error", (e) => errors.push(e));
  // A worker refuses BEFORE downloading a gigabyte it can never use, and tells the
  // host so it can re-plan now rather than waiting out the watchdog.
  await room._onMsg("host1", { t: "deal", model: "qwen3-0.6b", range: [14, 28], next: null, host: "host1", maxSeq: 2048 });
  const reply = room.sent.find((s) => s.t === "cannot-load");
  ok("a worker that cannot run the model answers cannot-load, not silence", !!reply, JSON.stringify(room.sent));
  ok("it never claims to be ready", !room.sent.some((s) => s.t === "ready"));
  ok("and it says why", /no WebGPU/.test(reply?.why || ""), reply?.why);
  ok("the device itself reports the problem too", errors.some((e) => /cannot run/.test(e)), JSON.stringify(errors));
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
