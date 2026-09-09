// EngineFactory: the one place room/room.js asks for an engine. It never imports
// CpuEngine or GpuEngineAdapter directly -- see blueprint section 5.4.
//
//   Room runtime
//       | simple, stable async engine contract
//       v
//   createEngine ── cpu-smollm ──► CpuEngine          (engine/cpu.mjs)
//       |
//       ├────────── dense-gguf ──► GpuEngineAdapter    (engine/gpu-adapter.mjs)
//       |                              └─► upstream DenseEngine
//       |
//       └────────── qwen35-gguf ─► not vendored yet — throws a clear, typed error
//
// Every engine kind returns an object satisfying the same four-call contract:
// embedRun(tokenId, pos), runHidden(x, pos), headFromHidden(x), reset() — plus
// layerCount, bytesLoaded, maxSeq and cache the room reads for progress/UI.

import { CpuEngine } from "./cpu.mjs";
import { GpuEngineAdapter } from "./gpu-adapter.mjs";

export class EngineNotAvailableError extends Error {
  constructor(descriptor, reason) {
    super(`${descriptor.label} is not available: ${reason}`);
    this.name = "EngineNotAvailableError";
    this.descriptor = descriptor;
  }
}

// opts: { layerRange, hasEmbed, hasHead, maxSeq, onProgress, device?, caps? }
export async function createEngine(descriptor, opts = {}) {
  switch (descriptor.engineKind) {
    case "cpu-smollm":
      return CpuEngine.load(descriptor.dir, opts);

    case "dense-gguf":
      return GpuEngineAdapter.load(descriptor, opts);

    case "qwen35-gguf":
      // Deliberately not a generic "unsupported model" message: the reason is
      // architectural (see UPSTREAM.md), not a missing URL, and the blueprint is
      // explicit that this must not be presented as "coming soon by config change".
      throw new EngineNotAvailableError(descriptor,
        "the Qwen 3.8 hybrid engine (engine/upstream/qwen35.js + wgsl/qwen35.js) is not vendored yet, " +
        "and the scheduler only models uniform dense layers. This is Phase 7/8 work, not a loader bug.");

    default:
      throw new EngineNotAvailableError(descriptor, `unknown engine kind "${descriptor.engineKind}"`);
  }
}
