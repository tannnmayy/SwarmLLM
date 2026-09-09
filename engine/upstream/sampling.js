// Imported from SwarmLLM — https://github.com/Nehanth/swarmllm
// Copyright (c) 2026 Nehanth Narendrula. MIT License; see THIRD_PARTY_NOTICES.md.
// Upstream commit 1c9763fcd7eb9c42865cd5937dceba678fe9d9a5. Functionally unmodified; this header is the only addition.

// Sampling helpers.

export function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}


// ---------- GPU kernel self-test ----------
// Runs a tiny synthetic Llama model through the f32, q8 and q4 kernel paths on
// the given device and compares logits with a CPU reference. Names the broken
// path on GPUs whose drivers/compilers disagree with the spec.
