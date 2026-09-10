# Third-party notices

This project includes source code from the projects below. Their licence terms are
reproduced here as those licences require.

---

## SwarmLLM

Two separate things in this repository come from
<https://github.com/Nehanth/swarmllm>:

- **The engine.** Files under `engine/upstream/` are imported, pinned to commit
  `1c9763fcd7eb9c42865cd5937dceba678fe9d9a5`, and used unmodified. Each carries a
  header naming its origin.
- **The interface design.** `index.html` and `room.html` adapt SwarmLLM's landing
  and room pages — the palette, the type scale, the component shapes, the ambient
  node-mesh canvas, the join-screen/room-screen split, the sidebar-plus-chat
  layout, peer cards, the per-device load card, chat bubbles and toasts. Both files
  carry a header saying so. The copy and the numbers are this project's own, and
  the code underneath them drives this project's own `Room`, scheduler and
  delivery resolver rather than SwarmLLM's.

```
MIT License

Copyright (c) 2026 Nehanth Narendrula

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime dependencies

| Package | Licence | Used for |
|---|---|---|
| `ws` | MIT | WebSocket signalling, host side only |
| `pptxgenjs` | MIT | deck generation, build tooling only |

Model weights and tokenizers are downloaded from their publishers at runtime under
their own licences and are not redistributed in this repository.
