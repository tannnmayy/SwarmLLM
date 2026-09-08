// Keep a device awake and in the foreground while it holds layers.
//
// This is not a nicety. A hidden or locked tab is throttled hard by the browser --
// measured here at roughly 3x slower on identical code -- and a throttled device in
// the chain sets the pace for every token the room produces. A phone that dims its
// screen mid-demo does not fail loudly; it just quietly makes everything worse.
//
// Two mechanisms, because no single one covers the devices in the room:
//   1. Screen Wake Lock. The correct API. Chrome/Android and Safari 16.4+.
//   2. A muted looping video. The old trick, for anything the first does not cover;
//      a playing video keeps the page active where a wake lock is unavailable.
//      (Technique noted from SwarmLLM's room, which does the same thing -- see
//      NOTICE.md. It is a well-known workaround, not their invention.)
//
// A wake lock is released by the browser whenever the page is hidden, so it has to be
// re-acquired on every return to visible rather than taken once at startup.

let lock = null;
let video = null;
let onVis = null;

export async function keepAwake() {
  const acquire = async () => {
    if (document.visibilityState !== "visible") return;
    try {
      if ("wakeLock" in navigator && !lock) {
        lock = await navigator.wakeLock.request("screen");
        lock.addEventListener("release", () => { lock = null; });
      }
    } catch { /* denied, or no user gesture yet: the video fallback still helps */ }
  };

  if (!video) {
    // 1x1 silent MP4. Muted + playsinline is what lets it autoplay without a gesture.
    video = document.createElement("video");
    video.setAttribute("playsinline", "");
    video.muted = true;
    video.loop = true;
    video.width = 1; video.height = 1;
    Object.assign(video.style, { position: "fixed", top: "-1px", left: "-1px", opacity: "0", pointerEvents: "none" });
    video.src = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAAr1tZGF0AAACrgYF//+q3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE1MiByMjg1NCBlOWE1OTAzIC0gSC4yNjQvTVBFRy00IEFWQyBjb2RlYyAtIENvcHlsZWZ0IDIwMDMtMjAxNyAtIGh0dHA6Ly93d3cudmlkZW9sYW4ub3JnL3gyNjQuaHRtbAAAAAFliIQAK//+9dHwgAAAAwAAAwAAAwAAAwAABRAAAAAgQZokbEK//jhAAAADAAADAAADAAADAAADAAADAAAOOAAAABBBnkJ4hX8AAAMAAAMAAA44AAAADwGeYXRC/wAAAwAAAwAADjkAAAAPAZ5jakL/AAADAAADAAAOOAAAABZBmmhJqEFomUwIT//98QAAAwAADjgAAAATQZ6GRREsK/8AAAMAAAMAAA45AAAADwGepXRC/wAAAwAAAwAADjgAAAAPAZ6nakL/AAADAAADAAAOOQAAABZBmqxJqEFsmUwIT//98QAAAwAADjkAAAATQZ7KRRUsK/8AAAMAAAMAAA44AAAADwGe6XRC/wAAAwAAAwAADjkAAAAPAZ7rakL/AAADAAADAAAOOAAAABZBmvBJqEFsmUwIT//98QAAAwAADjgAAAATQZ8ORRUsK/8AAAMAAAMAAA45AAAADwGfLXRC/wAAAwAAAwAADjgAAAAPAZ8vakL/AAADAAADAAAOOQAAABZBmzRJqEFsmUwIT//98QAAAwAADjkAAAATQZ9SRRUsK/8AAAMAAAMAAA44AAAADwGfcXRC/wAAAwAAAwAADjkAAAAPAZ9zakL/AAADAAADAAAOOAAAABZBm3hJqEFsmUwIT//98QAAAwAADjgAAAATQZ+WRRUsK/8AAAMAAAMAAA45AAAADwGftXRC/wAAAwAAAwAADjgAAAAPAZ+3akL/AAADAAADAAAOOQAAABZBm7xJqEFsmUwIT//98QAAAwAADjkAAAATQZ/aRRUsK/8AAAMAAAMAAA44AAAADwGf+XRC/wAAAwAAAwAADjkAAAAPAZ/7akL/AAADAAADAAAOOA==";
    document.body.appendChild(video);
  }
  try { await video.play(); } catch { /* autoplay refused; the wake lock may still hold */ }

  await acquire();
  if (!onVis) {
    // The browser drops the lock every time the page hides, so take it again on return.
    onVis = () => { if (document.visibilityState === "visible") acquire(); };
    document.addEventListener("visibilitychange", onVis);
  }
  return () => release();
}

export function release() {
  try { lock?.release(); } catch {}
  lock = null;
  if (onVis) { document.removeEventListener("visibilitychange", onVis); onVis = null; }
  if (video) { try { video.pause(); } catch {} video.remove(); video = null; }
}

export function isAwake() {
  return !!lock || !!(video && !video.paused);
}
