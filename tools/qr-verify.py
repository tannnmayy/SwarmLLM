# Decode the QR bitmaps written by tests/qr.test.mjs, using OpenCV's detector.
#
# Structural checks prove the patterns are in the right places. Only a decode proves
# a phone can actually read the thing, which is the only property that matters at a
# venue where somebody is holding up a laptop screen.
import json, sys, os
import cv2, numpy as np

here = os.path.join(os.path.dirname(__file__), "..", "tests", ".qr")
manifest = json.load(open(os.path.join(here, "manifest.json"), encoding="utf-8"))
det = cv2.QRCodeDetector()

ok = bad = 0
for e in manifest:
    img = cv2.imread(os.path.join(here, e["file"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  FAIL {e['file']}: could not read bitmap"); bad += 1; continue
    got, pts, _ = det.detectAndDecode(img)
    if got == e["text"]:
        print(f"  ok   v{e['version']}  decoded {len(got)} chars: {got[:52]}"); ok += 1
    else:
        print(f"  FAIL v{e['version']}  wanted {e['text']!r}\n              got {got!r}"); bad += 1

print(f"\n{ok} decoded, {bad} failed\n")
sys.exit(1 if bad else 0)
