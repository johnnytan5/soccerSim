"""Cut a window out of a recorded 3D clip (match_node --record) and (re)draw its caption strip.

  pixi run python scripts/cut_clip.py --in match.mp4 --t0 30 --t1 52 --out cut.mp4 \
      [--window-start 0] [--label "new brain"] [--situation rcap_bad] [--seed 1] [--poster cut.png]

--t0/--t1 are PLAY seconds; --window-start is the first PLAY second of the recording (its --record A:B start). With
--label/--situation/--seed the caption strip at the top is painted over with soccer_sim.overlay.draw_caption (clips
recorded before the caption fix in match_node have an empty strip). Encoded like the recorder (NVENC, libx264 fallback).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_sim import overlay as ov  # noqa: E402
from soccer_sim.video import encoder_args, find_ffmpeg, open_encoder  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--t0", type=float, required=True, help="first PLAY second")
    ap.add_argument("--t1", type=float, required=True, help="last PLAY second")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-start", type=float, default=0.0, help="first PLAY second of the recording")
    ap.add_argument("--label", default=None, help="caption: brain label (repaints the caption strip)")
    ap.add_argument("--situation", default=None, help="caption: network profile id")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--poster", default="", help="also write this PNG (the middle frame)")
    ap.add_argument("--cpu", action="store_true", help="encode with libx264 instead of NVENC")
    a = ap.parse_args(argv)

    cap = cv2.VideoCapture(a.src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = max(0, round((a.t0 - a.window_start) * fps))
    last = round((a.t1 - a.window_start) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    repaint = a.label is not None or a.situation is not None or a.seed is not None
    label, line = ov.caption_text(a.label or "", a.situation, a.seed)

    ffmpeg = find_ffmpeg()
    codec, _ = encoder_args(ffmpeg, force_cpu=a.cpu)
    enc = open_encoder(ffmpeg, a.out, codec, w, h, round(fps))
    mid = (first + last) // 2
    n = 0
    for i in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        if repaint:
            ov.draw_caption(img, 0, 0, w, label, line)
        enc.stdin.write(img.tobytes())
        if a.poster and i == mid:
            cv2.imwrite(a.poster, img)
        n += 1
    enc.stdin.close()
    rc = enc.wait()
    print(f"{n} frames ({n / fps:.1f} s) -> {a.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
