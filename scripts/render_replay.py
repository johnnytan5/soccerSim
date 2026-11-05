"""Top-down replay of one or two match traces (match_node --trace-out / batch_matches --trace) as an MP4.

  pixi run python scripts/render_replay.py --left a.trace.jsonl [--right b.trace.jsonl] --t0 15 --t1 40 --out clip.mp4 \
      [--title-left "new brain"] [--title-right "old brain"] [--poster clip.png]

Draws, to scale on a 1280x720 canvas (side by side with --right): the pitch, robots (circle in team colour, id + role
initial, heading tick, white-on-black ring for the robot with im_lead), the ball, lines between same-team robots whose
width and opacity follow the link health, a comm-mode badge per team, a red tint and banner during a blackout, and the
PLAY time. Numpy + cv2 per frame, piped to ffmpeg h264_nvenc (-preset p5 -cq 28), libx264 when NVENC is unavailable.
Colours: Okabe-Ito (home blue, away orange; comm mode green / yellow / vermillion), so it reads for colour-blind viewers.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_sim.config import FieldDims  # noqa: E402
from soccer_sim.field_geometry import field_lines  # noqa: E402
from soccer_sim.video import encoder_args, find_ffmpeg, open_encoder  # noqa: E402

W, H, FPS = 1280, 720, 25
HEADER_H, FOOTER_H = 46, 104
# BGR
TEAM = {"home": (178, 114, 0), "away": (0, 159, 230)}           # #0072B2 blue, #E69F00 orange
TEAM_TEXT = {"home": (255, 255, 255), "away": (0, 0, 0)}         # label colour with enough contrast on the disc
TEAM_NAME = {"home": "HOME", "away": "AWAY"}
PITCH_BG = (58, 74, 46)                                          # dark desaturated green, neutral for all colour vision
PITCH_LINE = (200, 200, 200)
PAGE_BG = (30, 30, 30)
MODE_COLOR = {"healthy": (115, 158, 0), "degraded": (66, 228, 240), "isolated": (0, 94, 213)}  # #009E73 #F0E442 #D55E00
MODE_TEXT = {"healthy": (255, 255, 255), "degraded": (0, 0, 0), "isolated": (255, 255, 255)}
MODE_RANK = {"healthy": 0, "degraded": 1, "isolated": 2}
BLACKOUT_BGR = (60, 60, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX
ROBOT_R_M, BALL_R_M, TICK_M = 0.5, 0.25, 0.95  # drawn larger than the real T2 so the label stays readable


def load_trace(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: empty trace")
    return {"rows": rows, "t": [r["t"] for r in rows]}


def _angle_lerp(a: float, b: float, f: float) -> float:
    return a + math.atan2(math.sin(b - a), math.cos(b - a)) * f


def state_at(trace: dict, t: float) -> dict:
    """The trace at PLAY time t: poses and the ball interpolated between the 10 Hz rows, everything else from the row before."""
    ts, rows = trace["t"], trace["rows"]
    i = min(max(bisect.bisect_right(ts, t) - 1, 0), len(rows) - 1)
    a = rows[i]
    b = rows[min(i + 1, len(rows) - 1)]
    span = ts[min(i + 1, len(rows) - 1)] - ts[i]
    f = 0.0 if span <= 1e-9 else min(max((t - ts[i]) / span, 0.0), 1.0)
    prev = {(r["team"], r["id"]): r for r in b["robots"]}
    robots = []
    for r in a["robots"]:
        n = prev.get((r["team"], r["id"]), r)
        robots.append({**r, "x": r["x"] + (n["x"] - r["x"]) * f, "y": r["y"] + (n["y"] - r["y"]) * f,
                       "theta": _angle_lerp(r["theta"], n["theta"], f)})
    ball = {k: a["ball"][k] + (b["ball"][k] - a["ball"][k]) * f for k in ("x", "y")}
    return {"ball": ball, "robots": robots, "blackout": a["blackout"], "covered": ts[0] - 0.2 <= t <= ts[-1] + 0.2}


class Panel:
    """One pitch drawn into the rectangle (x0, 0, w, H) of the canvas."""

    def __init__(self, x0: int, w: int, title: str, fd: FieldDims | None = None):
        self.x0, self.w, self.title = x0, w, title
        fd = fd or FieldDims()
        self.fd = fd
        avail_w, avail_h = w - 24, H - HEADER_H - FOOTER_H - 8
        self.scale = min(avail_w / (fd.length + 1.0), avail_h / (fd.width + 1.0))
        self.cx = w / 2  # panel-local: each panel is drawn into its own image, then pasted at x0
        pitch_h = (fd.width + 1.0) * self.scale
        self.cy = HEADER_H + 4 + max(0.0, (avail_h + FOOTER_H - 20 - pitch_h) / 2) + pitch_h / 2  # block centred below the title
        self.rect = (int(self.cx - (fd.length / 2 + 0.5) * self.scale), int(self.cy - (fd.width / 2 + 0.5) * self.scale),
                     int((fd.length + 1.0) * self.scale), int((fd.width + 1.0) * self.scale))
        self.static = self._static_layer()

    def px(self, x: float, y: float) -> tuple[int, int]:
        return int(round(self.cx + x * self.scale)), int(round(self.cy - y * self.scale))

    def _static_layer(self) -> np.ndarray:
        img = np.full((H, self.w, 3), PAGE_BG, np.uint8)
        shifted = self
        rx, ry, rw, rh = self.rect
        img[ry:ry + rh, rx:rx + rw] = PITCH_BG
        lw = max(1, int(round(0.07 * self.scale)))
        for (x0, y0), (x1, y1) in field_lines(self.fd, 48):
            cv2.line(img, shifted.px(x0, y0), shifted.px(x1, y1), PITCH_LINE, lw, cv2.LINE_AA)
        for sx in (-1, 1):  # goals
            gy = self.fd.goal_width / 2
            cv2.line(img, shifted.px(sx * self.fd.length / 2, -gy), shifted.px(sx * self.fd.length / 2, gy), (235, 235, 235),
                     lw * 3, cv2.LINE_AA)
        cv2.putText(img, self.title, (12, 31), FONT, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
        return img

    # ---- per frame ------------------------------------------------------------------------------------------
    def draw(self, canvas: np.ndarray, st: dict | None, t: float) -> None:
        img = self.static.copy()
        s = self.scale
        local = self
        if st is None or not st["covered"]:
            cv2.putText(img, "no trace data at this time", (self.w // 2 - 150, H // 2), FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            canvas[:, self.x0:self.x0 + self.w] = img
            return
        rx, ry, rw, rh = local.rect
        if st["blackout"]:  # red tint over the pitch
            roi = img[ry:ry + rh, rx:rx + rw]
            roi[:] = cv2.addWeighted(roi, 0.72, np.full_like(roi, BLACKOUT_BGR), 0.28, 0)
        robots = st["robots"]
        by_key = {(r["team"], r["id"]): r for r in robots}
        base = np.array(PITCH_BG, float)
        drawn = set()
        for r in robots:  # links: health of both directions, the weaker one counts
            for l in r.get("links") or []:
                o = by_key.get((r["team"], l.get("player")))
                key = frozenset(((r["team"], r["id"]), (r["team"], l.get("player"))))
                if o is None or key in drawn:
                    continue
                drawn.add(key)
                back = next((q for q in o.get("links") or [] if q.get("player") == r["id"]), None)
                h = min(float(l.get("health") or 0.0), float(back.get("health") or 0.0) if back else float(l.get("health") or 0.0))
                h = min(max(h, 0.0), 1.0)
                alpha = 0.12 + 0.78 * h
                colour = tuple(int(c) for c in (np.array(TEAM[r["team"]], float) * 0.55 + 255 * 0.45) * alpha + base * (1 - alpha))
                cv2.line(img, local.px(r["x"], r["y"]), local.px(o["x"], o["y"]), colour, 1 + int(round(5 * h)), cv2.LINE_AA)
        rr = max(8, int(ROBOT_R_M * s))
        for r in sorted(robots, key=lambda q: q["im_lead"] is True):
            c = local.px(r["x"], r["y"])
            if r.get("im_lead"):  # white ring with a black outline: visible on both team colours and on the red tint
                cv2.circle(img, c, rr + 6, (0, 0, 0), 7, cv2.LINE_AA)
                cv2.circle(img, c, rr + 6, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.circle(img, c, rr, TEAM[r["team"]], -1, cv2.LINE_AA)
            cv2.circle(img, c, rr, (0, 0, 0), 2, cv2.LINE_AA)
            tip = (int(c[0] + math.cos(r["theta"]) * TICK_M * s), int(c[1] - math.sin(r["theta"]) * TICK_M * s))
            cv2.line(img, c, tip, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(img, c, tip, (255, 255, 255), 2, cv2.LINE_AA)
            role = (r.get("player_role") or "?")[0].upper()
            label = f"{r['id']}{role}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.putText(img, label, (c[0] - tw // 2, c[1] + th // 2 - 1), FONT, 0.5, TEAM_TEXT[r["team"]], 1, cv2.LINE_AA)
        bc = local.px(st["ball"]["x"], st["ball"]["y"])
        br = max(5, int(BALL_R_M * s))
        cv2.circle(img, bc, br + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, bc, br, (255, 255, 255), -1, cv2.LINE_AA)
        # footer: badge per team, time, legend
        y = ry + rh + 10
        teams = sorted({r["team"] for r in robots}, key=lambda tm: tm != "home")
        bw = (self.w - 36) // max(len(teams), 1)
        for k, team in enumerate(teams):
            modes = [r.get("comm_mode") for r in robots if r["team"] == team and r.get("comm_mode")]
            worst = max(modes, key=lambda m: MODE_RANK.get(m, 3)) if modes else "unknown"
            n = sum(1 for m in modes if m == worst)
            text = f"{TEAM_NAME[team]}  comm: {worst}" + (f" ({n}/{len(modes)})" if modes and n != len(modes) else "")
            x = 12 + k * (bw + 12)
            cv2.rectangle(img, (x, y), (x + bw, y + 34), MODE_COLOR.get(worst, (120, 120, 120)), -1)
            cv2.rectangle(img, (x, y), (x + 12, y + 34), TEAM[team], -1)   # team stripe
            cv2.rectangle(img, (x, y), (x + bw, y + 34), (0, 0, 0), 1)
            cv2.putText(img, text, (x + 22, y + 24), FONT, 0.6, MODE_TEXT.get(worst, (255, 255, 255)), 1, cv2.LINE_AA)
        cv2.putText(img, f"t = {t:.1f} s", (12, y + 76), FONT, 1.1, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(img, "ring = lead   tick = heading   G/D/S = role   link width = health", (12, y + 100), FONT, 0.46,
                    (190, 190, 190), 1, cv2.LINE_AA)
        if st["blackout"]:
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + 30), (30, 30, 200), -1)
            cv2.putText(img, "TEAM LINK BLACKOUT", (rx + rw // 2 - 125, ry + 22), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        canvas[:, self.x0:self.x0 + self.w] = img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, help="trace JSONL (batch_matches --trace)")
    ap.add_argument("--right", default="", help="second trace for a side-by-side comparison")
    ap.add_argument("--t0", type=float, required=True, help="first PLAY second")
    ap.add_argument("--t1", type=float, required=True, help="last PLAY second")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title-left", default="")
    ap.add_argument("--title-right", default="")
    ap.add_argument("--poster", default="", help="also write this PNG (a frame at --poster-t, default the middle)")
    ap.add_argument("--poster-t", type=float, default=None)
    ap.add_argument("--cpu", action="store_true", help="encode with libx264 instead of NVENC")
    a = ap.parse_args(argv)
    if a.t1 <= a.t0:
        raise SystemExit("--t1 must be after --t0")
    traces = [load_trace(a.left)] + ([load_trace(a.right)] if a.right else [])
    titles = [a.title_left or Path(a.left).stem] + ([a.title_right or Path(a.right).stem] if a.right else [])
    widths = [W] if len(traces) == 1 else [W // 2, W // 2]
    panels, x = [], 0
    for w, title in zip(widths, titles):
        panels.append(Panel(x, w, title))
        x += w
    ffmpeg = find_ffmpeg()
    codec_args, codec = encoder_args(ffmpeg, a.cpu)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    enc = open_encoder(ffmpeg, a.out, codec_args, W, H, FPS)
    n = int(round((a.t1 - a.t0) * FPS))
    poster_t = a.poster_t if a.poster_t is not None else (a.t0 + a.t1) / 2
    poster_frame = min(max(int(round((poster_t - a.t0) * FPS)), 0), n)
    canvas = np.full((H, W, 3), PAGE_BG, np.uint8)
    import time
    start = time.perf_counter()
    for k in range(n + 1):
        t = a.t0 + k / FPS
        for panel, tr in zip(panels, traces):
            panel.draw(canvas, state_at(tr, t), t)
        if len(panels) == 2:
            cv2.line(canvas, (W // 2, 0), (W // 2, H), (90, 90, 90), 2)
        if a.poster and k == poster_frame:
            cv2.imwrite(a.poster, canvas)
        try:
            enc.stdin.write(canvas.tobytes())
        except BrokenPipeError:
            break
    enc.stdin.close()
    rc = enc.wait()
    took = time.perf_counter() - start
    if rc != 0:
        raise SystemExit(f"ffmpeg ({codec}) failed with {rc}")
    size = Path(a.out).stat().st_size
    print(f"{a.out}: {n + 1} frames, {codec}, {size / 1e6:.2f} MB, {took:.1f} s ({took / (n + 1) * 1e3:.1f} ms per frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
