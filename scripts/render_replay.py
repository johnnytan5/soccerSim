"""Top-down replay of one or two match traces (match_node --trace-out / batch_matches --trace) as an MP4.

  pixi run python scripts/render_replay.py --left a.trace.jsonl [--right b.trace.jsonl] --t0 15 --t1 40 --out clip.mp4 \
      [--title-left "new brain"] [--title-right "old brain"] [--situation rcap_bad] [--seed 1] [--layout side|stacked] \
      [--poster clip.png]

Draws, to scale: the pitch, robots (disc in team colour, id + role initial, heading tick, white-on-black ring for the robot
with im_lead), the ball, lines between same-team robots whose width and opacity follow the link health, a red tint and banner
during a blackout, and the PLAY time. Status overlays (soccer_sim/overlay.py, shared with the MuJoCo clip): a decision word and
outfield rank under each robot, a filled triangle above every chaser (decision in metrics.CHASE_DECISIONS), a red ring on every
chaser of a team with two or more, each robot's fused-ball belief as a hollow dot joined to the true ball, a flash ring on a ball
event with the reaction timer next to the ball (final time held ~1 s), a per-team HUD (chase flips, seconds with 2+ chasers,
seconds with no chaser, mean fused-ball error, worst comm mode; all counted from the clip start) and a caption (brain label,
network situation, seed) with a legend. A whole-game trace (--trace-all rows carry state, half, score) also shows the half and score next to the time (--t0/--t1 are PLAY seconds, which run on across the halves). Situation and seed default to the match JSON next to the trace.
Layout: --layout side puts two traces left/right (1280x720, 640 px panels); stacked puts them one above the other (1280x1440,
full-width panels, bigger pitch and one-line legend). Default: side for one trace, stacked for two.
Numpy + cv2 per frame, piped to ffmpeg h264_nvenc (-preset p5 -cq 28), libx264 when NVENC is unavailable.
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

from soccer_sim import overlay as ov  # noqa: E402
from soccer_sim.config import FieldDims  # noqa: E402
from soccer_sim.field_geometry import field_lines  # noqa: E402
from soccer_sim.overlay import (BLACKOUT_BGR, DUP_RING, FONT, PAGE_BG, TEAM, TEAM_TEXT, TEXT, TEXT_SCALE,  # noqa: E402
                                LabelPlacer)
from soccer_sim.video import encoder_args, find_ffmpeg, open_encoder  # noqa: E402

W, H, FPS = 1280, 720, 25   # canvas for one trace or side by side; --layout stacked makes it W x 2H
PITCH_BG = (58, 74, 46)     # dark desaturated green, neutral for all colour vision
PITCH_LINE = (200, 200, 200)
ROBOT_R_M, BALL_R_M, TICK_M = 0.5, 0.25, 0.95  # drawn larger than the real T2 so the label stays readable
HUD_ROW = ov.HUD_ROW
NONCHASER_SCALE = 0.55   # decision words of non-chasers, smaller than the chaser's (TEXT_SCALE)


def load_trace(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: empty trace")
    return {"rows": rows, "t": [r["t"] for r in rows]}


def _angle_lerp(a: float, b: float, f: float) -> float:
    return a + math.atan2(math.sin(b - a), math.cos(b - a)) * f


def state_at(trace: dict, t: float) -> dict:
    """The trace at PLAY time t: poses and the ball interpolated between the 10 Hz rows, everything else from the row before.
    "i" and "row" are the row before t, which the Panel feeds to its running counters once."""
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
    return {"ball": ball, "robots": robots, "blackout": a["blackout"], "covered": ts[0] - 0.2 <= t <= ts[-1] + 0.2,
            "i": i, "row": a}


class Panel:
    """One pitch with its caption, HUD and legend, drawn into the rectangle (x0, y0, w, H) of the canvas."""

    def __init__(self, x0: int, w: int, title: str, fd: FieldDims | None = None, situation: str | None = None, seed=None,
                 y0: int = 0):
        self.x0, self.y0, self.w, self.title = x0, y0, w, title
        self.fd = fd = fd or FieldDims()
        self.label, self.situation_line = ov.caption_text(title, situation, seed)
        self.cap_h = ov.caption_height(w, self.label, self.situation_line)
        self.wide = w >= 1000
        self.col_w = 300 if self.wide else 0      # wide panels: HUD, time and legend in a right-hand column
        self.block_w = (w - 36) // 2
        hud_h = HUD_ROW * 5
        if self.wide:
            self.legend_w = self.col_w - 20
            self.footer_h = 0
            avail_w, avail_h = w - self.col_w - 16, H - self.cap_h - 8
        else:           # HUD blocks, then time, then the legend (wrapped), all full width below the pitch
            self.legend_w = w - 24
            self.footer_h = hud_h + 38 + ov.legend_height(self.legend_w) + 10
            avail_w, avail_h = w - 24, H - self.cap_h - self.footer_h - 8
        self.scale = min(avail_w / (fd.length + 1.0), avail_h / (fd.width + 1.0))
        self.cx = (w - self.col_w) / 2  # panel-local: each panel is drawn into its own image, then pasted at (x0, y0)
        pitch_h = (fd.width + 1.0) * self.scale
        self.cy = self.cap_h + 4 + max(0.0, (avail_h - pitch_h) / 2) + pitch_h / 2
        self.rect = (int(self.cx - (fd.length / 2 + 0.5) * self.scale), int(self.cy - (fd.width / 2 + 0.5) * self.scale),
                     int((fd.length + 1.0) * self.scale), int((fd.width + 1.0) * self.scale))
        self.static = self._static_layer()
        self.counters, self.timer, self._fed = ov.RunningCounters(), ov.ReactionTimer(), None

    def px(self, x: float, y: float, z: float = 0.0) -> tuple[int, int]:
        return int(round(self.cx + x * self.scale)), int(round(self.cy - y * self.scale))

    def _static_layer(self) -> np.ndarray:
        img = np.full((H, self.w, 3), PAGE_BG, np.uint8)
        rx, ry, rw, rh = self.rect
        img[ry:ry + rh, rx:rx + rw] = PITCH_BG
        lw = max(1, int(round(0.07 * self.scale)))
        for (x0, y0), (x1, y1) in field_lines(self.fd, 48):
            cv2.line(img, self.px(x0, y0), self.px(x1, y1), PITCH_LINE, lw, cv2.LINE_AA)
        for sx in (-1, 1):  # goals
            gy = self.fd.goal_width / 2
            cv2.line(img, self.px(sx * self.fd.length / 2, -gy), self.px(sx * self.fd.length / 2, gy), (235, 235, 235),
                     lw * 3, cv2.LINE_AA)
        ov.draw_caption(img, 0, 0, self.w, self.label, self.situation_line)
        return img

    def _feed(self, st: dict) -> None:
        if st.get("row") is not None and st.get("i") != self._fed:
            self._fed = st.get("i")
            self.counters.update(st["row"])
            self.timer.update(st["row"])

    # ---- per frame ------------------------------------------------------------------------------------------
    def draw(self, canvas: np.ndarray, st: dict | None, t: float) -> None:
        img = self.static.copy()
        s = self.scale
        if st is None or not st["covered"]:
            cv2.putText(img, "no trace data at this time", (self.w // 2 - 150, H // 2), FONT, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            canvas[self.y0:self.y0 + H, self.x0:self.x0 + self.w] = img
            return
        self._feed(st)
        self._row = st.get("row") or {}
        rx, ry, rw, rh = self.rect
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
                alpha = 0.06 + 0.34 * h
                colour = tuple(int(c) for c in (np.array(TEAM[r["team"]], float) * 0.55 + 255 * 0.45) * alpha + base * (1 - alpha))
                cv2.line(img, self.px(r["x"], r["y"]), self.px(o["x"], o["y"]), colour, 1 + int(round(2 * h)), cv2.LINE_AA)
        ov.belief_marks(img, self.px, robots, st["ball"], radius=max(5, int(0.3 * s)))
        rr = max(8, int(ROBOT_R_M * s))
        dup = ov.duplicate_teams(robots)
        placer = LabelPlacer((rx + 2, ry + 2, rx + rw - 2, ry + rh - 2))
        bc = self.px(st["ball"]["x"], st["ball"]["y"])
        br = max(5, int(BALL_R_M * s))
        placer.reserve((bc[0] - br - 4, bc[1] - br - 4, bc[0] + br + 4, bc[1] + br + 4))
        for r in robots:
            c = self.px(r["x"], r["y"])
            placer.reserve((c[0] - rr - 4, c[1] - rr - 4, c[0] + rr + 4, c[1] + rr + 4))
        for r in sorted(robots, key=lambda q: q["im_lead"] is True):
            c = self.px(r["x"], r["y"])
            chaser = ov.is_chaser(r)
            if chaser and r["team"] in dup:  # red ring on the ground, outside the lead ring
                ov.ground_ring(img, self.px, r["x"], r["y"], ROBOT_R_M + 15 / s, (0, 0, 0), 7)
                ov.ground_ring(img, self.px, r["x"], r["y"], ROBOT_R_M + 15 / s, DUP_RING, 4)
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
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            cv2.putText(img, label, (c[0] - tw // 2, c[1] + th // 2 - 1), FONT, 0.6, TEAM_TEXT[r["team"]], 1, cv2.LINE_AA)
            if chaser:
                ov.chaser_marker(img, (c[0], c[1] - rr - 19), r["team"])
        for r in sorted(robots, key=lambda q: not ov.is_chaser(q)):  # decision word and rank; chasers get the best spot
            text = ov.label_text(r)
            if not text:
                continue
            c = self.px(r["x"], r["y"])
            ls = TEXT_SCALE if ov.is_chaser(r) else NONCHASER_SCALE
            tw, th = ov.text_size(text, ls, 2 if ov.is_chaser(r) else 1)
            gap = rr + 6 + (10 if r["team"] in dup and ov.is_chaser(r) else 0)
            cand = []
            for extra in (0, th + 6, 2 * (th + 6)):  # below, above, right, left; then the same one and two lines further out
                g = gap + extra
                cand += [(c[0] - tw // 2, c[1] + g), (c[0] - tw // 2, c[1] - g - th - 22), (c[0] + g, c[1] - th // 2),
                         (c[0] - g - tw, c[1] - th // 2)]
            x, y = placer.place(tw, th, cand)
            ov.put_text(img, text, (x, y + th - 5), ls, TEAM[r["team"]] if ov.is_chaser(r) else TEXT,
                        2 if ov.is_chaser(r) else 1)
        cv2.circle(img, bc, br + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, bc, br, (255, 255, 255), -1, cv2.LINE_AA)
        ov.event_marks(img, self.px, self.timer.display(t), (rx, ry, rx + rw, ry + rh), st["ball"])
        if st["blackout"]:
            cv2.rectangle(img, (rx, ry), (rx + rw, ry + 30), (30, 30, 200), -1)
            ov.put_text(img, "TEAM LINK BLACKOUT", (rx + rw // 2 - 125, ry + 22), 0.7, (255, 255, 255), 2, outline=False)
        self._footer(img, robots, t, ry + rh + 8)
        canvas[self.y0:self.y0 + H, self.x0:self.x0 + self.w] = img

    @staticmethod
    def game_text(row: dict) -> str:
        """'H1  2 - 1  PLAYING' from a whole-game trace row (state, half, score); '' for a PLAY-only trace."""
        sc = row.get("score")
        if not sc:
            return ""
        state = row.get("state")
        return f"H{row.get('half', 1)}  {sc.get('home', 0)} - {sc.get('away', 0)}" + (f"  {state}" if state and state != "PLAYING" else "")

    def _footer(self, img: np.ndarray, robots: list[dict], t: float, y: int) -> None:
        teams = sorted({r["team"] for r in robots}, key=lambda tm: tm != "home")
        hud_h = HUD_ROW * 5
        if self.wide:
            x, bw = self.w - self.col_w + 6, self.col_w - 14
            y = self.cap_h + 8
            for k, team in enumerate(teams):
                ov.draw_team_hud(img, x, y + k * (hud_h + 10), bw, team, self.counters.stats(team))
            y += len(teams) * (hud_h + 10)
            ov.put_text(img, f"t = {t:.1f} s", (x + 4, y + 30), 1.0, TEXT, 2)
            game = self.game_text(getattr(self, "_row", {}))
            if game:
                ov.put_text(img, game, (x + 4, y + 62), 0.8, TEXT, 2)
                y += 32
            ov.draw_legend(img, x, y + 44, self.legend_w)
            return
        for k, team in enumerate(teams):
            ov.draw_team_hud(img, 12 + k * (self.block_w + 12), y, self.block_w, team, self.counters.stats(team))
        ov.put_text(img, f"t = {t:.1f} s", (12, y + hud_h + 30), 1.0, TEXT, 2)
        ov.put_text(img, self.game_text(getattr(self, "_row", {})), (230, y + hud_h + 30), 0.8, TEXT, 2)
        ov.draw_legend(img, 12, y + hud_h + 38, self.legend_w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, help="trace JSONL (batch_matches --trace)")
    ap.add_argument("--right", default="", help="second trace for a side-by-side comparison")
    ap.add_argument("--t0", type=float, required=True, help="first PLAY second")
    ap.add_argument("--t1", type=float, required=True, help="last PLAY second")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title-left", default="")
    ap.add_argument("--title-right", default="")
    ap.add_argument("--situation", default="", help="network situation for the caption (default: profile in the match JSON next to the trace)")
    ap.add_argument("--seed", default="", help="seed for the caption (default: from the match JSON next to the trace)")
    ap.add_argument("--layout", choices=("auto", "side", "stacked"), default="auto",
                    help="two traces: side by side (1280x720) or stacked (1280x1440); auto = side for one trace, stacked for two")
    ap.add_argument("--poster", default="", help="also write this PNG (a frame at --poster-t, default the middle)")
    ap.add_argument("--poster-t", type=float, default=None)
    ap.add_argument("--cpu", action="store_true", help="encode with libx264 instead of NVENC")
    a = ap.parse_args(argv)
    if a.t1 <= a.t0:
        raise SystemExit("--t1 must be after --t0")
    traces = [load_trace(a.left)] + ([load_trace(a.right)] if a.right else [])
    titles = [a.title_left or Path(a.left).stem] + ([a.title_right or Path(a.right).stem] if a.right else [])
    metas = [ov.trace_meta(a.left)] + ([ov.trace_meta(a.right)] if a.right else [])
    layout = a.layout if a.layout != "auto" else ("stacked" if a.right else "side")
    stacked = len(traces) == 2 and layout == "stacked"
    canvas_h = H * 2 if stacked else H
    panels = []
    for k, (title, meta) in enumerate(zip(titles, metas)):
        kw = {"situation": a.situation or meta.get("situation"), "seed": a.seed if a.seed != "" else meta.get("seed")}
        if len(traces) == 1:
            panels.append(Panel(0, W, title, **kw))
        elif stacked:
            panels.append(Panel(0, W, title, y0=k * H, **kw))
        else:
            panels.append(Panel(k * (W // 2), W // 2, title, **kw))
    ffmpeg = find_ffmpeg()
    codec_args, codec = encoder_args(ffmpeg, a.cpu)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    enc = open_encoder(ffmpeg, a.out, codec_args, W, canvas_h, FPS)
    n = int(round((a.t1 - a.t0) * FPS))
    poster_t = a.poster_t if a.poster_t is not None else (a.t0 + a.t1) / 2
    poster_frame = min(max(int(round((poster_t - a.t0) * FPS)), 0), n)
    canvas = np.full((canvas_h, W, 3), PAGE_BG, np.uint8)
    import time
    start = time.perf_counter()
    for k in range(n + 1):
        t = a.t0 + k / FPS
        for panel, tr in zip(panels, traces):
            panel.draw(canvas, state_at(tr, t), t)
        if len(panels) == 2:
            cv2.line(canvas, (0, H), (W, H), (90, 90, 90), 2) if stacked else cv2.line(canvas, (W // 2, 0), (W // 2, H), (90, 90, 90), 2)
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
