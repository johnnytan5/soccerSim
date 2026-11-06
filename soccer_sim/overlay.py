"""Status overlays shared by the top-down replay (scripts/render_replay.py) and the MuJoCo clip (clip_recorder.py), so both
videos say the same thing.

Input is always a trace row (the dict batch_matches --trace writes, and match_node hands to ClipRecorder.offer):
  {"t", "ball": {x, y}, "blackout", "robots": [{id, team, x, y, theta, decision, player_role, outfield_rank, im_lead,
   comm_mode, fused_ball: {x, y, valid}, ...}]}

Numbers come from soccer_sim.metrics, never re-derived: RunningCounters and ReactionTimer each feed a private MetricsRecorder
with the row (same chasing(), keeper rule, MAX_DT_S, ball-event and reaction rules) and read its running totals, so the HUD at
the end of a window equals what metrics.py would report for that window.

Drawing helpers take a `to_px(x, y, z=0.0) -> (u, v) | None` so the same code serves a top-down map and a perspective camera.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from soccer_sim.metrics import REACTION_TIMEOUT_S, MetricsRecorder, RobotSample, Sample, chasing

# ---- colours (BGR, Okabe-Ito) and text ---------------------------------------------------------------------------
TEAM = {"home": (178, 114, 0), "away": (0, 159, 230)}           # #0072B2 blue, #E69F00 orange
TEAM_TEXT = {"home": (255, 255, 255), "away": (0, 0, 0)}         # label colour with enough contrast on the team colour
TEAM_NAME = {"home": "HOME", "away": "AWAY"}
MODE_COLOR = {"healthy": (115, 158, 0), "degraded": (66, 228, 240), "isolated": (0, 94, 213)}  # #009E73 #F0E442 #D55E00
MODE_TEXT = {"healthy": (255, 255, 255), "degraded": (0, 0, 0), "isolated": (255, 255, 255)}
MODE_RANK = {"healthy": 0, "degraded": 1, "isolated": 2}
BLACKOUT_BGR = (60, 60, 230)
DUP_RING = (40, 40, 235)          # red ring on every chaser of a team with two or more
FLASH = (80, 235, 255)            # ball-event flash and timer
PAGE_BG = (30, 30, 30)
TEXT = (245, 245, 245)
DIM = (190, 190, 190)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.72                 # HERSHEY_SIMPLEX cap height is ~22 px at scale 1.0, so 0.72 gives ~16 px
KEEPER_ROLE = "goal_keeper"

DECISION_WORD = {
    "chase": "CHASE", "assist": "ASSIST", "find": "FIND", "kick": "KICK", "adjust": "ADJUST", "dribble": "DRIBBLE",
    "cross": "CROSS", "auto_visual_kick": "V-KICK", "retreat": "RETREAT", "kickoff_spot": "KICKOFF", "cdm": "CDM",
    "weak_side_cover": "WS-COVER", "cover": "COVER", "standby": "STANDBY", "-": "", "": "",
}
LEGEND = [("tri", "chaser"), ("dup", "2+ chasers"), ("belief", "ball belief"), ("lead", "lead"), ("flash", "ball event")]


def short_decision(decision: str | None) -> str:
    d = decision or ""
    return DECISION_WORD.get(d, d.replace("_", " ").upper()[:8])


def robot_key(r: dict) -> str:
    return f"{'H' if r['team'] == 'home' else 'A'}{r['id']}"


def is_keeper(r: dict) -> bool:
    return r.get("player_role") == KEEPER_ROLE or r.get("id") == 1  # as match_node._record_metrics


def label_text(r: dict) -> str:
    """Decision word and outfield rank, e.g. 'CHASE #1' ('' when the brain published neither)."""
    word = short_decision(r.get("decision"))
    rank = r.get("outfield_rank")
    return " ".join(p for p in (word, f"#{rank}" if isinstance(rank, int) and rank > 0 else "") if p)


def is_chaser(r: dict) -> bool:
    return chasing(r.get("decision") or "")


def duplicate_teams(robots: list[dict]) -> set[str]:
    """Teams with two or more robots in the chase family (metrics' duplicate_decision rule: keeper included)."""
    n: dict[str, int] = {}
    for r in robots:
        if is_chaser(r):
            n[r["team"]] = n.get(r["team"], 0) + 1
    return {t for t, k in n.items() if k >= 2}


def fused_error(r: dict, ball: dict) -> float | None:
    f = r.get("fused_ball")
    if not f or not f.get("valid") or f.get("x") is None or f.get("y") is None:
        return None
    return math.hypot(f["x"] - ball["x"], f["y"] - ball["y"])


def sample_from_row(row: dict) -> Sample:
    ball = row["ball"]
    robots = [RobotSample(robot_key(r), r["team"], r["x"], r["y"], r["theta"], r.get("decision") or "", keeper=is_keeper(r),
                          comm_mode=r.get("comm_mode"), fused_err=fused_error(r, ball), role=r.get("player_role"),
                          rank=r.get("outfield_rank"), im_lead=r.get("im_lead"), role_own_view=r.get("role_own_view"))
              for r in row["robots"]]
    return Sample(row["t"], True, (ball["x"], ball["y"]), robots, blackout=bool(row.get("blackout")), ramp=bool(row.get("ramp")))


def trace_meta(trace_path: str) -> dict:
    """{'situation', 'seed', 'brain'} from the match JSON next to <stem>.trace.jsonl (batch_matches writes both); {} if absent."""
    p = Path(trace_path)
    js = p.with_name(p.name.replace(".trace.jsonl", ".json"))
    try:
        meta = json.loads(js.read_text()).get("meta", {})
    except (OSError, ValueError):
        return {}
    brains = {meta.get("home_brain"), meta.get("away_brain")} - {None}
    return {"situation": meta.get("profile"), "seed": meta.get("seed"),
            "brain": "-vs-".join(sorted(brains)) if brains else meta.get("brain")}


def caption_text(label: str, situation: str | None, seed) -> tuple[str, str]:
    """(brain label, 'network: <situation>   seed <n>')."""
    parts = []
    if situation:
        parts.append(f"network: {situation}")
    if seed not in (None, ""):
        parts.append(f"seed {seed}")
    return label, "   ".join(parts)


# ---- running numbers ---------------------------------------------------------------------------------------------
class RunningCounters:
    """Per-team totals since the first row fed (a clip's own start). Feed every row in time order with update()."""

    def __init__(self) -> None:
        self._rec = MetricsRecorder(True)
        self._worst: dict[str, str] = {}
        self.t = 0.0

    def update(self, row: dict) -> None:
        self._rec.add(sample_from_row(row))
        self.t = row["t"]
        for r in row["robots"]:
            m = r.get("comm_mode")
            if m and MODE_RANK.get(m, 3) >= MODE_RANK.get(self._worst.get(r["team"], "healthy"), 3):
                self._worst[r["team"]] = m

    def stats(self, team: str) -> dict:
        rec = self._rec
        members = [st for st in rec.robots.values() if st.team == team]
        play = rec.team_play_s.get(team, 0.0)
        err_sum = sum(st.err["fused"][0] for st in members)
        err_n = sum(st.err["fused"][1] for st in members)
        return {"chase_flips": sum(st.chase_flips for st in members), "dup_s": rec.dup_s.get(team, 0.0),
                "no_chaser_s": max(play - rec.team_chasers_s.get(team, 0.0), 0.0),
                "fused_err_m": err_sum / err_n if err_n else None, "worst_mode": self._worst.get(team, "unknown"),
                "play_s": play}


class ReactionTimer:
    """Ball events as metrics.py detects them (teleport or kick), the reaction clock of each team's nearest eligible robot, and
    what to show: a flash at the event, a running timer (the newest event of each team), and the final time for FINAL_HOLD_S after it
    stops (reaction, or the 8 s timeout). Events excluded by metrics (nearest robot already on the ball) flash but run no timer."""

    FINAL_HOLD_S = 1.0
    FLASH_S = 0.7

    def __init__(self) -> None:
        self._rec = MetricsRecorder(True)
        self._tracked: dict[int, dict] = {}  # id(_Event) -> record
        self._n_events = self._n_excluded = 0
        self.records: list[dict] = []
        self.t = 0.0

    def update(self, row: dict) -> None:
        s = sample_from_row(row)
        rec = self._rec
        rec.add(s)
        self.t = s.t
        live = {id(e): e for e in rec._open}
        for k, ev in live.items():
            if k not in self._tracked:
                self._tracked[k] = {"team": ev.team, "robot": ev.robot, "t0": ev.t0, "ball": s.ball, "end": None,
                                    "reaction_s": None, "excluded": False, "kind": ev.kind}
                self.records.append(self._tracked[k])
        for e in rec.excluded[self._n_excluded:]:
            self.records.append({"team": e["team"], "robot": e["robot"], "t0": s.t, "ball": s.ball, "end": s.t,
                                 "reaction_s": None, "excluded": True, "kind": e["kind"]})
        self._n_excluded = len(rec.excluded)
        for e in rec.events[self._n_events:]:
            match = next((r for r in self.records if r["end"] is None and not r["excluded"] and r["team"] == e["team"]
                          and r["robot"] == e["robot"] and abs(round(r["t0"], 2) - e["t"]) < 1e-6), None)
            if match:
                match["end"], match["reaction_s"] = s.t, e["reaction_s"]
        self._n_events = len(rec.events)
        self.records = [r for r in self.records if r["end"] is None or s.t - r["end"] < 5.0]

    def display(self, t: float) -> dict:
        """{'flashes': [{team, ball, age}], 'timers': [{team, robot, elapsed, running, censored}]} at time t."""
        flashes, latest = [], {}
        for r in self.records:
            age = t - r["t0"]
            if 0.0 <= age <= self.FLASH_S:
                flashes.append({"team": r["team"], "ball": r["ball"], "age": age / self.FLASH_S})
            if r["excluded"] or r["t0"] > t:
                continue
            if r["end"] is None or t - r["end"] <= self.FINAL_HOLD_S:  # running, or stopped less than FINAL_HOLD_S ago
                if r["team"] not in latest or r["t0"] >= latest[r["team"]]["t0"]:
                    latest[r["team"]] = r
        timers = []
        for r in latest.values():  # one timer per team: the newest event (older ones still count in metrics, they are just not drawn)
            if r["end"] is None:
                timers.append({"team": r["team"], "robot": r["robot"], "elapsed": max(t - r["t0"], 0.0), "running": True,
                               "censored": False})
            else:
                done = r["reaction_s"]
                timers.append({"team": r["team"], "robot": r["robot"], "elapsed": done if done is not None else REACTION_TIMEOUT_S,
                               "running": False, "censored": done is None})
        return {"flashes": flashes, "timers": timers}


# ---- drawing -----------------------------------------------------------------------------------------------------
def text_size(text: str, scale: float = TEXT_SCALE, thick: int = 1) -> tuple[int, int]:
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    return w, h + base


def put_text(img, text, org, scale=TEXT_SCALE, colour=TEXT, thick=1, outline=True) -> None:
    """Text with a dark outline so it reads on the pitch, the tint and the 3D scene."""
    if outline:
        cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, thick, cv2.LINE_AA)


def ground_ring(img, to_px, x, y, radius_m, colour, thick, z=0.02) -> None:
    pts = [to_px(x + radius_m * math.cos(a), y + radius_m * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    if all(p is not None for p in pts):
        cv2.polylines(img, [np.array(pts, np.int32)], True, colour, thick, cv2.LINE_AA)


def chaser_marker(img, apex_xy: tuple[int, int], team: str, size: int = 11) -> None:
    """Filled downward triangle whose tip is at apex_xy (put it just above the robot)."""
    x, y = apex_xy
    tri = np.array([(x, y), (x - size, y - int(size * 1.6)), (x + size, y - int(size * 1.6))], np.int32)
    cv2.fillConvexPoly(img, tri, TEAM[team], cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)


def belief_marks(img, to_px, robots: list[dict], ball: dict, ball_z: float = 0.1, radius: int = 7) -> None:
    """Each robot's fused-ball belief: hollow dot in team colour, a thin line to the true ball."""
    b = to_px(ball["x"], ball["y"], ball_z)
    for r in robots:
        f = r.get("fused_ball")
        if not f or not f.get("valid") or f.get("x") is None or f.get("y") is None:
            continue
        p = to_px(f["x"], f["y"], ball_z)
        if p is None:
            continue
        if b is not None:
            cv2.line(img, p, b, TEAM[r["team"]], 1, cv2.LINE_AA)
        cv2.circle(img, p, radius + 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, p, radius, TEAM[r["team"]], 2, cv2.LINE_AA)


def event_marks(img, to_px, shown: dict, bounds: tuple[int, int, int, int], ball: dict, ball_z: float = 0.1) -> None:
    """Flash ring at each new ball event, and the timers next to the ball (one line per timer), clamped into bounds (x0, y0, x1, y1)."""
    for fl in shown["flashes"]:
        p = to_px(fl["ball"][0], fl["ball"][1], 0.02)
        if p is not None:
            cv2.circle(img, p, int(18 + 60 * fl["age"]), (0, 0, 0), 6, cv2.LINE_AA)
            cv2.circle(img, p, int(18 + 60 * fl["age"]), FLASH, 3, cv2.LINE_AA)
    b = to_px(ball["x"], ball["y"], ball_z)
    if b is None:
        return
    x0, y0, x1, y1 = bounds
    for k, tm in enumerate(sorted(shown["timers"], key=lambda q: q["team"] != "home")):
        text = f"{tm['robot']} " + (f">{REACTION_TIMEOUT_S:.0f} s" if tm["censored"] else f"{tm['elapsed']:.1f} s")
        w, h = text_size(text)
        x = min(max(b[0] + 16, x0 + 4), x1 - w - 14)
        y = min(max(b[1] - 14 + k * (h + 8), y0 + h + 4), y1 - 6)
        cv2.rectangle(img, (x - 4, y - h), (x + w + 4, y + 6), (0, 0, 0) if tm["running"] else FLASH, -1)
        cv2.rectangle(img, (x - 4, y - h), (x + w + 4, y + 6), TEAM[tm["team"]], 2)
        cv2.putText(img, text, (x, y), FONT, TEXT_SCALE, FLASH if tm["running"] else (0, 0, 0), 1, cv2.LINE_AA)


class LabelPlacer:
    """Greedy placement of text boxes so they neither overlap each other nor the reserved boxes (robot discs)."""

    def __init__(self, bounds: tuple[int, int, int, int]):
        self.bounds, self.boxes = bounds, []

    def reserve(self, box: tuple[int, int, int, int]) -> None:
        self.boxes.append(box)

    def place(self, w: int, h: int, candidates: list[tuple[int, int]]) -> tuple[int, int]:
        """candidates: top-left corners in order of preference; returns the first free one (else the first inside bounds)."""
        bx0, by0, bx1, by1 = self.bounds
        fallback = None
        for x, y in candidates:
            x = min(max(x, bx0), bx1 - w)
            y = min(max(y, by0), by1 - h)
            if fallback is None:
                fallback = (x, y)
            if not any(x < b[2] and x + w > b[0] and y < b[3] and y + h > b[1] for b in self.boxes):
                self.boxes.append((x, y, x + w, y + h))
                return x, y
        self.boxes.append((fallback[0], fallback[1], fallback[0] + w, fallback[1] + h))
        return fallback


def hud_lines(st: dict) -> list[tuple[str, str]]:
    err = st["fused_err_m"]
    return [("chase flips", f"{st['chase_flips']}"), ("2+ chasers", f"{st['dup_s']:.1f} s"),
            ("no chaser", f"{st['no_chaser_s']:.1f} s"), ("ball error", "-" if err is None else f"{err:.2f} m")]


def draw_team_hud(img, x: int, y: int, w: int, team: str, st: dict, scale: float = TEXT_SCALE) -> int:
    """One team's counters in a block at (x, y) of width w: a header bar (team stripe + worst comm mode) and four rows.
    Returns the block height."""
    line = int(round(40 * scale))
    mode = st["worst_mode"]
    cv2.rectangle(img, (x, y), (x + w, y + line), MODE_COLOR.get(mode, (120, 120, 120)), -1)
    cv2.rectangle(img, (x, y), (x + 12, y + line), TEAM[team], -1)
    cv2.rectangle(img, (x, y), (x + w, y + line), (0, 0, 0), 1)
    head = f"{TEAM_NAME[team]}  " + ("comm: n/a" if mode == "unknown" else f"worst: {mode}")
    hs = scale
    while hs > 0.45 and text_size(head, hs)[0] > w - 28:
        hs -= 0.02
    cv2.putText(img, head, (x + 20, y + line - int(9 * scale / 0.72)), FONT, hs, MODE_TEXT.get(mode, (255, 255, 255)), 1, cv2.LINE_AA)
    rows = hud_lines(st)
    for k, (name, val) in enumerate(rows):
        base = y + line * (k + 2) - int(7 * scale / 0.72)
        put_text(img, name, (x + 6, base), scale, DIM)
        vw, _ = text_size(val, scale)
        put_text(img, val, (x + w - vw - 6, base), scale, TEXT)
    return line * (len(rows) + 1)


def draw_legend(img, x: int, y: int, max_w: int, scale: float = 0.6) -> int:
    """Icon + word legend wrapped into rows within max_w; returns the height used."""
    row_h = int(34 * scale / 0.6)
    cx, cy = x, y
    for icon, word in LEGEND:
        w, _ = text_size(word, scale)
        if cx + 30 + w > x + max_w and cx > x:
            cx, cy = x, cy + row_h
        ic = (cx + 12, cy + row_h // 2)
        if icon == "tri":
            chaser_marker(img, (ic[0], ic[1] + 8), "home", 8)
        elif icon == "dup":
            cv2.circle(img, ic, 9, DUP_RING, 3, cv2.LINE_AA)
        elif icon == "belief":
            cv2.circle(img, ic, 6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(img, ic, 5, TEAM["home"], 2, cv2.LINE_AA)
        elif icon == "lead":
            cv2.circle(img, ic, 9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.circle(img, ic, 9, (255, 255, 255), 2, cv2.LINE_AA)
        elif icon == "flash":
            cv2.circle(img, ic, 9, FLASH, 3, cv2.LINE_AA)
        cv2.putText(img, word, (cx + 28, cy + row_h // 2 + int(7 * scale / 0.6)), FONT, scale, TEXT, 1, cv2.LINE_AA)
        cx += 28 + w + 12
    return cy - y + row_h


def legend_height(max_w: int, scale: float = 0.6) -> int:
    """Rows the legend needs at width max_w (same wrapping rule as draw_legend)."""
    return draw_legend(np.zeros((200, max(max_w, 1) + 40, 3), np.uint8), 0, 0, max_w, scale)


def caption_height(w: int, label: str, situation_line: str) -> int:
    """44 px when label and situation share a line, 76 px when the situation goes on a second line."""
    lw, _ = text_size(label, 0.85, 2)
    sw, _ = text_size(situation_line)
    return 44 if not situation_line or 12 + lw + 24 + sw <= w - 8 else 76


def draw_caption(img, x: int, y: int, w: int, label: str, situation_line: str) -> int:
    """Caption strip: brain label (bold), then network situation and seed right-aligned on the same line when it fits, else on
    a second line. Returns the strip height."""
    h = caption_height(w, label, situation_line)
    cv2.rectangle(img, (x, y), (x + w, y + h), PAGE_BG, -1)
    cv2.putText(img, label, (x + 12, y + 32), FONT, 0.85, TEXT, 2, cv2.LINE_AA)
    if situation_line:
        sw, _ = text_size(situation_line)
        if h == 44:
            cv2.putText(img, situation_line, (x + w - sw - 10, y + 31), FONT, TEXT_SCALE, DIM, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, situation_line, (x + 12, y + 64), FONT, TEXT_SCALE, DIM, 1, cv2.LINE_AA)
    return h
