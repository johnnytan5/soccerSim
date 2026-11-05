"""Match metrics from a trace of samples (ground truth plus what each brain published). Pure Python, no ROS/MuJoCo.

A `Sample` is one instant of PLAY or not: the simulated clock, the ground-truth ball, and per robot its pose, its
decision (/brain/decision, else /speak) and, when the brain publishes them, its comm mode and ball errors against truth.
Every optional field may be missing (older brains publish no comm mode): the metric that needs it is then just absent.

Signals (all documented in docs/batch_matches.md):
  chasing        decision in CHASE_DECISIONS. The fallback "who is the striker right now" for brains that publish no role.
  role flip      PRIMARY: a robot's player_role (brain's /brain/world_model "brain" object) changing between two PLAY
                 samples; FALLBACK (baseline brain, no player_role): its chasing flag changing. `role_signal` says which.
  chase flip     a robot's chasing flag changing (always computed, decision based).
  lead change    a robot's im_lead changing (only where the brain publishes it).
  decision flip  any change of decision string between two PLAY samples.
  duplicate      PRIMARY (duplicate_striker_frac): PLAY time with >= 2 outfield robots of the team with im_lead true,
                 over the team's PLAY time with lead data; FALLBACK when no robot of the team publishes im_lead:
                 >= 2 robots chasing (decision based, keeper included). `duplicate_signal` says which. Also reported
                 separately: duplicate_lead_frac, duplicate_role_frac (>= 2 outfield robots with player_role striker
                 AND chasing) and duplicate_decision_frac.
  blackout       samples carry `blackout` (the bridge's view of the batch runner's blackout window); the summary then
                 has "segments": {"blackout": ..., "outside": ...} with the team metrics of each period alone.
  ball event     the ground-truth ball jumps more than TELEPORT_M between samples (a placement) or its speed rises
                 through KICK_SPEED_MS from below (a kick).
  reaction time  events where the nearest eligible robot already faces the ball within 30 degrees AND is already chasing
                 are excluded (counted in reaction_excluded), not scored 0. Otherwise from the event until the team's nearest robot (not the keeper; for a kick not a robot that was within
                 KICKER_M of the ball, which is most likely the kicker) faces the ball within 30 degrees, or switches
                 to a chasing decision it was not in at the event. Censored (None) after REACTION_TIMEOUT_S. Reported as median and
                 p90 over the uncensored events plus the censored count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

CHASE_DECISIONS = frozenset({"chase", "adjust", "kick", "dribble", "cross", "auto_visual_kick"})
TELEPORT_M = 1.0
KICK_SPEED_MS = 1.5
KICK_REARM_S = 2.0
KICKER_M = 0.7
FACING_RAD = math.radians(30.0)
REACTION_TIMEOUT_S = 8.0
MAX_DT_S = 0.5  # a gap longer than this between samples is not integrated


@dataclass
class RobotSample:
    key: str  # "H2", "A3"
    team: str  # "home" / "away"
    x: float
    y: float
    yaw: float
    decision: str = ""  # "" / "-" = nothing published yet
    keeper: bool = False
    comm_mode: str | None = None
    fused_err: float | None = None  # |fused ball - truth|, None = no valid estimate or no such field
    local_err: float | None = None
    role: str | None = None  # blackboard player_role, None = the brain does not publish it
    rank: int | None = None  # outfield_rank
    im_lead: bool | None = None
    role_own_view: bool | None = None


@dataclass
class Sample:
    t: float  # simulated seconds
    playing: bool
    ball: tuple[float, float]
    robots: list[RobotSample] = field(default_factory=list)
    blackout: bool = False  # inside the batch runner's blackout window


def chasing(decision: str) -> bool:
    return decision in CHASE_DECISIONS


def _angle_diff(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _facing(r: RobotSample, ball: tuple[float, float]) -> bool:
    return _angle_diff(math.atan2(ball[1] - r.y, ball[0] - r.x), r.yaw) <= FACING_RAD


def _known(decision: str) -> bool:
    return decision not in ("", "-")


class _Robot:
    def __init__(self, key: str, team: str):
        self.key, self.team = key, team
        self.play_s = 0.0
        self.chase_flips = 0
        self.role_changes = 0
        self.lead_changes = 0
        self.has_role = self.has_lead = False
        self.own_view_s = 0.0
        self.decision_flips = 0
        self.chase_s = 0.0
        self.modes: dict[str, float] = {}
        self.err = {"fused": [0.0, 0, 0], "local": [0.0, 0, 0]}  # sum, valid samples, all samples
        self.last_decision: str | None = None
        self.last_chasing: bool | None = None
        self.last_role: str | None = None
        self.last_lead: bool | None = None


@dataclass
class _Event:
    t0: float
    kind: str
    team: str
    robot: str
    decision0: str


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo), 3)


class MetricsRecorder:
    def __init__(self, _part: bool = False):
        self._parts = None if _part else {"blackout": MetricsRecorder(True), "outside": MetricsRecorder(True)}
        self.saw_blackout = False
        self.bo_window: list[float] | None = None  # [first, last] PLAY second (recorder clock) of a blackout sample
        self.robots: dict[str, _Robot] = {}
        self.play_s = 0.0
        self.team_play_s: dict[str, float] = {}
        self.dup_s: dict[str, float] = {}
        self.team_chasers_s: dict[str, float] = {}
        self.dup_lead_s: dict[str, float] = {}
        self.lead_s: dict[str, float] = {}
        self.dup_role_s: dict[str, float] = {}
        self.role_s: dict[str, float] = {}
        self.excluded: list[dict] = []
        self.events: list[dict] = []
        self.samples = 0
        self._open: list[_Event] = []
        self._prev: Sample | None = None
        self._prev_speed = 0.0
        self._kick_armed_at = -1e9

    # ---- feeding ----
    def add(self, s: Sample) -> None:
        """Feed one sample. The two period recorders see the other period as a stoppage (no integration, no memory)."""
        if self._parts is not None:
            self.saw_blackout = self.saw_blackout or s.blackout
            if s.blackout and s.playing:
                if self.bo_window is None:
                    self.bo_window = [self.play_s, self.play_s]
                self.bo_window[1] = self.play_s + (s.t - self._prev.t if self._prev is not None and 0 < s.t - self._prev.t <= MAX_DT_S else 0.0)
            for name, part in self._parts.items():
                part.add(replace(s, playing=s.playing and s.blackout == (name == "blackout")))
        self._add(s)

    def _add(self, s: Sample) -> None:
        self.samples += 1
        prev = self._prev
        dt = s.t - prev.t if prev is not None else 0.0
        valid_dt = 0.0 < dt <= MAX_DT_S
        both_playing = prev is not None and prev.playing and s.playing and valid_dt
        speed = 0.0
        if both_playing:
            jump = math.hypot(s.ball[0] - prev.ball[0], s.ball[1] - prev.ball[1])
            speed = jump / dt
            if jump > TELEPORT_M:
                self._start_events(s, prev, "teleport")
                speed = self._prev_speed  # the jump is not a speed
            elif speed > KICK_SPEED_MS >= self._prev_speed and s.t - self._kick_armed_at > KICK_REARM_S:
                self._kick_armed_at = s.t
                self._start_events(s, prev, "kick")
        self._prev_speed = speed
        self._check_events(s)
        if both_playing:
            self._integrate(s, dt)
        for r in s.robots:  # decision memory runs through stoppages so a flip across a pause is not counted twice
            st = self._robot(r)
            if not s.playing:
                st.last_decision = st.last_chasing = st.last_role = st.last_lead = None
        self._prev = s

    def _robot(self, r: RobotSample) -> _Robot:
        st = self.robots.get(r.key)
        if st is None:
            st = self.robots[r.key] = _Robot(r.key, r.team)
        return st

    def _integrate(self, s: Sample, dt: float) -> None:
        self.play_s += dt
        per_team: dict[str, int] = {}
        lead_n: dict[str, int] = {}
        role_n: dict[str, int] = {}
        has_lead: set[str] = set()
        has_role: set[str] = set()
        for r in s.robots:
            st = self._robot(r)
            st.play_s += dt
            self.team_play_s.setdefault(r.team, 0.0)
            per_team.setdefault(r.team, 0)
            if _known(r.decision):
                is_chase = chasing(r.decision)
                if st.last_decision is not None and r.decision != st.last_decision:
                    st.decision_flips += 1
                if st.last_chasing is not None and is_chase != st.last_chasing:
                    st.chase_flips += 1
                st.last_decision, st.last_chasing = r.decision, is_chase
                if is_chase:
                    st.chase_s += dt
                    per_team[r.team] += 1
            if r.role is not None:
                st.has_role = True
                if st.last_role is not None and r.role != st.last_role:
                    st.role_changes += 1
                st.last_role = r.role
                if not r.keeper:
                    has_role.add(r.team)
                    if r.role == "striker" and chasing(r.decision):
                        role_n[r.team] = role_n.get(r.team, 0) + 1
            if r.im_lead is not None:
                st.has_lead = True
                if st.last_lead is not None and r.im_lead != st.last_lead:
                    st.lead_changes += 1
                st.last_lead = r.im_lead
                if not r.keeper:
                    has_lead.add(r.team)
                    if r.im_lead:
                        lead_n[r.team] = lead_n.get(r.team, 0) + 1
            if r.role_own_view:
                st.own_view_s += dt
            st.modes[r.comm_mode or "unknown"] = st.modes.get(r.comm_mode or "unknown", 0.0) + dt
            for name, value in (("fused", r.fused_err), ("local", r.local_err)):
                acc = st.err[name]
                acc[2] += 1
                if value is not None:
                    acc[0] += value
                    acc[1] += 1
        for team in has_lead:
            self.lead_s[team] = self.lead_s.get(team, 0.0) + dt
            if lead_n.get(team, 0) >= 2:
                self.dup_lead_s[team] = self.dup_lead_s.get(team, 0.0) + dt
        for team in has_role:
            self.role_s[team] = self.role_s.get(team, 0.0) + dt
            if role_n.get(team, 0) >= 2:
                self.dup_role_s[team] = self.dup_role_s.get(team, 0.0) + dt
        for team, n in per_team.items():
            self.team_play_s[team] += dt
            if n >= 2:
                self.dup_s[team] = self.dup_s.get(team, 0.0) + dt
            if n >= 1:
                self.team_chasers_s[team] = self.team_chasers_s.get(team, 0.0) + dt

    # ---- ball events ----
    def _start_events(self, s: Sample, prev: Sample, kind: str) -> None:
        for team in sorted({r.team for r in s.robots}):
            cands = [r for r in s.robots if r.team == team and not r.keeper]
            if kind == "kick":
                cands = [r for r in cands if math.hypot(r.x - prev.ball[0], r.y - prev.ball[1]) > KICKER_M]
            if not cands:
                continue
            near = min(cands, key=lambda r: math.hypot(r.x - s.ball[0], r.y - s.ball[1]))
            if _facing(near, s.ball) and chasing(near.decision):  # already on it: nothing to react to
                self.excluded.append({"t": round(s.t, 2), "kind": kind, "team": team, "robot": near.key,
                                      "reaction_s": None, "excluded": True})
                continue
            self._open.append(_Event(s.t, kind, team, near.key, near.decision))

    def _check_events(self, s: Sample) -> None:
        if not self._open:
            return
        by_key = {r.key: r for r in s.robots}
        still = []
        for ev in self._open:
            r = by_key.get(ev.robot)
            reacted = False
            if r is not None:
                facing = _facing(r, s.ball)
                switched = chasing(r.decision) and r.decision != ev.decision0
                reacted = facing or switched
            if reacted:
                self.events.append({"t": round(ev.t0, 2), "kind": ev.kind, "team": ev.team, "robot": ev.robot,
                                    "reaction_s": round(s.t - ev.t0, 3)})
            elif s.t - ev.t0 > REACTION_TIMEOUT_S:
                self.events.append({"t": round(ev.t0, 2), "kind": ev.kind, "team": ev.team, "robot": ev.robot,
                                    "reaction_s": None})
            else:
                still.append(ev)
        self._open = still

    # ---- results ----
    def summary(self) -> dict:
        minutes = self.play_s / 60.0
        robots = {}
        for key, st in sorted(self.robots.items()):
            role_flips = st.role_changes if st.has_role else st.chase_flips
            robots[key] = {
                "team": st.team, "play_s": round(st.play_s, 2),
                "role_signal": "player_role" if st.has_role else "decision",
                "role_flips": role_flips, "role_flips_per_min": _rate(role_flips, st.play_s),
                "chase_flips": st.chase_flips, "chase_flips_per_min": _rate(st.chase_flips, st.play_s),
                "lead_changes": st.lead_changes if st.has_lead else None,
                "lead_changes_per_min": _rate(st.lead_changes, st.play_s) if st.has_lead else None,
                "role_own_view_frac": _frac(st.own_view_s, st.play_s) if st.has_role else None,
                "decision_flips": st.decision_flips, "decision_flips_per_min": _rate(st.decision_flips, st.play_s),
                "chase_frac": _frac(st.chase_s, st.play_s),
                "fused_ball_err_m": _mean(st.err["fused"]), "fused_ball_valid_frac": _valid(st.err["fused"]),
                "local_ball_err_m": _mean(st.err["local"]), "local_ball_valid_frac": _valid(st.err["local"]),
                "comm_mode_share": _share(st.modes),
            }
        teams = {}
        for team in sorted({st.team for st in self.robots.values()}):
            members = [st for st in self.robots.values() if st.team == team]
            tplay = self.team_play_s.get(team, 0.0)
            has_role = any(st.has_role for st in members)
            has_lead = any(st.has_lead for st in members)
            role_flips = sum(st.role_changes if has_role else st.chase_flips for st in members)
            chase_flips = sum(st.chase_flips for st in members)
            lead_changes = sum(st.lead_changes for st in members)
            scored = [e["reaction_s"] for e in self.events if e["team"] == team]
            excluded = sum(1 for e in self.excluded if e["team"] == team)
            done = sorted(v for v in scored if v is not None)
            modes: dict[str, float] = {}
            for st in members:
                for m, v in st.modes.items():
                    modes[m] = modes.get(m, 0.0) + v
            pooled = {n: [sum(st.err[n][i] for st in members) for i in range(3)] for n in ("fused", "local")}
            dup_lead = _frac(self.dup_lead_s.get(team, 0.0), self.lead_s.get(team, 0.0))
            dup_decision = _frac(self.dup_s.get(team, 0.0), tplay)
            teams[team] = {
                "robots": len(members),
                "role_signal": "player_role" if has_role else "decision",
                "role_flips": role_flips, "role_flips_per_min": _rate(role_flips, tplay),
                "chase_flips_per_min": _rate(chase_flips, tplay),
                "lead_changes": lead_changes if has_lead else None,
                "lead_changes_per_min": _rate(lead_changes, tplay) if has_lead else None,
                "decision_flips": sum(st.decision_flips for st in members),
                "decision_flips_per_min": _rate(sum(st.decision_flips for st in members), tplay),
                "duplicate_signal": "im_lead" if has_lead else "decision",
                "duplicate_striker_frac": dup_lead if has_lead else dup_decision,
                "duplicate_lead_frac": dup_lead if has_lead else None,
                "duplicate_role_frac": _frac(self.dup_role_s.get(team, 0.0), self.role_s.get(team, 0.0))
                if has_role else None,
                "duplicate_decision_frac": dup_decision,
                "chaser_present_frac": _frac(self.team_chasers_s.get(team, 0.0), tplay),
                "ball_events": len(scored) + excluded, "reactions": len(done),
                "reaction_excluded": excluded,
                "reaction_median_s": _percentile(done, 0.5), "reaction_p90_s": _percentile(done, 0.9),
                "reaction_mean_s": round(sum(done) / len(done), 3) if done else None,
                "reaction_censored": len(scored) - len(done),
                "fused_ball_err_m": _mean(pooled["fused"]), "fused_ball_valid_frac": _valid(pooled["fused"]),
                "local_ball_err_m": _mean(pooled["local"]), "local_ball_valid_frac": _valid(pooled["local"]),
                "comm_mode_share": _share(modes),
            }
        out = {"play_s": round(self.play_s, 2), "play_min": round(minutes, 3), "samples": self.samples,
               "teams": teams, "robots": robots, "ball_events": sorted(self.events + self.excluded, key=lambda e: e["t"])}
        if self._parts is not None and self.saw_blackout:
            if self.bo_window is not None:
                start, end = (round(v, 2) for v in self.bo_window)
                out["blackout_window"] = {"start_play_s": start, "end_play_s": end, "duration_s": round(end - start, 2)}
            out["segments"] = {}
            for name, part in self._parts.items():
                sub = part.summary()
                out["segments"][name] = {"play_s": sub["play_s"], "teams": sub["teams"]}
        return out


def _rate(n: int, secs: float) -> float | None:
    return round(n / (secs / 60.0), 3) if secs > 0 else None


def _frac(part: float, whole: float) -> float | None:
    return round(part / whole, 4) if whole > 0 else None


def _mean(acc: list) -> float | None:
    return round(acc[0] / acc[1], 4) if acc[1] else None


def _valid(acc: list) -> float | None:
    return round(acc[1] / acc[2], 4) if acc[2] else None


def _share(modes: dict[str, float]) -> dict[str, float]:
    total = sum(modes.values())
    return {m: round(v / total, 4) for m, v in sorted(modes.items())} if total > 0 else {}


# ---- one match -> one flat CSV row ----
CSV_FIELDS = ["profile", "brain", "match", "seed", "roster", "play_s", "wall_s", "goals_home", "goals_away", "goal_diff",
              "blackout_start_play_s", "blackout_end_play_s", "blackout_s"]
# appended after every other column, so rows added to a CSV written before they existed stay aligned
H2H_FIELDS = ["home_brain", "away_brain", "home_profile", "away_profile"]
TEAM_FIELDS = ["role_signal", "role_flips_per_min", "chase_flips_per_min", "lead_changes_per_min",
               "decision_flips_per_min", "duplicate_signal", "duplicate_striker_frac", "duplicate_lead_frac",
               "duplicate_role_frac", "duplicate_decision_frac", "chaser_present_frac",
               "ball_events", "reaction_excluded", "reaction_median_s", "reaction_p90_s", "reaction_censored",
               "fused_ball_err_m", "fused_ball_valid_frac", "local_ball_err_m", "local_ball_valid_frac"]
SEGMENT_FIELDS = ["role_flips_per_min", "lead_changes_per_min", "duplicate_striker_frac", "ball_events",
                  "reaction_excluded", "reaction_median_s", "reaction_p90_s", "reaction_censored"]
SEGMENTS = (("blackout", "bo"), ("outside", "out"))  # (summary key, CSV column prefix)
MODES = ("healthy", "degraded", "isolated", "unknown")  # extras (whatever the brain names them) go in the JSON only


def csv_columns(modes: tuple[str, ...] = MODES) -> list[str]:
    cols = list(CSV_FIELDS)
    for team in ("home", "away"):
        cols += [f"{team}_{f}" for f in TEAM_FIELDS] + [f"{team}_mode_{m}" for m in modes]
        cols += [f"{team}_{pre}_{f}" for _, pre in SEGMENTS for f in SEGMENT_FIELDS]
    return cols + H2H_FIELDS


def csv_row(meta: dict, summary: dict, modes: tuple[str, ...] = MODES) -> dict:
    row = {k: meta.get(k) for k in CSV_FIELDS}
    for team in ("home", "away"):
        t = summary.get("teams", {}).get(team, {})
        for f in TEAM_FIELDS:
            row[f"{team}_{f}"] = t.get(f)
        share = t.get("comm_mode_share", {})
        for m in modes:
            row[f"{team}_mode_{m}"] = share.get(m)
        for seg, pre in SEGMENTS:
            st = summary.get("segments", {}).get(seg, {}).get("teams", {}).get(team, {})
            for f in SEGMENT_FIELDS:
                row[f"{team}_{pre}_{f}"] = st.get(f)
    row.update({k: meta.get(k) for k in H2H_FIELDS})
    return row
