"""Simulated GameController: match clock, halves, score, kickoff after goals and out-of-bounds restarts.

Pure Python on a caller-supplied clock, so it can be tested without time passing. It works in the world
(home) frame: in the first half home defends the goal at -x. It never moves anything itself; it returns
events (put the ball here, swap sides) for the bridge to carry out. No penalties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import FieldDims
from .referee import SET_PLAYS, STATES, _lookup

INITIAL, READY, SET, PLAYING, FINISHED = (STATES[k] for k in ("INITIAL", "READY", "SET", "PLAYING", "FINISHED"))
HOME, AWAY = "home", "away"


def other(team: str) -> str:
    return AWAY if team == HOME else HOME


@dataclass(frozen=True)
class MatchCfg:
    half_secs: float = 600.0  # 2 x 10 min, the real rules
    continuous: bool = False  # no halves: the clock just counts up and nobody wins
    initial_secs: float = 3.0
    ready_secs: float = 20.0
    set_secs: float = 3.0
    halftime_secs: float = 5.0  # FINISHED before the second half's INITIAL
    set_play_stop_secs: float = 4.0  # the 'stopped' phase of a restart
    set_play_secs: float = 30.0  # then it must be taken within this long
    hold_initial: bool = False  # wait in INITIAL until READY is pressed (robots stand up first)
    first_kickoff: str = HOME
    teams: tuple = (HOME, AWAY)  # who is on the pitch: a restart never goes to a team that is not playing
    ball_radius: float = 0.105
    goal_height: float = 1.2
    ball_taken_dist: float = 0.25  # a restart ends once the ball has moved this far from where it was placed


@dataclass(frozen=True)
class PlaceBall:
    x: float
    y: float
    why: str = ""


@dataclass(frozen=True)
class Goal:
    team: str  # who scored


@dataclass(frozen=True)
class SwapSides:
    """Half-time: teams change ends. The bridge flips team frames and puts robots back on the sideline."""


@dataclass(frozen=True)
class Snapshot:
    state: int
    set_play: int
    kicking_team: str
    stopped: bool
    secondary_time: int
    first_half: bool
    secs_remaining: int
    score: dict
    awaiting_start: bool = False


@dataclass
class GameController:
    cfg: MatchCfg
    t0: float
    field_dims: FieldDims = field(default_factory=FieldDims)

    def __post_init__(self):
        self.state = INITIAL
        self.state_since = self.t0
        self.first_half = True
        self.score = {HOME: 0, AWAY: 0}
        self.play_elapsed = 0.0
        self._last_t = self.t0
        self.kicking_team = self.cfg.first_kickoff
        self.set_play = 0
        self.stopped = False
        self._sp_started = 0.0
        self._sp_ball: tuple[float, float] | None = None
        self.auto = not self.cfg.hold_initial
        self.awaiting_start = self.cfg.hold_initial
        self.last_event = ""

    def _opponent(self, team: str) -> str:
        """The team that gets the restart when `team` lost the ball, or the only team there is."""
        o = other(team)
        return o if o in self.cfg.teams else team

    # ---- geometry --------------------------------------------------------
    @property
    def home_attacks(self) -> int:
        """+1 when home attacks the goal at +x (first half), -1 in the second."""
        return 1 if self.first_half else -1

    def _goal_for(self, x: float, y: float, z: float) -> str | None:
        fd, r = self.field_dims, self.cfg.ball_radius
        if abs(x) <= fd.length / 2 + r:
            return None
        if abs(y) < fd.goal_width / 2 and z < self.cfg.goal_height:
            attacks_plus = x > 0
            return HOME if attacks_plus == (self.home_attacks == 1) else AWAY
        return None

    # ---- time ------------------------------------------------------------
    def _enter(self, state: int, now: float) -> None:
        self.state, self.state_since = state, now

    def update(self, now: float, ball: tuple[float, float, float], last_touch: str | None) -> list:
        """Advance the match to `now`. ball is (x, y, z) in the world frame; last_touch is a team or None."""
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        events: list = []
        c = self.cfg

        if self.state == PLAYING:
            self.play_elapsed += dt

        # restart bookkeeping: stop phase -> execute -> taken or expired
        if self.set_play:
            age = now - self._sp_started
            if self.stopped and age >= c.set_play_stop_secs:
                self.stopped = False
            sx, sy = self._sp_ball or (ball[0], ball[1])
            taken = math.hypot(ball[0] - sx, ball[1] - sy) > c.ball_taken_dist and not self.stopped
            if taken or age >= c.set_play_stop_secs + c.set_play_secs:
                self.set_play, self.stopped, self._sp_ball = 0, False, None

        if self.auto:
            events += self._advance(now)

        if self.state == PLAYING and not (self.set_play and self.stopped):
            events += self._ball_rules(now, ball, last_touch)
        return events

    def _advance(self, now: float) -> list:
        c, age = self.cfg, now - self.state_since
        if self.state == INITIAL and age >= c.initial_secs:
            self._enter(READY, now)
        elif self.state == READY and age >= c.ready_secs:
            self._enter(SET, now)
        elif self.state == SET and age >= c.set_secs:
            self._enter(PLAYING, now)
        elif self.state == PLAYING and not c.continuous and self.play_elapsed >= c.half_secs:
            self.set_play, self.stopped, self._sp_ball = 0, False, None
            self._enter(FINISHED, now)
            self.last_event = "half over" if self.first_half else "full time"
        elif self.state == FINISHED and self.first_half and not c.continuous and age >= c.halftime_secs:
            self.first_half = False
            self.play_elapsed = 0.0
            self.kicking_team = self._opponent(c.first_kickoff)
            self._enter(INITIAL, now)
            self.last_event = "second half"
            return [SwapSides(), PlaceBall(0.0, 0.0, "second half kickoff")]
        return []

    # ---- ball rules ------------------------------------------------------
    def _ball_rules(self, now: float, ball, last_touch: str | None) -> list:
        x, y, z = ball
        fd, r = self.field_dims, self.cfg.ball_radius
        scorer = self._goal_for(x, y, z)
        if scorer is not None:
            self.score[scorer] += 1
            self.kicking_team = self._opponent(scorer)
            self.set_play, self.stopped, self._sp_ball = 0, False, None
            self._enter(READY, now)
            self.auto = True
            self.last_event = f"goal {scorer}"
            return [Goal(scorer), PlaceBall(0.0, 0.0, "kickoff")]
        half_l, half_w = fd.length / 2, fd.width / 2
        if abs(y) > half_w + r and abs(x) <= half_l + r:
            team = self._opponent(last_touch or HOME)
            px = max(-half_l + 1.0, min(half_l - 1.0, x))
            py = math.copysign(half_w, y)
            return self._restart("THROW_IN", team, px, py, now)
        if abs(x) > half_l + r:
            defender = AWAY if (x > 0) == (self.home_attacks == 1) else HOME  # who defends the goal behind x
            attacker = other(defender)
            touched = last_touch or attacker
            sign = math.copysign(1.0, x)
            if defender not in self.cfg.teams:  # nobody defends this goal: the team that is there takes a goal kick
                return self._restart("GOAL_KICK", attacker, sign * (half_l - fd.goal_area_length),
                                     math.copysign(fd.goal_area_width / 2, y), now)
            if touched == defender:
                return self._restart("CORNER_KICK", attacker, sign * (half_l - 0.3), math.copysign(half_w - 0.3, y), now)
            return self._restart("GOAL_KICK", defender, sign * (half_l - fd.goal_area_length),
                                 math.copysign(fd.goal_area_width / 2, y), now)
        return []

    def _restart(self, kind: str, team: str, x: float, y: float, now: float) -> list:
        self.set_play = SET_PLAYS[kind]
        self.kicking_team = team
        self.stopped = True
        self._sp_started = now
        self._sp_ball = (x, y)
        self.last_event = f"{kind.lower()} {team}"
        return [PlaceBall(x, y, kind.lower())]

    # ---- manual control (Control tab) -------------------------------------
    def override(self, now: float, state=None, set_play=None, kicking=None, stopped=None) -> None:
        """Same fields as Referee.override; kicking is 'us' (home) / 'opponent' (away) or a team name."""
        resume = isinstance(state, str) and state.upper() == "SCHEDULE"
        new_state = None if state is None or resume else _lookup(STATES, state, "state")
        new_set_play = None if set_play is None else _lookup(SET_PLAYS, set_play, "set_play")
        if kicking is not None and kicking not in ("us", "opponent", HOME, AWAY):
            raise ValueError(f"kicking must be 'us' or 'opponent', got {kicking!r}")
        if stopped is not None and not isinstance(stopped, bool):
            raise ValueError("stopped must be a bool")

        if resume:
            self.auto = True
        elif new_state is not None:
            if self.awaiting_start and new_state == READY:
                self.awaiting_start, self.auto = False, True
            else:
                self.awaiting_start, self.auto = False, False
            self._enter(new_state, now)
        if new_set_play is not None:
            self.set_play = new_set_play
            self._sp_started = now
            self._sp_ball = None
            self.stopped = new_set_play != 0 if stopped is None else stopped
        elif stopped is not None:
            self.stopped = stopped
        if kicking is not None:
            self.kicking_team = HOME if kicking in ("us", HOME) else AWAY

    def finish_half(self, now: float) -> None:
        """Skip to the end of the current half (the Control tab's 'next half' button)."""
        self.auto = True
        self.awaiting_start = False
        self.play_elapsed = self.cfg.half_secs
        self._enter(PLAYING, now)

    # ---- output ----------------------------------------------------------
    def snapshot(self, now: float) -> Snapshot:
        c = self.cfg
        secs = 9999 if c.continuous else int(max(0.0, c.half_secs - self.play_elapsed))
        secondary = 0
        if self.set_play:
            limit = c.set_play_stop_secs if self.stopped else c.set_play_stop_secs + c.set_play_secs
            secondary = int(math.ceil(max(0.0, limit - (now - self._sp_started))))
        return Snapshot(self.state, self.set_play, self.kicking_team, self.stopped, secondary,
                        self.first_half, secs, dict(self.score), self.awaiting_start)
