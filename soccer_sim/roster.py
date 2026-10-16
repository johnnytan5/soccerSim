"""Who is on the pitch: players per team, their roles, entry spots and brain settings. Pure Python, no ROS/MuJoCo."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

HOME, AWAY = "home", "away"
TEAMS = (HOME, AWAY)
ROLES = ("goal_keeper", "striker", "defender")
GOALKEEPER_ID = 1  # brain makes this player id the keeper whatever `role` says
MAX_PLAYERS = 5
TEAM_NUMBERS = {HOME: 131, AWAY: 1}  # GameController team numbers (131 = config.yaml's game.team_id)

_DEFAULT_ROLES = {1: "goal_keeper", 2: "striker", 3: "striker", 4: "defender", 5: "defender"}
# Entry spot in the team's own frame: x along our half, which touchline. Brain's enter_field prior is
# x in [-L/2, -circle radius] just outside the touchline, facing the pitch.
_ENTRY = {1: (-9.0, "right"), 2: (-6.0, "left"), 3: (-6.0, "right"), 4: (-3.5, "left"), 5: (-3.5, "right")}
_TOUCHLINE_Y = 7.5


def default_role(player_id: int) -> str:
    return _DEFAULT_ROLES.get(player_id, "striker")


@dataclass(frozen=True)
class PlayerSpec:
    team: str
    player_id: int
    role: str

    @property
    def team_number(self) -> int:
        return TEAM_NUMBERS[self.team]

    @property
    def entry_side(self) -> str:
        return _ENTRY[self.player_id][1]

    def entry_pose(self) -> tuple[float, float, float]:
        """x, y, yaw in the player's own team frame."""
        x, side = _ENTRY[self.player_id]
        return (x, -_TOUCHLINE_Y, math.pi / 2) if side == "right" else (x, _TOUCHLINE_Y, -math.pi / 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerSpec":
        return cls(str(d.get("team", HOME)), int(d.get("player_id", 0)), str(d.get("role", "")))


def validate_roster(players: list[PlayerSpec]) -> list[str]:
    """Plain-language problems; empty means the roster can be played."""
    errs: list[str] = []
    if not players:
        return ["add at least one player"]
    for team in TEAMS:
        mine = [p for p in players if p.team == team]
        ids = [p.player_id for p in mine]
        label = "home" if team == HOME else "away"
        for p in mine:
            if not 1 <= p.player_id <= MAX_PLAYERS:
                errs.append(f"{label} player ids are 1-{MAX_PLAYERS} (got {p.player_id})")
            if p.role not in ROLES:
                errs.append(f"{label} player {p.player_id}: role must be one of {', '.join(ROLES)}")
        if len(set(ids)) != len(ids):
            errs.append(f"{label} team has two players with the same id")
        keepers = [p for p in mine if p.role == "goal_keeper"]
        if len(keepers) > 1:
            errs.append(f"{label} team can have only one goalkeeper")
        for p in keepers:
            if p.player_id != GOALKEEPER_ID:
                errs.append(f"{label} goalkeeper must be player {GOALKEEPER_ID}")
        for p in mine:
            if p.player_id == GOALKEEPER_ID and p.role != "goal_keeper":
                errs.append(f"{label} player {GOALKEEPER_ID} is always the goalkeeper")
    if not any(p.team == HOME for p in players):
        errs.append("the home team needs at least one player")
    return list(dict.fromkeys(errs))


def roster_from_json(text: str) -> list[PlayerSpec]:
    data = json.loads(text) if text else []
    if not isinstance(data, list):
        raise ValueError("roster must be a JSON list")
    return [PlayerSpec.from_dict(d) for d in data]


def roster_to_json(players: list[PlayerSpec]) -> str:
    return json.dumps([p.to_dict() for p in players], separators=(",", ":"))


def standard_team(team: str, ids=(1, 2, 3, 4, 5)) -> list[PlayerSpec]:
    return [PlayerSpec(team, i, default_role(i)) for i in ids]
