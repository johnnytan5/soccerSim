"""Play statistics of full games from their traces (batch_matches --half-minutes ... --trace), per game and team.

  pixi run python scripts/game_stats.py results/games/*.trace.jsonl [--out stats.json]

Over PLAYING rows only (rows with `state` other than PLAYING are skipped; traces without `state` count every row):
  possession   share of the time the robot nearest the ball (within 0.6 m) is on this team; time with nobody that close
               is "loose" and counts for neither team
  territory    share of the time the ball is in the opponent's half (home attacks +x in the first half; sides swap at
               half-time, read from the trace's `half` field)
  chasers      mean number of outfield robots in a chasing decision (soccer_sim.metrics.chasing)
  chase_share  share of outfield robot-time spent in a chasing decision
  first_touch  share of ball events (metrics' kick or teleport detection) after which this team's robot is the first within
               0.6 m of the ball
Goals and the team's brain come from the match JSON next to each trace.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_sim.metrics import chasing  # noqa: E402

NEAR_M = 0.6
TEAMS = ("home", "away")


def attack_sign(team: str, half: int | None) -> int:
    """+1 when this team attacks +x. Home attacks +x in the first half."""
    s = 1 if team == "home" else -1
    return -s if half == 2 else s


def game_stats(trace: Path) -> dict:
    rows = [json.loads(line) for line in trace.open()]
    meta_path = trace.with_name(trace.name.replace(".trace.jsonl", ".json"))
    meta = json.loads(meta_path.read_text())["meta"] if meta_path.exists() else {}
    acc = {t: {"poss": 0.0, "terr": 0.0, "chasers": 0.0, "chase_rt": 0.0, "robot_t": 0.0, "first": 0} for t in TEAMS}
    total = loose = 0.0
    events = 0
    pending = None  # last ball position at an event, waiting for a first touch
    prev = None
    for r in rows:
        if r.get("state", "PLAYING") not in ("PLAYING", "playing"):
            prev = None
            continue
        dt = 0.0 if prev is None else max(0.0, min(0.5, r["t"] - prev["t"]))
        prev = r
        if dt == 0.0:
            continue
        bx, by = r["ball"]["x"], r["ball"]["y"]
        near = min(r["robots"], key=lambda q: math.hypot(q["x"] - bx, q["y"] - by))
        dnear = math.hypot(near["x"] - bx, near["y"] - by)
        total += dt
        if dnear <= NEAR_M:
            acc[near["team"]]["poss"] += dt
        else:
            loose += dt
        half = r.get("half")
        for t in TEAMS:
            if bx * attack_sign(t, half) > 0:
                acc[t]["terr"] += dt
            out = [q for q in r["robots"] if q["team"] == t and q["id"] != 1]
            n = sum(1 for q in out if chasing(q.get("decision")))
            acc[t]["chasers"] += n * dt
            acc[t]["chase_rt"] += n * dt
            acc[t]["robot_t"] += len(out) * dt
        # ball events: a jump of the ball's speed, approximated by displacement over one row (> 1.5 m/s)
        speed = r.get("ball_speed")
        if speed is None and "_last_ball" in acc:
            lx, ly, lt = acc["_last_ball"]
            speed = math.hypot(bx - lx, by - ly) / max(1e-3, r["t"] - lt)
        acc["_last_ball"] = (bx, by, r["t"])
        if speed is not None and speed > 1.5 and pending is None:
            pending = True
            events += 1
        elif pending and dnear <= NEAR_M:
            acc[near["team"]]["first"] += 1
            pending = None
    acc.pop("_last_ball", None)
    out = {"game": trace.name.replace(".trace.jsonl", ""), "profile": meta.get("profile"), "seed": meta.get("seed"),
           "play_s": round(total, 1), "loose_share": round(loose / total, 3) if total else None, "ball_events": events,
           "teams": {}}
    for t in TEAMS:
        a = acc[t]
        out["teams"][t] = {
            "brain": meta.get(f"{t}_brain"),
            "goals": meta.get(f"goals_{t}"),
            "possession": round(a["poss"] / total, 3) if total else None,
            "territory": round(a["terr"] / total, 3) if total else None,
            "chasers": round(a["chasers"] / total, 3) if total else None,
            "chase_share": round(a["chase_rt"] / a["robot_t"], 3) if a["robot_t"] else None,
            "first_touch": round(a["first"] / events, 3) if events else None,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    games = [game_stats(Path(p)) for p in sorted(a.traces)]
    for g in games:
        line = "  ".join(f"{t}({g['teams'][t]['brain']}): goals {g['teams'][t]['goals']} poss {g['teams'][t]['possession']} "
                         f"terr {g['teams'][t]['territory']} chasers {g['teams'][t]['chasers']} "
                         f"first {g['teams'][t]['first_touch']}" for t in TEAMS)
        print(f"{g['game']}: {line}  loose {g['loose_share']}  events {g['ball_events']}")
    if a.out:
        Path(a.out).write_text(json.dumps(games, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
