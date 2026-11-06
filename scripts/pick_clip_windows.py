"""Pick the clip window that best shows one finding, from a new-brain trace and a pre-change trace of the same situation and seed.

  pixi run python scripts/pick_clip_windows.py --new new.trace.jsonl --old old.trace.jsonl --finding flips [--length 22] \
      [--step 1.0] [--team both|home|away] [--out window.json]

Findings (all numbers come from soccer_sim.overlay.RunningCounters / ReactionTimer, i.e. metrics.py's rules, over the window):
  flips      maximise the pre-change chase flips                         score = flips in the window
  dup        maximise the pre-change duplicate-chaser time (seconds)     score = seconds
  ball       maximise the fused-ball error gap, pre-change minus new     score = mean fused error (m)
  reaction   maximise the ball events with a reaction in BOTH brains     score = reacted events in the window (the
             optimised value is the smaller of the two counts; ties go to the larger sum)
  blackout   fixed: 5 s before the first blackout row to 5 s after the last one, clamped to the trace    score = duplicate seconds
Candidate starts run from the common start of the two traces in --step seconds; ties go to the lower score_new, then the earlier
start, so the result is deterministic. Writes (and prints) {finding, t0, t1, score_new, score_old}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_sim.metrics import MetricsRecorder  # noqa: E402
from soccer_sim.overlay import RunningCounters, sample_from_row  # noqa: E402

FINDINGS = ("flips", "dup", "ball", "reaction", "blackout")
BLACKOUT_MARGIN_S = 5.0


def load_rows(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: empty trace")
    return rows


def _window_counters(rows: list[dict], t0: float, t1: float, teams: tuple[str, ...]) -> dict:
    rc = RunningCounters()
    for r in rows:
        if t0 <= r["t"] <= t1:
            rc.update(r)
    st = [rc.stats(t) for t in teams]
    err_w = [(s["fused_err_m"], s) for s in st if s["fused_err_m"] is not None]
    return {"flips": sum(s["chase_flips"] for s in st), "dup": sum(s["dup_s"] for s in st),
            "ball": sum(e for e, _ in err_w) / len(err_w) if err_w else None}


def _reactions(rows: list[dict], teams: tuple[str, ...]) -> list[tuple[float, float]]:
    """(event start, reaction end) of every reacted ball event of the whole trace."""
    rec = MetricsRecorder(True)
    for r in rows:
        rec.add(sample_from_row(r))
    return [(e["t"], e["t"] + e["reaction_s"]) for e in rec.events if e["team"] in teams and e["reaction_s"] is not None]


def _blackout_window(rows: list[dict]) -> tuple[float, float] | None:
    ts = [r["t"] for r in rows if r.get("blackout")]
    return (ts[0], ts[-1]) if ts else None


def pick_window(new: list[dict], old: list[dict], finding: str, length: float = 22.0, step: float = 1.0,
                team: str = "both") -> dict:
    if finding not in FINDINGS:
        raise ValueError(f"finding must be one of {FINDINGS}")
    teams = ("home", "away") if team == "both" else (team,)
    lo, hi = max(new[0]["t"], old[0]["t"]), min(new[-1]["t"], old[-1]["t"])
    if finding == "blackout":
        bo = _blackout_window(new) or _blackout_window(old)
        if bo is None:
            raise SystemExit("no blackout flag in either trace")
        t0, t1 = max(bo[0] - BLACKOUT_MARGIN_S, lo), min(bo[1] + BLACKOUT_MARGIN_S, hi)
        a, b = _window_counters(new, t0, t1, teams), _window_counters(old, t0, t1, teams)
        return {"finding": finding, "t0": round(t0, 2), "t1": round(t1, 2), "score_new": round(a["dup"], 3),
                "score_old": round(b["dup"], 3)}
    if hi - lo < length:
        raise SystemExit(f"traces overlap for {hi - lo:.1f} s, shorter than --length {length}")
    ev_new, ev_old = (_reactions(new, teams), _reactions(old, teams)) if finding == "reaction" else ([], [])
    best, best_key = None, None
    k = 0
    while lo + k * step + length <= hi + 1e-9:
        t0 = lo + k * step
        t1 = t0 + length
        k += 1
        if finding == "reaction":
            sn = float(sum(1 for s, e in ev_new if s >= t0 and e <= t1))
            so = float(sum(1 for s, e in ev_old if s >= t0 and e <= t1))
            key = (min(sn, so), sn + so, -t0)
        else:
            a, b = _window_counters(new, t0, t1, teams), _window_counters(old, t0, t1, teams)
            f = {"flips": "flips", "dup": "dup", "ball": "ball"}[finding]
            sn, so = a[f], b[f]
            if finding == "ball":
                if sn is None or so is None:
                    continue
                key = (so - sn, -sn, -t0)
            else:
                key = (so, -sn, -t0)
        if best_key is None or key > best_key:
            best_key, best = key, (t0, t1, sn, so)
    if best is None:
        raise SystemExit("no candidate window had data for this finding")
    t0, t1, sn, so = best
    return {"finding": finding, "t0": round(t0, 2), "t1": round(t1, 2), "score_new": round(sn, 3), "score_old": round(so, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="new-brain trace JSONL")
    ap.add_argument("--old", required=True, help="pre-change trace JSONL (same situation and seed)")
    ap.add_argument("--finding", required=True, choices=FINDINGS)
    ap.add_argument("--length", type=float, default=22.0, help="window length in PLAY seconds (not used by blackout)")
    ap.add_argument("--step", type=float, default=1.0, help="candidate start spacing, seconds")
    ap.add_argument("--team", default="both", choices=("both", "home", "away"))
    ap.add_argument("--out", default="", help="write the JSON here (always printed)")
    a = ap.parse_args(argv)
    res = pick_window(load_rows(a.new), load_rows(a.old), a.finding, a.length, a.step, a.team)
    text = json.dumps(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
