"""Run headless matches under team-network profiles and collect match metrics (soccer_sim.metrics) into a CSV.

  pixi run python scripts/batch_matches.py --profiles good,rcap_bad,terrible,blackout:10 --matches 3 --duration 120 \
      --brain current --out results/current.csv

Each match is `scripts/run_match.sh` (bridge + one brain per robot, ground-truth ball, no viewer), started the way the
launcher starts it (soccer_launcher.command / netns.session: namespaces, netem profile on every robot, torn down again).
The profile applies to every robot of both teams. The bridge records the metrics itself and exits after --duration
seconds of simulated PLAY. Per match: <out stem>_<profile>_<brain>_<n>.json (full metrics) and one CSV row (--out).

Profile `blackout:T@O` (T seconds of total team-link loss on every robot, starting O seconds after the first PLAY; O
defaults to 20): the runner polls the flag file the bridge writes at PLAY start, applies `netem.sh apply blackout`, reverts
to good after T and tells the bridge the wall-clock window (a JSON file) so the metrics split in/outside the blackout.
Plain `blackout` (no T) never reverts. The other profiles are applied before kick-off as before.

--brain baseline runs an older brain: build it once into a second workspace (see docs/batch_matches.md):
  ROBOTEDGE_ROOT=<worktree of 8d241a2> SS_WS_DIR=$PWD/ros_ws_baseline PACKAGES=brain pixi run bash scripts/setup_ws.sh
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from soccer_launcher import command, netns  # noqa: E402
from soccer_launcher.config import _match, _players  # noqa: E402
from soccer_sim import metrics  # noqa: E402


DEFAULT_BLACKOUT_OFFSET_S = 20.0
_BLACKOUT_RE = re.compile(r"^blackout(?::(\d+(?:\.\d+)?))?(?:@(\d+(?:\.\d+)?))?$")


def parse_blackout(profile: str) -> tuple[float | None, float] | None:
    """'blackout:15@20' -> (15.0, 20.0); 'blackout:10' -> (10.0, 20.0); 'blackout' -> (None, 20.0); else None."""
    m = _BLACKOUT_RE.match(profile)
    if not m:
        return None
    return (float(m.group(1)) if m.group(1) else None, float(m.group(2)) if m.group(2) else DEFAULT_BLACKOUT_OFFSET_S)


def valid_profile(profile: str) -> bool:
    return parse_blackout(profile) is not None or netns.valid_profile(profile)


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles", default="good", help="comma list: none, good, rcap_bad, terrible, blackout[:T[@O]]")
    p.add_argument("--matches", type=int, default=1, help="matches per profile")
    p.add_argument("--duration", type=float, default=120.0, help="simulated seconds of PLAY to record per match")
    p.add_argument("--brain", choices=("current", "baseline"), default="current")
    p.add_argument("--ws", default="", help="colcon workspace of the brain (default ros_ws; ros_ws_baseline for baseline)")
    p.add_argument("--baseline-root", default=os.environ.get("BASELINE_ROBOTEDGE_ROOT", ""),
                   help="Robotedge checkout of the baseline brain (commit 8d241a2), for config.yaml and behaviour trees")
    p.add_argument("--out", default="results/batch.csv")
    p.add_argument("--players", type=int, default=5, help="players per team (5 = 5v5; ids 1..N, player 1 is the keeper)")
    p.add_argument("--yolo", action="store_true", help="YOLO ball detection from the rendered camera (slow) instead of ground truth")
    p.add_argument("--ready-secs", type=float, default=6.0, help="READY phase before kick-off")
    p.add_argument("--seed", type=int, default=1, help="first match's seed; the next matches add 1")
    p.add_argument("--domain-base", type=int, default=150, help="ROS domain of the first robot (avoid other running sims)")
    p.add_argument("--timeout-extra", type=float, default=150.0, help="wall seconds allowed beyond duration + ready")
    p.add_argument("--logs", default="", help="directory for per-match logs (default: next to --out)")
    return p.parse_args(argv)


def kill_group(pid: int) -> None:
    for sig, wait in ((signal.SIGINT, 6.0), (signal.SIGTERM, 4.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        end = time.monotonic() + wait
        while time.monotonic() < end:
            try:
                os.killpg(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.2)


def _drive_blackout(spec, window, flag_path: Path, bo_path: Path, cfg, log):
    """One poll step of the PLAY-keyed blackout. The window is only written (and the netem profile only changed)
    after the PLAY flag file exists; start/end are the wall clock at which the profile was actually in force."""
    duration, offset = spec
    if window is not None and window.get("end") is not None:
        return window
    try:
        flag = json.loads(flag_path.read_text())
    except (OSError, ValueError):
        return window  # PLAY has not started
    now = time.time()
    if window is None:
        if now < flag["wall"] + offset:
            return None
        if not netns._netem(["apply", "blackout"], cfg, log):
            raise SystemExit("could not apply the blackout")
        window = {"start": time.time(), "end": None, "play_wall": flag["wall"]}
        _write_json(bo_path, window)
        log(f"blackout started {window['start'] - flag['wall']:.1f} s after PLAY")
        return window
    if duration is not None and now >= window["start"] + duration:
        window["end"] = time.time()  # recorded before the revert command: those samples are certainly blacked out
        _write_json(bo_path, window)
        if not netns._netem(["apply", "good"], cfg, log):
            raise SystemExit("could not revert the blackout")
        log(f"blackout ended {window['end'] - window['start']:.1f} s after it started")
    return window


def run_one(args, profile: str, n: int, out: Path, logs: Path, rcap_root: str, ws: Path) -> dict | None:
    ids = tuple(range(1, args.players + 1))
    tag = f"{profile.replace(':', '')}_{args.brain}_{n}"
    json_path = out.with_name(f"{out.stem}_{tag}.json")
    flag_path = json_path.with_suffix(".play")
    bo_path = json_path.with_suffix(".blackout")
    blackout = parse_blackout(profile)
    log_path = logs / f"{out.stem}_{tag}.log"
    meta = {"profile": profile, "brain": args.brain, "match": n, "seed": args.seed + n,
            "roster": f"{args.players}v{args.players}"}
    extra = (f"--web-port -1 --ready-secs {args.ready_secs:g} --seed {meta['seed']} --metrics-out {json_path} "
             f"--metrics-play-secs {args.duration:g} "
             f"--play-flag {flag_path} --blackout-file {bo_path}")
    cfg = _match(_players(ids, ids), continuous=True, network_profile="good" if blackout else profile, viewer=False, simulator="mujoco",
                 ros_domain_id=args.domain_base, extra_bridge_args=extra)
    if args.yolo:
        cfg.ball_detector = "yolo"
    errs = cfg.validate()
    if errs:
        raise SystemExit("; ".join(errs))
    for stale in (json_path, flag_path, bo_path):
        stale.unlink(missing_ok=True)
    t0 = time.monotonic()
    proc, timed_out = None, False
    window: dict | None = None  # {"start": wall, "end": wall | None}, mirrored to the bridge (--blackout-file)
    log = lambda text: print(f"  {text}", flush=True)  # noqa: E731
    try:
        with netns.session(cfg, log) as run_cfg:
            if netns.active(cfg) and not netns.active(run_cfg):
                raise SystemExit("network namespaces unavailable here; refusing to run a profile without them")
            argv, env = command.build_command(run_cfg, rcap_root)
            env = {**env, "SS_WS_DIR": str(ws)}
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, **env}, start_new_session=True,
                                        stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
                limit = args.duration + args.ready_secs + args.timeout_extra
                while proc.poll() is None and time.monotonic() - t0 < limit and not json_path.exists():
                    if blackout is not None:
                        window = _drive_blackout(blackout, window, flag_path, bo_path, run_cfg, log)
                    time.sleep(0.2)
                time.sleep(1.0 if json_path.exists() else 0.0)
                timed_out = proc.poll() is None and not json_path.exists()
    finally:
        if proc is not None:
            kill_group(proc.pid)
    wall = time.monotonic() - t0
    if blackout is not None:
        meta["blackout_wall"] = window
        meta["blackout_spec"] = {"duration_s": blackout[0], "offset_s": blackout[1]}
    if not json_path.exists():
        print(f"  no metrics for {tag} ({'timed out' if timed_out else 'match exited early'}); see {log_path}", flush=True)
        return None
    doc = json.loads(json_path.read_text())
    meta.update(play_s=doc["play_s"], wall_s=round(wall, 1), goals_home=doc["score"]["home"],
                goals_away=doc["score"]["away"], goal_diff=doc["score"]["home"] - doc["score"]["away"])
    bw = doc.get("blackout_window") or {}
    meta.update(blackout_start_play_s=bw.get("start_play_s"), blackout_end_play_s=bw.get("end_play_s"),
                blackout_s=bw.get("duration_s"))
    doc["meta"] = {**doc.get("meta", {}), **meta}
    json_path.write_text(json.dumps(doc, indent=1))
    return metrics.csv_row(meta, doc)


def main(argv=None) -> int:
    args = parse_args(argv)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    bad = [p for p in profiles if not valid_profile(p)]
    if bad:
        raise SystemExit(f"bad profile(s): {', '.join(bad)}")
    if args.brain == "baseline":
        if not args.baseline_root:
            raise SystemExit("--brain baseline needs --baseline-root (or $BASELINE_ROBOTEDGE_ROOT)")
        rcap_root = str(Path(args.baseline_root).resolve())
    else:
        rcap_root = os.environ.get("ROBOTEDGE_ROOT", "")
        if not rcap_root:
            raise SystemExit("run through pixi (pixi run python scripts/batch_matches.py ...)")
    ws = Path(args.ws or ("ros_ws_baseline" if args.brain == "baseline" else "ros_ws"))
    ws = (ROOT / ws).resolve() if not ws.is_absolute() else ws
    if not (ws / "install" / "setup.bash").exists():
        raise SystemExit(f"{ws} has no install/: build it first (see --help)")
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs) if args.logs else out.parent
    logs.mkdir(parents=True, exist_ok=True)
    if any(p != "none" for p in profiles):
        lacks = netns.missing()
        if lacks:
            raise SystemExit("cannot use network profiles here: " + "; ".join(lacks))
    columns = metrics.csv_columns()
    new = not out.exists()
    done = 0
    for profile in profiles:
        for n in range(args.matches):
            print(f"[batch] {args.brain} {profile} match {n + 1}/{args.matches} ({args.players}v{args.players}, "
                  f"{args.duration:g} s of play)", flush=True)
            row = run_one(args, profile, n, out, logs, rcap_root, ws)
            if row is None:
                continue
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                if new:
                    w.writeheader()
                    new = False
                w.writerow(row)
            done += 1
            print(f"  play {row['play_s']} s in {row['wall_s']} s wall; goals {row['goals_home']}-{row['goals_away']}",
                  flush=True)
    print(f"[batch] {done} matches -> {out}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
