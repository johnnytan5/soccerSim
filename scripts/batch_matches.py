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

Head-to-head: --home-brain / --away-brain (current|baseline, default --brain) put different brains on the two teams,
--home-network / --away-network give one team its own profile (e.g. --away-network blackout isolates the away team for
the whole match). Swap sides by a second run with home and away exchanged. See docs/batch_matches.md.

--brain baseline runs an older brain: build it once into a second workspace (see docs/batch_matches.md):
  ROBOTEDGE_ROOT=<worktree of 8d241a2> SS_WS_DIR=$PWD/ros_ws_baseline PACKAGES=brain pixi run bash scripts/setup_ws.sh
"""
from __future__ import annotations

import argparse
import csv
import re
import json
import concurrent.futures
import queue
import shutil
import threading
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
    p.add_argument("--home-brain", choices=("current", "baseline"), default=None, help="default: --brain")
    p.add_argument("--away-brain", choices=("current", "baseline"), default=None, help="default: --brain")
    p.add_argument("--home-network", default="", help="network profile of the home robots only (default: the scenario profile)")
    p.add_argument("--away-network", default="", help="network profile of the away robots only; permanent 'blackout' isolates the team")
    p.add_argument("--ws", default="", help="colcon workspace of the brain (default ros_ws; ros_ws_baseline for baseline)")
    p.add_argument("--baseline-root", default=os.environ.get("BASELINE_ROBOTEDGE_ROOT", ""),
                   help="Robotedge checkout of the baseline brain (commit 8d241a2), for config.yaml and behaviour trees")
    p.add_argument("--out", default="results/batch.csv")
    p.add_argument("--players", type=int, default=5, help="players per team (5 = 5v5; ids 1..N, player 1 is the keeper)")
    p.add_argument("--yolo", action="store_true", help="YOLO ball detection from the rendered camera (slow) instead of ground truth")
    p.add_argument("--trace", action="store_true", help="also write <stem>.trace.jsonl next to each match JSON (10 Hz PLAY trace)")
    p.add_argument("--record", default="", metavar="A:B",
                   help="also render PLAY seconds A:B as <stem>.mp4 (MuJoCo, free camera above the pitch, 720p/25 fps, GPU GL, NVENC); "
                        "implies --trace. Which matches: --record-matches")
    p.add_argument("--record-matches", default="0", help="match numbers (0-based, per profile) to record, comma list or 'all'")
    p.add_argument("--parallel", type=int, default=1, metavar="K",
                   help="run K matches at once, each in its own network slot (SS_SLOT, br-sim<n>), ROS domains spaced by 20 "
                        "(max 4 with --domain-base 150); the sim must keep tick_hz >= 45 or the matches are slowed down")
    p.add_argument("--ready-secs", type=float, default=6.0, help="READY phase before kick-off")
    p.add_argument("--seed", type=int, default=1, help="first match's seed; the next matches add 1")
    p.add_argument("--domain-base", type=int, default=150, help="ROS domain of the first robot (avoid other running sims)")
    p.add_argument("--timeout-extra", type=float, default=150.0, help="wall seconds allowed beyond duration + ready")
    p.add_argument("--logs", default="", help="directory for per-match logs (default: next to --out)")
    return p.parse_args(argv)


def team_brains(args) -> tuple[str, str]:
    return args.home_brain or args.brain, args.away_brain or args.brain


def brain_plan(home: str, away: str, current_root: str, baseline_root: str, ws_current: Path, ws_baseline: Path) -> dict:
    """How to start the brains of a match. Same brain on both teams: the old way (one workspace overlay, one checkout).
    Mixed: the sourced ros_ws brain stays the default and the baseline team's brain_node is started by absolute path from
    the baseline install, with the baseline checkout's config.yaml/trees (match.launch.py home_/away_brain_install|root).
    "cfg" holds LaunchConfig fields ({team}_brain_root: soccer_launcher.command then emits {team}_brain_root:= and a
    {team}_brain_install:= pointing at ros_ws_brains/<name>); "installs" is where the baseline build really is
    (ros_ws_baseline/install), substituted for that argument by command_for()."""
    if home == away:
        base = home == "baseline"
        return {"rcap_root": baseline_root if base else current_root, "ws": ws_baseline if base else ws_current,
                "cfg": {}, "installs": {}}
    cfg, installs = {}, {}
    for team, brain in (("home", home), ("away", away)):
        if brain == "baseline":
            cfg[f"{team}_brain_root"] = baseline_root
            installs[team] = str(ws_baseline / "install")
    return {"rcap_root": current_root, "ws": ws_current, "cfg": cfg, "installs": installs}


def command_for(cfg, rcap_root: str, plan: dict) -> tuple[list[str], dict[str, str]]:
    """soccer_launcher.command.build_command with the baseline team's brain_install and the workspace overlay
    (SS_WS_DIR, a colon-separated list for run_match.sh) pointed at the batch runner's own builds."""
    argv, env = command.build_command(cfg, rcap_root)
    for team, install in plan["installs"].items():
        argv = [f"{team}_brain_install:={install}" if a.startswith(f"{team}_brain_install:=") else a for a in argv]
    env["SS_WS_DIR"] = str(plan["ws"])  # replaces the launcher's ros_ws_brains/<name> overlay, which is not built here
    return argv, env


def team_overrides(profile: str, home_net: str, away_net: str, players: int) -> tuple[str, str, str]:
    """(network_overrides, home_profile, away_profile) for the scenario `profile` with optional per-team profiles."""
    over = [f"{team}{i}={net}" for team, net in (("home", home_net), ("away", away_net)) if net for i in range(1, players + 1)]
    return ",".join(over), home_net or profile, away_net or profile


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
    """One poll step of the PLAY-keyed blackout. The window file is what the bridge classifies samples with
    (soccer_sim.metrics.window_state), all in monotonic seconds:
      start         taken BEFORE `netem.sh apply blackout` begins
      apply_s       how long that apply took; [start, start+apply_s] is a ramp (counted in neither segment)
      revert_begin  taken BEFORE the revert begins; the dark period is [start+apply_s, revert_begin)
      end, revert_s taken AFTER the revert returned; [revert_begin, end] is a ramp too
    The blackout lasts `duration` seconds from the moment the apply finished."""
    duration, offset = spec
    if window is not None and window.get("end") is not None:
        return window
    try:
        flag = json.loads(flag_path.read_text())
    except (OSError, ValueError):
        return window  # PLAY has not started
    now = time.monotonic()
    if window is None:
        if now < flag["wall"] + offset:
            return None
        window = {"start": now, "apply_s": None, "revert_begin": None, "end": None, "revert_s": None,
                  "play_wall": flag["wall"]}
        _write_json(bo_path, window)  # start is on disk before the first tc command runs
        if not netns._netem(["apply", "blackout"], cfg, log):
            raise SystemExit("could not apply the blackout")
        window["apply_s"] = time.monotonic() - window["start"]
        _write_json(bo_path, window)
        log(f"blackout started {window['start'] - flag['wall']:.1f} s after PLAY (apply took {window['apply_s']:.2f} s)")
        return window
    if duration is not None and window["revert_begin"] is None and now >= window["start"] + window["apply_s"] + duration:
        window["revert_begin"] = time.monotonic()
        _write_json(bo_path, window)
        ok = netns._netem(["apply", "good"], cfg, log)
        window["end"] = time.monotonic()  # after the revert returned: every sample before this may still be impaired
        window["revert_s"] = window["end"] - window["revert_begin"]
        _write_json(bo_path, window)
        if not ok:
            raise SystemExit("could not revert the blackout")
        log(f"blackout dark for {window['revert_begin'] - window['start'] - window['apply_s']:.1f} s, revert took {window['revert_s']:.2f} s")
    return window


def validate_parallel(parallel: int, profiles: list[str], home_net: str, away_net: str) -> None:
    """--parallel K>1 runs each match in its own netem slot, so every robot must be namespaced: a `none` profile runs on the
    host network, where parallel matches would hear each other's team broadcasts."""
    if parallel <= 1:
        return
    bad = [p for p in profiles if "none" in (home_net or p, away_net or p)]
    if bad:
        raise SystemExit("--parallel > 1 needs every robot in a network namespace: profile 'none' (host network) would let "
                         "concurrent matches hear each other; use 'good' instead, or --parallel 1")


def match_tag(profile: str, brain_tag: str, n: int, home_net: str = "", away_net: str = "") -> str:
    """Per-match output tag; per-team networks are part of it so different --home/--away-network runs never overwrite each other."""
    nets = "".join(f"_{team}-{net.replace(':', '')}" for team, net in (("home", home_net), ("away", away_net)) if net)
    return f"{profile.replace(':', '')}_{brain_tag}{nets}_{n}"


STOP = threading.Event()  # set by the SIGINT/SIGTERM handler: no new matches, running ones end
_LOCK = threading.Lock()
RUNNING: list = []  # match processes (subprocess.Popen) currently running
ACTIVE: list = []  # LaunchConfigs whose netns session is up


def shutdown(signum=None, frame=None) -> None:
    """SIGINT/SIGTERM: kill every running match's process group, then remove the namespaces of the slots in use."""
    STOP.set()
    with _LOCK:
        procs, cfgs = list(RUNNING), list(ACTIVE)
    for proc in procs:
        kill_group(proc.pid)
    for cfg in cfgs:
        if netns.active(cfg):
            netns.tear_down(cfg)
    if signum is not None:
        raise SystemExit(128 + signum)


RECORD_ALLOWANCE_S = 600.0  # extra wall seconds for the clip render when --record is set
DOMAIN_SPACING = 20  # a match uses up to 10 consecutive ROS domains
MAX_DOMAIN = 232


def slot_plan(args, k: int) -> tuple[str, int]:
    """(netem slot, ROS domain base) of the k-th concurrent match. One match alone keeps today's names and base."""
    if args.parallel <= 1:
        return "", args.domain_base
    return str(k), args.domain_base + DOMAIN_SPACING * k


def run_one(args, profile: str, n: int, out: Path, logs: Path, plan: dict, k: int = 0) -> dict | None:
    rcap_root, ws = plan["rcap_root"], plan["ws"]
    home_brain, away_brain = team_brains(args)
    slot, domain_base = slot_plan(args, k)
    ids = tuple(range(1, args.players + 1))
    brain_tag = home_brain if home_brain == away_brain else f"{home_brain}-vs-{away_brain}"
    tag = match_tag(profile, brain_tag, n, args.home_network, args.away_network)
    json_path = out.with_name(f"{out.stem}_{tag}.json")
    flag_path = json_path.with_suffix(".play")
    bo_path = json_path.with_suffix(".blackout")
    blackout = parse_blackout(profile)
    log_path = logs / f"{out.stem}_{tag}.log"
    overrides, home_profile, away_profile = team_overrides(profile, args.home_network, args.away_network, args.players)
    meta = {"profile": profile, "brain": brain_tag, "match": n, "home_brain": home_brain, "away_brain": away_brain,
            "home_profile": home_profile, "away_profile": away_profile, "seed": args.seed + n,
            "roster": f"{args.players}v{args.players}"}
    extra = (f"--web-port -1 --ready-secs {args.ready_secs:g} --seed {meta['seed']} --metrics-out {json_path} "
             f"--metrics-play-secs {args.duration:g} "
             f"--play-flag {flag_path} --blackout-file {bo_path}")
    if slot:  # count only this slot's team broadcasts (the other slots' bridges listen on their own subnets)
        extra += f" --packet-bind {netns.slot_names(slot)['broadcast']}"
    recording = bool(args.record and (args.record_matches == "all" or n in {int(x) for x in args.record_matches.split(",")}))
    if recording:
        extra += f" --record {json_path.with_suffix('.mp4')} --record-window {args.record}"
    if args.trace or args.record:
        extra += f" --trace-out {json_path.with_suffix('.trace.jsonl')}"
    cfg = _match(_players(ids, ids), continuous=True, network_profile="good" if blackout else profile,
                 network_overrides=overrides, viewer=False, simulator="mujoco",
                 ros_domain_id=domain_base, extra_bridge_args=extra)
    cfg.netns_slot = slot
    for k_, v in plan["cfg"].items():
        setattr(cfg, k_, v)
    if args.yolo:
        cfg.ball_detector = "yolo"
    errs = cfg.validate()
    if errs:
        raise SystemExit("; ".join(errs))
    for stale in (json_path, flag_path, bo_path, json_path.with_suffix(".mp4")):
        stale.unlink(missing_ok=True)
    t0 = time.monotonic()
    proc, timed_out = None, False
    window: dict | None = None  # {"start": wall, "end": wall | None}, mirrored to the bridge (--blackout-file)
    log = lambda text: print(f"  {text}", flush=True)  # noqa: E731
    with _LOCK:
        ACTIVE.append(cfg)  # the signal handler tears these sessions down
    try:
        with netns.session(cfg, log) as run_cfg:
            if netns.active(cfg) and not netns.active(run_cfg):
                raise SystemExit("network namespaces unavailable here; refusing to run a profile without them")
            argv, env = command_for(run_cfg, rcap_root, plan)
            try:
                with open(log_path, "w") as lf:
                    proc = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, **env}, start_new_session=True,
                                            stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT)
                    with _LOCK:
                        RUNNING.append(proc)
                    limit = args.duration + args.ready_secs + args.timeout_extra + (RECORD_ALLOWANCE_S if recording else 0.0)
                    while proc.poll() is None and time.monotonic() - t0 < limit and not json_path.exists() and not STOP.is_set():
                        if blackout is not None:
                            window = _drive_blackout(blackout, window, flag_path, bo_path, run_cfg, log)
                        time.sleep(0.2)
                    if json_path.exists() and recording:  # the metrics are on disk; let the clip render finish (bounded)
                        wait_end = time.monotonic() + RECORD_ALLOWANCE_S
                        while proc.poll() is None and time.monotonic() < wait_end and not STOP.is_set():
                            time.sleep(0.5)
                    time.sleep(1.0 if json_path.exists() else 0.0)
                    timed_out = proc.poll() is None and not json_path.exists()
            finally:
                # the match process group dies BEFORE the namespaces are torn down (leaving the `with` below)
                if proc is not None:
                    kill_group(proc.pid)
                    with _LOCK:
                        if proc in RUNNING:
                            RUNNING.remove(proc)
    finally:
        with _LOCK:
            if cfg in ACTIVE:
                ACTIVE.remove(cfg)
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
    if window:
        meta.update(blackout_apply_s=round(window["apply_s"], 3) if window.get("apply_s") is not None else None,
                    blackout_revert_s=round(window["revert_s"], 3) if window.get("revert_s") is not None else None)
    bw = doc.get("blackout_window") or {}
    meta.update(blackout_start_play_s=bw.get("start_play_s"), blackout_end_play_s=bw.get("end_play_s"),
                blackout_s=bw.get("duration_s"))
    log(f"{tag}: tick_hz mean {doc.get('tick_hz_mean')} (last {doc.get('tick_hz_last')})")
    doc["meta"] = {**doc.get("meta", {}), **meta}
    json_path.write_text(json.dumps(doc, indent=1))
    return metrics.csv_row(meta, doc)


def check_csv_header(out: Path, columns: list[str]) -> None:
    """Appending rows to a CSV whose header lacks columns would misalign them: refuse with a clear error."""
    if not out.exists() or out.stat().st_size == 0:
        return
    with open(out, newline="") as f:
        header = next(csv.reader(f), [])
    if header != columns:
        missing = [c for c in columns if c not in header]
        raise SystemExit(f"{out} has a different header than this version writes (missing: {', '.join(missing) or 'none, order differs'}); "
                         "write to a new --out file or migrate the CSV (rewrite it with the new header) first")


def _gpu_reexec(args, argv) -> None:
    """--yolo: run the whole batch in the pixi `gpu` environment (onnxruntime-gpu) when it is installed and this
    interpreter's onnxruntime has no CUDA provider. SIM_YOLO_PROVIDER (auto/cuda/cpu) still decides in the matches."""
    if not args.yolo or os.environ.get("SIM_YOLO_PROVIDER", "auto") == "cpu" or os.environ.get("SS_GPU_REEXEC"):
        return
    try:
        import onnxruntime
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            return
    except ImportError:
        pass
    pixi = shutil.which("pixi")
    if pixi and (ROOT / ".pixi" / "envs" / "gpu").is_dir():
        print("[batch] --yolo: switching to the pixi gpu environment (onnxruntime-gpu)", flush=True)
        os.execvpe(pixi, [pixi, "run", "-e", "gpu", "python", *sys.argv[:1], *(sys.argv[1:] if argv is None else argv)],
                   {**os.environ, "SS_GPU_REEXEC": "1"})
    print("[batch] --yolo: no gpu environment (pixi install -e gpu); YOLO runs on the CPU", flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    _gpu_reexec(args, argv)
    if args.record and not re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", args.record):
        raise SystemExit("--record takes PLAY seconds as A:B, e.g. 20:40")
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    bad = [p for p in profiles if not valid_profile(p)]
    if bad:
        raise SystemExit(f"bad profile(s): {', '.join(bad)}")
    home, away = team_brains(args)
    for net in (args.home_network, args.away_network):
        if net and not netns.valid_profile(net):
            raise SystemExit(f"bad team network profile {net!r} (none, good, rcap_bad, terrible, blackout[:T], lat<ms>_loss<pct>)")
    if (args.home_network or args.away_network) and any(parse_blackout(p) for p in profiles):
        raise SystemExit("--home-network/--away-network cannot be combined with a blackout:T@O scenario profile")
    baseline_root = ""
    if "baseline" in (home, away):
        if not args.baseline_root:
            raise SystemExit("a baseline brain needs --baseline-root (or $BASELINE_ROBOTEDGE_ROOT)")
        baseline_root = str(Path(args.baseline_root).resolve())
    current_root = os.environ.get("ROBOTEDGE_ROOT", "")
    if "current" in (home, away) and not current_root:
        raise SystemExit("run through pixi (pixi run python scripts/batch_matches.py ...)")
    validate_parallel(args.parallel, profiles, args.home_network, args.away_network)
    if not 1 <= args.parallel <= (MAX_DOMAIN - args.domain_base - 9) // DOMAIN_SPACING + 1:
        raise SystemExit(f"--parallel {args.parallel}: ROS domains {args.domain_base} + 20*k + 9 must stay <= {MAX_DOMAIN}")

    def abs_ws(w):
        w = Path(w)
        return w if w.is_absolute() else (ROOT / w).resolve()

    uses_baseline = "baseline" in (home, away)
    ws_baseline = abs_ws(args.ws or "ros_ws_baseline") if uses_baseline else None
    ws_current = abs_ws(args.ws if args.ws and not uses_baseline else "ros_ws")  # --ws names the baseline build when there is one
    plan = brain_plan(home, away, current_root, baseline_root, ws_current, ws_baseline)
    for w in {plan["ws"], ws_baseline} - {None}:
        if not (w / "install" / "setup.bash").exists():
            raise SystemExit(f"{w} has no install/: build it first (see --help)")
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs) if args.logs else out.parent
    logs.mkdir(parents=True, exist_ok=True)
    if any(p != "none" for p in profiles) or args.home_network or args.away_network:
        lacks = netns.missing()
        if lacks:
            raise SystemExit("cannot use network profiles here: " + "; ".join(lacks))
    columns = metrics.csv_columns()
    check_csv_header(out, columns)
    state = {"new": not out.exists(), "done": 0}
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    write_lock = threading.Lock()
    jobs = [(profile, n) for profile in profiles for n in range(args.matches)]

    def work(job, k):
        profile, n = job
        if STOP.is_set():
            return
        print(f"[batch] {'/'.join(team_brains(args))} {profile} match {n + 1}/{args.matches} ({args.players}v{args.players}, "
              f"{args.duration:g} s of play" + (f", slot {k}" if args.parallel > 1 else "") + ")", flush=True)
        row = run_one(args, profile, n, out, logs, plan, k)
        if row is None:
            return
        with write_lock:
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=columns)
                if state["new"]:
                    w.writeheader()
                    state["new"] = False
                w.writerow(row)
            state["done"] += 1
            print(f"  [{profile} #{n}] play {row['play_s']} s in {row['wall_s']} s wall; goals {row['goals_home']}-{row['goals_away']}", flush=True)

    if args.parallel <= 1:
        for job in jobs:
            work(job, 0)
    else:
        free = queue.Queue()
        for k in range(args.parallel):
            free.put(k)

        def slotted(job):
            k = free.get()
            try:
                work(job, k)
            finally:
                free.put(k)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(pool.map(slotted, jobs))
    done = state["done"]
    print(f"[batch] {done} matches -> {out}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
