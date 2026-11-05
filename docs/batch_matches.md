# Batch matches and match metrics

```
pixi run python scripts/batch_matches.py --profiles good,rcap_bad,terrible,blackout:10@20 --matches 3 --duration 120 \
    --brain current --out results/current.csv
```

Each match: bridge + one brain per robot (5v5 by default, `--players N` for N a side), ground-truth ball (`--yolo` for the
camera), no viewer, one netem profile on every robot (see netem.md; needs passwordless sudo). `--duration` is simulated
seconds of PLAY (after `--ready-secs`, default 6). Output: one CSV row per match (`--out`, appended) and
`<out stem>_<profile>_<brain>_<n>.json` (per robot, per team, every ball event) plus the match log next to it.
Uses ROS domains from `--domain-base` (150) and no web port, so it can run beside a normal launch.

Options added later: `--trace`, `--record A:B`, `--parallel K`, `--yolo` (GPU), profiles `lat<ms>_loss<pct>` (netem.md). Each is below.

## Trace (`--trace`) and replay

`--trace` makes the bridge write `<stem>.trace.jsonl` next to the match JSON (`match_node --trace-out`, which needs
`--metrics-out`: it reuses the metrics sampling, `MatchBridge._record_metrics`, so it shares its definitions). One JSON row per
0.1 s of PLAY (simulated time; nothing during READY/SET/stoppages): `t` (PLAY seconds), `ball` (ground truth x, y, vx, vy),
`blackout`, and `robots`: `id`, `team`, truth pose `x, y, theta`, `decision`, `player_role`, `im_lead`, `outfield_rank`,
`role_own_view`, `comm_mode`, `links` (per teammate: `player`, `health`, `latency_ms`, `loss`, from the robot's world model),
`fused_ball` and `own_ball` (`x, y, valid`, world frame). Fields the brain does not publish are `null`. About 2 MB per 60 s of 5v5.

`scripts/render_replay.py --left a.trace.jsonl [--right b.trace.jsonl] --t0 15 --t1 40 --out clip.mp4 --title-left "new" --title-right
"baseline" [--poster clip.png]` renders a top-down replay (pitch to scale, team-coloured robots with id + role initial and a
heading tick, white ring = `im_lead`, link lines whose width/opacity follow the health, comm-mode badge per team, red tint and
banner in a blackout, PLAY time) at 1280x720, 25 fps, numpy + cv2 (~6 ms per frame for two panels), piped to ffmpeg
`h264_nvenc -preset p5 -cq 28` (libx264 if NVENC is missing, or `--cpu`). Colours are Okabe-Ito (home blue, away orange; comm
mode green / yellow / vermillion). A 20 s two-panel clip is about 1 MB.

## MuJoCo 3D clip (`--record A:B`)

`--record 20:40` renders PLAY seconds 20-40 of the first match of each profile (`--record-matches 0,2` or `all` for others) to
`<stem>.mp4` (and implies `--trace`). `match_node --record out.mp4 --record-window 20:40`. A free camera above the pitch at an
oblique angle (azimuth 90, elevation -46, 18.5 m), the stripped viewer model (cheap team-coloured bodies), 720p, 25 fps, a label
per robot (H2 striker LEAD), a legend with the clock, comm mode per team and a blackout banner, h264_nvenc.

The sim does not render. `soccer_sim_bridge/clip_recorder.py` runs the renderer as a second process; the sim only queues the pose
(25 Hz, a pickle, ~3 KB, never blocks) and the process renders when the match is over, so the match keeps its tick rate and no frame
is dropped. GL: `SIM_RECORD_GL=cpu` (default, Mesa llvmpipe) or
`gpu` (the system driver, D3D12 on WSL).

Render cost per frame (720p, 10 stripped robots, readback included), measured in the renderer process: GPU path (Mesa D3D12 on this
WSL, RTX 4070 Ti) median 251 ms, p95 313 ms; llvmpipe median 36 ms, p95 56 ms in the real 60 s match (about 66 ms in a standalone test while
the machine was busier). The WSL D3D12 readback stall dominates the GPU number, so `gpu` is ten times over the 10 ms budget and *slower* than software
here, which is why the default is `cpu`. Both exceed 10 ms, so the renderer is a separate process fed 25 Hz poses and renders after the match instead of
inside the step loop. Overlay (cv2 labels) 0.7 ms and the pipe write to NVENC ffmpeg 2.8 ms per frame. Effect on the sim: the 60 s verification match with `--record 20:40` ran at `tick_hz_mean`
50.0 (a match without recording: 50.0); the render of 502 frames (~20 s) happens after PLAY and delays the match JSON by about 20-30 s. A 20 s clip is
about 1.2 MB.

## YOLO on the GPU

`pixi run -e gpu ...` is the default environment plus onnxruntime-gpu (CUDA 13 wheel; conda-forge cudnn 9, cublas, cufft, curand,
cudart; the Windows driver's libcuda comes from /usr/lib/wsl/lib). The default environment keeps the CPU onnxruntime (`pixi install -e gpu`
once). `SIM_YOLO_PROVIDER=auto|cuda|cpu` (default auto: CUDA when onnxruntime has it, else CPU; `cuda` fails if it does not);
the provider in use is logged (`[yolo] onnxruntime provider in use: ...`); CUDA runs with 1 intra-op thread, CPU with 4.
`batch_matches --yolo` re-runs itself under `pixi run -e gpu` when the gpu environment exists and the current interpreter has no CUDA
provider (`SIM_YOLO_PROVIDER=cpu` or `SS_GPU_REEXEC` prevents it). Model latency, 320x320 input, median of 300 runs: CPU
(4 threads) 8.7 ms, CUDA 2.6 ms (p95 4.2 ms); letterbox preprocessing 0.15 ms.

## Parallel matches (`--parallel K`)

K matches at once, match k in slot k: network slot `SS_SLOT=k` (netem.md: `br-sim<k>`, `ss<k>-rN`, subnet 10.77.<k+1>.0/24), ROS domain
base `--domain-base + 20 k` (domains 150-159, 170-179, ...; at most 4 with base 150 because of the 232 limit), the bridge's packet counter
bound to the slot's broadcast address, and its own flag/trace/JSON files (named by profile and match number). No web port. K=1
(default) keeps the original names and base.

Measured, two 30 s 5v5 `good` matches at once (16 cores): mean `tick_hz` over PLAY 37.1 and 39.0 (target 50; a single match: 50.0).
Both are below 45, so the brains see a slowed-down world: **default K=1**. `tick_hz_mean` (and `tick_hz_last`) are in each match JSON
and printed by the runner.

## Head-to-head matches (`--home-brain`, `--away-brain`, `--home-network`, `--away-network`)

Both teams normally run the same brain, so the goal difference is 0 by symmetry. Head-to-head puts different brains and/or
different network conditions on the two teams:

```
pixi run python scripts/batch_matches.py --profiles good,rcap_bad --matches 10 --duration 120 \
    --home-brain current --away-brain baseline --baseline-root <worktree of 8d241a2> --out results/h2h1_a.csv
```

- `--home-brain` / `--away-brain` (`current`|`baseline`) default to `--brain`. Same brain on both teams: unchanged behaviour
  (`--brain baseline` overlays `ros_ws_baseline` for everyone). Mixed: `ros_ws` stays the sourced workspace and the baseline
  team's robots are started with `home_|away_brain_install:=<ros_ws_baseline>/install` and `home_|away_brain_root:=<baseline
  checkout>` (match.launch.py; the launcher's per-team brain arguments, with the install dir pointed at the baseline build and
  `SS_WS_DIR` set to the runner's workspace for run_match.sh's overlay): that team's `brain_node` is run by absolute path from the baseline install, with the baseline
  checkout's `config.yaml`, behaviour trees and vision config. `--ws` names the baseline workspace then. Teams use different
  UDP ports (10000 + team), so protocol v7 vs v8 does not matter. Both teams get the same `brain_params`/settings.
- `--home-network P` / `--away-network P` give one team's five robots profile P (per-robot overrides, `home1..5=P`); the
  scenario profile from `--profiles` applies to the other team. A permanent `blackout` (no `:T`) isolates the team for the
  whole match. Not combinable with a `blackout:T@O` scenario profile (that one is applied to all robots by the runner).
- Output names carry `<home>-vs-<away>` instead of the brain. The CSV/JSON get `home_brain`, `away_brain`, `home_profile`,
  `away_profile` (last four CSV columns, so older CSVs stay aligned); the `brain` column is `current-vs-baseline` etc., so
  `summarise_batch.py --main` ignores mixed rows.
- Swapping sides: run a second set with home and away exchanged (and the same seeds), writing to another CSV.

Summary: `pixi run python scripts/summarise_batch.py --h2h results/h2h1_a.csv results/h2h1_b.csv --perspective current
[--out h2h.json]`. `--perspective home` (default) reports the CSV's home team; `current`/`baseline` follow that brain across
both runs so swapped sets pool. Per group (scenario profile, brains, team networks): n = matches, goal difference and goals
for/against per match (mean, Student-t 95% CI), W/D/L, exact two-sided binomial sign test on wins vs losses (draws dropped)
and a Wilcoxon signed-rank p on the goal differences (normal approximation).

**H2H-1: new vs pre-change, good and rcap_bad.** Set A: `--home-brain current --away-brain baseline --profiles good,rcap_bad`.
Set B: the same with `--home-brain baseline --away-brain current` (other `--out`). Summarise both with
`--h2h A.csv B.csv --perspective current`.

**H2H-2: new isolated vs new healthy.** `--brain current --profiles good --away-network blackout --out results/h2h2_a.csv`
(away isolated), then `--home-network blackout` (home isolated) for the swapped set; summarise with `--perspective home` on
the second set, or read the first set's goal difference negated (the isolated team is `away`).

## Baseline brain (Robotedge 8d241a2, protocol v7)

```
git -C $ROBOTEDGE_ROOT worktree add --detach /path/to/baseline 8d241a2
ROBOTEDGE_ROOT=/path/to/baseline SS_WS_DIR=$PWD/ros_ws_baseline PACKAGES=brain pixi run bash scripts/setup_ws.sh
pixi run python scripts/batch_matches.py --brain baseline --baseline-root /path/to/baseline ...
```

`SS_WS_DIR` (default `ros_ws`) is the colcon workspace `setup_ws.sh` builds into and `run_match.sh` overlays on `ros_ws`
(the bridge stays from `ros_ws`; only the brain comes from the overlay). `ros_ws_baseline/` is gitignored.

## Blackout keyed to PLAY: `blackout:T@O`

`blackout:T@O` cuts the team link of every robot (loss 100% both ways) for T seconds, starting O seconds (default 20) after
the first PLAY of the match, then returns to `good`. Plain `blackout` never reverts. The profiles are applied by the batch
runner, not at launch:

1. the bridge writes `<out stem>_<profile>_<brain>_<n>.play` (JSON, wall clock) when it records its first PLAY sample;
2. the runner polls it, waits O wall seconds, runs `netem.sh apply blackout`, and writes `<...>.blackout`
   (`{"start": wall, "end": null}`) when the profile is in force; at start + T it writes `end` and runs `netem.sh apply good`
   (its own timer, not netem.sh's background job);
3. the bridge re-reads the `.blackout` file (~5 Hz) and marks each PLAY sample inside `[start, end)` as blackout.

The window is in seconds of CLOCK_MONOTONIC (time.monotonic, shared by the runner and the bridge; this WSL's wall clock steps by seconds, which once shortened a 15 s window to 10 PLAY s and flagged stray samples), so if the sim runs slower than real time fewer than T PLAY seconds fall inside it. The
real window is recorded in the match JSON as `blackout_window` (`start_play_s`, `end_play_s`, `duration_s`, in PLAY seconds
of the recorder) and in `meta.blackout_wall`; the CSV has `blackout_start_play_s`, `blackout_end_play_s`, `blackout_s`.
The summary has `segments.blackout` and `segments.outside` (team metrics computed over those samples only; a robot's
memory is reset at each boundary, so a flip across the boundary counts in neither). CSV columns: `<team>_bo_*` and
`<team>_out_*` for role_flips_per_min, lead_changes_per_min, duplicate_striker_frac, ball_events, reaction_excluded,
reaction_median_s, reaction_p90_s, reaction_censored. Empty when the profile has no blackout.

## Metrics (soccer_sim/metrics.py, recorded in the bridge at 25 Hz, in simulated time, PLAY only)

The bridge reads each brain's `/brain/world_model` (via `world_view.to_world`): comm mode, ball estimates and, for brains
that publish them, the `brain` object (`player_role`, `outfield_rank`, `im_lead`, `ball_owner_id`, `role_own_view`; Robotedge
commit d6f87d7 and later). It also reads the published decision (`/brain/decision`, else `/speak`) and the ground truth from
the world. Missing fields (the baseline brain has none of the above) give `unknown` / empty cells and the decision-based
fallback; the CSV column `<team>_role_signal` (`player_role` or `decision`) and `<team>_duplicate_signal` (`im_lead` or
`decision`) say which signal produced the number.

| Metric | Definition |
|---|---|
| chasing | decision in chase, adjust, kick, dribble, cross, auto_visual_kick. The fallback striker signal |
| role flips / min | PRIMARY: changes of a robot's `player_role` between consecutive PLAY samples; FALLBACK (no `player_role`): changes of its chasing flag. Team value = all robots' flips / PLAY minutes. Column `role_signal` |
| chase flips / min | changes of the chasing flag, always computed (decision based) |
| lead changes / min | changes of `im_lead` per robot (a separate count from role flips); empty without `im_lead` |
| decision flips / min | any change of decision string, same counting |
| duplicate striker (`duplicate_striker_frac`) | PRIMARY, lead based: PLAY time with >= 2 outfield (non-keeper) robots of the team with `im_lead` true / team PLAY time with lead data. FALLBACK for a team that publishes no `im_lead` (baseline brain): PLAY time with >= 2 chasing robots (keeper included) / PLAY time. Column `duplicate_signal` says which |
| `duplicate_lead_frac`, `duplicate_role_frac`, `duplicate_decision_frac` | the three variants separately: >= 2 outfield robots with `im_lead`; >= 2 outfield robots with `player_role` striker AND chasing; >= 2 chasing robots (decision only) |
| ball event | ground-truth ball moves > 1 m between samples (teleport) or its speed rises through 1.5 m/s (kick) |
| reaction time | event to the team's nearest outfield robot (for a kick: not one within 0.7 m of the ball before it moved) facing the ball within 30 degrees, or switching to a chasing decision. Events where that robot already faces the ball within 30 degrees AND is already chasing (decision) at event time are excluded, not scored 0: `reaction_excluded` counts them (they stay in `ball_events`, flagged `excluded` in the JSON). `reaction_median_s` and `reaction_p90_s` (linear interpolation) are over the uncensored scored events; `reaction_censored` counts events with no reaction within 8 s |
| ball error | mean distance to the true ball of the fused / own (local) estimate, pooled over samples with a valid estimate, plus the valid fraction |
| comm mode share | fraction of PLAY time in each `comm_mode` of the robots' world model |
| goals, goal_diff | the GameController score at the end (home minus away). The packet-budget zeroing is not applied |
