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

The window is in wall-clock seconds, so if the sim runs slower than real time fewer than T PLAY seconds fall inside it. The
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
