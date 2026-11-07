#!/usr/bin/env bash
# Batch 4: new brain (Robotedge 8958303) vs pre-change brain (8d241a2), both rerun on the same seeds,
# after the blackout-window ramp fix and the chase-family reaction metric (soccerSim 1d659d8).
set -u
cd "$(dirname "$0")/../.."
B=results/batch4
BASE=${BASELINE_ROBOTEDGE_ROOT:-}  # the 8d241a2 checkout; needed only by phases that run the baseline brain
PY="$HOME/.pixi/bin/pixi run python scripts/batch_matches.py"
# PHASES="main_current sweep_current ..." runs only those phases (default: all).
run() { local name=$1; shift; if [ -n "${PHASES:-}" ] && [[ " $PHASES " != *" $name "* ]]; then echo "=== $name skipped"; return; fi; echo "=== $name $(date +%T)"; $PY "$@" --out $B/$name.csv > $B/$name.out 2>&1; echo "=== $name rc=$? $(date +%T)"; }
MAIN=(--profiles good,rcap_bad,terrible,blackout:30@30 --matches 8 --duration 120 --seed 1 --trace --record 25:45 --record-matches 0)
run main_current  "${MAIN[@]}" --brain current
run main_baseline "${MAIN[@]}" --brain baseline --baseline-root "$BASE"
SWEEP=(--profiles lat0_loss15,lat150_loss15,lat300_loss15,lat600_loss15,lat1000_loss15,lat150_loss0,lat150_loss30,lat150_loss45,lat150_loss60 --matches 2 --duration 90 --seed 11 --trace)
run sweep_current  "${SWEEP[@]}" --brain current
run sweep_baseline "${SWEEP[@]}" --brain baseline --baseline-root "$BASE"
YOLO=(--profiles rcap_bad,blackout:30@30 --matches 3 --duration 120 --seed 21 --yolo --trace)
run yolo_current  "${YOLO[@]}" --brain current
run yolo_baseline "${YOLO[@]}" --brain baseline --baseline-root "$BASE"
H2H=(--profiles good,rcap_bad --matches 3 --duration 120 --seed 31 --trace)
run h2h1_a "${H2H[@]}" --home-brain current --away-brain baseline --baseline-root "$BASE"
run h2h1_b "${H2H[@]}" --home-brain baseline --away-brain current --baseline-root "$BASE"
run h2h2_a --profiles good --matches 3 --duration 120 --seed 31 --trace --away-network blackout
run h2h2_b --profiles good --matches 3 --duration 120 --seed 31 --trace --home-network blackout
echo "=== done $(date +%T)"
