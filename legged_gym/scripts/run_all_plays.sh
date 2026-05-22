#!/usr/bin/env bash
# Batch re-play for all 4 policies × {static, walk@0.5} (+OOD), ckpt 11000.
# Each invocation auto-aligns env/encoder dims from saved cfg, auto-names
# the .mat and plots dir, and exits headless.
#
# Run from repo root after:
#   conda activate pointfoot_legged_gym
#   export ROBOT_TYPE=WF_TRON1A
#
# Stop on first failure so a config issue surfaces early instead of compounding.
set -euo pipefail

ENV="--task=wheelfoot_flat --headless --num_envs 20 --checkpoint 11000"
PYTHON_BIN="${PYTHON:-python}"
TRAIN_RANGE="--load_mass_min 2 --load_mass_max 4"
OOD_LOW="--load_mass_min 1 --load_mass_max 2"
OOD_HIGH="--load_mass_min 4 --load_mass_max 6"
STATIC="--cmd_vx 0 --cmd_vy 0 --cmd_yaw 0"
WALK="--cmd_vx 0.5 --cmd_vy 0 --cmd_yaw 0"

MAIN=exper_qs_resi_load_boost_3_seed_42
LB6=exper_qs_resi_load_boost_6
DIRECT=exper_qs_noresi_load_boost_3
HIST=exper_history_only_load_boost_3

run () {  # run <label> <args...>
  local label=$1; shift
  echo "============================================================"
  echo ">>> [$(date +%H:%M:%S)] $label"
  echo "============================================================"
  "$PYTHON_BIN" legged_gym/scripts/play.py $ENV "$@"
}

# --- main lb=3 (3 seeds static + 1 walk) ---
run "main lb=3 static seed=42"  --load_run $MAIN --seed 42 $STATIC $TRAIN_RANGE
run "main lb=3 static seed=43"  --load_run $MAIN --seed 43 $STATIC $TRAIN_RANGE
run "main lb=3 static seed=44"  --load_run $MAIN --seed 44 $STATIC $TRAIN_RANGE
run "main lb=3 walk   seed=42"  --load_run $MAIN --seed 42 $WALK   $TRAIN_RANGE

# --- main lb=6 (static + walk) ---
run "main lb=6 static seed=42"  --load_run $LB6  --seed 42 $STATIC $TRAIN_RANGE
run "main lb=6 walk   seed=42"  --load_run $LB6  --seed 42 $WALK   $TRAIN_RANGE

# --- OOD low [1,2] kg, main lb=3 ---
run "OOD-low static"            --load_run $MAIN --seed 42 $STATIC $OOD_LOW
run "OOD-low walk"              --load_run $MAIN --seed 42 $WALK   $OOD_LOW

# --- OOD high [4,6] kg, main lb=3 ---
run "OOD-high static"           --load_run $MAIN --seed 42 $STATIC $OOD_HIGH
run "OOD-high walk"             --load_run $MAIN --seed 42 $WALK   $OOD_HIGH

# --- direct (no residual) lb=3 ---
run "direct static"             --load_run $DIRECT --seed 42 $STATIC $TRAIN_RANGE
run "direct walk"               --load_run $DIRECT --seed 42 $WALK   $TRAIN_RANGE

# --- history_only lb=3 ---
run "history_only static"       --load_run $HIST   --seed 42 $STATIC $TRAIN_RANGE
run "history_only walk"         --load_run $HIST   --seed 42 $WALK   $TRAIN_RANGE

echo "============================================================"
echo "all 14 plays done."
echo "============================================================"
