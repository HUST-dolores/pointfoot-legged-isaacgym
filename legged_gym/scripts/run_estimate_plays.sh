#!/usr/bin/env bash
# 给 estimate-guided 宽基底(2-30kg)补全 ExpC 数据,复刻 model/source/rl 同一配方。
set -uo pipefail
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A

RUN=Jun06_00-18-09_wide2-30_estimate_guided_seed1
COMMON="--task=wheelfoot_flat --headless --num_envs 30 --checkpoint 16000 --load_run $RUN --seed 1 --load_mass_min 2 --load_mass_max 30 --load_hold --flat_terrain"
PYBIN="${PYTHON:-python}"

run () { local label=$1; shift
  echo "============================================================"
  echo ">>> [$(date +%H:%M:%S)] $label"
  echo "============================================================"
  "$PYBIN" legged_gym/scripts/play.py $COMMON "$@" || echo "!!! $label FAILED (continue)"
}

run "walk flat (slope0 + 估计扫描)"  --cmd_vx 0.5
for S in 8 12 16 20 24 28; do run "walk slope${S}"  --cmd_vx 0.5 --slope_deg $S; done
run "static drift"  --cmd_vx 0
run "estop 1.5"     --cmd_vx 0 --estop_vx 1.5

echo "===== ESTIMATE PLAYS DONE (9 runs) ====="
