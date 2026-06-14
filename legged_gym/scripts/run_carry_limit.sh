#!/usr/bin/env bash
# 搬运极限实验:平地 vx=0.5 负重行走,负载扫到 [2,60]kg(含 OOD),找各变体存活极限。
set -uo pipefail
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A
PYBIN="${PYTHON:-python}"

# 变体 -> "run_dir ckpt"
run_one () { # run_one <label> <run> <ckpt>
  echo ">>> [$(date +%H:%M:%S)] $1 (load[2,60] flat vx0.5)"
  "$PYBIN" legged_gym/scripts/play.py --task=wheelfoot_flat --headless --num_envs 30 \
    --load_run "$2" --checkpoint "$3" --seed 1 \
    --load_mass_min 2 --load_mass_max 60 --load_hold --flat_terrain --cmd_vx 0.5 \
    || echo "!!! $1 FAILED (continue)"
}

run_one "Model-guided"    Jun04_23-18-34_wide2-30_model_guided_seed1    11000
run_one "Estimate-guided" Jun06_00-18-09_wide2-30_estimate_guided_seed1 16000
run_one "Source-guided"   Jun05_17-50-01_wide2-30_source_guided_seed42  16000
run_one "RL-only"         Jun05_03-01-38_wide2-30_rl_only_seed1         11000

echo "===== CARRY LIMIT SWEEP DONE (4 runs) ====="
