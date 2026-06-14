#!/usr/bin/env bash
# 验证建议:把 C-2(定点漂移)和 C-3(急停)搬到斜坡上,看负载扰动 ∝ m·sinβ 是否放大估计优势。
# C-2': 静止保持 @15°;C-3': 下坡急停 @10°。负载 [2,30]。
set -uo pipefail
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A
PYBIN="${PYTHON:-python}"
COMMON="--task=wheelfoot_flat --headless --num_envs 30 --seed 1 --load_mass_min 2 --load_mass_max 30 --load_hold --flat_terrain"

play () { # play <label> <run> <ckpt> <extra...>
  local lbl=$1 run=$2 ck=$3; shift 3
  echo ">>> [$(date +%H:%M:%S)] $lbl"
  "$PYBIN" legged_gym/scripts/play.py $COMMON --load_run "$run" --checkpoint "$ck" "$@" || echo "!!! $lbl FAILED"
}

for spec in \
  "Model    Jun04_23-18-34_wide2-30_model_guided_seed1    11000" \
  "Estimate Jun06_00-18-09_wide2-30_estimate_guided_seed1 16000" \
  "Source   Jun05_17-50-01_wide2-30_source_guided_seed42  16000" \
  "RLonly   Jun05_03-01-38_wide2-30_rl_only_seed1         11000"; do
  set -- $spec; V=$1; RUN=$2; CK=$3
  play "C2' static@15 $V"  "$RUN" "$CK" --cmd_vx 0   --slope_deg 15
  play "C3' estop@10  $V"  "$RUN" "$CK" --cmd_vx 0   --estop_vx 1.5 --slope_deg 10
done
echo "===== C2C3 SLOPE DONE ====="
