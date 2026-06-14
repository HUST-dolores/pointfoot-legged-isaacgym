#!/usr/bin/env bash
# 给 source-guided 宽基底(2-30kg)补全 ExpC 数据:质量扫描+斜坡+静态漂移+急停。
# 完全复刻 model/rl 新基底的配方(num_envs 30, ckpt 11000, load[2,30], --load_hold --flat_terrain)。
# play.py 自动按 source run 对齐架构(n_obs=36)。串行、headless、--exit_after_save 自动开。
set -uo pipefail
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A

RUN=Jun05_17-50-01_wide2-30_source_guided_seed42
COMMON="--task=wheelfoot_flat --headless --num_envs 30 --checkpoint 16000 --load_run $RUN --seed 1 --load_mass_min 2 --load_mass_max 30 --load_hold --flat_terrain"
PYBIN="${PYTHON:-python}"

run () { # run <label> <extra args...>
  local label=$1; shift
  echo "============================================================"
  echo ">>> [$(date +%H:%M:%S)] $label"
  echo "============================================================"
  "$PYBIN" legged_gym/scripts/play.py $COMMON "$@" || echo "!!! $label FAILED (continue)"
}

# 1) walk 平地:质量估计扫描 + 斜坡0 参考
run "walk flat (slope0 + 估计扫描)"  --cmd_vx 0.5
# 2-7) walk 斜坡
for S in 8 12 16 20 24 28; do
  run "walk slope${S}"  --cmd_vx 0.5 --slope_deg $S
done
# 8) 静态漂移
run "static drift"  --cmd_vx 0
# 9) 急停 1.5
run "estop 1.5"     --cmd_vx 0 --estop_vx 1.5

echo "============================================================"
echo "===== SOURCE PLAYS DONE (9 runs) ====="
echo "============================================================"
