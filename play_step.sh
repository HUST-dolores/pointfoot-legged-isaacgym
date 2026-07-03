#!/bin/bash
# 看训练好的【迈步策略】走路（本机开窗口实时看）。轮子自由、腿做步态。
# 用法:  bash play_step.sh [前进速度]     默认 0.8 m/s
#   bash play_step.sh 0.8    # 中速前进
#   bash play_step.sh 0.3    # 慢走(更容易看清抬腿)
#   bash play_step.sh 1.5    # 快走
# 播放中按 v 切换渲染同步；鼠标拖动转视角。
set -u
VX=${1:-0.8}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
cd "$REPO"
export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
echo "[play] 迈步策略 Jul01_11-15-52_ ckpt15000, cmd_vx=$VX"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat \
    --load_run Jul01_11-15-52_ --checkpoint 15000 --num_envs 1 --cmd_vx "$VX"
