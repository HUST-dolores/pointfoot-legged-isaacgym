#!/bin/bash
# 看 B【纯涌现基线】策略（本机开窗口）。B=v3 纯涌现，指标显示双支撑为主=偏"滚"（对照组）。
# 用法: bash play_stepB.sh [cmd_roll]   默认 0.2（同样命令它迈步，看它是不是还是滚）
set -u
VX=${1:-0.2}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[demo] config 已还原"; }
trap restore EXIT INT TERM
sed -i 's/^        use_gait_phase = True/        use_gait_phase = False/' "$CFG"
sed -i 's/^        cmd_step_range = .*/        cmd_step_range = [0.4, 0.4]/' "$CFG"
cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
echo "[demo] B 基线策略 Jul02_00-25-00_ ckpt10000  (cmd_roll=$VX, cmd_step=0.4) —— 对照:它大概率还是滚"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat --load_run Jul02_00-25-00_ --checkpoint 10000 --num_envs 1 --cmd_vx "$VX"
