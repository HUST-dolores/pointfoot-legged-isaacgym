#!/bin/bash
# 看 v8-robust2【地形版真·交替迈步】(从v8微调+粗糙/斜坡地形,诊断确认0.50/0.50交替). 用法: bash play_stepV8robust.sh [cmd_roll] 默认0.2
set -u
VX=${1:-0.2}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[demo] config 已还原"; }
trap restore EXIT INT TERM
sed -i 's/^        use_gait_phase = .*/        use_gait_phase = True/' "$CFG"
sed -i 's/^        use_gait_stepping = .*/        use_gait_stepping = True/' "$CFG"
sed -i 's/^        cmd_step_range = .*/        cmd_step_range = [0.4, 0.4]/' "$CFG"
sed -i 's/frequencies = \[1.5, 2.5\]/frequencies = [1.0, 1.5]/' "$CFG"
sed -i 's/^        num_rows = 1  #/        num_rows = 10  #/' "$CFG"
sed -i 's/^        terrain_proportions = .*/        terrain_proportions = [0.5, 0.5, 0.0, 0.0, 0.0]/' "$CFG"
cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
echo "[demo] v8-robust2 地形版交替迈步 Jul02_16-11-09_ ckpt14500 (cmd_roll=$VX)"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat --load_run Jul02_16-11-09_ --checkpoint 14500 --num_envs 1 --cmd_vx "$VX"
