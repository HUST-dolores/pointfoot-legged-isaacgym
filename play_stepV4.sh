#!/bin/bash
# 看 v4【迈步】策略走路（本机开窗口实时看）。v4=反滚动配方，指标显示单脚支撑为主=真迈步。
# 用法: bash play_stepV4.sh [cmd_roll]   默认 0.2（低滚动，让迈步分量主导，看得清抬腿）
#   bash play_stepV4.sh 0.0   # 纯迈步(不许滚)
#   bash play_stepV4.sh 0.4   # 混合(滚+迈)
set -u
VX=${1:-0.2}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[demo] config 已还原"; }
trap restore EXIT INT TERM
# v4 是 emergent(use_gait_phase=False)+cmd_step；对上 obs 维 + 强制迈步命令 cmd_step=0.4
sed -i 's/^        use_gait_phase = True/        use_gait_phase = False/' "$CFG"
sed -i 's/^        cmd_step_range = .*/        cmd_step_range = [0.4, 0.4]/' "$CFG"
cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
echo "[demo] v4 迈步策略 Jul02_02-08-56_ ckpt8500  (cmd_roll=$VX, cmd_step=0.4 强制迈步)"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat --load_run Jul02_02-08-56_ --checkpoint 8500 --num_envs 1 --cmd_vx "$VX"
