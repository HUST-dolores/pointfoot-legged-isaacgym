#!/bin/bash
# 看 v8【真·交替迈步】策略（本机开窗口）。交替诊断已确认:左右脚各50%摆动、支撑0.42/0.42均衡、每集换脚36次。
# 破局奖励=weight_shift(按相位教重心转移)+相位门控swing_phase. 用法: bash play_stepV8.sh [cmd_roll]  默认0.2
#   bash play_stepV8.sh 0.0   # 纯迈步(不许滚,看左右轮流最清)
#   bash play_stepV8.sh 0.2   # 略带滚+迈步
set -u
VX=${1:-0.2}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[demo] config 已还原"; }
trap restore EXIT INT TERM
# v8 是 use_gait_phase=True + use_gait_stepping=True; 对上 obs 维 + 强制迈步命令
sed -i 's/^        use_gait_phase = .*/        use_gait_phase = True/' "$CFG"
sed -i 's/^        use_gait_stepping = .*/        use_gait_stepping = True/' "$CFG"
sed -i 's/^        cmd_step_range = .*/        cmd_step_range = [0.4, 0.4]/' "$CFG"
sed -i 's/frequencies = \[1.5, 2.5\]/frequencies = [1.0, 1.5]/' "$CFG"
cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
echo "[demo] v8 真·交替迈步 Jul02_12-12-17_ ckpt10000  (cmd_roll=$VX, cmd_step=0.4)"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat --load_run Jul02_12-12-17_ --checkpoint 10000 --num_envs 1 --cmd_vx "$VX"
