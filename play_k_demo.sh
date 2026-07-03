#!/bin/bash
# 看 loadgen generalist 在【固定电机 k】+【固定负载】下的姿态/硬撑（本地开窗口实时看）。
# 用法:  bash play_k_demo.sh [k] [load_kg]
#   bash play_k_demo.sh 0.4 50   # 弱电机(32N·m) + 重载50kg → 看它挺直腿硬撑、膝盖顶到上限
#   bash play_k_demo.sh 1.0 50   # 标称电机(80N·m) + 重载50kg → 对照，明显轻松
#   bash play_k_demo.sh 0.4 10   # 弱电机 + 轻载 → 看它蹲着(力臂大反而膝盖更饱和)
# 自动备份/还原 config，退出即恢复。需在有显示器的本机终端里跑(不要加 --headless)。
set -u
K=${1:-0.4}; LOAD=${2:-50}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp)
cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[demo] config 已还原"; }
trap restore EXIT INT TERM

# 强制：电机设计开 + critic 含 motor 维(匹配训练ckpt) + 固定 k
sed -i 's/^\( *use_motor_design_in_critic = \).*/\1True/'   "$CFG"
sed -i 's/^\( *randomize_motor_design = \).*/\1True/'       "$CFG"
sed -i "s/^\( *motor_torque_scale_range = \).*/\1[$K, $K]/" "$CFG"
echo "[demo] k=$K, load=${LOAD}kg  (config 已临时改，退出会还原)"
grep -nE "use_motor_design_in_critic = |randomize_motor_design = |motor_torque_scale_range = " "$CFG" | head

cd "$REPO"
export ROBOT_TYPE=WF_TRON1A
# 激活本地 conda 环境（交互 shell 里 python 可能不在 PATH）
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat \
    --load_run Jun30_23-49-24_ --checkpoint 16000 --num_envs 1 \
    --load_mass_min "$LOAD" --load_mass_max "$LOAD" --keep_load --cmd_vx 0.5
