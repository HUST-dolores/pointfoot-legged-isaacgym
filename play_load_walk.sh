#!/bin/bash
# ============================================================================
# 负载行走 demo —— 机器人边走边扛【恒定负载】。用 Ch5 负载估计策略
#   (model_guided wide2-30，负载范围 2~30kg，最能扛/最稳的一档)。
#
# 用法:  bash play_load_walk.sh [cmd_vx] [load_kg]     默认 0.5 m/s、15 kg
#   bash play_load_walk.sh              # 0.5 m/s 扛 15kg
#   bash play_load_walk.sh 0.8 25       # 0.8 m/s 扛 25kg
#   bash play_load_walk.sh 0.0 30       # 原地站立扛满 30kg(看它压不压得住)
#
# ★为什么要脚本而不是一行命令:
#   本仓库 config 现在停在"迈步"状态(use_gait_stepping/use_gait_phase=True，
#   给 obs 多加 3 维)，而负载权重是【无迈步维】的 48 维 obs。play 的自动对齐
#   只恢复 use_qs_in_obs / num_observations，【不会】关这俩迈步 flag，直接跑会
#   obs 维对不上、加载报错。本脚本临时把两 flag 关掉，退出时 trap 自动还原 config。
# ============================================================================
set -u
VX=${1:-0.5}          # 前进速度指令 m/s
M=${2:-15}            # 负载质量 kg (min=max=恒定)
RUN=Jun13_23-58-05_wide2-30_model_guided_seed3
CKPT=16000
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py

BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[load-demo] config 已还原(迈步 flag 复位)"; }
trap restore EXIT INT TERM

# 临时关掉迈步 obs 维(负载权重无此维)；use_qs_in_obs/num_observations 由 play 自动从存档恢复
sed -i 's/^        use_gait_stepping = .*/        use_gait_stepping = False  # temp-off for load play/' "$CFG"
sed -i 's/^        use_gait_phase = .*/        use_gait_phase = False  # temp-off for load play/' "$CFG"

cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"

echo "[load-demo] 负载行走  $RUN  ckpt$CKPT   cmd_vx=$VX m/s   负载=${M}kg(全程恒定)"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat \
  --load_run "$RUN" --checkpoint "$CKPT" --num_envs 1 \
  --cmd_vx "$VX" --load_mass_min "$M" --load_mass_max "$M" --load_hold
