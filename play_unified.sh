#!/bin/bash
# ============================================================================
# A 统一策略 play —— 【一个策略】命令切换 站/滚 和 迈步行走。
#   run Jul03_15-59-41_  ckpt 13000  (reward+162, ep~1750, 行为诊断双确认)
#
# 用法:  bash play_unified.sh [mode]     mode = stand | roll | step | mix   默认 step
#   bash play_unified.sh stand   # 站立: cmd_step=0, cmd_vx=0   → 两轮着地原地平衡(不迈不滚)
#   bash play_unified.sh roll    # 滚行: cmd_step=0, cmd_vx=0.5 → 两轮着地向前滚(纯轮式)
#   bash play_unified.sh step    # 行走: cmd_step=0.4,cmd_vx=0.2 → 左右交替迈步(点足式)
#   bash play_unified.sh mix     # 混合: cmd_step=0.3,cmd_vx=0.4 → 边滚边迈
#
# 关键: cmd_step=0 → 滚(两轮着地) ; cmd_step>0 → 迈(交替抬脚)。同一个策略、只换命令。
# 窗口内按 v 切渲染同步、ESC 退出。脚本临时改 config,退出自动还原。
# ============================================================================
set -u
ARG1=${1:-step}
case "$ARG1" in
  stand) STEP=0.0; VX=0.0;;
  roll)  STEP=0.0; VX=0.5;;
  step)  STEP=0.4; VX=0.2;;
  mix)   STEP=0.3; VX=0.4;;
  *) STEP=$ARG1; VX=${2:-0.0};;   # 直接传数值: bash play_unified.sh <cmd_step> <cmd_vx>  (支持负数)
esac
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[unified] config 已还原"; }
trap restore EXIT INT TERM
# A 是 use_gait_stepping/phase=True; 固定 cmd_step=$STEP(关纯滚随机), gait 频率对上 A 训练值
sed -i 's/^        use_gait_phase = .*/        use_gait_phase = True/' "$CFG"
sed -i 's/^        use_gait_stepping = .*/        use_gait_stepping = True/' "$CFG"
sed -i 's/^        cmd_step_zero_prob = .*/        cmd_step_zero_prob = 0.0/' "$CFG"
sed -i "s/^        cmd_step_range = .*/        cmd_step_range = [$STEP, $STEP]/" "$CFG"
sed -i 's/frequencies = \[1.5, 2.5\]/frequencies = [1.0, 1.5]/' "$CFG"
cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
echo "[unified] cmd_step=$STEP  cmd_roll=$VX   (Jul03_15-59-41_ ckpt13000)"
"$PY" legged_gym/scripts/play.py --task=wheelfoot_flat --load_run Jul03_15-59-41_ --checkpoint 13000 --num_envs 1 --cmd_vx="$VX"
