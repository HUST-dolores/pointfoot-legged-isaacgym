#!/bin/bash
# Roll out v8 headless and RECORD a physically-valid stepping reference -> npz.
# Uses the SAME config edits as play_stepV8.sh so the gait matches the confirmed demo.
# Usage: bash collect_v8_ref.sh [cmd_vx] [out_npz]   defaults: 0.2  /tmp/v8_ref.npz
set -u
VX=${1:-0.2}
OUT=${2:-/tmp/v8_ref.npz}
REPO=/home/xu/limx_rl/pointfoot-legged-gym
CFG=$REPO/legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BK=$(mktemp); cp "$CFG" "$BK"
restore(){ cp "$BK" "$CFG"; rm -f "$BK"; echo "[collect] config 已还原"; }
trap restore EXIT INT TERM
# identical to play_stepV8.sh
sed -i 's/^        use_gait_phase = .*/        use_gait_phase = True/' "$CFG"
sed -i 's/^        use_gait_stepping = .*/        use_gait_stepping = True/' "$CFG"
sed -i 's/^        cmd_step_range = .*/        cmd_step_range = [0.4, 0.4]/' "$CFG"
sed -i 's/frequencies = \[1.5, 2.5\]/frequencies = [1.0, 1.5]/' "$CFG"
cd "$REPO"; export ROBOT_TYPE=WF_TRON1A
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"; [ -f "$CONDA_SH" ] && source "$CONDA_SH" && conda activate pointfoot_legged_gym
PY="python"; command -v python >/dev/null 2>&1 || PY="$HOME/anaconda3/envs/pointfoot_legged_gym/bin/python"
export COLLECT_CMD_VX="$VX" COLLECT_STEPS="600" COLLECT_WARMUP="60" COLLECT_OUT="$OUT"
echo "[collect] v8 ckpt10000  cmd_vx=$VX -> $OUT  (headless)"
"$PY" legged_gym/scripts/collect_v8_ref.py --task=wheelfoot_flat --load_run Jul02_12-12-17_ --checkpoint 10000 --num_envs 1 --headless
