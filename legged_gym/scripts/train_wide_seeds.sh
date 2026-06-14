#!/bin/bash
# 训练 4 个变体 × 多 seed（可选宽负载范围），用于 ①多 seed 坐实结论 ②Ch5 宽负载重训。
# 每个变体临时改 config 的消融标志 + 负载范围，训练完退出时自动还原 config（trap）。
# 用法: bash legged_gym/scripts/train_wide_seeds.sh
# 改下面 4 个变量即可。每个训练约数小时——休息时跑。
set -u
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "[train_wide_seeds] ERROR: neither '$PYTHON_BIN' nor python3 found in PATH" >&2
    exit 1
  fi
fi
echo "[train_wide_seeds] Python: $PYTHON_BIN ($(command -v "$PYTHON_BIN"))"

# ===== 可调参数（可用环境变量覆盖，如 VARIANTS=source_guided bash ...）=====
LOAD_MAX="${LOAD_MAX:-30}"          # 负载范围上界 add_load_range=[2,LOAD_MAX]
MAX_ITER="${MAX_ITER:-11000}"      # 训练迭代数（~4h/个）
SEEDS="${SEEDS:-1}"                # 训练 seed
VARIANTS="${VARIANTS:-model_guided rl_only}"   # 变体: model_guided|estimate_guided|source_guided|rl_only
# ====================

CFG=legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BAK="/tmp/wf_cfg_backup_$$.py"
cp "$CFG" "$BAK"
restore(){ cp "$BAK" "$CFG"; echo "[train_wide_seeds] 已还原 config"; }
trap restore EXIT INT TERM

# 变体 → (qs resid torq)
flags_model_guided="True True True"
flags_estimate_guided="True False True"
flags_source_guided="False False True"
flags_rl_only="False False False"

for V in $VARIANTS; do
  eval "F=\$flags_$V"
  for S in $SEEDS; do
    "$PYTHON_BIN" legged_gym/scripts/set_variant_cfg.py $F $LOAD_MAX || { echo "set cfg 失败,跳过"; continue; }
    RUN="wide2-${LOAD_MAX}_${V}_seed${S}"
    echo "===== TRAIN $RUN  (flags=$F load=[2,$LOAD_MAX] iter=$MAX_ITER) ====="
    "$PYTHON_BIN" legged_gym/scripts/train.py --task=wheelfoot_flat --headless \
      --seed "$S" --run_name "$RUN" --max_iterations "$MAX_ITER"
  done
done
echo "===== ALL TRAINING DONE ====="
# config 由 trap 自动还原
