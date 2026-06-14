#!/usr/bin/env bash
# 重训 source-guided 宽负载基底(2-30kg)。上次 seed1 优化坍缩(ep长度120→61),
# 换种子重训。config 与训练成功的 model/rl 完全一致,仅架构标志为 source。
# 退出时自动还原 config(trap),不污染工作区。
#
# 用法:
#   bash legged_gym/scripts/retrain_source.sh
# 想跑多个种子保险(串行,挑存活的那个),把 SEEDS 写成 "42 45 7":
#   SEEDS="42 45" bash legged_gym/scripts/retrain_source.sh
# 后台跑:
#   nohup bash legged_gym/scripts/retrain_source.sh > /tmp/retrain_source.log 2>&1 &
set -uo pipefail
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A

# ===== 可调 =====
SEEDS="${SEEDS:-42}"          # 上次崩的是 seed1;换 42(旧成功 source 用 45,也可)
LOAD_MAX="${LOAD_MAX:-30}"    # add_load_range = [2, LOAD_MAX]
MAX_ITER="${MAX_ITER:-16000}" # 旧成功 source 训到 16000;比 model/rl 的 11000 多给余量
# 如想试 hyperparam,训练前手动改 wheelfoot_flat_config.py 的 entropy_coef / learning_rate
# ================

CFG=legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BAK="/tmp/wf_cfg_backup_retrain_$$.py"
cp "$CFG" "$BAK"
restore(){ cp "$BAK" "$CFG"; echo "[retrain_source] 已还原 config"; }
trap restore EXIT INT TERM

# source-guided 架构标志:qs=False resid=False torq=True (resid 在 qs=False 时是 no-op)
python legged_gym/scripts/set_variant_cfg.py False False True "$LOAD_MAX" \
  || { echo "set_variant_cfg 失败,退出"; exit 1; }

for S in $SEEDS; do
  RUN="wide2-${LOAD_MAX}_source_guided_seed${S}"
  echo "============================================================"
  echo ">>> [$(date +%H:%M:%S)] TRAIN $RUN  (qs=F resid=F torq=T, load=[2,$LOAD_MAX], iter=$MAX_ITER)"
  echo "============================================================"
  python legged_gym/scripts/train.py --task=wheelfoot_flat --headless \
    --seed "$S" --run_name "$RUN" --max_iterations "$MAX_ITER" \
    || echo "!!! seed $S 训练异常退出(继续下一个)"
done

echo "============================================================"
echo "===== RETRAIN SOURCE DONE (seeds: $SEEDS) ====="
echo "训练后看健康度:grep 'Mean episode length' 训练日志,应升到 ~700+(像 model/rl)"
echo "============================================================"
# config 由 trap 自动还原
