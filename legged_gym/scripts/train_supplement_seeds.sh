#!/usr/bin/env bash
# 96h+ 无人值守:四变体 × 多个全新 seed 补充训练(用于多 seed mean±std 定稿)。
# 基于 train_wide_seeds.sh,改进:
#   ① seed 外层 / 变体内层「交织」—— 每轮 4 变体各 +1 seed,即使中途停机覆盖也均衡;
#   ② 跳过已训完的 run(final ckpt 存在)—— 断电后重跑同一条命令可续;
#   ③ 每个 run 独立日志 + 主进度日志,单个训练崩溃不影响后续;
#   ④ 退出(正常/Ctrl-C/被杀)自动还原 config(trap);
#   ⑤ num_envs / MAX_ITER 对齐现有 seed2/3 口径(ckpt16000),保证可池化。
#
# 用法(后台跑,关终端不掉):
#   cd /home/xu/limx_rl/pointfoot-legged-gym
#   nohup bash legged_gym/scripts/train_supplement_seeds.sh > /tmp/supp_train_master.log 2>&1 &
#
# 回来后看进度:   tail -n 60 /tmp/supp_train_master.log
# 看某个 run 详情: tail -n 40 /tmp/supp_train_logs/wide2-30_model_guided_seed5.log
# 停止:            kill <脚本PID>   (会自动还原 config)
set -uo pipefail
cd /home/xu/limx_rl/pointfoot-legged-gym
export ROBOT_TYPE=WF_TRON1A

PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN=python3

# ===== 可调(可用环境变量覆盖)=====
LOAD_MAX="${LOAD_MAX:-30}"                          # add_load_range = [2, LOAD_MAX]
MAX_ITER="${MAX_ITER:-16000}"                       # 与现有 seed2/3 对齐(ckpt16000),~5.8h/个
SEEDS="${SEEDS:-5 6 7 8 9 10}"                      # 全新 seed,避开已用的 1 2 3 4 42
VARIANTS="${VARIANTS:-model_guided estimate_guided source_guided rl_only}"
LOGDIR="logs/wheelfoot_flat/WF_TRON1A"
RUNLOGDIR="/tmp/supp_train_logs"
# ==================================
mkdir -p "$RUNLOGDIR"

CFG=legged_gym/envs/wheelfoot_flat/wheelfoot_flat_config.py
BAK="/tmp/wf_cfg_backup_supp_$$.py"
cp "$CFG" "$BAK"
restore(){ cp "$BAK" "$CFG"; echo "[supp] $(date '+%F %T') 已还原 config"; }
trap restore EXIT INT TERM

# 变体 → (use_qs_in_obs  use_residual_learning  use_torques_in_obs)
flags_model_guided="True True True"
flags_estimate_guided="True False True"
flags_source_guided="False False True"
flags_rl_only="False False False"

done_count=0; skip_count=0; fail_count=0
echo "[supp] $(date '+%F %T') START  seeds=[$SEEDS] variants=[$VARIANTS] iter=$MAX_ITER load=[2,$LOAD_MAX]"

# seed 外层 / 变体内层 —— 每轮让 4 变体各 +1 seed
for S in $SEEDS; do
  for V in $VARIANTS; do
    RUN="wide2-${LOAD_MAX}_${V}_seed${S}"
    # 已训完则跳过(目录名带日期前缀,用 glob 匹配尾部 RUN)
    if ls "$LOGDIR"/*"${RUN}"/model_${MAX_ITER}.pt >/dev/null 2>&1; then
      echo "[supp] $(date '+%F %T') SKIP  $RUN  (已存在 model_${MAX_ITER}.pt)"
      skip_count=$((skip_count+1)); continue
    fi
    eval "F=\$flags_$V"
    if ! "$PYTHON_BIN" legged_gym/scripts/set_variant_cfg.py $F "$LOAD_MAX"; then
      echo "[supp] $(date '+%F %T') CFGFAIL $RUN  (set_variant_cfg 失败,跳过)"
      fail_count=$((fail_count+1)); continue
    fi
    LOG="$RUNLOGDIR/${RUN}.log"
    echo "[supp] $(date '+%F %T') TRAIN $RUN  (flags=$F load=[2,$LOAD_MAX] iter=$MAX_ITER) -> $LOG"
    if "$PYTHON_BIN" legged_gym/scripts/train.py --task=wheelfoot_flat --headless \
         --seed "$S" --run_name "$RUN" --max_iterations "$MAX_ITER" > "$LOG" 2>&1; then
      echo "[supp] $(date '+%F %T') DONE  $RUN"
      done_count=$((done_count+1))
    else
      echo "[supp] $(date '+%F %T') FAIL  $RUN  (异常退出,见 $LOG 末尾;继续下一个)"
      fail_count=$((fail_count+1))
    fi
  done
done

echo "[supp] $(date '+%F %T') ALL DONE  done=$done_count skip=$skip_count fail=$fail_count"
# config 由 trap 自动还原
