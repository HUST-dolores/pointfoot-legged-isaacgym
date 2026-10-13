#!/usr/bin/env bash
set -u

GPU="${1:?usage: $0 GPU RUN CKPT OUT [SEED THIGH SHANK MOTOR]}"
RUN="${2:?missing run}"
CKPT="${3:?missing checkpoint}"
OUT="${4:?missing output directory}"
SEED="${5:-20260724}"
THIGH="${6:-1.0}"
SHANK="${7:-1.0}"
MOTOR="${8:-1.0}"
TASKS="${PHYS_EVAL_TASKS_OVERRIDE:-clearance,slope,payload,speed}"
REPLICAS="${PHYS_EVAL_REPLICAS_OVERRIDE:-32}"
CLEARANCE_LEVELS="${PHYS_EVAL_CLEARANCE_LEVELS_OVERRIDE:-0}"
SLOPE_LEVELS="${PHYS_EVAL_SLOPE_LEVELS_OVERRIDE:-0,3,6,9,12,15,18,21}"
PAYLOAD_LEVELS="${PHYS_EVAL_PAYLOAD_LEVELS_OVERRIDE:-0,5,10,15,20,25,30,35}"
SPEED_LEVELS="${PHYS_EVAL_SPEED_LEVELS_OVERRIDE:-0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export ROBOT_TYPE=WF_TRON1A

mkdir -p "${OUT}"
cd "${ROOT}" || exit 2
date --iso-8601=seconds >"${OUT}/STARTED"

run_metric() {
    local task="$1"
    local levels="$2"
    local task_out="${OUT}/${task}"
    mkdir -p "${task_out}"
    env \
        CUDA_VISIBLE_DEVICES="${GPU}" \
        PHYS_EVAL_TASK="${task}" \
        PHYS_EVAL_LEVELS="${levels}" \
        PHYS_EVAL_REPLICAS="${REPLICAS}" \
        PHYS_EVAL_NUM_ENVS=256 \
        PHYS_EVAL_SETTLE_S=2.0 \
        PHYS_EVAL_DURATION_S=8.0 \
        PHYS_EVAL_RUN="${RUN}" \
        PHYS_EVAL_CKPT="${CKPT}" \
        PHYS_EVAL_SEED="${SEED}" \
        PHYS_EVAL_THIGH="${THIGH}" \
        PHYS_EVAL_SHANK="${SHANK}" \
        PHYS_EVAL_MOTOR="${MOTOR}" \
        PHYS_EVAL_PREFIX="physeval_${CKPT}_${task}_$$_${GPU}" \
        PHYS_EVAL_OUT="${task_out}" \
        "${PY}" legged_gym/scripts/physical_capability_eval.py --headless \
        >"${task_out}/console.log" 2>&1
}

status=0
IFS=',' read -r -a task_list <<<"${TASKS}"
for task in "${task_list[@]}"; do
    [[ "${status}" -eq 0 ]] || break
    case "${task}" in
        clearance) run_metric clearance "${CLEARANCE_LEVELS}" || status=$? ;;
        slope) run_metric slope "${SLOPE_LEVELS}" || status=$? ;;
        payload) run_metric payload "${PAYLOAD_LEVELS}" || status=$? ;;
        speed) run_metric speed "${SPEED_LEVELS}" || status=$? ;;
        *)
            printf 'unknown PHYS_EVAL_TASKS_OVERRIDE entry: %s\n' "${task}" >&2
            status=2
            ;;
    esac
done

printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUT}/DONE"
fi
exit "${status}"
