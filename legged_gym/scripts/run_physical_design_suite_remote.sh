#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?usage: $0 GPU LABEL THIGH SHANK MOTOR OUT_ROOT [SEED]}"
LABEL="${2:?missing label}"
THIGH="${3:?missing thigh scale}"
SHANK="${4:?missing shank scale}"
MOTOR="${5:?missing motor scale}"
OUT_ROOT="${6:?missing output root}"
SEED="${7:-20260724}"

: "${PHYS_SUITE_CLEARANCE_UPDATES:?set PHYS_SUITE_CLEARANCE_UPDATES}"
: "${PHYS_SUITE_SLOPE_UPDATES:?set PHYS_SUITE_SLOPE_UPDATES}"
: "${PHYS_SUITE_PAYLOAD_UPDATES:?set PHYS_SUITE_PAYLOAD_UPDATES}"
: "${PHYS_SUITE_SPEED_UPDATES:?set PHYS_SUITE_SPEED_UPDATES}"
: "${PHYS_SUITE_JUMP_UPDATES:?set PHYS_SUITE_JUMP_UPDATES}"
: "${PHYS_SUITE_SLOPE_LEVELS:?set PHYS_SUITE_SLOPE_LEVELS}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
LOCO_TRAIN="${ROOT}/legged_gym/scripts/run_physical_adaptation_remote.sh"
LOCO_EVAL="${ROOT}/legged_gym/scripts/run_physical_metric_eval_remote.sh"
JUMP_TRAIN="${ROOT}/legged_gym/scripts/run_physical_jump_adaptation_remote.sh"
JUMP_EVAL="${ROOT}/legged_gym/scripts/run_physical_jump_checkpoint_eval_remote.sh"
OUT="${OUT_ROOT}/${LABEL}"

export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export ROBOT_TYPE=WF_TRON1A

mkdir -p "${OUT}"
if [[ -f "${OUT}/DONE" ]]; then
    echo "[suite] ${LABEL} already complete"
    exit 0
fi

finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
    if [[ "${status}" -eq 0 ]]; then
        date --iso-8601=seconds >"${OUT}/DONE"
    fi
    exit "${status}"
}
trap finish EXIT

date --iso-8601=seconds >"${OUT}/STARTED"
printf '%s,%s,%s,%s,%s,%s\n' \
    "label" "thigh_scale" "shank_scale" "motor_scale" "seed" "gpu" \
    >"${OUT}/design.csv"
printf '%s,%s,%s,%s,%s,%s\n' \
    "${LABEL}" "${THIGH}" "${SHANK}" "${MOTOR}" "${SEED}" "${GPU}" \
    >>"${OUT}/design.csv"
printf '%s,%s,%s,%s,%s\n' \
    "clearance_updates" "slope_updates" "payload_updates" \
    "speed_updates" "jump_updates" >"${OUT}/budgets.csv"
printf '%s,%s,%s,%s,%s\n' \
    "${PHYS_SUITE_CLEARANCE_UPDATES}" "${PHYS_SUITE_SLOPE_UPDATES}" \
    "${PHYS_SUITE_PAYLOAD_UPDATES}" "${PHYS_SUITE_SPEED_UPDATES}" \
    "${PHYS_SUITE_JUMP_UPDATES}" >>"${OUT}/budgets.csv"
printf '%s\n' \
    "slope_levels=${PHYS_SUITE_SLOPE_LEVELS}" \
    "slope_total_vel_error_reward=${PHYS_SUITE_SLOPE_ERROR_REWARD:--4.0}" \
    "slope_total_vel_sigma=${PHYS_SUITE_SLOPE_SIGMA:-0.2}" \
    >"${OUT}/suite_config.env"

read_training_result() {
    local result_path="$1"
    "${PY}" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); print(d["run_name"], d["final_iteration"])' \
        "${result_path}"
}

run_locomotion_task() {
    local task="$1"
    local updates="$2"
    local task_root="${OUT}/${task}"
    local train_out="${task_root}/train"
    local eval_out="${task_root}/eval"
    mkdir -p "${task_root}"

    if [[ ! -f "${train_out}/DONE" || ! -f "${train_out}/result.json" ]]; then
        case "${task}" in
            clearance)
                PHYS_TRAIN_TASK=clearance \
                PHYS_TRAIN_CLEARANCE_LEVELS=0.22,0.22,0.22,0.22,0.22,0.22,0.22,0.22 \
                    bash "${LOCO_TRAIN}" \
                    "${GPU}" "${updates}" \
                    "${LABEL}_${task}_s${SEED}_u${updates}" \
                    "${train_out}" "${THIGH}" "${SHANK}" "${MOTOR}" \
                    "${SEED}" 4096
                ;;
            slope)
                PHYS_TRAIN_TASK=slope \
                PHYS_TRAIN_SLOPE_LEVELS="${PHYS_SUITE_SLOPE_LEVELS}" \
                PHYS_TRAIN_TOTAL_VEL_ERROR_REWARD="${PHYS_SUITE_SLOPE_ERROR_REWARD:--4.0}" \
                PHYS_TRAIN_TOTAL_VEL_SIGMA="${PHYS_SUITE_SLOPE_SIGMA:-0.2}" \
                    bash "${LOCO_TRAIN}" \
                    "${GPU}" "${updates}" \
                    "${LABEL}_${task}_s${SEED}_u${updates}" \
                    "${train_out}" "${THIGH}" "${SHANK}" "${MOTOR}" \
                    "${SEED}" 4096
                ;;
            payload|speed)
                PHYS_TRAIN_TASK="${task}" \
                    bash "${LOCO_TRAIN}" \
                    "${GPU}" "${updates}" \
                    "${LABEL}_${task}_s${SEED}_u${updates}" \
                    "${train_out}" "${THIGH}" "${SHANK}" "${MOTOR}" \
                    "${SEED}" 4096
                ;;
            *)
                echo "unsupported locomotion task ${task}" >&2
                return 2
                ;;
        esac
    fi

    local run checkpoint
    read -r run checkpoint < <(
        read_training_result "${train_out}/result.json"
    )
    if [[ ! -f "${eval_out}/DONE" || ! -f "${eval_out}/result.json" ]]; then
        bash "${LOCO_EVAL}" \
            "${GPU}" "${task}" "${run}" "${checkpoint}" "${eval_out}" \
            "${SEED}" "${THIGH}" "${SHANK}" "${MOTOR}"
    fi
}

run_jump_task() {
    local updates="${PHYS_SUITE_JUMP_UPDATES}"
    local task_root="${OUT}/jump"
    local train_out="${task_root}/train"
    local eval_out="${task_root}/eval"
    mkdir -p "${task_root}"

    if [[ ! -f "${train_out}/DONE" || ! -f "${train_out}/result.json" ]]; then
        bash "${JUMP_TRAIN}" \
            "${GPU}" "${updates}" \
            "${LABEL}_jump_s${SEED}_u${updates}" \
            "${train_out}" "${THIGH}" "${SHANK}" "${MOTOR}" \
            "${SEED}" 4096
    fi
    local run checkpoint
    read -r run checkpoint < <(
        read_training_result "${train_out}/result.json"
    )
    if [[ ! -f "${eval_out}/DONE" || ! -f "${eval_out}/result.json" ]]; then
        bash "${JUMP_EVAL}" \
            "${GPU}" "${run}" "${checkpoint}" "${eval_out}" \
            "${SEED}" "${THIGH}" "${SHANK}" "${MOTOR}"
    fi
}

run_locomotion_task clearance "${PHYS_SUITE_CLEARANCE_UPDATES}"
run_locomotion_task slope "${PHYS_SUITE_SLOPE_UPDATES}"
run_locomotion_task payload "${PHYS_SUITE_PAYLOAD_UPDATES}"
run_locomotion_task speed "${PHYS_SUITE_SPEED_UPDATES}"
run_jump_task
