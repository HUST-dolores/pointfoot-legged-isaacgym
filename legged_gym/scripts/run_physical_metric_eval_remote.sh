#!/usr/bin/env bash
set -u

GPU="${1:?usage: $0 GPU TASK RUN CKPT OUT [SEED THIGH SHANK MOTOR]}"
TASK="${2:?missing task}"
RUN="${3:?missing run}"
CKPT="${4:?missing checkpoint}"
OUT="${5:?missing output directory}"
SEED="${6:-20260724}"
THIGH="${7:-1.0}"
SHANK="${8:-1.0}"
MOTOR="${9:-1.0}"

case "${TASK}" in
    clearance)
        LEVELS=0
        ;;
    slope)
        LEVELS=0,3,6,9,12,15,18,21
        ;;
    payload)
        LEVELS=0,5,10,15,20,25,30,35
        ;;
    speed)
        LEVELS=0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6
        ;;
    *)
        echo "TASK must be clearance, slope, payload, or speed" >&2
        exit 2
        ;;
esac
LEVELS="${PHYS_EVAL_LEVELS_OVERRIDE:-${LEVELS}}"
REPLICAS="${PHYS_EVAL_REPLICAS_OVERRIDE:-32}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export ROBOT_TYPE=WF_TRON1A

mkdir -p "${OUT}"
cd "${ROOT}" || exit 2
date --iso-8601=seconds >"${OUT}/STARTED"

env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PHYS_EVAL_TASK="${TASK}" \
    PHYS_EVAL_LEVELS="${LEVELS}" \
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
    PHYS_EVAL_PREFIX="physeval_${CKPT}_${TASK}_$$_${GPU}" \
    PHYS_EVAL_OUT="${OUT}" \
    "${PY}" legged_gym/scripts/physical_capability_eval.py --headless \
    >"${OUT}/console.log" 2>&1
status=$?

printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUT}/DONE"
fi
exit "${status}"
