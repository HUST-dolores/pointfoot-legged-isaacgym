#!/usr/bin/env bash
set -u

GPU="${1:?usage: $0 GPU UPDATES TAG OUT [THIGH SHANK MOTOR SEED NUM_ENVS]}"
UPDATES="${2:?missing updates}"
TAG="${3:?missing tag}"
OUT="${4:?missing output directory}"
THIGH="${5:-1.0}"
SHANK="${6:-1.0}"
MOTOR="${7:-1.0}"
SEED="${8:-20260724}"
NUM_ENVS="${9:-4096}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export ROBOT_TYPE=WF_TRON1A

mkdir -p "${OUT}"
cd "${ROOT}" || exit 2

date --iso-8601=seconds >"${OUT}/STARTED"
env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PHYS_TRAIN_TASK="${PHYS_TRAIN_TASK:-all}" \
    PHYS_TRAIN_NUM_ENVS="${NUM_ENVS}" \
    PHYS_TRAIN_UPDATES="${UPDATES}" \
    PHYS_TRAIN_SEED="${SEED}" \
    PHYS_TRAIN_TAG="${TAG}" \
    PHYS_TRAIN_THIGH="${THIGH}" \
    PHYS_TRAIN_SHANK="${SHANK}" \
    PHYS_TRAIN_MOTOR="${MOTOR}" \
    PHYS_TRAIN_LEARNING_RATE="${PHYS_TRAIN_LEARNING_RATE:-0.0001}" \
    PHYS_TRAIN_SCHEDULE="${PHYS_TRAIN_SCHEDULE:-fixed}" \
    PHYS_TRAIN_ENTROPY_COEF="${PHYS_TRAIN_ENTROPY_COEF:-0.001}" \
    PHYS_TRAIN_EST_LEARNING_RATE="${PHYS_TRAIN_EST_LEARNING_RATE:-0.0}" \
    PHYS_TRAIN_TOTAL_VEL_REWARD="${PHYS_TRAIN_TOTAL_VEL_REWARD:-4.0}" \
    PHYS_TRAIN_TOTAL_VEL_SIGMA="${PHYS_TRAIN_TOTAL_VEL_SIGMA:-0.05}" \
    PHYS_TRAIN_TOTAL_VEL_ERROR_REWARD="${PHYS_TRAIN_TOTAL_VEL_ERROR_REWARD:--1.0}" \
    PHYS_TRAIN_RESULT_JSON="${OUT}/result.json" \
    "${PY}" legged_gym/scripts/train_physical_capability.py --headless \
    >"${OUT}/console.log" 2>&1
status=$?
printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUT}/DONE"
fi
exit "${status}"
