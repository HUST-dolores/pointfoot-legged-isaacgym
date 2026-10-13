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
    PHYS_JUMP_NUM_ENVS="${NUM_ENVS}" \
    PHYS_JUMP_UPDATES="${UPDATES}" \
    PHYS_JUMP_SEED="${SEED}" \
    PHYS_JUMP_TAG="${TAG}" \
    PHYS_JUMP_THIGH="${THIGH}" \
    PHYS_JUMP_SHANK="${SHANK}" \
    PHYS_JUMP_MOTOR="${MOTOR}" \
    PHYS_JUMP_RESUME_NOISE_STD="${PHYS_JUMP_RESUME_NOISE_STD:-0.5}" \
    PHYS_JUMP_LEARNING_RATE="${PHYS_JUMP_LEARNING_RATE:-0.001}" \
    PHYS_JUMP_SCHEDULE="${PHYS_JUMP_SCHEDULE:-adaptive}" \
    PHYS_JUMP_ENTROPY_COEF="${PHYS_JUMP_ENTROPY_COEF:-0.005}" \
    PHYS_JUMP_EST_LEARNING_RATE="${PHYS_JUMP_EST_LEARNING_RATE:-0.001}" \
    PHYS_JUMP_RESULT_JSON="${OUT}/result.json" \
    "${PY}" legged_gym/scripts/train_physical_jump.py --headless \
    >"${OUT}/console.log" 2>&1
status=$?
printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUT}/DONE"
fi
exit "${status}"
