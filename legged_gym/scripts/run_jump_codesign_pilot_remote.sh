#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?gpu index required}"
LABEL="${2:?candidate label required}"
THIGH="${3:?thigh scale required}"
SHANK="${4:?shank scale required}"
MOTOR="${5:?motor scale required}"
SEED="${6:-20260723}"
STEPS="${7:-2400}"
REPLICAS="${8:-16}"

REPO=/home/ps/mydrive/xu/pointfoot-legged-gym
PYENV=/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin
OUT_ROOT=/home/ps/mydrive/xu/jump_codesign_pilot_2026-07-23
OUT_DIR="${OUT_ROOT}/${LABEL}"

mkdir -p "${OUT_DIR}"
export PATH="${PYENV}:${PATH}"
export ROBOT_TYPE=WF_TRON1A
export CUDA_VISIBLE_DEVICES="${GPU}"
export JUMP_LOAD_ESTIMATION=0
export JUMP_LOAD_MODE=attached
export JUMPFWD_COMMAND_STAGE=narrow
export JUMPFWD_RESIDUAL_REFERENCE_STAGE=narrow
export JUMPFWD_KNEE_HEIGHT_RESIDUAL=0.30
export JUMPFWD_KNEE_HEIGHT_BIAS=0.10
export JUMPFWD_HIP_DISTANCE_RESIDUAL=0.45
export JUMPFWD_HIP_DISTANCE_BIAS=0.60
export JUMPFWD_HIP_DISTANCE_VEL_RESIDUAL=0.0
export JUMPFWD_HIP_DISTANCE_VEL_BIAS=0.0
export JUMPFWD_HIP_LANDING_RESIDUAL=-0.75
export JUMPFWD_HIP_LANDING_BIAS=-0.75
export JUMPFWD_RESIDUAL_PROFILE=step
export JUMPFWD_RUN=Jul11_21-43-33_jumpfwd23_residual_s79
export JUMPFWD_CKPT=4000
export JUMPFWD_GRID_HEIGHTS=0.10,0.13,0.16
export JUMPFWD_GRID_DISTANCES=0.05,0.095,0.14
export JUMPFWD_GRID_REPLICAS="${REPLICAS}"
export JUMPFWD_STEPS="${STEPS}"
export JUMPFWD_SEED="${SEED}"
export JUMPFWD_NOMINAL_EVAL=1
export JUMPFWD_EVAL_THIGH="${THIGH}"
export JUMPFWD_EVAL_SHANK="${SHANK}"
export JUMPFWD_EVAL_MOTOR="${MOTOR}"
export JUMPFWD_EVAL_PREFIX="jumpcd_${LABEL}_${SEED}"
export JUMPFWD_CSV="${OUT_DIR}/jumps.csv"
export JUMPFWD_SUMMARY="${OUT_DIR}/summary.json"

cd "${REPO}"
{
    echo "[${LABEL}] START $(date '+%F_%T')"
    echo "gpu=${GPU} seed=${SEED} steps=${STEPS} replicas=${REPLICAS}"
    echo "design thigh=${THIGH} shank=${SHANK} motor=${MOTOR}"
} | tee "${OUT_DIR}/wrapper.log"

python legged_gym/scripts/measure_jumpfwd.py \
    --task=wheelfoot_jump_forward --headless --seed "${SEED}" \
    2>&1 | tee "${OUT_DIR}/measure.log"

echo "[${LABEL}] DONE $(date '+%F_%T')" | tee -a "${OUT_DIR}/wrapper.log"
