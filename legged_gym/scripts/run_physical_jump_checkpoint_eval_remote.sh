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
STEPS="${JUMPFWD_STEPS_OVERRIDE:-2400}"
HEIGHTS="${JUMPFWD_GRID_HEIGHTS_OVERRIDE:-0.10,0.11,0.12,0.13,0.14,0.15,0.16}"
DISTANCES="${JUMPFWD_GRID_DISTANCES_OVERRIDE:-0.05,0.065,0.08,0.095,0.11,0.125,0.14}"
REPLICAS="${JUMPFWD_GRID_REPLICAS_OVERRIDE:-16}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export ROBOT_TYPE=WF_TRON1A

mkdir -p "${OUT}"
cd "${ROOT}" || exit 2
date --iso-8601=seconds >"${OUT}/STARTED"

env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    JUMPFWD_RUN="${RUN}" \
    JUMPFWD_CKPT="${CKPT}" \
    JUMPFWD_STEPS="${STEPS}" \
    JUMPFWD_GRID_HEIGHTS="${HEIGHTS}" \
    JUMPFWD_GRID_DISTANCES="${DISTANCES}" \
    JUMPFWD_GRID_REPLICAS="${REPLICAS}" \
    JUMPFWD_SEED="${SEED}" \
    JUMPFWD_NOMINAL_EVAL=1 \
    JUMPFWD_FLAT_EVAL=1 \
    JUMPFWD_EVAL_FRICTION=0.8 \
    JUMPFWD_EVAL_THIGH="${THIGH}" \
    JUMPFWD_EVAL_SHANK="${SHANK}" \
    JUMPFWD_EVAL_MOTOR="${MOTOR}" \
    JUMPFWD_EVAL_PREFIX="physjump_${CKPT}_$$_${GPU}" \
    JUMPFWD_CSV="${OUT}/jumps.csv" \
    JUMPFWD_SUMMARY="${OUT}/result.json" \
    "${PY}" legged_gym/scripts/measure_jumpfwd.py --headless \
    >"${OUT}/console.log" 2>&1
status=$?

printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUT}/DONE"
fi
exit "${status}"
