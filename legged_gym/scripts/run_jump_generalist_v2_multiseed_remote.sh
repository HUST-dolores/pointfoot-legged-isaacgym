#!/usr/bin/env bash
set -u

GPU="${1:?usage: $0 GPU RUN CKPT OUT_ROOT SEED}"
RUN="${2:?missing run}"
CKPT="${3:?missing checkpoint}"
OUT_ROOT="${4:?missing output root}"
SEED="${5:?missing seed}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
EVAL="${ROOT}/legged_gym/scripts/run_physical_jump_checkpoint_eval_remote.sh"
CROSS="${ROOT}/legged_gym/scripts/run_jump_generalist_v2_cross_eval_remote.sh"

bash "${EVAL}" \
    "${GPU}" \
    "${RUN}" \
    "${CKPT}" \
    "${OUT_ROOT}/nominal" \
    "${SEED}" \
    1.0 \
    1.0 \
    1.0 || exit $?

bash "${CROSS}" \
    "${GPU}" \
    "${RUN}" \
    "${CKPT}" \
    "${OUT_ROOT}" \
    "${SEED}"
