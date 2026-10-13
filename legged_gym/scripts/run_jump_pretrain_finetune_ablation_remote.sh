#!/usr/bin/env bash
set -u

REPO="/home/ps/mydrive/xu/pointfoot-legged-gym"
PYTHON="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
ROOT="/home/ps/mydrive/xu/codesign_runs/jump_pretrain_finetune_ablation_2026-07-30"

mkdir -p "${ROOT}"
cd "${REPO}" || exit 2
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"

"${PYTHON}" \
    legged_gym/scripts/run_jump_pretrain_finetune_ablation.py \
    --root "${ROOT}" \
    --gpu-ids 0,1 \
    --seeds 20260801,20260802,20260803 \
    --updates 400 \
    --train-envs 2048 \
    --checkpoints 0,25,50,100,200,400 \
    >"${ROOT}/driver.log" 2>&1
status=$?

printf '%s\n' "${status}" >"${ROOT}/DRIVER_EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${ROOT}/DRIVER_DONE"
fi
exit "${status}"
