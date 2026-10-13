#!/usr/bin/env bash
set -u

REPO="/home/ps/mydrive/xu/pointfoot-legged-gym"
PYTHON="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
ROOT="/home/ps/mydrive/xu/codesign_runs/nested_four_objective_sixd_2026-07-30"
OUT="${ROOT}/precompute_ready_tasks"

mkdir -p "${OUT}"
rm -f "${OUT}/DRIVER_EXIT_CODE" "${OUT}/DRIVER_DONE"
cd "${REPO}" || exit 2
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export NESTED_BO_REPO="${REPO}"
export NESTED_BO_PYTHON="${PYTHON}"
export NESTED_BO_ROOT="${ROOT}"
export NESTED_BO_OUTPUT="${ROOT}/nested_bo"
export SIXD_GPUS="0,0"

"${PYTHON}" \
    legged_gym/scripts/precompute_nested_ready_tasks.py \
    --tasks clearance,payload,speed \
    --gpu-workers 0,0 \
    --poll-seconds 300 \
    >"${OUT}/driver.log" 2>&1
status=$?

printf '%s\n' "${status}" >"${OUT}/DRIVER_EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUT}/DRIVER_DONE"
fi
exit "${status}"
