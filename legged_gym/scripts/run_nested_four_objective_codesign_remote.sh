#!/usr/bin/env bash
set -u

REPO="/home/ps/mydrive/xu/pointfoot-legged-gym"
PYTHON="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
ROOT="/home/ps/mydrive/xu/codesign_runs/nested_four_objective_sixd_2026-07-30"
OUTPUT="${ROOT}/nested_bo"

mkdir -p "${OUTPUT}"
rm -f "${OUTPUT}/DRIVER_EXIT_CODE" "${OUTPUT}/DRIVER_DONE"
cd "${REPO}" || exit 2
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export NESTED_BO_REPO="${REPO}"
export NESTED_BO_PYTHON="${PYTHON}"
export NESTED_BO_ROOT="${ROOT}"
export NESTED_BO_OUTPUT="${OUTPUT}"
export SIXD_GPUS="0,1"

"${PYTHON}" \
    legged_gym/scripts/run_nested_four_objective_codesign.py \
    --poll-seconds 300 \
    >"${OUTPUT}/driver.log" 2>&1
status=$?

printf '%s\n' "${status}" >"${OUTPUT}/DRIVER_EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${OUTPUT}/DRIVER_DONE"
fi
exit "${status}"
