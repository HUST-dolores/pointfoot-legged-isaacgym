#!/usr/bin/env bash
set -u

REPO="/home/ps/mydrive/xu/pointfoot-legged-gym"
PYTHON="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
ROOT="/home/ps/mydrive/xu/codesign_runs/nested_four_objective_sixd_2026-07-30/generalists"

mkdir -p "${ROOT}"
cd "${REPO}" || exit 2
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"

"${PYTHON}" \
    legged_gym/scripts/run_four_task_staged_generalists.py \
    --root "${ROOT}" \
    --gpu-ids 0,1 \
    --tasks clearance,payload,speed,jump \
    --seed 20260804 \
    --stage-updates 7000 \
    --flat-envs 4096 \
    --jump-envs 4096 \
    --morphology-buckets 64 \
    --save-interval 250 \
    >"${ROOT}/driver.log" 2>&1
status=$?

printf '%s\n' "${status}" >"${ROOT}/DRIVER_EXIT_CODE"
if [[ "${status}" -eq 0 ]]; then
    date --iso-8601=seconds >"${ROOT}/DRIVER_DONE"
fi
exit "${status}"
