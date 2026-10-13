#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?usage: $0 GPU WORKER_INDEX WORKER_COUNT CANDIDATE_CSV OUT_ROOT [SEEDS]}"
WORKER_INDEX="${2:?missing zero-based worker index}"
WORKER_COUNT="${3:?missing worker count}"
CANDIDATE_CSV="${4:?missing candidate CSV}"
OUT_ROOT="${5:?missing output root}"
SEEDS_CSV="${6:-20260727,20260728}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
TRAIN_QUEUE="${ROOT}/legged_gym/scripts/run_physical_replication_queue_remote.sh"
EVAL_QUEUE="${ROOT}/legged_gym/scripts/run_physical_eval_refinement_remote.sh"
PIPELINE_OUT="${OUT_ROOT}/_pipeline_gpu${GPU}_worker${WORKER_INDEX}"
REFINED_OUT="${OUT_ROOT}/_refined_eval"

mkdir -p "${PIPELINE_OUT}"
rm -f "${PIPELINE_OUT}/EXIT_CODE" "${PIPELINE_OUT}/DONE"

export PHYS_SUITE_CLEARANCE_UPDATES="${PHYS_SUITE_CLEARANCE_UPDATES:-100}"
export PHYS_SUITE_SLOPE_UPDATES="${PHYS_SUITE_SLOPE_UPDATES:-200}"
export PHYS_SUITE_PAYLOAD_UPDATES="${PHYS_SUITE_PAYLOAD_UPDATES:-100}"
export PHYS_SUITE_SPEED_UPDATES="${PHYS_SUITE_SPEED_UPDATES:-400}"
export PHYS_SUITE_JUMP_UPDATES="${PHYS_SUITE_JUMP_UPDATES:-400}"
export PHYS_SUITE_SLOPE_LEVELS="${PHYS_SUITE_SLOPE_LEVELS:-0,0,0,1.5,3,6,9,12}"

finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "${status}" >"${PIPELINE_OUT}/EXIT_CODE"
    if [[ "${status}" -eq 0 ]]; then
        date --iso-8601=seconds >"${PIPELINE_OUT}/DONE"
    fi
    exit "${status}"
}
trap finish EXIT

date --iso-8601=seconds >"${PIPELINE_OUT}/STARTED"
bash "${TRAIN_QUEUE}" \
    "${GPU}" "${WORKER_INDEX}" "${WORKER_COUNT}" \
    "${CANDIDATE_CSV}" "${OUT_ROOT}" "${SEEDS_CSV}"
date --iso-8601=seconds >"${PIPELINE_OUT}/TRAIN_QUEUE_DONE"

while true; do
    failed=0
    complete=0
    for worker in $(seq 0 $((WORKER_COUNT - 1))); do
        marker=("${OUT_ROOT}"/_queue_gpu*_worker"${worker}"/EXIT_CODE)
        if (( ${#marker[@]} == 1 )) && [[ -f "${marker[0]}" ]]; then
            if [[ "$(<"${marker[0]}")" != "0" ]]; then
                failed=1
                break
            fi
            [[ -f "${marker[0]%/EXIT_CODE}/DONE" ]] && complete=$((complete + 1))
        fi
    done
    if (( failed )); then
        echo "another training worker failed" >&2
        exit 4
    fi
    (( complete == WORKER_COUNT )) && break
    sleep 30
done

date --iso-8601=seconds >"${PIPELINE_OUT}/ALL_TRAIN_QUEUES_DONE"
PHYS_REFINEMENT_SOURCE_ROOT="${OUT_ROOT}" \
    bash "${EVAL_QUEUE}" \
        "${GPU}" "${WORKER_INDEX}" "${WORKER_COUNT}" \
        "${CANDIDATE_CSV}" "${REFINED_OUT}" "${SEEDS_CSV}"
date --iso-8601=seconds >"${PIPELINE_OUT}/EVAL_QUEUE_DONE"
