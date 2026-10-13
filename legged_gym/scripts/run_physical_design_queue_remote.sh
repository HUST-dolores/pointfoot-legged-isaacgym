#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?usage: $0 GPU WORKER_INDEX WORKER_COUNT CANDIDATE_CSV OUT_ROOT [SEED]}"
WORKER_INDEX="${2:?missing zero-based worker index}"
WORKER_COUNT="${3:?missing worker count}"
CANDIDATE_CSV="${4:?missing candidate CSV}"
OUT_ROOT="${5:?missing output root}"
SEED="${6:-20260724}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
SUITE="${ROOT}/legged_gym/scripts/run_physical_design_suite_remote.sh"
QUEUE_OUT="${OUT_ROOT}/_queue_gpu${GPU}_worker${WORKER_INDEX}"

if (( WORKER_COUNT < 1 || WORKER_INDEX < 0 || WORKER_INDEX >= WORKER_COUNT )); then
    echo "require 0 <= WORKER_INDEX < WORKER_COUNT" >&2
    exit 2
fi
if [[ ! -f "${CANDIDATE_CSV}" ]]; then
    echo "candidate CSV not found: ${CANDIDATE_CSV}" >&2
    exit 2
fi

mkdir -p "${QUEUE_OUT}"
if [[ -f "${QUEUE_OUT}/DONE" ]]; then
    echo "[queue] GPU ${GPU} worker ${WORKER_INDEX} already complete"
    exit 0
fi

finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "${status}" >"${QUEUE_OUT}/EXIT_CODE"
    if [[ "${status}" -eq 0 ]]; then
        date --iso-8601=seconds >"${QUEUE_OUT}/DONE"
    fi
    exit "${status}"
}
trap finish EXIT

date --iso-8601=seconds >"${QUEUE_OUT}/STARTED"
cp "${CANDIDATE_CSV}" "${QUEUE_OUT}/candidates.csv"
printf '%s\n' \
    "gpu=${GPU}" \
    "worker_index=${WORKER_INDEX}" \
    "worker_count=${WORKER_COUNT}" \
    "seed=${SEED}" \
    >"${QUEUE_OUT}/worker.env"

while IFS=, read -r label thigh shank motor role; do
    [[ -n "${label}" ]] || continue
    printf '%s %s\n' "$(date --iso-8601=seconds)" "${label}" \
        >>"${QUEUE_OUT}/candidate_start.log"
    bash "${SUITE}" \
        "${GPU}" "${label}" "${thigh}" "${shank}" "${motor}" \
        "${OUT_ROOT}" "${SEED}"
    printf '%s %s\n' "$(date --iso-8601=seconds)" "${label}" \
        >>"${QUEUE_OUT}/candidate_done.log"
done < <(
    awk -F, -v worker="${WORKER_INDEX}" -v count="${WORKER_COUNT}" \
        'NR > 1 && ((NR - 2) % count) == worker { print }' \
        "${CANDIDATE_CSV}"
)
