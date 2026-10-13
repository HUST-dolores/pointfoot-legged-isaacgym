#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?usage: $0 GPU WORKER_INDEX WORKER_COUNT CANDIDATE_CSV OUT_ROOT [SEEDS]}"
WORKER_INDEX="${2:?missing zero-based worker index}"
WORKER_COUNT="${3:?missing worker count}"
CANDIDATE_CSV="${4:?missing candidate CSV}"
OUT_ROOT="${5:?missing output root}"
SEEDS_CSV="${6:-20260725,20260726}"

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

IFS=',' read -r -a SEEDS <<<"${SEEDS_CSV}"
if (( ${#SEEDS[@]} < 1 )); then
    echo "at least one seed is required" >&2
    exit 2
fi
for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ || "${seed}" -eq 1 ]]; then
        echo "invalid deterministic seed: ${seed}" >&2
        exit 2
    fi
done

mkdir -p "${QUEUE_OUT}"
if [[ -f "${QUEUE_OUT}/DONE" ]]; then
    echo "[replication-queue] GPU ${GPU} worker ${WORKER_INDEX} complete"
    exit 0
fi
rm -f "${QUEUE_OUT}/EXIT_CODE"

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
    "seeds=${SEEDS_CSV}" \
    >"${QUEUE_OUT}/worker.env"

candidate_index=0
while IFS=, read -r label thigh shank motor role; do
    [[ -n "${label}" ]] || continue
    for seed_index in "${!SEEDS[@]}"; do
        seed="${SEEDS[seed_index]}"
        if (( (candidate_index + seed_index) % WORKER_COUNT != WORKER_INDEX )); then
            continue
        fi
        seed_root="${OUT_ROOT}/seed_${seed}"
        printf '%s %s seed=%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${seed}" \
            >>"${QUEUE_OUT}/candidate_start.log"
        bash "${SUITE}" \
            "${GPU}" "${label}" "${thigh}" "${shank}" "${motor}" \
            "${seed_root}" "${seed}"
        printf '%s %s seed=%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${seed}" \
            >>"${QUEUE_OUT}/candidate_done.log"
    done
    candidate_index=$((candidate_index + 1))
done < <(awk -F, 'NR > 1 { print }' "${CANDIDATE_CSV}")
