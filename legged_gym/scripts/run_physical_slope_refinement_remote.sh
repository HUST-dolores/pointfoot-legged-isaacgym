#!/usr/bin/env bash
set -Eeuo pipefail

GPU="${1:?usage: $0 GPU WORKER_INDEX WORKER_COUNT CANDIDATE_CSV OUT_ROOT [SEEDS]}"
WORKER_INDEX="${2:?missing zero-based worker index}"
WORKER_COUNT="${3:?missing worker count}"
CANDIDATE_CSV="${4:?missing candidate CSV}"
OUT_ROOT="${5:?missing output root}"
SEEDS_CSV="${6:-20260724,20260725,20260726}"

ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
EVAL="${ROOT}/legged_gym/scripts/run_physical_checkpoint_eval_remote.sh"
FIRST_ROOT="/home/ps/mydrive/xu/physical_codesign_v1_2026-07-24"
REPLICATION_ROOT="/home/ps/mydrive/xu/physical_codesign_shortlist_v1_2026-07-24"
LEVELS="${PHYS_SLOPE_REFINEMENT_LEVELS:-0,1,2,3,4,5,6,7,8,9,10,11,12}"
REPLICAS="${PHYS_SLOPE_REFINEMENT_REPLICAS:-128}"
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
mkdir -p "${QUEUE_OUT}"

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
    "levels=${LEVELS}" \
    "replicas=${REPLICAS}" \
    >"${QUEUE_OUT}/worker.env"

candidate_index=0
while IFS=, read -r label thigh shank motor role; do
    [[ -n "${label}" ]] || continue
    for seed_index in "${!SEEDS[@]}"; do
        seed="${SEEDS[seed_index]}"
        if (( (candidate_index + seed_index) % WORKER_COUNT != WORKER_INDEX )); then
            continue
        fi

        source_root="${REPLICATION_ROOT}"
        candidate_rel="seed_${seed}/${label}"
        if [[ "${seed}" == "20260724" ]]; then
            source_root="${FIRST_ROOT}"
            candidate_rel="${label}"
        fi
        manifest="${source_root}/${candidate_rel}/slope/train/result.json"
        if [[ ! -f "${manifest}" ]]; then
            echo "missing slope training manifest: ${manifest}" >&2
            exit 3
        fi
        read -r run checkpoint < <(
            "${PY}" - "${manifest}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
print(result["run_name"], result["final_iteration"])
PY
        )

        out="${OUT_ROOT}/seed_${seed}/${label}"
        if [[ -f "${out}/DONE" ]]; then
            continue
        fi
        printf '%s %s seed=%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${seed}" \
            >>"${QUEUE_OUT}/candidate_start.log"
        PHYS_EVAL_TASKS_OVERRIDE=slope \
        PHYS_EVAL_SLOPE_LEVELS_OVERRIDE="${LEVELS}" \
        PHYS_EVAL_REPLICAS_OVERRIDE="${REPLICAS}" \
            bash "${EVAL}" \
                "${GPU}" "${run}" "${checkpoint}" "${out}" \
                "${seed}" "${thigh}" "${shank}" "${motor}"
        printf '%s %s seed=%s\n' \
            "$(date --iso-8601=seconds)" "${label}" "${seed}" \
            >>"${QUEUE_OUT}/candidate_done.log"
    done
    candidate_index=$((candidate_index + 1))
done < <(awk -F, 'NR > 1 { print }' "${CANDIDATE_CSV}")
