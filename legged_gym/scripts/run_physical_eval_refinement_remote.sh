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
SOURCE_ROOT_OVERRIDE="${PHYS_REFINEMENT_SOURCE_ROOT:-}"
REPLICAS="${PHYS_REFINEMENT_REPLICAS:-128}"
SLOPE_LEVELS="${PHYS_REFINEMENT_SLOPE_LEVELS:-0,1,2,3,4,5,6,7,8,9,10,11,12}"
PAYLOAD_LEVELS="${PHYS_REFINEMENT_PAYLOAD_LEVELS:-0,5,10,15,20,25,30,35,40,45}"
SPEED_LEVELS="${PHYS_REFINEMENT_SPEED_LEVELS:-0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6}"
SLOPE_VERTICAL_SCALE="${PHYS_REFINEMENT_SLOPE_VERTICAL_SCALE:-0.001}"
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
rm -f "${QUEUE_OUT}/EXIT_CODE" "${QUEUE_OUT}/DONE"

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
    "replicas=${REPLICAS}" \
    "slope_levels=${SLOPE_LEVELS}" \
    "payload_levels=${PAYLOAD_LEVELS}" \
    "speed_levels=${SPEED_LEVELS}" \
    "slope_vertical_scale=${SLOPE_VERTICAL_SCALE}" \
    >"${QUEUE_OUT}/worker.env"

job_index=0
while IFS=, read -r label thigh shank motor role; do
    [[ -n "${label}" ]] || continue
    for seed in "${SEEDS[@]}"; do
        if [[ -n "${SOURCE_ROOT_OVERRIDE}" ]]; then
            source_root="${SOURCE_ROOT_OVERRIDE}"
            candidate_rel="seed_${seed}/${label}"
        else
            source_root="${REPLICATION_ROOT}"
            candidate_rel="seed_${seed}/${label}"
            if [[ "${seed}" == "20260724" ]]; then
                source_root="${FIRST_ROOT}"
                candidate_rel="${label}"
            fi
        fi
        for task in slope payload speed; do
            if (( job_index % WORKER_COUNT != WORKER_INDEX )); then
                job_index=$((job_index + 1))
                continue
            fi
            job_index=$((job_index + 1))

            manifest="${source_root}/${candidate_rel}/${task}/train/result.json"
            if [[ ! -f "${manifest}" ]]; then
                echo "missing ${task} training manifest: ${manifest}" >&2
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

            out="${OUT_ROOT}/seed_${seed}/${label}/${task}"
            if [[ -f "${out}/DONE" ]]; then
                continue
            fi
            printf '%s %s seed=%s task=%s\n' \
                "$(date --iso-8601=seconds)" "${label}" "${seed}" "${task}" \
                >>"${QUEUE_OUT}/job_start.log"
            PHYS_EVAL_TASKS_OVERRIDE="${task}" \
            PHYS_EVAL_SLOPE_LEVELS_OVERRIDE="${SLOPE_LEVELS}" \
            PHYS_EVAL_PAYLOAD_LEVELS_OVERRIDE="${PAYLOAD_LEVELS}" \
            PHYS_EVAL_SPEED_LEVELS_OVERRIDE="${SPEED_LEVELS}" \
            PHYS_EVAL_REPLICAS_OVERRIDE="${REPLICAS}" \
            PHYS_EVAL_SLOPE_VERTICAL_SCALE="${SLOPE_VERTICAL_SCALE}" \
                bash "${EVAL}" \
                    "${GPU}" "${run}" "${checkpoint}" "${out}" \
                    "${seed}" "${thigh}" "${shank}" "${motor}"
            printf '%s %s seed=%s task=%s\n' \
                "$(date --iso-8601=seconds)" "${label}" "${seed}" "${task}" \
                >>"${QUEUE_OUT}/job_done.log"
        done
    done
done < <(awk -F, 'NR > 1 { print }' "${CANDIDATE_CSV}")
