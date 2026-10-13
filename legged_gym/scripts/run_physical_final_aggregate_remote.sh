#!/usr/bin/env bash
set -Eeuo pipefail

FINAL_ROOT="${1:-/home/ps/mydrive/xu/physical_codesign_finalists_v1_2026-07-25}"

ROOT="/home/ps/mydrive/xu"
REPO="${ROOT}/pointfoot-legged-gym"
PY="${ROOT}/miniconda3/envs/cdesign/bin/python"
CANDIDATES="${REPO}/legged_gym/scripts/physical_codesign_finalists_v1.csv"
AGGREGATE="${REPO}/legged_gym/scripts/aggregate_physical_codesign.py"
MULTISEED="${REPO}/legged_gym/scripts/aggregate_physical_codesign_multiseed.py"
REPORT="${REPO}/legged_gym/scripts/make_physical_codesign_report.py"
FIRST_ROOT="${ROOT}/physical_codesign_v1_2026-07-24"
SHORTLIST_ROOT="${ROOT}/physical_codesign_shortlist_v1_2026-07-24"
EARLY_REFINED_ROOT="${ROOT}/physical_codesign_eval_refinement_v1_2026-07-25"
OUT="${FINAL_ROOT}/_final_aggregate"

mkdir -p "${OUT}"
rm -f "${OUT}/EXIT_CODE" "${OUT}/DONE"

finish() {
    local status=$?
    trap - EXIT
    printf '%s\n' "${status}" >"${OUT}/EXIT_CODE"
    if [[ "${status}" -eq 0 ]]; then
        date --iso-8601=seconds >"${OUT}/DONE"
    fi
    exit "${status}"
}
trap finish EXIT

date --iso-8601=seconds >"${OUT}/STARTED"
for worker in 0 1; do
    marker=("${FINAL_ROOT}"/_pipeline_gpu*_worker"${worker}"/DONE)
    if (( ${#marker[@]} != 1 )) || [[ ! -f "${marker[0]}" ]]; then
        echo "finalist pipeline worker ${worker} is incomplete" >&2
        exit 3
    fi
done

for seed in 20260724 20260725 20260726 20260727 20260728; do
    case "${seed}" in
        20260724)
            data_root="${FIRST_ROOT}"
            eval_root="${EARLY_REFINED_ROOT}/seed_${seed}"
            ;;
        20260725|20260726)
            data_root="${SHORTLIST_ROOT}/seed_${seed}"
            eval_root="${EARLY_REFINED_ROOT}/seed_${seed}"
            ;;
        *)
            data_root="${FINAL_ROOT}/seed_${seed}"
            eval_root="${FINAL_ROOT}/_refined_eval/seed_${seed}"
            ;;
    esac
    seed_out="${OUT}/seed_${seed}"
    "${PY}" "${AGGREGATE}" \
        --root "${data_root}" \
        --candidates "${CANDIDATES}" \
        --out "${seed_out}" \
        --expected-seed "${seed}" \
        --eval-override-root "${eval_root}"
done

multiseed_out="${OUT}/multiseed_20260724_28"
"${PY}" "${MULTISEED}" \
    --input "20260724=${OUT}/seed_20260724" \
    --input "20260725=${OUT}/seed_20260725" \
    --input "20260726=${OUT}/seed_20260726" \
    --input "20260727=${OUT}/seed_20260727" \
    --input "20260728=${OUT}/seed_20260728" \
    --candidates "${CANDIDATES}" \
    --out "${multiseed_out}"

"${PY}" "${REPORT}" \
    --aggregate-dir "${multiseed_out}" \
    --out "${OUT}/paper_report" \
    --title "Five-seed physical co-design results"

printf '%s\n' \
    "seeds=20260724,20260725,20260726,20260727,20260728" \
    "candidates=${CANDIDATES}" \
    "multiseed=${multiseed_out}" \
    >"${OUT}/aggregate.env"
