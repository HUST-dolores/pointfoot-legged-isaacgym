#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 locomotion|jump [output_dir]}"
OUT="${2:-/home/ps/mydrive/xu/physical_metrics_nominal_2026-07-24}"
ROOT="/home/ps/mydrive/xu/pointfoot-legged-gym"
PY="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin/python"
export PATH="/home/ps/mydrive/xu/miniconda3/envs/cdesign/bin:${PATH}"
export ROBOT_TYPE=WF_TRON1A

mkdir -p "${OUT}"
cd "${ROOT}"

run_locomotion_metric() {
    local task="$1"
    local levels="$2"
    local task_out="${OUT}/${task}"
    mkdir -p "${task_out}"
    env \
        CUDA_VISIBLE_DEVICES=0 \
        PHYS_EVAL_TASK="${task}" \
        PHYS_EVAL_LEVELS="${levels}" \
        PHYS_EVAL_REPLICAS=32 \
        PHYS_EVAL_NUM_ENVS=256 \
        PHYS_EVAL_SETTLE_S=2.0 \
        PHYS_EVAL_DURATION_S=8.0 \
        PHYS_EVAL_RUN=Jul05_00-02-15_ \
        PHYS_EVAL_CKPT=27000 \
        PHYS_EVAL_SEED=20260724 \
        PHYS_EVAL_THIGH=1.0 \
        PHYS_EVAL_SHANK=1.0 \
        PHYS_EVAL_MOTOR=1.0 \
        PHYS_EVAL_PREFIX="physnom_${task}" \
        PHYS_EVAL_OUT="${task_out}" \
        "${PY}" legged_gym/scripts/physical_capability_eval.py --headless \
        >"${task_out}/console.log" 2>&1
}

run_jump_metric() {
    local task_out="${OUT}/jump"
    mkdir -p "${task_out}"
    env \
        CUDA_VISIBLE_DEVICES=1 \
        JUMPFWD_RUN=Jul11_21-43-33_jumpfwd23_residual_s79 \
        JUMPFWD_CKPT=4000 \
        JUMPFWD_STEPS=2400 \
        JUMPFWD_GRID_HEIGHTS=0.10,0.11,0.12,0.13,0.14,0.15,0.16 \
        JUMPFWD_GRID_DISTANCES=0.05,0.065,0.08,0.095,0.11,0.125,0.14 \
        JUMPFWD_GRID_REPLICAS=16 \
        JUMPFWD_SEED=20260724 \
        JUMPFWD_NOMINAL_EVAL=1 \
        JUMPFWD_FLAT_EVAL=1 \
        JUMPFWD_EVAL_FRICTION=0.8 \
        JUMPFWD_EVAL_THIGH=1.0 \
        JUMPFWD_EVAL_SHANK=1.0 \
        JUMPFWD_EVAL_MOTOR=1.0 \
        JUMPFWD_EVAL_PREFIX=physnom_jump \
        JUMPFWD_CSV="${task_out}/jumps.csv" \
        JUMPFWD_SUMMARY="${task_out}/result.json" \
        "${PY}" legged_gym/scripts/measure_jumpfwd.py --headless \
        >"${task_out}/console.log" 2>&1
}

case "${MODE}" in
    locomotion)
        run_locomotion_metric clearance 0
        run_locomotion_metric slope 0,3,6,9,12,15,18,21
        run_locomotion_metric payload 0,5,10,15,20,25,30,35
        run_locomotion_metric speed 0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6
        ;;
    jump)
        run_jump_metric
        ;;
    *)
        echo "unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac

date --iso-8601=seconds >"${OUT}/${MODE}.DONE"
