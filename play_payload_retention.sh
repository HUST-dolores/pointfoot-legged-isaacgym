#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export ROBOT_TYPE=WF_TRON1A
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
PYTHON="${PYTHON:-/home/xu/anaconda3/envs/pointfoot_legged_gym/bin/python}"
cd "$ROOT"
exec "$PYTHON" -m legged_gym.scripts.sliding_lab.retention_eval \
  --checkpoint "$ROOT/logs/payload_retention_20261018/selected_policy.pt" \
  --viewer --duration 30 --seed 12345 --direction forward --dv .65 --clean-exit "$@"
