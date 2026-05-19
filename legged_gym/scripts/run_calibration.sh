#!/bin/bash
# 跑完整的负载估计标定流程：4 个质量各采一次数据，最后一次性拟合
#
# 用法:
#   ./legged_gym/scripts/run_calibration.sh [run_name] [checkpoint]
# 例:
#   ./legged_gym/scripts/run_calibration.sh test32 10000

set -e

RUN="${1:-test32}"
CKPT="${2:-10000}"
MASSES=(2.0 3.333 4.667 6.0)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTDIR="${REPO_ROOT}/logs/calibration/$(date +%b%d_%H-%M-%S)"
mkdir -p "$OUTDIR"

echo "[calib] run=$RUN  checkpoint=$CKPT"
echo "[calib] outdir=$OUTDIR"
echo "[calib] masses=${MASSES[*]}"
echo

for M in "${MASSES[@]}"; do
    echo "===== mass=$M ====="
    python "${REPO_ROOT}/legged_gym/scripts/collect_calibration_data.py" \
        --task=wheelfoot_flat --load_run="$RUN" --checkpoint="$CKPT" \
        --headless --mass="$M" --output_dir="$OUTDIR"
done

echo
echo "===== fitting ====="
python "${REPO_ROOT}/legged_gym/scripts/fit_load_estimation.py" \
    --inputs "$OUTDIR"/calib_m*.npz | tee "$OUTDIR/fit_report.txt"

echo
echo "[calib] done. report: $OUTDIR/fit_report.txt"
