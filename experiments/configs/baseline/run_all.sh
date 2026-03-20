#!/bin/bash
# Run all baseline experiment batches (GAIA, HLE, GPQA, AIME, MuSiQue).
#
# Usage (from project root):
#   ./experiments/configs/baseline/run_all.sh           # Submit all baseline configs to SLURM
#   ./experiments/configs/baseline/run_all.sh --local   # Run all baseline configs locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

for folder in gaia hle gpqa aime musique; do
    run_script="$SCRIPT_DIR/$folder/run_all.sh"
    if [ -f "$run_script" ]; then
        echo ""
        echo "========== $folder =========="
        "$run_script" "$@"
    fi
done

echo ""
echo "All baseline experiment batches submitted."

