#!/bin/bash
# Run all 1_milestone experiments (GAIA, HLE, GPQA, AIME).
#
# Usage (from project root):
#   ./experiments/configs/1_milestone/run_all.sh           # Submit all to SLURM
#   ./experiments/configs/1_milestone/run_all.sh --local   # Run all locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

for folder in gaia hle gpqa aime; do
    run_script="$SCRIPT_DIR/$folder/run_all.sh"
    if [ -f "$run_script" ]; then
        echo ""
        echo "========== $folder =========="
        "$run_script" "$@"
    fi
done

echo ""
echo "All 1_milestone experiment batches submitted."
