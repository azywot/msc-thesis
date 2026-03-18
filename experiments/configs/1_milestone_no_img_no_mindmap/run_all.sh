#!/bin/bash
# Run all 1_milestone_no_img_no_mindmap experiments (GAIA, HLE, GPQA, AIME, MuSiQue).
#
# Usage (from project root):
#   ./experiments/configs/1_milestone_no_img_no_mindmap/run_all.sh           # Submit all to SLURM
#   ./experiments/configs/1_milestone_no_img_no_mindmap/run_all.sh --local   # Run all locally

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

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
echo "All 1_milestone_no_img_no_mindmap experiment batches submitted."
