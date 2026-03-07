#!/bin/bash
# Run all 1_milestone GAIA experiments.
#
# Usage (from project root):
#   ./experiments/configs/1_milestone/gaia/run_all.sh           # Submit to SLURM
#   ./experiments/configs/1_milestone/gaia/run_all.sh --local   # Run locally

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG_FOLDER="experiments/configs/1_milestone/gaia"

cd "$PROJECT_ROOT" || exit 1
exec "$PROJECT_ROOT/experiments/scripts/run_all_in_folder.sh" "$CONFIG_FOLDER" "$@"
