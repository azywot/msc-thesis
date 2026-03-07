#!/bin/bash
# Run all YAML experiment configs in a given folder.
#
# Usage:
#   ./run_all_in_folder.sh <config_folder> [--local]
#
# Examples:
#   ./run_all_in_folder.sh experiments/configs/1_milestone/gaia
#   ./run_all_in_folder.sh experiments/configs/1_milestone/gaia --local
#
# By default: submits each config to SLURM via launch_experiment.sh
# With --local: runs each config sequentially with python run_experiment.py

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_folder> [--local]"
    echo ""
    echo "  config_folder  Path to folder containing YAML configs (e.g. experiments/configs/1_milestone/gaia)"
    echo "  --local        Run sequentially with python instead of submitting to SLURM"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/configs/1_milestone/gaia"
    echo "  $0 experiments/configs/1_milestone/hle --local"
    exit 1
fi

CONFIG_FOLDER="$1"
RUN_LOCAL=false

if [ "${2:-}" = "--local" ]; then
    RUN_LOCAL=true
fi

# Resolve paths relative to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$(cd "$PROJECT_ROOT" && cd "$CONFIG_FOLDER" 2>/dev/null || echo "$CONFIG_FOLDER")"

if [ ! -d "$CONFIG_DIR" ]; then
    # Try relative to current dir
    CONFIG_DIR="$(cd "$(dirname "$CONFIG_FOLDER")" && pwd)/$(basename "$CONFIG_FOLDER")"
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config folder not found: $CONFIG_FOLDER"
    exit 1
fi

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT}"
cd "$PROJECT_DIR" || exit 1

shopt -s nullglob
CONFIGS=("$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml)
shopt -u nullglob

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: No YAML configs found in $CONFIG_DIR"
    exit 1
fi

echo "========================================"
echo "Running ${#CONFIGS[@]} experiments from: $CONFIG_DIR"
echo "Mode: $([ "$RUN_LOCAL" = true ] && echo "local (python)" || echo "SLURM (sbatch)")"
echo "========================================"

for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || continue
    echo ""
    echo ">>> $(basename "$cfg")"
    if [ "$RUN_LOCAL" = true ]; then
        python scripts/run_experiment.py --config "$cfg" || {
            echo "Warning: $cfg failed (exit $?)"
        }
    else
        ./jobs/scripts/launch_experiment.sh "$cfg" || {
            echo "Warning: $cfg submission failed (exit $?)"
        }
        # Stagger submissions to avoid concurrent model downloads / cache contention
        sleep 30
    fi
done

echo ""
echo "========================================"
echo "Done. $([ "$RUN_LOCAL" = true ] && echo "Check output directories for results." || echo "Monitor jobs with: squeue -u \$USER")"
echo "========================================"
