#!/bin/bash
# Run all YAML experiment configs in a folder, recursively.
#
# Usage (from project root):
#   ./experiments/scripts/run_all_in_folder.sh <folder> [--local]
#
# Examples:
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline/gaia
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/1_milestone_no_img_no_mindmap_AgentFlow --local
#
# By default: submits each config to SLURM via launch_experiment.sh
# With --local: runs each config sequentially with python run_experiment.py

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder> [--local]"
    echo ""
    echo "  folder   Path to a folder containing YAML configs (searched recursively)"
    echo "  --local  Run sequentially with python instead of submitting to SLURM"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/configs/baseline"
    echo "  $0 experiments/configs/baseline/gaia"
    echo "  $0 experiments/configs/1_milestone_no_img_no_mindmap_AgentFlow --local"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Resolve config folder (accept relative to project root or absolute)
CONFIG_FOLDER="$1"
if [[ "$CONFIG_FOLDER" != /* ]]; then
    CONFIG_FOLDER="$PROJECT_ROOT/$CONFIG_FOLDER"
fi

if [ ! -d "$CONFIG_FOLDER" ]; then
    echo "Error: folder not found: $1"
    exit 1
fi

RUN_LOCAL=false
if [ "${2:-}" = "--local" ]; then
    RUN_LOCAL=true
fi

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

cd "${PROJECT_DIR:-$PROJECT_ROOT}" || exit 1

# Collect all YAML files recursively, sorted
mapfile -t CONFIGS < <(find "$CONFIG_FOLDER" -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: no YAML configs found under $1"
    exit 1
fi

echo "========================================"
echo "Folder : $1"
echo "Configs: ${#CONFIGS[@]}"
echo "Mode   : $([ "$RUN_LOCAL" = true ] && echo "local (python)" || echo "SLURM (sbatch)")"
echo "========================================"

for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || continue
    echo ""
    echo ">>> ${cfg#$PROJECT_ROOT/}"
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
echo "Done. $([ "$RUN_LOCAL" = true ] && echo "Check output directories for results." || echo "Monitor with: squeue -u \$USER")"
echo "========================================"
