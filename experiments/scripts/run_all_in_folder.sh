#!/bin/bash
# Run all YAML experiment configs in a folder, recursively.
#
# Usage (from project root):
#   ./experiments/scripts/run_all_in_folder.sh <folder> [--dataset <name>] [--local]
#
# Examples:
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline/gaia
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline --dataset math500
#   ./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline --dataset math500 --local
#
# By default: submits each config to SLURM via launch_experiment.sh
# With --local: runs each config sequentially with python run_experiment.py
# With --dataset <name>: only run configs whose parent directory matches <name>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder> [--dataset <name>] [--local]"
    echo ""
    echo "  folder           Path to a folder containing YAML configs (searched recursively)"
    echo "  --dataset <name> Only run configs under a subdirectory named <name>"
    echo "  --local          Run sequentially with python instead of submitting to SLURM"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/configs/baseline"
    echo "  $0 experiments/configs/baseline/gaia"
    echo "  $0 experiments/configs/qwen3/baseline --dataset math500"
    echo "  $0 experiments/configs/qwen3/baseline --dataset math500 --local"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Resolve config folder (accept relative to project root or absolute)
CONFIG_FOLDER="$1"
shift
if [[ "$CONFIG_FOLDER" != /* ]]; then
    CONFIG_FOLDER="$PROJECT_ROOT/$CONFIG_FOLDER"
fi

if [ ! -d "$CONFIG_FOLDER" ]; then
    echo "Error: folder not found: $1"
    exit 1
fi

RUN_LOCAL=false
DATASET_FILTER=""
while [ $# -gt 0 ]; do
    case "$1" in
        --local)
            RUN_LOCAL=true
            shift
            ;;
        --dataset)
            DATASET_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

# Filter by dataset (parent directory name) if requested
if [ -n "$DATASET_FILTER" ]; then
    filtered=()
    for cfg in "${CONFIGS[@]}"; do
        if [ "$(basename "$(dirname "$cfg")")" = "$DATASET_FILTER" ]; then
            filtered+=("$cfg")
        fi
    done
    CONFIGS=("${filtered[@]}")
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: no YAML configs found under $CONFIG_FOLDER${DATASET_FILTER:+ (dataset=$DATASET_FILTER)}"
    exit 1
fi

echo "========================================"
echo "Folder : $CONFIG_FOLDER"
[ -n "$DATASET_FILTER" ] && echo "Dataset: $DATASET_FILTER"
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
