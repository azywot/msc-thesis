#!/bin/bash
# Launch experiment: generate job file and submit to SLURM
#
# Usage: ./launch_experiment.sh <config_file>
#
# Example: ./launch_experiment.sh experiments/configs/gaia/baseline.yaml

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file>"
    echo "Example: $0 experiments/configs/gaia/baseline.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Load .env if present (PROJECT_DIR, ENV_NAME, etc.)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Project dir and conda env (from env / .env)
PROJECT_DIR="${PROJECT_DIR:-$HOME/azywot/msc-thesis}"
ENV_NAME="${ENV_NAME:-agent_engine}"

# If we're not in a Slurm job already, load the same modules we use in job scripts.
# (On many HPC systems, `module` is available on login nodes too; if not, these are no-ops.)
if command -v module >/dev/null 2>&1; then
    module purge || true
    module load 2025 || true
    module load Miniconda3/25.5.1-1 || true
    module load CUDA/12.8.0 || true
fi

cd "$PROJECT_DIR" || exit 1

# Activate conda env if conda is available
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source activate "$ENV_NAME" || true
fi

# Prevent Python from importing ~/.local site-packages (common source of mismatched torch/numpy)
export PYTHONNOUSERSITE=1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "========================================"
echo "Launching Experiment"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo ""

# Generate job file (capture JOB_FILE from script output for consistency)
echo "Generating SLURM job file..."
GEN_OUT=$(python jobs/scripts/generate_job.py "$CONFIG_FILE")
echo "$GEN_OUT"
JOB_FILE=$(echo "$GEN_OUT" | grep "^JOB_FILE=" | cut -d= -f2-)

if [ -z "$JOB_FILE" ] || [ ! -f "$JOB_FILE" ]; then
    echo "Error: Failed to generate job file"
    exit 1
fi

echo "Generated: $JOB_FILE"
echo ""

# Submit job
echo "Submitting to SLURM..."
sbatch "$JOB_FILE"

echo ""
echo "Job submitted successfully!"
echo "Monitor with: squeue -u $USER"
echo "Cancel with: scancel <job_id>"
