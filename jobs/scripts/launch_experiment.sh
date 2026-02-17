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

# Generate job file
echo "Generating SLURM job file..."
python jobs/scripts/generate_job.py "$CONFIG_FILE"

# Get experiment name from config
EXPERIMENT_NAME=$(grep "^name:" "$CONFIG_FILE" | sed 's/name: *"\?\([^"]*\)"\?/\1/' | tr -d '"')
JOB_FILE="jobs/generated/${EXPERIMENT_NAME}.job"

if [ ! -f "$JOB_FILE" ]; then
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
