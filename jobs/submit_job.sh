#!/bin/bash
# Convenience script to submit common jobs
#
# Usage: ./submit_job.sh <job_type> [options]
#
# Job types:
#   setup           - Setup conda environment
#   download        - Download all datasets
#   test            - Run simple test example
#   export          - Export conda environment
#   experiment      - Run full experiment (requires config file)
#
# Examples:
#   ./submit_job.sh setup
#   ./submit_job.sh download
#   ./submit_job.sh test
#   ./submit_job.sh experiment experiments/configs/datasets/gaia/baseline.yaml

set -e

# Load .env if present (PROJECT_DIR, ENV_NAME, DATA_DIR, API keys, etc.)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Default values (PROJECT_DIR and ENV_NAME read from env / .env above)
PROJECT_DIR="${PROJECT_DIR:-$HOME/azywot/msc-thesis}"
ENV_NAME="${ENV_NAME:-agent_engine}"
DATA_DIR="${DATA_DIR:-/scratch-shared/$USER/data}"

show_usage() {
    echo "Usage: $0 <job_type> [options]"
    echo ""
    echo "Job types:"
    echo "  setup           - Setup conda environment"
    echo "  download        - Download all datasets"
    echo "  test            - Run simple test example"
    echo "  export          - Export conda environment"
    echo "  experiment      - Run full experiment (requires config file)"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 download"
    echo "  $0 test"
    echo "  $0 export"
    echo "  $0 experiment experiments/configs/datasets/gaia/baseline.yaml"
    echo ""
    echo "Environment variables (from env or .env):"
    echo "  PROJECT_DIR     - Project directory (default: \$HOME/azywot/msc-thesis)"
    echo "  ENV_NAME        - Conda environment name (default: agent_engine)"
    echo "  DATA_DIR        - Data directory (default: /scratch-shared/\$USER/data)"
}

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

JOB_TYPE="$1"
shift

case "$JOB_TYPE" in
    setup)
        echo "Submitting setup job..."
        mkdir -p out/setup
        sbatch \
            --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_NAME="$ENV_NAME" \
            jobs/001_setup.job
        ;;

    download)
        echo "Submitting dataset download job..."
        mkdir -p out/datasets
        sbatch \
            --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_NAME="$ENV_NAME",DATA_DIR="$DATA_DIR" \
            jobs/002_download_datasets.job
        ;;

    test)
        echo "Submitting simple test job..."
        mkdir -p out/test
        sbatch \
            --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_NAME="$ENV_NAME" \
            jobs/003_run_examples.job
        ;;

    export)
        echo "Submitting environment export job..."
        mkdir -p out/export_env
        sbatch \
            --export=ALL,PROJECT_DIR="$PROJECT_DIR",ENV_NAME="$ENV_NAME" \
            jobs/004_export_env.job
        ;;

    experiment)
        if [ $# -eq 0 ]; then
            echo "Error: experiment job requires config file"
            echo "Usage: $0 experiment <config_file>"
            exit 1
        fi
        CONFIG_FILE="$1"
        echo "Submitting experiment job with config: $CONFIG_FILE"
        ./jobs/scripts/launch_experiment.sh "$CONFIG_FILE"
        ;;

    *)
        echo "Error: Unknown job type: $JOB_TYPE"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "Job submitted successfully!"
echo "Monitor with: squeue -u $USER"
echo "Cancel with: scancel <job_id>"
echo "View logs in: out/<job_type>/"
