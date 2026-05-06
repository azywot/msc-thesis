#!/bin/bash
#SBATCH --job-name=cosmas-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/%j_train.log
#SBATCH --error=jobs/logs/%j_train.err

set -euo pipefail

module load 2023
module load CUDA/12.1.1

conda activate cosmas-train

cd "$HOME/azywot/msc-thesis"

mkdir -p jobs/logs

echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"

# ── 1. Start VERL server in background ──────────────────────────────────────
echo "Starting VERL server..."
python scripts/launch_verl.py --config train/config.yaml \
    > jobs/logs/${SLURM_JOB_ID}_verl.log 2>&1 &
VERL_PID=$!

# Give Ray + vLLM time to initialise (adjust if cold-start takes longer)
echo "Waiting 60s for VERL server to be ready..."
sleep 60

# ── 2. Start rollout workers ─────────────────────────────────────────────────
echo "Starting rollout workers..."
python scripts/train_orchestrator.py --config train/config.yaml

# ── 3. Wait for VERL ─────────────────────────────────────────────────────────
wait $VERL_PID
echo "Training complete."
