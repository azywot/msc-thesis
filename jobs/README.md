# Quick Start Guide (HPC / SLURM)

Complete setup in 4 steps.

---

## Step 1: First Time Setup (One-time)

```bash
cd $HOME/thesis/msc-thesis

# 1) Setup environment 
sbatch jobs/001_setup.job

# 2) Create `.env` with API keys
cp .env.example .env
nano .env  # or vim, emacs, etc.

# 3) Download datasets
sbatch jobs/002_download_datasets.job

# 4) Test setup 
sbatch jobs/003_test_simple.job

# Monitor jobs / logs
squeue -u $USER
```

`.env` minimum (required):

```bash
SERPER_API_KEY=your_serper_key_here
```

Optional:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
```

---

## Job Files & Outputs (Reference)

| File | Purpose | Typical output |
|------|---------|----------------|
| `jobs/001_setup.job` | Create conda env `agent_engine` + install project | `out/setup/msc_thesis_env_setup_<job_id>.log` |
| `jobs/002_download_datasets.job` | Download all benchmark datasets | `out/datasets/download_datasets_<job_id>.log` |
| `jobs/003_test_simple.job` |
| `jobs/004_export_env.job` | Export conda env YAMLs | `out/export_env/export_env_<job_id>.log` + `jobs/env_exports/*.yml` |
| `jobs/submit_job.sh` | Wrapper for all jobs | prints job id + paths |

Optional overrides (via `sbatch --export=ALL,...`):
- `ENV_NAME` (default: `agent_engine`)
- `PROJECT_DIR` (default: `$HOME/thesis/msc-thesis`)
- `DATA_DIR` (default: `/scratch-shared/$USER/data`)

## Step 2: Running Experiments

```bash
# Recommended: convenience wrapper
./jobs/submit_job.sh experiment experiments/configs/gaia/baseline.yaml

# Alternative: call launcher directly
./jobs/scripts/launch_experiment.sh experiments/configs/gaia/baseline.yaml
```

### Example Experiment Configs

```bash
# GPQA experiment
./jobs/submit_job.sh experiment experiments/configs/gpqa/baseline.yaml

# Math datasets
./jobs/submit_job.sh experiment experiments/configs/math/baseline.yaml
```

---

## Step 3: Monitoring

```bash
# Check your jobs
squeue -u $USER

# Watch logs in real-time (written to the experiment `output_dir/`)
# Example: ./experiments/results/gaia_baseline/gaia_qwen3_baseline_<job_id>.log
tail -f experiments/results/<experiment_output_dir>/<experiment_name>_<job_id>.log

# Check job efficiency after completion
seff <job_id>
```

---

## Step 4: Analyzing Results

```bash
# Analyze results from single experiment
python scripts/analyze_results.py experiments/results/<experiment_name>/results.json

# Analyze by level (for GAIA)
python scripts/analyze_results.py \
    experiments/results/gaia_baseline/results.json \
    --by-level

# Analyze tool usage
python scripts/analyze_results.py \
    experiments/results/gaia_baseline/results.json \
    --tools
```

Notes:
- The runner also writes **legacy-format** files for compatibility:
  - `<split>.<month>.<day>,<hour>:<min>.json`
  - `<split>.<month>.<day>,<hour>:<min>.metrics.json`
- To log to W&B: set `use_wandb: true` + `wandb_project: ...` in the experiment YAML, and provide `WANDB_API_KEY`.

---

## SLURM Reference

```bash
scontrol show job <job_id>          # Job details
sacct -j <job_id>                   # Accounting info (after / during run)
scancel <job_id>                    # Cancel one job
scancel -u $USER                    # Cancel all your jobs
scancel -n <job_name>               # Cancel by name
```

---

## Create Your Own Experiment Config

Copy an existing config, then edit a few key fields (`name`, model, GPUs, dataset split, `output_dir`):

```bash
cp experiments/configs/gaia/baseline.yaml experiments/configs/gaia/my_experiment.yaml
nano experiments/configs/gaia/my_experiment.yaml

./jobs/submit_job.sh experiment experiments/configs/gaia/my_experiment.yaml
```

---

## Troubleshooting

### Job fails immediately
```bash
# Check the log
cat out/<job_type>/<job_name>_<job_id>.log

# Common issues:
# 1. Environment not set up → run ./jobs/submit_job.sh setup
# 2. API keys not set → check .env file
# 3. Dataset not downloaded → run ./jobs/submit_job.sh download
```

### Out of memory
```bash
# Solution 1: Use more GPUs
# In config: increase tensor_parallel_size and gpu_ids

# Solution 2: Use H100 instead of A100
# In job file: change partition to gpu_h100

# Solution 3: Use smaller model
# In config: use Qwen3-14B instead of Qwen3-32B
```

### Job pending forever
```bash
# Check cluster status
sinfo

# Check your job priority
sprio -j <job_id>

# Use different partition if needed
# gpu_a100 usually has more availability
```

---

## Need More Help?

- **Project README**: `README.md`

---

**Last Updated**: 2025-02-17  
**Cluster**: Snellius (SURF)
