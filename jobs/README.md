# jobs/

SLURM job scripts for the Snellius cluster.

For setup instructions, job file descriptions, and how to submit experiments see the
[main README](../README.md#hpc--cluster-setup).

## Files

| File | Purpose |
|------|---------|
| `001_setup.job` | Create conda env + install project |
| `002_download_datasets.job` | Download benchmark datasets |
| `003_run_examples.job` | Run simple examples to verify if everything works as expected |
| `004_export_env.job` | Export conda env YAMLs to `env_exports/` |
| `005_export_prompts.job` | Export prompt templates + tool schemas to JSON |
| `006_create_configs.job` | Regenerate all experiment YAML configs via `scripts/generate_configs.py` |
| `007_add_bigcodebench_libs.job` | Install BigCodeBench runtime dependencies into the conda env |
| `cosmas_train_pip_reconcile.sh` | Post-install pip pin reconciliation for the `cosmas-train` env (antlr4, peft<0.19) |
| `environment_train.yml` | Conda env spec for `cosmas-train` (RL fine-tuning) — pins torch / vllm / verl / transformers |
| `submit_job.sh` | Convenience wrapper — generates a job file from a config and submits it |
| `scripts/generate_job.py` | Generate a SLURM `.job` file from an experiment YAML |
| `templates/experiment.job.j2` | Jinja2 SLURM job template |
| `generated/` | Generated `.job` files (git-ignored) |
| `env_exports/` | Exported conda environment YAMLs |

### `fine_tuning/` — orchestrator RL fine-tuning pipeline (see [src/fine_tuning/README.md](../src/fine_tuning/README.md))

| File | Purpose |
|------|---------|
| `000_create_environment.job` | Build `cosmas-train` conda env from `environment_train.yml` + flash-attn + flashinfer (set `REBUILD_ENV=1` to wipe + rebuild) |
| `001_prepare_data.job` | Build GRPO training/val/test parquets under `data/training/` |
| `002_inspect_data.job` | Sample rows from training parquets for inspection/debugging |
| `003_smoke_4b.job` | Smoke-test the fine-tuning pipeline with Qwen3-4B (2 GPUs) |
| `004_smoke_8b.job` | Smoke-test the fine-tuning pipeline with Qwen3-8B (3 GPUs: GPU 0 = sub-agent, GPUs 1–2 = VERL N_GPUS=2) |
| `004_smoke_8b_load.job` | Verifies LoRA mid-run resume end-to-end (loads a saved checkpoint, runs 2 new steps, asserts log signals + new checkpoints) |
| `005_train.job` | Full orchestrator fine-tuning run — Qwen3-8B LoRA, 2 epochs, 4×H100 GPUs, 48h walltime |

### `gepa/` — GEPA-based prompt optimization runs

| File | Purpose |
|------|---------|
| `000_prep_gepa_data.job`, `005_prep_gepa_data.job` | Build GEPA datasets |
| `001_install_gepa_deps.job` | Install GEPA Python deps into the env |
| `002_smoke_gepa.job`, `003_smoke_gepa_gpu.job` | GEPA pipeline smoke tests (CPU / GPU) |
| `004_run_gepa.job`, `006_run_gepa_gaia.job`, `007_run_gepa_math.job` | Full GEPA optimization runs per benchmark |
