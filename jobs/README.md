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
| `008_prepare_fine_tuning_data.job` | Create `cosmas-train` conda env + build GRPO training/val/test parquets |
| `009_test_small_ft_example.job` | Smoke-test the fine-tuning pipeline end-to-end with Qwen3-4B (2 GPUs) |
| `010_ft_orchestrator.job` | Full orchestrator fine-tuning run — Qwen3-8B, 5 epochs, 4×H100 GPUs |
| `011_sample_train_parquet.job` | Sample rows from training parquets for inspection/debugging |
| `012_smoke_8b.job` | Smoke-test the fine-tuning pipeline with Qwen3-8B (3 GPUs: GPU 0 = sub-agent, GPUs 1–2 = VERL N_GPUS=2); checkpoints to scratch-shared |
| `submit_job.sh` | Convenience wrapper — generates a job file from a config and submits it |
| `scripts/generate_job.py` | Generate a SLURM `.job` file from an experiment YAML |
| `templates/experiment.job.j2` | Jinja2 SLURM job template |
| `generated/` | Generated `.job` files (git-ignored) |
| `env_exports/` | Exported conda environment YAMLs |
