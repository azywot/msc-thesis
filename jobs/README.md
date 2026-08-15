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

### `fine_tuning/` — orchestrator training pipelines (see [src/fine_tuning/README.md](../src/fine_tuning/README.md))

#### RL (GRPO)

| File | Purpose |
|------|---------|
| `000_create_environment.job` | Build `cosmas-train` conda env from `environment_train.yml` + flash-attn + flashinfer (set `REBUILD_ENV=1` to wipe + rebuild) |
| `001_prepare_data.job` | Build GRPO training/val/test parquets under `data/training/` |
| `002_inspect_data.job` | Sample rows from training parquets for inspection/debugging |
| `003_smoke_4b.job` | Smoke-test the fine-tuning pipeline with Qwen3-4B (2 GPUs) |
| `004_smoke_8b.job` | Smoke-test the fine-tuning pipeline with Qwen3-8B (3 GPUs: GPU 0 = sub-agent, GPUs 1–2 = VERL N_GPUS=2) |
| `004_smoke_8b_load.job` | Verifies LoRA mid-run resume end-to-end (loads a saved checkpoint, runs 2 new steps, asserts log signals + new checkpoints) |
| `005_train.job` | Full orchestrator GRPO run — Qwen3-8B, 2 epochs, 4×H100 GPUs, 48h walltime; LoRA by default, full-parameter via `USE_LORA: "false"` in `experiments/configs/fine_tuning/config.yaml` |

#### SFT (distillation)

| File | Purpose |
|------|---------|
| `006_collect_sft_data.job` | Run Qwen3-32B teacher (ORCHESTRATOR_ONLY thinking, sub-agent mode) on the 1800 GRPO training questions and save correct trajectories as SFT training data (4×H100) |
| `007_run_tests_for_sft_folded.job` | CPU-only verification suite for the folded-format pipeline (tests, pre-flight gate, gate trip-wire) |
| `007_train_sft_folded.job` | SFT distillation of Qwen3-8B orchestrator from the Qwen3-32B teacher trajectories, memory-folded prompt format — LoRA rank 64, ~187 steps, 2×H100 |
| `007_train_sft_full.job` | Same, full parameter (no LoRA) — every weight trained, 4×H100 |

### `grpo_inference/` — GRPO / SFT / base-model evaluation jobs

Self-contained eval jobs (require only `sbatch`, no manual pre-steps) comparing three orchestrator
checkpoints on the same AgentFlow setup (Qwen3-8B orchestrator + Qwen3-1.7B sub-agents,
`thinking_mode: NO`, `direct_tool_call: false`):

| File | Purpose |
|------|---------|
| `GRPO_eval_aime_qwen8B_sub1_7b_none.job` | AIME eval with the GRPO LoRA-adapted Qwen3-8B orchestrator |
| `GRPO_eval_gaia_qwen8B_sub1_7b_none.job` | GAIA eval with the GRPO LoRA-adapted Qwen3-8B orchestrator |
| `SFT_eval_aime_qwen8B_sub1_7b_none.job` | AIME eval with the SFT LoRA-adapted Qwen3-8B orchestrator (`global_step_90`); merges FSDP shards into a PEFT adapter on first run if needed |
| `SFT_eval_gaia_qwen8B_sub1_7b_none.job` | GAIA eval with the SFT LoRA-adapted Qwen3-8B orchestrator; same one-time merge step |
| `BASE_eval_aime_qwen8B_sub1_7b_none.job` | AIME eval with the unmodified (no-adapter) Qwen3-8B orchestrator — no-fine-tuning baseline |
| `BASE_eval_gaia_qwen8B_sub1_7b_none.job` | GAIA eval with the unmodified (no-adapter) Qwen3-8B orchestrator — no-fine-tuning baseline |
| `configs/aime/`, `configs/gaia/` | Per-job experiment YAMLs (`qwen8B_sub1_7b_none.yaml` = GRPO, `qwen8B_sft_sub1_7b_none.yaml` = SFT, `qwen8B_base_sub1_7b_none.yaml` = base) |

Results land in `experiments/results/{grpo,sft,base}_inference/{aime,gaia}/qwen8B_sub1_7b_none/`.

### `gepa/` — GEPA-based prompt optimization runs

| File | Purpose |
|------|---------|
| `000_prep_gepa_data.job`, `005_prep_gepa_data.job` | Build GEPA datasets |
| `001_install_gepa_deps.job` | Install GEPA Python deps into the env |
| `002_smoke_gepa.job`, `003_smoke_gepa_gpu.job` | GEPA pipeline smoke tests (CPU / GPU) |
| `004_run_gepa.job`, `006_run_gepa_gaia.job`, `007_run_gepa_math.job` | Full GEPA optimization runs per benchmark |
