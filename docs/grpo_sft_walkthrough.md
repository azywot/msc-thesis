# Fine-Tuning Walkthrough

This guide walks through running the orchestrator fine-tuning pipeline from scratch on Snellius.

For a general project overview (setup, running experiments, config reference), see [`README.md`](../README.md) and [`CLAUDE.md`](../CLAUDE.md).

**What this actually trains:** GRPO (Group Relative Policy Optimisation - an RL method). The model learns by rolling out full agentic trajectories on real questions and receiving a reward based on answer correctness. We fine-tune Qwen3-8B with LoRA adapters so it becomes a better orchestrator in the CoSMAS multi-agent framework.

---

## Contents

1. [Quick overview of moving parts](#quick-overview-of-moving-parts)
2. [Prerequisites](#prerequisites)
3. [Step-by-step: running the GRPO pipeline](#step-by-step-running-the-pipeline)
   - [Step 0 - Create the conda environment](#step-0---create-the-conda-environment)
   - [Step 1 - Prepare training data](#step-1---prepare-training-data)
   - [Step 2 - Smoke test (4B)](#step-2---smoke-test-4b-fast-sanity-check)
   - [Step 3 - Smoke test (8B)](#step-3---smoke-test-8b-production-memory-layout)
   - [Step 4 - Full training run](#step-4---full-training-run)
4. [After training: run inference](#after-training-run-inference)
5. [What to change for your own setup](#what-to-change-for-your-own-setup)
6. [Key config parameters explained](#key-config-parameters-explained)
7. [Adapting to SFT (supervised fine-tuning)](#adapting-to-sft-supervised-fine-tuning)
   - [How the pipeline changes](#how-the-pipeline-changes)
   - [Step 1 - Collect trajectory data](#step-1---collect-trajectory-data)
   - [Step 2 - Format the data](#step-2---format-the-data)
   - [Step 3 - Write a VERL SFT config](#step-3---write-a-verl-sft-config)
   - [Step 4 - Launch](#step-4---launch)
   - [SFT checkpoints and inference](#sft-checkpoints-and-inference)
8. [Troubleshooting](#troubleshooting)
9. [File map](#file-map)

---

## Quick overview of moving parts

```
┌───────────────────────────────────────────────────────────────────┐
│  SLURM node (4× H100 NVL, ~94 GB each)                            │
│                                                                   │
│  GPU 0  ┬─ frozen sub-agent vLLM server  (Qwen3-1.7B, port 9998)  │
│          └─ VERL actor shard + rollout vLLM (shared, util=0.70)   │
│  GPU 1–3 ─ VERL actor/ref shards + rollout vLLM (util=0.70)       │
│                                                                   │
│  Python process 1: launch_verl.py  → Ray + FSDP training loop     │
│  Python process 2: train_orchestrator.py → rollout workers        │
│     (N=4 async workers, each sends questions to AgentFlow         │
│      on :9999, collects trajectories, sends to VERL)              │
└───────────────────────────────────────────────────────────────────┘
```

Training data: 900 Search-R1 (multi-hop web search) + 900 DeepMath questions.
Algorithm: GRPO, n=8 rollouts per question, 2 epochs ≈ 112 gradient steps.
Checkpoints: LoRA adapters written to `/scratch-shared/$USER/fine_tuning/lora_adapters/`.

---

## Prerequisites

### 1. HuggingFace access

You need a HF token with access to the gated models and datasets:

- Accept terms for `Qwen/Qwen3-8B` and `Qwen/Qwen3-1.7B` on HuggingFace
- Accept terms for `gaia-benchmark/GAIA` (if running GAIA evals later)

### 2. API keys

Copy `.env.example` to `.env` and fill in:

```
SERPER_API_KEY=...      # required - rollout workers use web search
WANDB_API_KEY=...       # strongly recommended - tracks training curves
HF_TOKEN=...            # required - model downloads
```

At minimum you need `SERPER_API_KEY` (or `TAVILY_API_KEY`).

### 3. Project directory

The job scripts default to `$HOME/azywot/msc-thesis`. If your clone is somewhere else, set `PROJECT_DIR` in `.env`:

```
PROJECT_DIR=/path/to/your/msc-thesis
```

---

## Step-by-step: running the pipeline

All jobs are under `jobs/fine_tuning/` and submitted with `sbatch`.
Logs land in `out/fine_tuning/` (relative to the project root).

### Step 0 - Create the conda environment

```bash
sbatch jobs/fine_tuning/000_create_environment.job
```

Creates the `cosmas-train` conda environment with the exact pinned stack:
Python 3.11, PyTorch 2.7, vLLM 0.10.1.1, VERL 0.7.1, flash-attn 2.8.3.

This job needs a GPU node (flash-attn's wheel selection inspects the local CUDA headers).
Takes ~25 minutes. Check `out/fine_tuning/create_env_<JOBID>.log` - it should end with lines like:

```
PyTorch:      2.7.x  CUDA available=True
vLLM:         0.10.1.1
Training imports OK
```

> **Do not skip this job.** Using a different environment (e.g. the main `agent_engine` env) will fail because the training stack requires specific VERL+vLLM version pins.

### Step 1 - Prepare training data

```bash
sbatch jobs/fine_tuning/001_prepare_data.job
```

Downloads Search-R1 (HotpotQA + NQ) and DeepMath-103K from HuggingFace, then writes VERL-compatible parquet files:

```
data/training/train/combined_train.parquet   # 1800 questions (train)
data/training/val/val_combined.parquet       # 50 questions (val)
data/training/smoke/...                      # small smoke subset (used by smoke jobs)
```

Takes ~30–60 minutes depending on download speed. Check `out/fine_tuning/prepare_data_<JOBID>.log`.

### Step 2 - Smoke test (4B, fast sanity check)

```bash
sbatch jobs/fine_tuning/003_smoke_4b.job
```

Runs one epoch of GRPO on 16 questions using **Qwen3-4B** (not 8B) on 2 GPUs. Faster to iterate.
Checks: imports, reward function, config, parquet schema, checkpoint written.

Takes ~30–45 minutes. You want to see `PASS - checkpoint found.` near the end.

### Step 3 - Smoke test (8B, production memory layout)

```bash
sbatch jobs/fine_tuning/004_smoke_8b.job
```

Same check but with **Qwen3-8B** on 3 GPUs (1 sub-agent + 2 VERL), which matches the production memory layout. This catches OOM issues you would hit in the full run.

Takes ~45–90 minutes. Again look for `PASS - checkpoint found.`

> Skip to Step 4 only after both smoke tests pass.

### Step 4 - Full training run

```bash
sbatch jobs/fine_tuning/005_train.job
```

Full run: 4 H100 GPUs, 1800 questions, 2 epochs, n=8 rollouts per question, LoRA rank=64.
Wall-time allocation: 72 hours.

Four log files appear in `out/fine_tuning/orchestrator_ft/`:
- `ft_<JOBID>.log` - main job log (startup, readiness polling)
- `ft_<JOBID>_verl.log` - VERL + Ray training loop
- `ft_<JOBID>_subagent.log` - frozen sub-agent vLLM server
- `ft_<JOBID>_gpu.csv` - GPU utilisation every 30 s

Training posts to W&B under project `msc-thesis-fine-tuning`, experiment `qwen3-8b-grpo-search-math-v2`. Watch `val/reward_mean` to track progress.

Checkpoints are saved every 5 steps to:
```
/scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math-v2/<run-tag>/global_step_N/actor/lora_adapter/
```

---

## After training: run inference

### Direct LoRA inference (no merge needed)

The configs under `experiments/configs/qwen3/lora_inference/` are the normal way to evaluate. vLLM loads the base Qwen3-8B model and applies the adapter at inference time - no merging required.

**You must update `lora_adapter_path`** in each config to point at your checkpoint:

```yaml
models:
  orchestrator:
    lora_adapter_path: "/scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math-v2/<run-tag>/global_step_N/actor/lora_adapter"
```

Find your latest checkpoint with:
```bash
find /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math-v2 \
    -maxdepth 3 -name 'global_step_*' -type d | sort -V | tail -1
```

Then run as a normal experiment:

```bash
python scripts/run_experiment.py \
    --config experiments/configs/qwen3/lora_inference/gaia/qwen8B_sub1_7b_none.yaml
```

Configs exist for GAIA, GPQA, HLE, AIME, and MuSiQue.
The `_none` suffix = thinking disabled; `_orchestrator` = thinking enabled for the orchestrator only.

### Merge LoRA into base model (optional)

Only needed if you want a standalone model (e.g. to deploy without the adapter file, or to use a backend that doesn't support LoRA):

```bash
LATEST=$(find /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math-v2 \
    -maxdepth 3 -name 'global_step_*' -type d | sort -V | tail -1)

python scripts/merge_lora.py \
    --checkpoint "$LATEST" \
    --base-model Qwen/Qwen3-8B \
    --output-dir /scratch-shared/$USER/fine_tuning/merged/qwen3-8b-grpo-merged
```

This creates a self-contained HF model directory you can load with `AutoModelForCausalLM.from_pretrained`.

---

## What to change for your own setup

### Change the experiment name

In `experiments/configs/fine_tuning/config.yaml`, update:

```yaml
env:
  EXPERIMENT_NAME: 'your-experiment-name'   # used for checkpoint dirs and W&B
  PROJECT_NAME: 'your-wandb-project'
```

This also affects where checkpoints land (`/scratch-shared/$USER/fine_tuning/lora_adapters/<EXPERIMENT_NAME>/`).

### Change the base model

To use a different model (e.g. Qwen3-4B), update in `config.yaml`:

```yaml
env:
  BASE_MODEL: 'Qwen/Qwen3-4B'
```

And in the corresponding smoke config if you want the smoke tests to match.
Also update the sub-agent model if needed (`SUBAGENT_MODEL`).

### Change the training data mix

Edit `jobs/fine_tuning/001_prepare_data.job` - the key flags to `src/fine_tuning/data/prepare.py`:

```
--n-search   N    # number of Search-R1 questions (web search tasks)
--n-math     N    # number of DeepMath questions
--hotpot-ratio R  # fraction of search questions drawn from HotpotQA (vs NQ)
--deepmath-min-difficulty D  # filter DeepMath by difficulty (1–5; 3 is default)
```

After changing, re-run `001_prepare_data.job` before smoke or training jobs.

### Change LoRA hyperparameters

In `config.yaml` under the `lora:` section:

```yaml
lora:
  rank: 64        # higher rank = more capacity, more memory
  alpha: 64       # scaling = alpha/rank; keeping alpha=rank gives scale 1.0
  target_modules: "all-linear"
```

### Disable LoRA (full fine-tuning)

Set `USE_LORA: "false"` in `config.yaml`. Full FT uses ~24 GB more per GPU for the optimizer. Requires adjusting `gpu_memory_utilization` from `0.70` → `0.45` (already in the config comments) - but `launch_verl.py` handles this automatically based on `USE_LORA`.

### Resume from a checkpoint

Uncomment and fill in the resume lines near the bottom of `config.yaml`:

```yaml
python_args:
  trainer.resume_mode: 'resume_path'
  trainer.resume_from_path: '/scratch-shared/$USER/fine_tuning/lora_adapters/<exp>/<run>/global_step_N'

lora:
  resume_adapter_path: '/scratch-shared/$USER/fine_tuning/lora_adapters/<exp>/<run>/global_step_N/actor/lora_adapter'
```

Both need to point at the same `global_step_N` directory.

---

## Key config parameters explained

| Parameter | Default | What it does |
|---|---|---|
| `actor_rollout_ref.rollout.n` | 8 | Number of rollout samples per question (GRPO group size) |
| `data.train_batch_size` | 32 | Questions per gradient step (= 32 × 8 = 256 rollouts) |
| `data.max_prompt_length` | 16384 | Max tokens in prompt; truncated+zero-reward if exceeded |
| `data.max_response_length` | 2048 | Max tokens the model generates per turn |
| `TOOL_STEPS` | 5 | Max agentic turns per rollout episode |
| `N_WORKERS` | 4 | Parallel rollout workers (IO-bound; more = faster data collection) |
| `trainer.save_freq` | 5 | Save checkpoint every N gradient steps |
| `actor_rollout_ref.actor.optim.lr` | 1e-6 | Learning rate |
| `algorithm.adv_estimator` | grpo | RL algorithm (GRPO = no critic, uses group baseline) |

---

## Adapting to SFT (supervised fine-tuning)

GRPO trains on the model's own rollouts using a reward signal. SFT trains on a fixed dataset of **(prompt, correct response)** pairs using cross-entropy loss - simpler, faster, and useful for behaviour cloning (e.g. distilling a stronger model's trajectories).

### How the pipeline changes

| | GRPO (current) | SFT |
|---|---|---|
| **Data** | Questions only; model generates answers at training time | Pre-collected conversations with ground-truth completions |
| **Loss** | Policy gradient (GRPO objective) over rollout tokens | Cross-entropy on assistant turns only |
| **During training** | Sub-agent vLLM + AgentFlow server + rollout workers + VERL | Just VERL SFT trainer (FSDP); no sub-agent, no rollout workers |
| **GPU pressure** | vLLM rollout + actor + ref all colocated | Actor only (+ FSDP optimizer); ~2 GPUs sufficient |
| **Config** | `actor_rollout_ref`, `algorithm`, reward fn | `model`, `trainer`, `optim`, `data.messages_key` |
| **Launcher** | `launch_verl.py` + `train_orchestrator.py` | `torchrun verl/trainer/sft_trainer.py` or the Ray variant |

### Step 1 - Collect trajectory data

SFT needs conversations where the orchestrator's responses are known-good. Three options:

**A. Filter winning GRPO rollouts (cheapest).** The rollout workers already collect full episode traces. Add a post-processing step after the GRPO run: extract episodes with reward = 1 from the raw rollout data and write them as parquet. This reuses data you already have.

**B. Use a stronger oracle model.** Run the existing `run_experiment.py` with a GPT-4 or Claude config as the orchestrator on your training questions, collect `raw_results.json`, then extract conversations where the answer is correct. Higher quality data but costs API tokens.

**C. Run the LoRA-adapted model.** After a GRPO run, use the LoRA checkpoint as the orchestrator in `run_experiment.py`, collect trajectories, and filter by correctness. This gives you on-policy SFT data from the already-improved model.

In all cases the result is parquet rows with a `messages` column - a list of chat-message dicts covering the full orchestrator conversation (system prompt, planning turn, tool calls, final answer). The SFT loss is applied only to assistant turns.

### Step 2 - Format the data

VERL's SFT dataset class (`verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`) expects a parquet where each row has a `messages` column — a list of chat-message dicts covering the full multi-turn trajectory.

In the MAS setup, a trajectory includes tool calls and their results, so the full structure looks like:

```python
{
  "messages": [
    {"role": "system",    "content": "<system prompt>"},
    {"role": "user",      "content": "<question>"},
    # planning turn — orchestrator reasons and decides which tools to call
    {"role": "assistant", "content": "<planning output>\n<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"...\"}}</tool_call>"},
    # tool result — what the sub-agent / tool returned
    {"role": "tool",      "content": "<tool_response>...</tool_response>"},
    # orchestrator may call another tool
    {"role": "assistant", "content": "<tool_call>{\"name\": \"code_generator\", \"arguments\": {\"task\": \"...\"}}</tool_call>"},
    {"role": "tool",      "content": "<tool_response>...</tool_response>"},
    # final answer turn
    {"role": "assistant", "content": "<final reasoning and answer>"},
  ]
}
```

**The `tool_call` / `tool_response` content must be preserved verbatim** from the raw trajectory — these are the Qwen3 native tool-call tags that the chat template understands. Do not strip or reformat them.

**> TODO: data collection script.** A script to extract MAS trajectories from `raw_results.json` runs, filter by correctness, and write the parquet is not yet written. It needs to reconstruct the `messages` list from the stored conversation turns (including sub-agent tool results), then serialise to parquet with a `messages` column.

The tokenizer's chat template is applied turn-by-turn; loss is masked to zero on `system`, `user`, and `tool` tokens — **only the orchestrator's `assistant` turns are trained on**. This is correct: the model should learn to produce good plans and tool calls, not to reproduce tool outputs.

The key config field is `data.messages_key: messages` (matches the column name above).

### Step 3 - Write a VERL SFT config

The template lives at `verl/trainer/config/sft_trainer_engine.yaml` inside the installed package. A minimal override for this project would look like:

```yaml
# experiments/configs/fine_tuning/config_sft.yaml
defaults:
  - sft_trainer_engine     # inherits from VERL's defaults

data:
  train_files: data/training/sft/train.parquet
  val_files:   data/training/sft/val.parquet
  messages_key: messages
  max_length: 16384
  truncation: truncate
  train_batch_size: 32
  micro_batch_size_per_gpu: 2

trainer:
  project_name: msc-thesis-sft
  experiment_name: qwen3-8b-sft-v1
  total_epochs: 3
  save_freq: 50
  test_freq: 50
  n_gpus_per_node: 2
  default_local_dir: /scratch-shared/$USER/fine_tuning/sft/qwen3-8b-sft-v1

model:
  path: Qwen/Qwen3-8B
  # LoRA — keeps trainable params to ~250 MB vs ~16 GB full FT, fits on 2 GPUs
  lora_rank: 64            # match the GRPO rank so adapters are comparable
  lora_alpha: 64           # alpha = rank → scaling factor 1.0
  target_modules: all-linear

optim:
  lr: 2.0e-5      # SFT uses a higher LR than GRPO's 1e-6; LoRA adapters start from zero
  warmup_steps: 20
```

**Why LoRA for SFT?** Without LoRA, FSDP SFT on Qwen3-8B needs ~16 GB base weights + ~16 GB gradients + ~32 GB Adam state ≈ 64 GB/GPU, which fills a single H100. With LoRA rank=64 all-linear, only ~250 MB of adapter params are trained; the base model weights are frozen and held in bf16. That fits the whole training pass on 2 GPUs instead of 4, and the adapter-only checkpoint is ~500 MB rather than ~47 GB of FSDP shards.

### Step 4 - Launch

No sub-agent server, no AgentFlow, no rollout workers. The trainer is launched directly with `torchrun`. Two approaches:

**Option A — Hydra overrides on the command line** (safest, no config file needed):
```bash
torchrun --nproc_per_node=2 -m verl.trainer.sft_trainer \
    data.train_files=data/training/sft/train.parquet \
    data.val_files=data/training/sft/val.parquet \
    data.messages_key=messages \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    model.path=Qwen/Qwen3-8B \
    model.lora_rank=64 \
    model.lora_alpha=64 \
    model.target_modules=all-linear \
    optim.lr=2e-5 \
    optim.warmup_steps=20 \
    trainer.project_name=msc-thesis-sft \
    trainer.experiment_name=qwen3-8b-sft-v1 \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.n_gpus_per_node=2 \
    "trainer.default_local_dir=/scratch-shared/$USER/fine_tuning/sft/qwen3-8b-sft-v1"
```

**Option B — YAML override config:** Put `config_sft.yaml` inside VERL's own config directory (`verl/trainer/config/`) so Hydra can resolve the `defaults: [sft_trainer_engine]` chain, then:
```bash
torchrun --nproc_per_node=2 -m verl.trainer.sft_trainer \
    --config-name config_sft
```
This is cleaner but requires writing into the installed package, which is fragile.

Or use the Ray variant (`verl.trainer.sft_trainer_ray`) if you want multi-node or Ray-managed workers — same config schema, same dataset.

A SLURM job needs only 2 GPUs (vs 4 for GRPO), no 9998/9999 ports, no `SERPER_API_KEY` check at training time. The job body is just: load modules → activate `cosmas-train` → `torchrun ...`.

### SFT checkpoints and inference

VERL's SFT trainer writes checkpoints under `trainer.default_local_dir`. With LoRA it saves the PEFT adapter format — the same format GRPO produces:

```
<default_local_dir>/global_step_N/
  model/
    lora_adapter/
      adapter_model.safetensors   # ~500 MB
      adapter_config.json         # rank, alpha, target_modules
  optimizer/...
  extra_state/...
```

This means the SFT adapter can be plugged straight into the existing `lora_inference` configs without any changes to the inference side:

```yaml
models:
  orchestrator:
    lora_adapter_path: "/scratch-shared/$USER/fine_tuning/sft/qwen3-8b-sft-v1/global_step_N/model/lora_adapter"
```

Note the path differs from GRPO (`model/lora_adapter/` vs GRPO's `actor/lora_adapter/`) — double-check which subdirectory `adapter_model.safetensors` lands in after your first run.

### What changes vs GRPO

**Not needed during the training job itself:**
- `scripts/launch_verl.py` and `scripts/train_orchestrator.py` — replaced by `torchrun verl/trainer/sft_trainer`
- The Ray actor pool, rollout vLLM instances, and AgentFlow server on port 9999
- 4 GPUs — 2 is enough for the training step with LoRA

**Still needed for data collection** (Step 1 above — running the MAS to gather trajectories):
- The full MAS stack: sub-agent vLLM, AgentFlow, `run_experiment.py`
- `SERPER_API_KEY` / `TAVILY_API_KEY` — web search tool is active during rollouts
- VERL itself if collecting from GRPO winning rollouts

---

## Troubleshooting

### Job fails immediately at import check

```
Training imports OK   ← this line missing
```

The `cosmas-train` env is broken. Re-run `000_create_environment.job` with `REBUILD_ENV=1` in `.env`.

### Sub-agent vLLM exits during startup

Check `ft_<JOBID>_subagent.log`. Common cause: another job is already using GPU 0 on the node, or the model download failed (check HF_TOKEN).

### AgentFlow readiness poll times out (45 min)

Check `ft_<JOBID>_verl.log`. Common causes:
- Ray failed to start - look for `RayActorError` or `Address already in use`
- Flash-attn version mismatch - look for `ImportError`
- CUDA OOM during vLLM init - look for `CUDA out of memory`

Ray logs are under `/scratch-local/$USER.$SLURM_JOB_ID/ray/`.

### NCCL deadlock / 1-hour watchdog

Symptom: job hangs at step N, then `NCCL watchdog` timeout after 1 hour.
The `use_dynamic_bsz=False` fix in `scripts/launch_verl.py` should prevent this, but if you hit it again, check that `launch_verl.py` is not being bypassed and that the config file has:

```yaml
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz: false
actor_rollout_ref.ref.log_prob_use_dynamic_bsz: false
```

### CUDA error at sampled_token_ids / flash_attn

These are caused by prefix caching being active despite `enable_prefix_caching: false` in the config. This is a known VERL bug (the `false` bool is silently dropped). The fix is already in `scripts/launch_verl.py` (injects `no_enable_prefix_caching=True`). If you see it, make sure you haven't overridden `launch_verl.py`.

### Checkpoints not found after training

Full-FT checkpoints go to `/scratch-shared/$USER/msc-thesis/fine_tuning/<exp>/`.
LoRA checkpoints go to `/scratch-shared/$USER/fine_tuning/lora_adapters/<exp>/`.
Check `USE_LORA` in your config and `USE_SCRATCH_CHECKPOINTS` (default `true` for the full run, `false` for smoke tests which write to `experiments/results/`).

---

## File map

```
jobs/fine_tuning/
  000_create_environment.job   # create cosmas-train conda env
  001_prepare_data.job         # download + write training parquets
  002_inspect_data.job         # optional: print dataset stats
  003_smoke_4b.job             # fast smoke test (Qwen3-4B, 2 GPUs)
  004_smoke_8b.job             # production-layout smoke test (Qwen3-8B, 3 GPUs)
  005_train.job                # full training run (Qwen3-8B LoRA, 4 GPUs)

experiments/configs/fine_tuning/
  config.yaml                  # production training config (edit this)
  config_smoke.yaml            # 4B smoke config
  config_smoke8b.yaml          # 8B smoke config

experiments/configs/qwen3/lora_inference/
  gaia/   gpqa/   hle/   aime/   musique/
    qwen8B_sub1_7b_none.yaml        # LoRA inference, thinking off
    qwen8B_sub1_7b_orchestrator.yaml # LoRA inference, thinking on

scripts/
  launch_verl.py          # starts VERL + Ray (process 1)
  train_orchestrator.py   # starts rollout workers (process 2)
  merge_lora.py           # merges adapter into base weights
  count_lora_params.py    # prints trainable param count (dry run)
  test_ft_smoke.py        # pre-flight checks run by smoke jobs

src/fine_tuning/
  agentflow/              # VERL integration: LitAgent, Trainer, reward function
  data/prepare.py         # dataset download + parquet writer
```
