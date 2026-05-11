# Fine-Tuning the CoSMAS Orchestrator

**Target:** Qwen3-8B orchestrator only. Sub-agents (web search analyser, code generator) are not gradient-trained — their token generations are treated as environment interactions by VERL and do not enter the GRPO objective. Sub-agents run on a **separate, frozen vLLM server** (`Qwen/Qwen3-1.7B`, port 9998) that is started before training and never updated, eliminating train/eval distribution shift for sub-agents.

**Method:** GRPO reinforcement learning via the AgentFlow training stack (VERL backend, LoRA adapters).

**Motivation:** Failure analysis across 2,534 MAS failures identifies *direct reasoning without action* as the dominant failure mode (39.4%). The orchestrator answers from parametric knowledge instead of delegating to a sub-agent. GRPO on retrieval-intensive (Search-R1) and math-intensive (DeepMath) training data creates gradient pressure toward tool use by making tool-less rollouts lose.

Full alignment analysis: `docs/failure_modes_fine_tuning_alignment.md`
Design spec: `docs/superpowers/specs/2026-05-06-orchestrator-finetuning-design.md`

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Step-by-Step Guide](#3-step-by-step-guide)
4. [File Map](#4-file-map)
5. [Config Reference](#5-config-reference)
6. [W&B Metrics](#6-wb-metrics)
7. [Things to Watch](#7-things-to-watch)
8. [Merging LoRA and Evaluating](#8-merging-lora-and-evaluating)
9. [Troubleshooting](#9-troubleshooting)
10. [Design Decisions](#10-design-decisions)

---

## 1. Architecture

```
Training time
────────────────────────────────────────────────────────────────

          ┌──────────────────────────────────────────────────┐
          │  agentflow.verl  (VERL backend, Ray cluster)     │
          │                                                  │
          │  AgentFlowTrainer(RayPPOTrainer)                 │
          │  ├── GRPO advantage estimator (n=8 rollouts)     │
          │  ├── FSDP actor: Qwen3-8B + LoRA rank-64         │
          │  ├── Reference policy: Qwen3-8B (frozen)         │
          │  └── AgentModeDaemon  :9999                      │
          └──────────────┬───────────────────────────────────┘
                         │  HTTP  (tasks ↓ / rewards ↑)
          ┌──────────────▼───────────────────────────────────┐
          │  OrchestratorRollout (LitAgent)                  │
          │  ├── AgenticOrchestrator  ← model being trained  │
          │  │     thinking_mode: ORCHESTRATOR_ONLY          │
          │  ├── WebSearchTool  → sub-agent LLM @ :9998      │
          │  ├── CodeGeneratorTool → sub-agent LLM @ :9998   │
          │  │   (sub-agents use a SEPARATE frozen server:   │
          │  │    Qwen3-1.7B, never updated during training)  │
          │  └── OrchestratorReward  ← binary via metrics.py │
          └──────────────────────────────────────────────────┘

Inference (unchanged after training)
────────────────────────────────────────────────────────────────

  model_merger.py  →  merged HF model  →  path_or_id in YAML
                                           ↓
                                     VLLMProvider (no changes)
```

**Key point:** only the orchestrator path is directly optimized by GRPO (LoRA updates on the actor policy). Sub-agents run on a separate frozen vLLM server (`Qwen/Qwen3-1.7B`) that never changes during training — the orchestrator is always evaluated against the same stable tool interface. Tools, evaluation scripts, and experiment configs are untouched. After training, merge the LoRA adapter and point any existing YAML at the merged model path.

---

## 2. Prerequisites

### Conda environments

Two separate environments are required and intentionally kept separate:

| Environment | Purpose | vLLM |
|---|---|---|
| `cosmas-train` | Training (VERL, rollout workers) | 0.9.2 |
| `agent_engine` | Inference and evaluation | 0.12.0 |

The version split is a hard constraint, not a convenience choice. VERL 0.5.0 requires
vLLM ~0.9.x, but the inference stack pins vLLM 0.12.0. AgentFlow adds a third
constraint (its requirements declare vllm==0.8.5), making a three-way compatibility
fix necessary to consolidate. The latest VERL (0.7.1) may support newer vLLM, but
upgrading risks breaking AgentFlow's internal VERL wrappers (`agentflow.verl.*`) in
untested ways.

Create the training environment:
```bash
conda env create -f jobs/environment_train.yml
```

### Environment variables

Set these in your Snellius login script (`~/.bashrc`) or in the SLURM job before launching:

```bash
export SERPER_API_KEY=<your_key>      # or TAVILY_API_KEY
export WANDB_API_KEY=<your_key>
export HF_TOKEN=<your_token>          # for gated HuggingFace datasets
```

**`SERPER_API_KEY` (or `TAVILY_API_KEY`) is required at runtime.** Rollout workers call the search API on every training question. Missing this key causes an immediate `EnvironmentError`.

### AgentFlow

Must be cloned and installed into the training environment:
```bash
git clone https://github.com/shin-ee-chen/AgentFlow $HOME/azywot/AgentFlow
conda activate cosmas-train
pip install -e $HOME/azywot/AgentFlow/agentflow
```

The `jobs/environment_train.yml` installs it from the local path automatically if the clone is already present.

---

## 3. Step-by-Step Guide

### Step 1 — Prepare training and validation data

**SLURM (recommended):**
```bash
sbatch jobs/008_prepare_fine_tuning_data.job
```

**Local (for testing):**
```bash
conda activate cosmas-train
python src/fine_tuning/data/prepare.py \
    --n-search 900 --n-math 900 \
    --n-val-search 100 --n-val-math 100 \
    --n-test-search 100 --n-test-math 100 \
    --search-source both --hotpot-ratio 0.85 \
    --deepmath-min-difficulty 5 \
    --output-dir data/training --seed 42
```

This downloads Search-R1 (`PeterJinGo/nq_hotpotqa_train`) and DeepMath-103K (`zwhe99/DeepMath-103K`) from HuggingFace and writes:

```
data/training/
├── train/
│   └── combined_train.parquet    1800 rows  (900 search + 900 math, shuffled)
├── val/
│   ├── val_search.parquet         100 rows  (held-out NQ + HotpotQA)
│   ├── val_deepmath.parquet       100 rows  (held-out DeepMath, difficulty ≥ 5)
│   └── val_combined.parquet       200 rows  (both merged — offline analysis only)
└── test/
    ├── test_search.parquet        100 rows  (held-out NQ + HotpotQA)
    ├── test_deepmath.parquet      100 rows  (held-out DeepMath, difficulty ≥ 5)
    └── test_combined.parquet      200 rows  (both merged — final reporting only)
```

Job 008 also writes a smoke subset to `data/training/smoke/` for use in Step 2.

**Critical:** rows are carved in order — test first, then val, then train. All three splits share the same source proportions (85% HotpotQA / 15% NQ; 50/50 search/math). There is zero overlap across splits.

### Step 2 — Run the smoke test

Verifies the full pipeline end-to-end on tiny data (16 questions, 2 rollouts, 1 epoch) before committing 24 hours of A100 time.

```bash
sbatch jobs/009_test_small_ft_example.job
```

The job runs in two phases:
1. **Pre-flight checks** (CPU, no VERL): imports, reward routing for all data sources, config parsing, parquet schema validation, `OrchestratorRollout` instantiation.
2. **Mini training run**: 1 epoch on `experiments/configs/train/config_smoke.yaml`. Asserts a checkpoint was written. W&B is enabled (project `cosmas-rl-finetuning-smoke`) so logging is verified too.

Run the pre-flight checks locally at any time (no GPU, no VERL needed):
```bash
conda activate cosmas-train
python scripts/test_ft_smoke.py \
    --data-dir data/training/smoke \
    --config   experiments/configs/train/config_smoke.yaml
```

### Step 3 — Full training run

```bash
sbatch jobs/010_ft_orchestrator.job
```

Or manually (three terminals, after activating `cosmas-train` in all):
```bash
# Terminal 1 — frozen sub-agent server (start first, never needs restarting)
vllm serve Qwen/Qwen3-1.7B --port 9998 --tensor-parallel-size 1 --gpu-memory-utilization 0.15
export SUBAGENT_ENDPOINT=http://localhost:9998/v1

# Terminal 2 — VERL server (start after sub-agent server is up)
python scripts/launch_verl.py --config experiments/configs/train/config.yaml

# Terminal 3 — rollout workers (start after VERL vLLM is up, ~120s)
python scripts/train_orchestrator.py --config experiments/configs/train/config.yaml
```

Training runs for 5 epochs. Checkpoints every epoch:
```
experiments/results/training/qwen3-8b-grpo-search-math/
├── config.yaml                        copy of config at run start
├── checkpoint_step_1/actor/lora_weights.pt
├── checkpoint_step_2/actor/lora_weights.pt
├── checkpoint_step_3/actor/lora_weights.pt
├── checkpoint_step_4/actor/lora_weights.pt
├── checkpoint_step_5/actor/lora_weights.pt
└── checkpoint_best/                   → symlink to best val checkpoint
```

The SLURM launcher (`jobs/010_ft_orchestrator.job`) then prunes checkpoint payloads at the end to keep only:
- latest checkpoint step (`checkpoint_step_<max_epoch>/`)
- best validation checkpoint target (`checkpoint_best` symlink target)

### Step 4 — Merge LoRA and evaluate

```bash
conda activate cosmas-train
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path experiments/results/training/qwen3-8b-grpo-search-math/checkpoint_best/actor/lora_weights.pt \
    --output_dir experiments/results/training/qwen3-8b-grpo-search-math/merged_model/
```

Then update an existing experiment YAML to use the fine-tuned model:
```yaml
models:
  orchestrator:
    path_or_id: /path/to/experiments/results/training/qwen3-8b-grpo-search-math/merged_model/
    # all other fields unchanged
```

No changes to `VLLMProvider`, `AgenticOrchestrator`, or evaluation scripts.

---

## 4. File Map

```
msc-thesis/
│
├── src/fine_tuning/
│   ├── __init__.py              lazy imports (heavy deps optional)
│   ├── config.py                FinetuningConfig dataclass
│   ├── reward.py                OrchestratorReward — binary via metrics.py
│   ├── rollout.py               OrchestratorRollout(LitAgent)
│   ├── trainer.py               unused stub (agentflow.Trainer used directly)
│   └── data/
│       └── prepare.py           download + split + write parquet files
│
├── scripts/
│   ├── launch_verl.py           starts VERL server (reads experiments/configs/train/config.yaml)
│   ├── train_orchestrator.py    starts rollout workers (connects to VERL + frozen sub-agent server)
│   └── test_ft_smoke.py         pre-flight checks — runs without GPU or VERL
│
├── experiments/configs/train/
│   ├── config.yaml              full training config (5 epochs, 4×A100)
│   └── config_smoke.yaml        smoke-test config (1 epoch, 16 samples)
│
├── jobs/
│   ├── 008_prepare_fine_tuning_data.job   SLURM: prepare data/training/
│   ├── 009_test_small_ft_example.job      SLURM: smoke test
│   ├── 010_ft_orchestrator.job            SLURM: full training run
│   └── environment_train.yml              conda env spec (cosmas-train)
│
├── data/training/               created by job 008
│   ├── train/combined_train.parquet
│   ├── val/val_search.parquet
│   │   val/val_deepmath.parquet
│   │   val/val_combined.parquet
│   └── test/test_search.parquet
│        test/test_deepmath.parquet
│        test/test_combined.parquet
│
└── docs/
    ├── fine_tuning_README.md              this file
    ├── failure_modes_fine_tuning_alignment.md
    └── superpowers/specs/2026-05-06-orchestrator-finetuning-design.md
```

---

## 5. Config Reference

### `experiments/configs/train/config.yaml` — `env` block

| Key | Value | Notes |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | HuggingFace model ID or local path |
| `SUBAGENT_MODEL` | `Qwen/Qwen3-1.7B` | Frozen sub-agent model (separate vLLM server at port 9998) |
| `SUBAGENT_ENDPOINT` | `http://localhost:9998/v1` | URL of the frozen sub-agent vLLM server |
| `N_GPUS` | `4` | Must match `#SBATCH --gres=gpu:a100:4` |
| `ROLLOUT_TP_SIZE` | `2` | Tensor parallelism for rollout vLLM (2 GPUs per shard) |
| `EXPERIMENT_NAME` | `qwen3-8b-grpo-search-math` | Checkpoint dir name and W&B run name |
| `PROJECT_NAME` | `cosmas-rl-finetuning` | W&B project |
| `BASE_DATA_DIR` | `data/training` | Root of train/val parquet files |
| `ENABLE_TOOLS` | `["web_search", "code_generator"]` | Tools available to orchestrator during rollout |
| `TOOL_STEPS` | `5` | Max tool calls per rollout episode |
| `THINKING_MODE` | `ORCHESTRATOR_ONLY` | Must match eval condition — see §10 |
| `TRAIN_TEMPERATURE` | `0.7` | Sampling temperature for training rollouts |
| `TEST_TEMPERATURE` | `0.0` | Greedy decoding for validation rollouts |
| `N_WORKERS` | `1` | Number of parallel rollout worker processes |

### `experiments/configs/train/config.yaml` — `python_args` block (key parameters)

| Key | Value | Notes |
|---|---|---|
| `data.train_batch_size` | `32` | Questions per training step |
| `data.train_max_samples` | `128` | Questions per epoch (subset of full dataset) |
| `data.max_prompt_length` | `18432` | Max prompt tokens (system prompt + memory + question) |
| `data.max_response_length` | `4096` | Max response tokens per rollout; doubled from AgentFlow default to cover multi-turn orchestrator trajectories + thinking traces |
| `actor_rollout_ref.rollout.n` | `8` | Rollouts per question (GRPO group size) |
| `actor_rollout_ref.model.lora_rank` | `64` | LoRA rank (~130 MB checkpoints) |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.001` | KL penalty weight (keeps policy close to reference) |
| `trainer.total_epochs` | `5` | Training epochs |
| `trainer.save_freq` | `1` | Save checkpoint every epoch |
| `trainer.test_freq` | `1` | Run validation every epoch |
| `trainer.val_before_train` | `true` | Runs validation before epoch 1 (baseline measurement) |

### Smoke test differences (`experiments/configs/train/config_smoke.yaml`)

| Parameter | Full config | Smoke config |
|---|---|---|
| `data.train_max_samples` | 128 | 16 |
| `rollout.n` | 8 | 2 |
| `total_epochs` | 5 | 1 |
| `TOOL_STEPS` | 5 | 2 |
| `data.max_prompt_length` | 18432 | 8192 |
| `train_batch_size` | 32 | 4 |
| `ppo_micro_batch_size_per_gpu` | 4 | 1 |
| `BASE_DATA_DIR` | `data/training` | `data/training/smoke` |
| `EXPERIMENT_NAME` | `qwen3-8b-grpo-search-math` | `qwen3-8b-grpo-smoke` |
| `PROJECT_NAME` | `cosmas-rl-finetuning` | `cosmas-rl-finetuning-smoke` |

---

## 6. W&B Metrics

VERL logs to the project set in `PROJECT_NAME`. `data.val_files` is a two-element list so VERL runs validation separately per file:

| Metric | Source | What it tells you |
|---|---|---|
| `val_0/reward_mean` | `val_search.parquet` | Search accuracy (NQ + HotpotQA) — confirms `web_search` tool use is improving |
| `val_1/reward_mean` | `val_deepmath.parquet` | Math accuracy (DeepMath) — confirms `code_generator` tool use is improving |
| `actor/reward_mean` | training rollouts | Average reward across both domains per step |
| `actor/reward_std` | training rollouts | High std = good signal diversity; near-zero = all rollouts winning or all losing |
| `actor/kl_divergence` | GRPO | Should stay low; spike means policy drifting too far from reference |
| `actor/pg_loss` | GRPO | Policy gradient loss; should decrease over epochs |
| `val_0/reward_mean` before epoch 1 | baseline | Tells you where the base model starts (from `trainer.val_before_train: true`) |

**Healthy training signal:** `val_0` and `val_1` both rising. `actor/reward_std > 0` (not all rollouts tied). `kl_divergence` stable and small.

**Problem signals and fixes:**

| Signal | Likely cause | Fix |
|---|---|---|
| `val_1/reward_mean` near 0, `val_0` rising | DeepMath truncation | Increase `data.max_response_length` to `8192` |
| Both val metrics flat for 2+ epochs | All rollouts winning or losing | Check `actor/reward_std`; if near 0, training data is too easy or too hard |
| `actor/kl_divergence` spike | Policy diverging | Increase `kl_loss_coef` from `0.001` to `0.01` |
| W&B run missing | `WANDB_API_KEY` not set | Set in login script before `sbatch` |

Per-domain breakdown is also available offline from the rollout JSON files saved during training. Each record at `experiments/results/training/<run>/rollout_data/val/idx_*/rollout_*.json` contains a `data_source` field.

---

## 7. Things to Watch

### Thinking traces and the response budget

With `THINKING_MODE: ORCHESTRATOR_ONLY`, Qwen3-8B generates a `<think>...</think>` block before every action. On NQ/HotpotQA and DeepMath training questions these are typically 300–800 tokens (simple questions) to ~1500 tokens (harder DeepMath). Combined with tool calls and synthesis, most rollouts fit within the 4096 token budget.

**Risk:** the hard tail of DeepMath. If the model thinks extensively before dispatching code, total tokens can approach or exceed 4096. When `data.truncation: 'truncate'` fires, the answer token is lost → reward = 0. Spurious zeros on DeepMath questions push the model away from long thinking traces.

**Detection:** watch `val_1/reward_mean` in epoch 1. If it stays near zero while `val_0` and `actor/reward_mean` are rising, truncation is likely. **Fix:** change `data.max_response_length` from `4096` to `8192` and relaunch.

### OOM on 4×A100

If the job crashes with CUDA OOM, try in order:
1. Reduce `actor_rollout_ref.rollout.gpu_memory_utilization` from `0.6` to `0.55`
2. Halve `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` from `4` to `2`
3. Halve `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` from `4` to `2`

### VERL startup time

vLLM loading Qwen3-8B typically takes 45–90 seconds. The SLURM job sleeps 60 seconds before starting rollout workers. If you see rollout workers failing to connect, check the VERL log (`ft_<jobid>_verl.log`) and increase the sleep in `jobs/010_ft_orchestrator.job` to 120 seconds.

### Search API rate limits

With `N_WORKERS: 1` and `data.train_batch_size: 32`, up to 32 concurrent search calls can fire per training step. Serper's free tier is 2,500 queries/month. Budget accordingly, or set `TOOL_STEPS: 3` to reduce search volume without changing the training objective meaningfully.

---

## 8. Merging LoRA and Evaluating

After training, the checkpoint contains only the LoRA adapter weights (~130 MB). Merge them into the base model before running inference:

```bash
conda activate cosmas-train

RUN="qwen3-8b-grpo-search-math"
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path  experiments/results/training/${RUN}/checkpoint_best/actor/lora_weights.pt \
    --output_dir experiments/results/training/${RUN}/merged_model/
```

The merged model is a standard HuggingFace checkpoint. Use it in any existing experiment config:

```yaml
# experiments/configs/qwen3/agentflow/qwen3_8b_ft_gaia.yaml
models:
  orchestrator:
    name: "Qwen3-8B-FT"
    family: "qwen3"
    path_or_id: "/path/to/experiments/results/training/qwen3-8b-grpo-search-math/merged_model/"
    role: "orchestrator"
# everything else unchanged
```

Run evaluation exactly as for any other model:
```bash
python scripts/run_experiment.py --config experiments/configs/qwen3/agentflow/qwen3_8b_ft_gaia.yaml
python scripts/analyze_results.py experiments/results/<run>/raw_results.json --by-level --tools
```

---

## 9. Troubleshooting

| Problem | Diagnostic | Fix |
|---|---|---|
| `EnvironmentError: SERPER_API_KEY must be set` | Missing search API key | Export `SERPER_API_KEY` before launching rollout workers |
| `EnvironmentError: SUBAGENT_ENDPOINT must be set` | Frozen sub-agent server not started | Start `vllm serve Qwen/Qwen3-1.7B --port 9998` and export `SUBAGENT_ENDPOINT=http://localhost:9998/v1` |
| Rollout workers fail to connect to VERL | VERL not ready yet | Increase sleep in job script from 60 to 120s; check `*_verl.log` |
| `ModuleNotFoundError: agentflow` | Wrong conda env | `source activate cosmas-train` |
| `ModuleNotFoundError: verl` | Wrong vLLM version env | Inference env (`agent_engine`) doesn't have verl — use `cosmas-train` for training |
| Parquet schema error in rollout | Stale data files | Re-run job 008 with the current `prepare.py` |
| `val_combined.parquet not found` | Old config pointing at old file | `data.val_files` now uses `val_search.parquet` + `val_deepmath.parquet`; `val_combined.parquet` is offline only |
| Checkpoint not found after training | Training crashed before `save_freq` | Lower `trainer.save_freq` to `1`; check error log |
| W&B shows only one val metric | Old VERL version | VERL 0.5.0 required; check `pip show verl` in training env |

---

## 10. Design Decisions

### Why GRPO and not PPO?

GRPO has no value network, which reduces memory and simplifies the training loop. AgentFlow uses it; we replicate the setup exactly to minimize deviation from a tested baseline.

### Why Search-R1 + DeepMath?

These are the two domains where tool use is demonstrably necessary:
- **NQ / HotpotQA** (Search-R1): specific entity-level facts the model doesn't reliably hold in memory. Direct reasoning → reward = 0 on most questions. GRPO pushes the model toward `web_search`.
- **DeepMath**: competition math with exact numerical answers. Arithmetic drift in natural-language reasoning → reward = 0. GRPO pushes toward `code_generator`.

Both directly target the dominant failure mode (direct reasoning without action) in the domains where it is correctable. GPQA/HLE expert-science failures are structurally different (web search doesn't help on google-proof questions) and are not the primary training target.

### Why do sub-agents use a separate frozen server rather than sharing the VERL endpoint?

**Train/eval consistency.** At evaluation time, sub-agents run the base `Qwen/Qwen3-1.7B` model through the standard `VLLMProvider`. If sub-agents instead shared the VERL endpoint during training, they would call the evolving actor snapshot at each step, meaning the orchestrator is trained against an unstable, constantly-shifting tool interface. By locking sub-agents to a separate frozen server, the tool interface the orchestrator trains against is identical to the tool interface it sees at eval time — the base model behaviour.

**Memory efficiency.** A 1.7B frozen sub-agent server uses ~3–4 GB of GPU memory (with `--gpu-memory-utilization 0.15` on a 40 GB A100). This fits alongside the VERL actor/rollout stack for 8B without OOM. Using the full 8B VERL endpoint would not add any quality benefit for the narrow sub-agent tasks (retrieve-and-summarise, write-and-execute).

**Why not replicate AgentFlow's shared-endpoint design?**
AgentFlow's `vllm-local-<BASE_MODEL>` points all agents (Planner, Verifier, Executor, Coder) to the same VERL vLLM. This creates a train/eval mismatch for sub-agents: at eval time they use the base model; during training they used the evolving snapshot. The msc-thesis design eliminates this mismatch by freezing sub-agents at a fixed checkpoint from day one.

### Why is thinking enabled during training?

Train/eval consistency. The thesis results show `ORCHESTRATOR_ONLY` thinking is the dominant performance driver; the fine-tuned model will be evaluated with thinking enabled. Training without it creates a distribution mismatch.

More importantly: with thinking enabled, the "direct reasoning without action" failure appears in training rollouts. The model reasons in its `<think>` trace, reaches a confident-but-wrong answer, skips the tool call, and gets reward = 0. GRPO can push it to dispatch a tool instead. Without thinking, this failure pattern is invisible to the gradient.

### Why two separate val files?

A single combined val file would report one `val/reward_mean` mixing search accuracy (qa mode) and math accuracy (gen mode). Using two separate files gives `val_0/reward_mean` and `val_1/reward_mean` in W&B, making it possible to see whether *both* tool types are being learned or just one. `val_combined.parquet` is kept for offline analysis only.

### Why a separate test split?

The val split is used for checkpoint selection (choosing which epoch to keep). Using the same data for both selection and final reporting would inflate the reported numbers — the selected checkpoint is the one that happened to score best on that held-out set. The test split is never seen by the training loop or the checkpoint selector; it is only used once, for final metric reporting after the best checkpoint is chosen. All three splits share the same source proportions so test statistics are representative of what the model was trained on.

### Why not AIME for validation?

AIME is one of the five evaluation benchmarks reported in the thesis. Using it for checkpoint selection (choosing which epoch to keep) would introduce selection bias — the best checkpoint would be chosen based on the same questions you later report results on. Held-out DeepMath questions are in-distribution with the math training data, never appear in the evaluation suite, and give an honest learning curve.
