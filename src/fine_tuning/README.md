# RL Fine-Tuning Pipeline

GRPO-based reinforcement learning pipeline for fine-tuning the orchestrator model (**Qwen3-8B**).
Only the orchestrator is trained; sub-agents run on a **separate, frozen** vLLM server and are
never updated, so the tool interface seen during training is identical to the one seen at evaluation.

**Motivation:** Failure analysis across 2,534 MAS failures identifies *direct reasoning without action*
as the dominant failure mode. The orchestrator answers from parametric knowledge instead of
delegating to a sub-agent. GRPO on retrieval-intensive (Search-R1 / HotpotQA+NQ) and math-intensive
(DeepMath) training data *hopefully* creates pressure toward tool use — tool-less rollouts lose reward,
tool-using rollouts win.

Full failure-mode analysis: `docs/failure_modes_fine_tuning_alignment.md`

---

## TL;DR

| What | Detail |
|---|---|
| **Model trained** | Qwen3-8B orchestrator only (sub-agents frozen at Qwen3-1.7B) |
| **Method** | Flow GRPO — final reward propagated to every turn in the trajectory (planning + tool calls + synthesis) |
| **Reward** | Binary: 1.0 if correct, 0.0 otherwise (same `evaluate_answer()` as benchmark eval) |
| **Training data** | 1800 questions: 900 Search-R1 (HotpotQA 85% + NQ 15%) + 900 DeepMath (difficulty ≥ 3) |
| **Val / Test** | 50-row val (20 search + 10 math + 20 AIME) for checkpoint selection; 200-row test held out for final reporting |
| **LoRA** | Rank 64, alpha 64, all-linear; adapter ~250–500 MB vs ~16 GB full FT |
| **Effective LR** | `1e-5` (set by `launch_verl.py`; config.yaml shows `1e-6` which is the full-FT baseline) |
| **Hardware** | 4 × H100 NVL (94 GB each): GPU 0 hosts frozen sub-agent + VERL, GPUs 1–3 VERL only |
| **Rollouts** | 8 per question during training (GRPO group); 1 per question during validation (greedy) |
| **Validation** | Every 10 steps on `val_combined.parquet` (50 rows); `best_checkpoint/` symlink updated when val reward improves |
| **Checkpoints** | Only **latest** and **best** adapter dirs kept; all others deleted asynchronously after rotation |
| **Run time** | 2 epochs, ~112 steps; SLURM budget 48 h (observed ~20 min/step → ~40 h end-to-end) |
| **Launch** | `sbatch jobs/fine_tuning/005_train.job` (after smoke test passes: `004_smoke_8b.job`) |
| **Merge & eval** | `python scripts/merge_lora.py --checkpoint <best_checkpoint> --base-model Qwen/Qwen3-8B --output-dir <path>` |

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Training Data](#3-training-data)
4. [Running Training](#4-running-training)
5. [Reward Design: Flow GRPO](#5-reward-design-flow-grpo)
6. [Training Configuration](#6-training-configuration)
7. [GPU Allocation](#7-gpu-allocation)
8. [W&B Metrics](#8-wb-metrics)
9. [Checkpoint Layout](#9-checkpoint-layout)
10. [Merge LoRA and Evaluate](#10-merge-lora-and-evaluate)
11. [Troubleshooting](#11-troubleshooting)
12. [Design Decisions](#12-design-decisions)

---

## 1. Architecture

```
Training time
────────────────────────────────────────────────────────────────

          ┌──────────────────────────────────────────────────┐
          │  agentflow.verl  (VERL 0.7.1 backend, Ray)       │
          │                                                  │
          │  AgentFlowTrainer(RayPPOTrainer)                 │
          │  ├── GRPO advantage estimator (n=8 rollouts)     │
          │  ├── FSDP actor: Qwen3-8B + LoRA rank-64         │
          │  ├── Reference policy: separate ref worker       │
          │  │   (ref_in_actor not in effect; KL via         │
          │  │    disable_adapter() on ref worker)           │
          │  └── AgentModeDaemon  :9999                      │
          └──────────────┬───────────────────────────────────┘
                         │  HTTP  (tasks ↓ / rewards ↑)
          ┌──────────────▼───────────────────────────────────┐
          │  OrchestratorRollout (LitAgent)                  │
          │  ├── AgenticOrchestrator  ← model being trained  │
          │  │     thinking_mode: NO  (current config;       │
          │  │     ORCHESTRATOR_ONLY recommended - see §12)  │
          │  ├── WebSearchTool  → sub-agent LLM @ :9998      │
          │  ├── CodeGeneratorTool → sub-agent LLM @ :9998   │
          │  │   (sub-agents: Qwen3-1.7B, frozen server,     │
          │  │    never updated during training)             │
          │  └── OrchestratorReward  ← binary via metrics.py │
          └──────────────────────────────────────────────────┘

Inference (unchanged after training)
────────────────────────────────────────────────────────────────

  merge lora_adapter/ → merged HF model → path_or_id in YAML
                                           ↓
                                     VLLMProvider (no changes)
```

GRPO optimises only the orchestrator (LoRA updates on actor). Sub-agent token generations are
treated as environment interactions - they never enter the GRPO objective. After training, merge
the LoRA adapter and point any existing experiment YAML at the merged model path; tools, evaluators,
and run scripts are untouched.

---

## 2. Prerequisites

### Conda environments

Two separate environments are required. **Never mix them.**

| Environment | Purpose | Key packages |
|---|---|---|
| `cosmas-train` | Training: VERL, rollout workers, AgentFlow | VERL 0.7.1, vLLM 0.17.0, Python 3.12 |
| `agent_engine` | Inference and evaluation | vLLM 0.12.0 |

The split is a hard constraint: VERL 0.7.1 requires vLLM 0.17.0; the inference stack pins 0.12.0.

Create the training environment (Snellius):
```bash
sbatch jobs/fine_tuning/000_create_environment.job
# or locally:
conda env create -f jobs/fine_tuning/environment_train.yml
```

### AgentFlow

AgentFlow is vendored into this repo at `src/fine_tuning/agentflow/`. No external clone is needed -
it is installed as part of the project when you run:
```bash
pip install -e ".[training]"
```
The `000_create_environment.job` does this automatically.

### Environment variables

Set in your Snellius login script (`~/.bashrc`) or in the SLURM job before launching:

| Variable | Required for | Notes |
|---|---|---|
| `SERPER_API_KEY` or `TAVILY_API_KEY` | Rollout workers (every training step) | Missing → immediate `EnvironmentError` |
| `SUBAGENT_ENDPOINT` | Rollout workers | Set after starting the frozen sub-agent server (default `http://localhost:9998/v1`) |
| `WANDB_API_KEY` | W&B logging | Optional but strongly recommended |
| `HF_TOKEN` | Gated HuggingFace datasets | Required for DeepMath download |

---

## 3. Training Data

### Data composition

| Split | Search-R1 rows | DeepMath rows | AIME rows | Total | Purpose |
|---|---|---|---|---|---|
| Train | 900 (85% HotpotQA / 15% NQ) | 900 (difficulty ≥ 3) | - | 1800 | GRPO training |
| Val | 20 | 10 | 20 | 50 | Checkpoint selection (VERL reads `val_combined.parquet`) |
| Test | 100 | 100 | - | 200 | Final reporting only, never used during training |

**Why this mix:**
- **HotpotQA (85 %)** requires multi-hop evidence aggregation → strong retrieval-policy signal.
- **NQ (15 %)** adds single-hop diversity; higher share dilutes the multi-hop signal.
- **DeepMath difficulty ≥ 3** produces cleaner GRPO signal; easy problems resolve in one step with near-zero gradient.
- **AIME in val** gives an early-warning signal for AIME-flavoured regressions during training. The AIME val sample must remain disjoint from the held-out AIME eval set used for final reporting.

**Contamination guarantee:** rows are carved in order - test first, then val, then train. Zero cross-split overlap.

### Preparing the data

**SLURM (recommended):**
```bash
sbatch jobs/fine_tuning/001_prepare_data.job
```

**Locally:**
```bash
conda activate cosmas-train
python src/fine_tuning/data/prepare.py \
    --n-search 900 --n-math 900 \
    --n-val-search 20 --n-val-math 10 --n-val-aime 20 \
    --aime-jsonl-path data/AIME/train.jsonl \
    --n-test-search 100 --n-test-math 100 \
    --search-source both --hotpot-ratio 0.85 \
    --deepmath-min-difficulty 3 \
    --output-dir data/training --seed 42
```

Files written to `data/training/`:
```
train/combined_train.parquet      1800 rows  (shuffled, 900 search + 900 math)
val/val_search.parquet              20 rows  NQ + HotpotQA - offline reference
val/val_deepmath.parquet            10 rows  DeepMath (difficulty ≥ 3) - offline reference
val/val_aime.parquet                20 rows  sampled from data/AIME/train.jsonl - offline reference
val/val_combined.parquet            50 rows  all three merged ← VERL reads this
test/test_search.parquet           100 rows  held-out Search-R1
test/test_deepmath.parquet         100 rows  held-out DeepMath (difficulty ≥ 3)
test/test_combined.parquet         200 rows  both merged - final reporting only
```

<!-- **Note on Search-R1 reproducibility:** `_download_search_r1()` uses HuggingFace streaming mode with
reservoir-buffer shuffle. The same `--seed` produces a *similar* but not bit-for-bit identical subset
across runs (shard download order can vary). `DeepMath` uses non-streaming shuffle and is fully
reproducible. -->

---

## 4. Running Training

### Step 1 - Smoke test (run before the full production run)

Verifies the full pipeline end-to-end: **Qwen3-8B** (same model as production), LoRA, 8 training
samples, 2 rollouts, 1 epoch, on **3 H100 GPUs** (1 sub-agent + 2 VERL).

```bash
sbatch jobs/fine_tuning/004_smoke_8b.job
```

Pre-flight checks run automatically (imports, reward routing, parquet schema). A passing smoke test
prints `ALL 9 checks passed` and completes 2 gradient steps with a checkpoint saved.

### Step 2 - Full training run

```bash
sbatch jobs/fine_tuning/005_train.job
```

**2 epochs**, 1800 training questions, n=8 rollouts, **4 H100 GPUs** (1 sub-agent + 4 VERL), step-based
checkpointing every 10 steps. SLURM budget **48 h**. From a 1-hour calibration run (`ft_23005301`) one
training step took **~20 min** (1216 s; `timing_s/gen` ≈ 1091 s — generation dominates at ~90%). With
~112 steps (56 steps/epoch × 2 epochs) the end-to-end estimate is **~40 h**, leaving ~8 h of headroom in
the 48 h budget for bootstrap (~15 min), `val_before_train` (~5–10 min), 11 intermediate validations,
and 11 checkpoint saves. If a run spills the budget, the LoRA mid-run resume path is verified — resubmit
with `trainer.resume_mode=resume_path`, `trainer.resume_from_path=<…>/latest_checkpoint`, and
`lora.resume_adapter_path=<…>/latest_checkpoint/actor/lora_adapter` (see `004_smoke_8b_load.job`).

**Manual launch** (three terminals, after `conda activate cosmas-train` in all):
```bash
# Terminal 1 - frozen sub-agent server (start first, GPU 0 only)
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B \
    --port 9998 --tensor-parallel-size 1 --gpu-memory-utilization 0.08 --max-model-len 16384
export SUBAGENT_ENDPOINT=http://localhost:9998/v1

# Terminal 2 - VERL server (after sub-agent is up)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/launch_verl.py \
    --config experiments/configs/fine_tuning/config.yaml

# Terminal 3 - rollout workers (after VERL vLLM is ready, ~60–90 s)
python scripts/train_orchestrator.py --config experiments/configs/fine_tuning/config.yaml
```

### Job reference

| Job | GPUs | Purpose |
|---|---|---|
| `jobs/fine_tuning/000_create_environment.job` | CPU | Create `cosmas-train` conda env |
| `jobs/fine_tuning/001_prepare_data.job` | CPU | Download datasets, write parquet files |
| `jobs/fine_tuning/002_inspect_data.job` | CPU | Verify parquet schema and row counts |
| `jobs/fine_tuning/003_smoke_4b.job` | 2 H100 | Smoke test with Qwen3-4B (fast sanity check) |
| `jobs/fine_tuning/004_smoke_8b.job` | 3 H100 | Smoke test with Qwen3-8B (production code path) |
| `jobs/fine_tuning/005_train.job` | 4 H100 | Full 5-epoch training run |

---

## 5. Reward Design: Flow GRPO

### What Flow GRPO does

The orchestrator produces a **multi-turn trajectory** per rollout: a planning turn, one or
more tool-call turns, and a final synthesis turn. VERL captures each turn as a `Triplet`
(prompt token IDs, response token IDs, reward).

**Flow GRPO assigns the same final sparse reward to every triplet in the trajectory.**
This is critical because:

- The **planning step** is where the "direct reasoning without action" failure occurs - the model
  decides whether to call a tool at all. Without gradient signal on the planning triplet, training
  cannot reinforce correct tool-dispatch decisions.
- The **tool-call formulation** (which tool, which query) is also part of the policy. Reward only
  on the synthesis step lets the model get credit for stumbling to a good answer via a bad tool call.
- All triplets from the same rollout share the same `uid`, so GRPO advantage normalisation
  (within the n=8 group) is consistent across all turns.

GRPO advantage normalisation itself is unchanged - Flow GRPO only affects *which turns receive
the reward*, not how advantages are computed.

### Reward function

`OrchestratorReward` (`reward.py`) returns **binary**: **1.0** if correct, **0.0** otherwise.
It calls `evaluate_answer(prediction, ground_truth)` from `agent_engine/datasets/evaluators/metrics.py`.

- `ground_truth` is `task["result"]`, which is the **first** golden answer from the dataset's
  `golden_answers` list.
- NQ and HotpotQA questions sometimes have 2–4 valid aliases. A prediction matching a non-first
  alias scores 0.0 at training time but 1.0 at evaluation time (where all aliases are checked).
  Effect is minor - `evaluate_answer` uses containment matching and most aliases are covered.

### Answer format

The thesis inference stack uses `\boxed{ANSWER}` as the final-answer format (parsed by
`extract_answer()` in `parsing.py`). AgentFlow's rollout code appends `<answer>...</answer>` tags
- this suffix is **not injected** in `_build_rollout_question()`. The reward function calls
`extract_answer()` which looks for `\boxed{}`, so injecting `<answer>` tags would cause reward = 0.

---

## 6. Training Configuration

All configuration lives in `experiments/configs/fine_tuning/config.yaml`.
`launch_verl.py` reads it, resolves the `env:` block, and forwards `python_args` to VERL
as Hydra overrides. **When `USE_LORA=true`, `launch_verl.py` additionally overrides several
defaults** - see the LoRA overrides column.

### `env:` block

| Key | Production value | Notes |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | HuggingFace model ID or local path |
| `SUBAGENT_MODEL` | `Qwen/Qwen3-1.7B` | Frozen sub-agent (separate server, port 9998) |
| `N_GPUS` | `4` | VERL GPU count (CUDA_VISIBLE_DEVICES=0,1,2,3) |
| `ROLLOUT_TP_SIZE` | `1` | Rollout vLLM tensor parallelism |
| `EXPERIMENT_NAME` | `qwen3-8b-grpo-search-math` | Checkpoint dir name + W&B run name |
| `PROJECT_NAME` | `msc-thesis-fine-tuning` | W&B project |
| `BASE_DATA_DIR` | `data/training` | Root of train/val parquet files |
| `ENABLE_TOOLS` | `["web_search", "code_generator"]` | Tools available during rollout |
| `TOOL_STEPS` | `5` | Max tool calls per rollout episode |
| `THINKING_MODE` | `NO` | **Currently set to NO.** See §12 for why `ORCHESTRATOR_ONLY` is recommended. |
| `TRAIN_TEMPERATURE` | `0.7` | Sampling temperature for training rollouts |
| `TEST_TEMPERATURE` | `0.0` | Greedy decoding for validation |
| `N_WORKERS` | `4` | Parallel rollout worker processes |
| `USE_LORA` | `"true"` | Default is LoRA; full-FT requires explicit `"false"` |
| `USE_SCRATCH_CHECKPOINTS` | `"true"` | LoRA adapters → `/scratch-shared/$USER/fine_tuning/lora_adapters/` |
| `SAVE_OPTIMIZER` | `"true"` | Save Adam moments + LR-scheduler state (tiny for LoRA; enables resume) |

### Key `python_args:` (selected; see config.yaml for full list)

| Key | Production value | LoRA override (launch_verl.py) | Notes |
|---|---|---|---|
| `data.train_batch_size` | `32` | - | Questions per training step |
| `data.train_max_samples` | `1800` | - | Full training split per epoch |
| `data.max_prompt_length` | `16384` | - | Bump to 20480 if `prompt_length/clip_ratio > 0` in VERL logs |
| `data.max_response_length` | `2048` | - | Bump to 3072 if switching to `THINKING_MODE: ORCHESTRATOR_ONLY` |
| `actor_rollout_ref.rollout.n` | `8` | - | GRPO group size (rollouts per question) |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | **`1e-5`** | LoRA trains ~1% of params; 10× higher LR is standard |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.01` | - | KL penalty; scale proportionally if LR is increased |
| `actor_rollout_ref.actor.clip_ratio_low/high` | `0.2 / 0.3` | - | PPO clip range |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.45` | **`0.70`** | LoRA optimizer savings (~24 GB/GPU vs full-FT) free headroom for a larger KV pool and higher rollout throughput |
| `actor_rollout_ref.rollout.max_model_len` | `18432` | - | 16384 prompt + 2048 response |
| `actor_rollout_ref.rollout.max_num_seqs` | `128` | - | Decode parallelism cap |
| `actor_rollout_ref.rollout.free_cache_engine` | `false` | **`true`** | LoRA requires KV flush between rollout and training (adapter swap invalidates cached prefixes) |
| `actor_rollout_ref.actor.use_dynamic_bsz` | - | **`true`** | Pack sequences up to `ppo_max_token_len_per_gpu` |
| `trainer.total_epochs` | `2` | - | Reduced from 5 to fit 48 h walltime at observed ~20 min/step |
| `trainer.save_freq` | `10` | - | Step-based (~6 saves/epoch) |
| `trainer.test_freq` | `10` | - | Step-based (~6 evals/epoch) |
| `trainer.val_before_train` | `true` | - | Baseline measurement at step 0 |

### LoRA parameters (`lora:` block)

| Key | Value | Notes |
|---|---|---|
| `rank` | `64` | ~130 MB adapter vs ~16 GB full-FT weights |
| `alpha` | `64` | alpha = rank → scaling factor 1.0 (neutral initialisation) |
| `target_modules` | `all-linear` | Applies LoRA to all linear layers |
| `resume_adapter_path` | _(unset)_ | Set to a saved `lora_adapter/` path for warm restarts |

**LoRA learning rate note:** 1e-6 in config.yaml is for full-FT. `launch_verl.py` overrides to
**1e-5** when `USE_LORA=true`. This is the actual LR used during training.

### Smoke test differences (`config_smoke8b.yaml`)

| Parameter | Production (`config.yaml`) | Smoke 8B (`config_smoke8b.yaml`) |
|---|---|---|
| `N_GPUS` | 4 | 2 |
| `data.train_max_samples` | 1800 | 8 |
| `actor_rollout_ref.rollout.n` | 8 | 2 |
| `trainer.total_epochs` | 2 | 1 |
| `TOOL_STEPS` | 5 | 2 |
| `data.max_prompt_length` | 16384 | 4096 |
| `trainer.save_freq / test_freq` | 10 | 1 |
| `USE_SCRATCH_CHECKPOINTS` | `"true"` | `"false"` (adapters fit in GPFS home) |
| `N_WORKERS` | 4 | 1 |
| `BASE_DATA_DIR` | `data/training` | `data/training/smoke` |

### Max turns: training vs evaluation

| Setting | Training (`OrchestratorRollout`) | Evaluation (`run_experiment.py`) |
|---|---|---|
| `max_turns` | 5 | 15 (from config YAML) |
| `max_tokens` (sub-agent) | 2048 | 8192 (from config) |
| Temperature (orchestrator) | 0.7 train / 0.0 val | 0.0 greedy |
| Planning turn | enabled | enabled (unless `baseline: true`) |

Fewer turns during training keeps rollouts short and reduces token-budget overflow risk. Training
questions (NQ, HotpotQA, DeepMath) typically resolve in 1–3 tool calls; the distribution shift
at eval time (more turns available) does not harm the trained policy.

---

## 7. GPU Allocation

Training uses **4 × H100 NVL GPUs (~94 GB each)**.

### Memory envelope

| Component | GPU 0 | GPUs 1–3 |
|---|---|---|
| Frozen sub-agent (Qwen3-1.7B, util=0.08) | ~7.5 GB | - |
| VERL vLLM (Qwen3-8B model, util=0.70) | ~66 GB KV+model | ~66 GB |
| FSDP actor shard (LoRA) | ~4 GB base + <1 GB adapter | ~4 GB + <1 GB |
| Ref shard (separate ref worker) | ~4 GB | ~4 GB |
| Activations + misc | ~8 GB | ~8 GB |
| **Total** | **~93.5 GB / 94 GB (0.5 GB headroom)** | **~86 GB / 94 GB** |

GPU 0 is the tight one — it hosts both the sub-agent and VERL, with only ~0.5 GB headroom. Watch
`out/fine_tuning/orchestrator_ft/ft_<jobid>_gpu.csv` (the `nvidia-smi` sidecar started by
`005_train.job`) during the first 10 minutes. If GPU 0 memory exceeds 92 GB, drop sub-agent
`--gpu-memory-utilization` from `0.08` to `0.06`.

### Throughput tuning

The training loop is rollout-bound: ~60 % of each step is `timing_s/gen` (HTTP → VERL vLLM,
HTTP → sub-agent, Serper API). The knobs below maximise GPU utilisation.

| Knob | Production value | Rationale |
|---|---|---|
| `N_WORKERS` | `4` | Fills vLLM's continuous batcher; single-worker serialised 256 in-flight episodes through one Python client |
| `gpu_memory_utilization` (LoRA) | `0.70` | LoRA frees ~28 GB vs full-FT; growing the KV pool lifts rollout throughput |
| `max_num_seqs` | `128` | Decode parallelism cap; paged KV means this doesn't multiply by `max_model_len` |
| Sub-agent `--gpu-memory-utilization` | `0.08` | ~7.5 GB total (~4 GB KV) for the frozen 1.7B server; GPU 0 budget is tight (0.5 GB headroom) — drop to 0.06 if GPU 0 > 92 GB |

### Health-check metrics (first production run)

These should stay at zero. Non-zero means the cap is too tight and silently corrupts the GRPO signal:

| Metric (in VERL log) | Non-zero fix |
|---|---|
| `prompt_length/clip_ratio` | Bump `data.max_prompt_length`: 16384 → 20480 |
| `response_length/clip_ratio` | Bump `data.max_response_length`: 2048 → 3072 |
| `n_dropped_sample_because_of_prompt` | Same as above (prompt cap) |
| `n_trunc_sample_because_of_response` | Same as above (response cap) |
| `n_dropped_sample_because_of_mini_batch` | Tune `ppo_max_token_len_per_gpu` (LoRA override: 45056) |

---

## 8. W&B Metrics

VERL logs to project `msc-thesis-fine-tuning` under run name `qwen3-8b-grpo-search-math`.
`val_before_train: true` runs a validation pass at step 0, giving a baseline before any gradient update.
Validation and checkpoint saves happen every 10 steps (step-based mode, not epoch boundaries).

### Key metrics

| Metric | Source | What it tells you |
|---|---|---|
| `val/reward` | `val_combined.parquet` (50 rows: 20 search + 10 math + 20 AIME) | Main checkpoint-selection signal; used by `_rotate_checkpoints` to maintain `best_checkpoint/` symlink |
| `actor/reward_mean` | Training rollouts | Mean reward across both domains per step |
| `actor/reward_std` | Training rollouts | Near-zero → all rollouts tied (bad); should stay > 0 |
| `actor/kl_divergence` | GRPO | Should stay small; spike = policy drifting from reference |
| `actor/pg_loss` | GRPO | Policy gradient loss; should decrease over epochs |

**Note:** `actor/reward_mean` is a combined average - no per-domain breakdown in W&B for training
rollouts. Per-domain breakdown requires offline analysis of rollout JSONs (see below).

### Rollout JSONs (disk)

Every episode is saved to:
```
experiments/results/fine_tuning/<experiment>/<run-tag>/rollout_data/train|val/idx_<N>/rollout_<uuid8>.json
```

Each record:
```json
{
  "idx": 42,
  "rollout_id": "...",
  "data_source": "hotpotqa",
  "question": "...",
  "groundtruth": "...",
  "answer_extracted": "...",
  "reward": 1.0,
  "output_messages": [...],
  "timestamp": "2026-05-11T..."
}
```

Metrics you can compute offline from the JSONs:

| Plot / metric | How |
|---|---|
| Reward by domain per epoch | Group by `data_source` + epoch; mean `reward` |
| Reward distribution histogram | Histogram of `reward` over all 8 rollouts per question |
| Tool call counts per type | Count tool-call messages in `output_messages` |
| Average turns to solution | Count assistant turns in `output_messages` |
| Thinking trace length (tokens) | Extract `<think>...</think>` blocks from assistant messages |
| Reward by DeepMath difficulty | Join on `extra_info.difficulty` from the parquet |
| Pass@k curves | k ∈ {1, 2, 4, 8} - fraction of questions with ≥ 1 correct rollout |

**No analysis script yet.** When writing plots, create `scripts/plots/ft_rollout_analysis.py`
(pattern: `scripts/plots/efficiency_plots.py`).

### Problem signals and fixes

| Signal | Likely cause | Fix |
|---|---|---|
| `val/reward` flat for 2+ epochs | All rollouts winning or losing | Check `actor/reward_std`; if near 0, training data may be too easy or too hard |
| `actor/kl_divergence` spike | Policy diverging | Scale `kl_loss_coef` up proportionally (e.g. 0.05 at lr=1e-5) |
| DeepMath rollouts drop to reward=0 while search improves | Response truncation | Increase `data.max_response_length` to `8192` |
| W&B run missing | `WANDB_API_KEY` not set | Set in login script before `sbatch` |

---

## 9. Checkpoint Layout

### Paths

| Config | Checkpoint base |
|---|---|
| `config.yaml` (`USE_SCRATCH_CHECKPOINTS: "true"`, LoRA) | `/scratch-shared/$USER/fine_tuning/lora_adapters/<experiment>/<run-tag>/` |
| `config_smoke8b.yaml` (`USE_SCRATCH_CHECKPOINTS: "false"`) | `experiments/results/fine_tuning/<experiment>/<run-tag>/` |

The run tag `<DD-MM-YYYY_HH-MM-SLURM_JOB_ID>` is printed at VERL startup and written to log files.
Rollout JSONs always land in `experiments/results/fine_tuning/<experiment>/<run-tag>/rollout_data/`.

### Checkpoint directory tree (LoRA run, verl 0.7.1)

```
<ckpt_base>/qwen3-8b-grpo-search-math/<run-tag>/
│
├── latest_checkpointed_iteration.txt   # Last saved global step (e.g. "10").
│                                       # VERL reads this to auto-resume.
│
├── latest_checkpoint -> global_step_10/  # Symlink → most recently saved step.
│
├── best_checkpoint -> global_step_20/    # Symlink → step with highest val/reward so far.
│                                         # Updated only when a fresh val improves on the best.
│
├── best_checkpoint_info.json             # {"epoch": 2, "step": 20, "val_reward": 0.47}
│
└── global_step_<N>/
    ├── data.pt                         # Dataloader state (RNG + sampler position for exact resume)
    └── actor/
        ├── lora_adapter/               # LoRA adapter (HF PEFT format, ~250–500 MB)
        │   ├── adapter_model.safetensors
        │   └── adapter_config.json
        ├── optim_world_size_*_rank_*.pt    # Adam optimizer state (tiny for LoRA, ~10s of MB)
        ├── extra_state_world_size_*_rank_*.pt  # LR scheduler + RNG state
        ├── fsdp_config.json
        └── huggingface/                # HF tokenizer (always saved)
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── ...
```

**Note:** For LoRA runs, VERL does **not** write `model_world_size_*_rank_*.pt` (the full model
state dict). The adapter delta is in `lora_adapter/`. Only the two checkpoint dirs referenced by
`latest_checkpoint/` and `best_checkpoint/` are retained - older dirs are deleted asynchronously
after rotation.

### What to keep vs. discard

| File | Keep for inference | Keep for resuming training |
|---|---|---|
| `lora_adapter/` | Yes (merge input) | Yes |
| `optim_world_size_*_rank_*.pt` | No | Yes |
| `extra_state_*_rank_*.pt` | No | Yes |
| `fsdp_config.json` | No | Yes |
| `huggingface/` | Yes (tokenizer) | Yes |
| `data.pt` | No | Yes |
| `latest_checkpointed_iteration.txt` | No | Yes |

### Resuming training

Leave `trainer.resume_from_path` unset - VERL auto-resumes from `latest_checkpointed_iteration.txt`.
For LoRA warm-restart from a specific adapter (e.g. `best_checkpoint/`), set `lora.resume_adapter_path`
in `config.yaml` to the resolved `<ckpt_dir>/actor/lora_adapter/` path.

---

## 10. Merge LoRA and Evaluate

After training, the checkpoint contains only the LoRA adapter delta (`actor/lora_adapter/`).
Merge it into the base model before running inference:

```bash
conda activate cosmas-train

# Production run (USE_SCRATCH_CHECKPOINTS=true):
RUN_TAG="<DD-MM-YYYY_HH-MM-JOBID>"   # printed by launch_verl.py at startup
CKPT="/scratch-shared/${USER}/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math/${RUN_TAG}/best_checkpoint"

python scripts/merge_lora.py \
    --checkpoint "${CKPT}" \
    --base-model Qwen/Qwen3-8B \
    --output-dir "/scratch-shared/${USER}/fine_tuning/merged_models/qwen3-8b-grpo-${RUN_TAG}/"

# Smoke run (USE_SCRATCH_CHECKPOINTS=false):
CKPT="experiments/results/fine_tuning/qwen3-8b-grpo-smoke/${RUN_TAG}/best_checkpoint"

python scripts/merge_lora.py \
    --checkpoint "${CKPT}" \
    --base-model Qwen/Qwen3-8B \
    --output-dir "experiments/results/fine_tuning/qwen3-8b-grpo-smoke/${RUN_TAG}/merged_model/"
```

`scripts/merge_lora.py` reads hyperparameters (rank, alpha, target modules) directly from
`actor/lora_adapter/adapter_config.json` - no need to pass them on the CLI.

The merged model is a standard HuggingFace checkpoint. Use it in any experiment config:

```yaml
models:
  orchestrator:
    name: "Qwen3-8B-FT"
    family: "qwen3"
    path_or_id: "/scratch-shared/<user>/fine_tuning/merged_models/qwen3-8b-grpo-<run-tag>/"
    role: "orchestrator"
    # all other fields (tensor_parallel_size, gpu_ids, etc.) unchanged
```

Run evaluation exactly as for any other model:
```bash
python scripts/run_experiment.py --config experiments/configs/qwen3/agentflow/qwen3_8b_ft_gaia.yaml
python scripts/analyze_results.py experiments/results/<run>/raw_results.json --by-level --tools
```

No changes to `VLLMProvider`, `AgenticOrchestrator`, or evaluation scripts.

---

## 11. Troubleshooting

| Problem | Diagnostic | Fix |
|---|---|---|
| `EnvironmentError: SERPER_API_KEY must be set` | Missing search API key | Export `SERPER_API_KEY` before launching rollout workers |
| `EnvironmentError: SUBAGENT_ENDPOINT must be set` | Frozen sub-agent server not started | Start `vllm serve Qwen/Qwen3-1.7B --port 9998` first; export `SUBAGENT_ENDPOINT` |
| Rollout workers fail to connect to VERL | VERL not ready yet | Increase sleep in job script (60 → 120 s); check `ft_<jobid>_verl.log` |
| `ModuleNotFoundError: agentflow` | Wrong conda env | `conda activate cosmas-train` |
| `ModuleNotFoundError: verl` | Running in inference env | `agent_engine` env doesn't have verl - use `cosmas-train` |
| Parquet schema error in rollout | Stale data files from older `prepare.py` | Re-run `001_prepare_data.job` |
| GPU 0 OOM | Sub-agent + VERL hitting 94 GB | Drop sub-agent `--gpu-memory-utilization` to `0.06` |
| `val/reward` near 0 on DeepMath while search improves | Response truncation | Increase `data.max_response_length` to `8192` |
| W&B shows no val metrics | `WANDB_API_KEY` not exported | Set in login script before `sbatch` |
| Checkpoint not found on resume | Training crashed before first save | Lower `trainer.save_freq` to `5`; check VERL log for the step count |

---

## 12. Design Decisions

### Algorithm: GRPO (not PPO)

GRPO has no value network, reducing memory and simplifying the training loop. AgentFlow uses it; we
replicate the setup to minimise deviation from a tested baseline.

### Training data: Search-R1 + DeepMath

These target the two domains where tool use is demonstrably necessary and the reward signal is clean:

- **NQ / HotpotQA (Search-R1):** Specific entity-level facts the model doesn't hold in memory.
  Direct reasoning → reward = 0 on most questions. GRPO pushes toward `web_search`.
- **DeepMath:** Competition math with exact numerical answers. Arithmetic drift in natural-language
  reasoning → reward = 0. GRPO pushes toward `code_generator`.

GPQA/HLE expert-science failures are structurally different (web search doesn't reliably help on
google-proof questions) and are not the primary training target.

### Sub-agents: separate frozen server (not shared with VERL)

At evaluation time, sub-agents run the base `Qwen/Qwen3-1.7B` through the standard `VLLMProvider`.
If sub-agents instead shared the VERL endpoint, they would call the evolving actor snapshot at each
step - training the orchestrator against an unstable, changing tool interface. The frozen server
makes the tool interface at training time **identical** to the tool interface at eval time.

A 1.7B server uses ~7.5 GB at util=0.08, fitting alongside the 8B VERL stack. Using 8B for
sub-agents adds no quality benefit for the narrow sub-agent tasks (retrieve-and-summarise,
write-and-execute).

### LoRA rank 64 (not full fine-tune)

Checkpoints are ~250–500 MB (adapter only) vs ~16 GB for full fine-tune. Adapter-only optimizer
state saves ~24 GB per GPU vs full-FT. A separate ref worker is still spawned (ref_in_actor is not
in effect in the custom entrypoint), but KL divergence is computed correctly via `disable_adapter()`
on the ref worker. The ~24 GB optimizer saving allows `gpu_memory_utilization=0.70` for a larger
KV pool and higher rollout throughput.

### Thinking mode: NO (current config) vs ORCHESTRATOR_ONLY (recommended)

**Current config:** `THINKING_MODE: NO`.

**Problem:** The dominant failure mode is the orchestrator reasoning in its head and skipping a
tool call. With `THINKING_MODE: NO`, this failure is invisible to the gradient - the model just
outputs an answer directly, reward = 0, but GRPO can't distinguish "skipped a tool call" from
"tried but produced the wrong answer". With `THINKING_MODE: ORCHESTRATOR_ONLY`, the model reasons
in its `<think>` trace, reaches a confident-but-wrong answer, skips the tool call, and gets
reward = 0 - GRPO can now push it to dispatch a tool instead.

**Train/eval consistency also favours `ORCHESTRATOR_ONLY`**: thesis results show thinking is the
dominant performance driver, and the fine-tuned model will be evaluated with thinking enabled.

**If switching:** bump `data.max_response_length` from `2048` to `3072` to accommodate thinking
traces (~500–1500 extra tokens). Watch `response_length/clip_ratio` in the VERL log.

### Reward: binary exact-match

Binary reward is directly comparable to benchmark accuracy numbers. The reward function reuses
`evaluate_answer()` from `metrics.py`, ensuring training reward and eval metric are computed
identically.

### Validation: val_combined.parquet (single file → one W&B series)

VERL reads a single `val_combined.parquet` to log one `val/reward` series used for checkpoint
selection. The 20 AIME rows in the val set provide an early-warning signal for AIME regressions
during training. Per-domain breakdowns are available offline from the per-domain `val_*.parquet`
files or from rollout JSONs via the `data_source` field.

### Test split: held out entirely

The val split selects the best checkpoint. Using the same data for checkpoint selection and final
reporting inflates the reported numbers. The test split (100 Search-R1 + 100 DeepMath) is never
touched during training and is used only once, for final metric reporting after checkpoint selection.

---

## File Map

```
msc-thesis/
├── src/fine_tuning/
│   ├── __init__.py              lazy imports (heavy deps optional)
│   ├── reward.py                OrchestratorReward - binary via metrics.py
│   ├── rollout.py               OrchestratorRollout(LitAgent) - VERL rollout worker
│   └── data/
│       └── prepare.py           Download + split + write parquet files
│
│   agentflow/                   AgentFlow training stack (VERL integration)
│   ├── trainer.py               AgentFlowTrainer - extends RayPPOTrainer
│   ├── reward.py                Flow GRPO reward assignment (all triplets in trajectory)
│   ├── runner.py                Rollout worker entry point
│   ├── server.py                AgentModeDaemon HTTP server
│   └── verl/
│       ├── config.yaml          Hydra base config (extends verl ppo_trainer)
│       ├── trainer.py           VERL trainer subclass
│       ├── dataset.py           VERL dataset adapter (reads parquet)
│       ├── daemon.py            Async rollout daemon (task dispatch + reward collection)
│       └── async_server.py      PatchedvLLMServer (vLLM V1 LoRA compatibility)
│
├── scripts/
│   ├── launch_verl.py           Starts VERL training server (reads config.yaml, applies LoRA overrides)
│   ├── train_orchestrator.py    Starts rollout workers (connects to VERL + sub-agent)
│   ├── merge_lora.py            Merges LoRA adapter into base model (reads adapter_config.json for hparams)
│   └── test_ft_smoke.py         Pre-flight checks (no GPU/VERL needed)
│
├── experiments/configs/fine_tuning/
│   ├── config.yaml              Full training config (2 epochs, 4×H100)
│   ├── config_smoke.yaml        4B smoke (2 GPUs, 1 epoch, 8 samples)
│   └── config_smoke8b.yaml      8B smoke (3 GPUs, 1 epoch, 8 samples) ← production code path
│
├── jobs/fine_tuning/
│   ├── 000_create_environment.job   SLURM: create cosmas-train conda env
│   ├── 001_prepare_data.job         SLURM: download + write data/training/
│   ├── 002_inspect_data.job         SLURM: verify parquet schema + row counts
│   ├── 003_smoke_4b.job             SLURM: 4B smoke test (2 GPUs)
│   ├── 004_smoke_8b.job             SLURM: 8B smoke test (3 GPUs) ← run before 005
│   └── 005_train.job                SLURM: full 5-epoch training (4 GPUs, 48h)
│
└── data/training/               Created by job 001
    ├── train/combined_train.parquet   1800 rows
    ├── val/val_search.parquet          20 rows
    │   val/val_deepmath.parquet        10 rows
    │   val/val_aime.parquet            20 rows
    │   val/val_combined.parquet        50 rows ← VERL reads this
    └── test/test_search.parquet       100 rows
         test/test_deepmath.parquet    100 rows
         test/test_combined.parquet    200 rows (final reporting only)
```
