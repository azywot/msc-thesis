# fine_tuning — RL Fine-Tuning Pipeline

GRPO-based reinforcement learning pipeline for fine-tuning the CoSMAS orchestrator (Qwen3-8B).
Only the orchestrator is trained; sub-agents (web search analyser, code generator) remain frozen.

See the design spec: `docs/superpowers/specs/2026-05-06-orchestrator-finetuning-design.md`
See the failure-mode rationale: `docs/failure_modes_fine_tuning_alignment.md`

---

## Structure

```
fine_tuning/
├── config.py        # FinetuningConfig dataclass — LoRA, GRPO, paths
├── reward.py        # OrchestratorReward — binary exact-match via metrics.py
├── rollout.py       # OrchestratorRollout(LitAgent) — wraps AgenticOrchestrator for VERL
├── trainer.py       # Unused stub (training uses agentflow.Trainer directly)
└── data/
    └── prepare.py   # Download Search-R1 + DeepMath, write VERL parquet files
```

---

## Quick Start

### 1. Prepare training data

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

This writes:
- `data/training/train/combined_train.parquet` — 1800 mixed questions (900 Search-R1 + 900 DeepMath, shuffled)
- `data/training/val/val_search.parquet` — 100 held-out Search-R1 (NQ + HotpotQA)
- `data/training/val/val_deepmath.parquet` — 100 held-out DeepMath (difficulty ≥ 5)
- `data/training/val/val_combined.parquet` — both merged (offline analysis only)
- `data/training/test/test_search.parquet` — 100 held-out Search-R1 (same proportions as train)
- `data/training/test/test_deepmath.parquet` — 100 held-out DeepMath (difficulty ≥ 5)
- `data/training/test/test_combined.parquet` — both merged (final reporting only, never used during training)

Source proportions (85% HotpotQA / 15% NQ; 50/50 search/math) are identical across all three splits.
Rows are carved in order — test first, then val, then train — so there is zero cross-split contamination.

**Note:** AIME is an evaluation benchmark and must not be used for checkpoint selection (selection bias).
The test split is held out entirely and used only once, for final metric reporting after checkpoint selection.

### 2. Start training (Snellius)

```bash
sbatch jobs/010_ft_orchestrator.job
```

Or manually (three terminals, after `conda activate cosmas-train` in each):

```bash
# Terminal 1 — frozen sub-agent server (start first, never needs restarting)
vllm serve Qwen/Qwen3-1.7B --port 9998 --tensor-parallel-size 1 --gpu-memory-utilization 0.15

# Terminal 2 — VERL server (after sub-agent server is up)
python scripts/launch_verl.py --config experiments/configs/train/config.yaml

# Terminal 3 — rollout workers (after VERL vLLM is up, ~120s)
# SUBAGENT_ENDPOINT is read from config.yaml env block (default: http://localhost:9998/v1)
python scripts/train_orchestrator.py --config experiments/configs/train/config.yaml
```

### 3. Merge LoRA and evaluate

When `USE_LORA=true`, merge the adapter before inference:

```bash
conda activate cosmas-train
# Find the run tag printed by launch_verl.py at startup (also in the SLURM log)
RUN_TAG="<DD-MM-YYYY_HH-MM-JOBID>"
CKPT_STEP="experiments/results/training/qwen3-8b-grpo-search-math/${RUN_TAG}/global_step_<N>"

python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path "${CKPT_STEP}/actor/model_world_size_1_rank_0.pt" \
    --output_dir "experiments/results/training/qwen3-8b-grpo-search-math/${RUN_TAG}/merged_model/"
```

When `USE_LORA=false` (current default), `model_world_size_1_rank_0.pt` is the full model — no merge needed, load directly with `from_pretrained`.

Then update any experiment YAML:
```yaml
models:
  orchestrator:
    path_or_id: /path/to/experiments/results/training/<experiment>/<run-tag>/merged_model/
    # all other fields (family, role, tensor_parallel_size, etc.) unchanged
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Algorithm | GRPO, n=8 rollouts | No value network; same as AgentFlow |
| Training data | Search-R1 (NQ+HotpotQA) + DeepMath-103K | Targets the two dominant failure modes: direct reasoning without action on retrieval tasks and math tasks |
| Training data mix | 85% HotpotQA / 15% NQ within Search-R1; 50/50 search/math overall | HotpotQA requires multi-hop evidence aggregation → stronger retrieval-policy signal than single-hop NQ; DeepMath difficulty ≥ 5 → harder problems produce cleaner GRPO reward signal |
| Validation | 100 held-out Search-R1 + 100 held-out DeepMath | AIME is an eval benchmark — must not be used for checkpoint selection; two separate val files so W&B logs `val_0/reward_mean` (search) and `val_1/reward_mean` (math) independently |
| Test split | 100 held-out Search-R1 + 100 held-out DeepMath | Held out entirely; used only for final reporting after checkpoint selection via val; same source proportions as train and val |
| Reward | Binary exact-match via `metrics.py` | Directly comparable to benchmark numbers |
| Model weights | LoRA rank-64, all-linear | ~130 MB checkpoints vs ~16 GB full fine-tune |
| Thinking mode | `THINKING_MODE: NO` (current config) | **Verify before training.** Config is set to `NO`. The recommended value is `ORCHESTRATOR_ONLY` — it matches the evaluation condition and exposes the "direct reasoning without action" failure to the gradient (model reasons in `<think>`, skips tool call, gets reward=0). Training with `NO` removes that signal. |
| Response budget | `max_response_length: 4096` | Full multi-turn orchestrator rollout is longer than AgentFlow's single-step Planner; thinking traces add ~500–1500 tokens per rollout |

---

## Logging and Analysis

Training progress is captured in two places.

### W&B (live, per epoch)

VERL logs automatically via `trainer.logger: ['console', 'wandb']` into the project set by `PROJECT_NAME`.
`val_before_train: true` runs a validation pass before epoch 1, giving a baseline measurement at epoch 0.

| Metric | Source | What it tells you |
|---|---|---|
| `val_0/reward_mean` | `val_search.parquet` | Accuracy on held-out Search-R1 (NQ + HotpotQA) — `web_search` improving |
| `val_1/reward_mean` | `val_deepmath.parquet` | Accuracy on held-out DeepMath — `code_generator` improving |
| `actor/reward_mean` | training rollouts | Mean reward across both domains per step |
| `actor/reward_std` | training rollouts | Diversity signal — near-zero means all rollouts tied (bad) |
| `actor/kl_divergence` | GRPO | Should stay low; spike = policy drifting from reference |
| `actor/pg_loss` | GRPO | Policy gradient loss; should fall over epochs |

**Gap:** `actor/reward_mean` is the combined average — W&B does not get per-domain breakdown for training rollouts. That requires offline analysis of the rollout JSONs (see below).

### Rollout JSONs (disk, per episode)

Every episode is persisted to `experiments/results/training/<experiment>/<run-tag>/rollout_data/train|val/idx_N/rollout_<uuid8>.json`. With `rollout_n=8`, each training question produces 8 files (one per GRPO sample).

Each record contains:
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

**What you can compute offline from the JSONs:**

| Plot / metric | How |
|---|---|
| Reward by domain per epoch | Group records by `data_source` + epoch; mean `reward` |
| Reward distribution histogram (per epoch) | Histogram of `reward` over all 8 rollouts per question |
| Tool call counts (`web_search` vs `code_generator`) | Count tool-call messages in `output_messages` |
| Average turns to solution | Count assistant turns in `output_messages` |
| Thinking trace length (tokens) | Extract `<think>...</think>` content from assistant messages |
| Reward by DeepMath difficulty | Join on `extra_info.difficulty` from the parquet |
| Pass@k curves | k ∈ {1,2,4,8} — fraction of questions with ≥1 correct rollout |

**No analysis script exists yet for the JSONs.** When writing plots, create `scripts/plots/ft_rollout_analysis.py` — pattern matches `scripts/plots/efficiency_plots.py` (loads JSON files, produces matplotlib figures).

---

## Watch: Thinking Traces and the Response Budget

With `THINKING_MODE: ORCHESTRATOR_ONLY`, Qwen3-8B generates a `<think>...</think>`
block before every action. On the training data (NQ, HotpotQA, DeepMath) these
traces are typically 300–800 tokens for simple questions and up to ~1500 tokens
for harder DeepMath problems. Combined with tool calls and synthesis, most rollouts
fit comfortably within the 4096 token budget.

The risk is the hard tail of DeepMath: if the model thinks extensively before
dispatching code, the total can approach or exceed 4096. When truncation happens,
`data.truncation: 'truncate'` silently cuts the trajectory and the answer token
is lost → reward = 0. A cluster of spurious zeros on DeepMath questions will push
the model away from long thinking traces — the opposite of what you want.

**How to detect:** in the first epoch, watch `val/reward_mean` on the DeepMath
val split in W&B. If it drops or stays near zero while training reward is rising,
truncation is likely. Fix: increase `data.max_response_length` to `8192` in
`experiments/configs/train/config.yaml` and relaunch.

---

## Checkpoint Layout

VERL writes checkpoints to a unique run directory set by `trainer.default_local_dir` in `launch_verl.py`.
The run tag `<DD-MM-YYYY_HH-MM-JOBID>` is printed at startup and shared by checkpoints and rollout data.

| Config | Checkpoint base |
|---|---|
| `config_smoke.yaml` (`USE_SCRATCH_CHECKPOINTS: false`) | `experiments/results/training/<experiment>/<run-tag>/` |
| `config.yaml` (`USE_SCRATCH_CHECKPOINTS: true`) | `/scratch-shared/$USER/msc-thesis/training/<experiment>/<run-tag>/` |

Rollout JSONs always land in `experiments/results/training/<experiment>/<run-tag>/rollout_data/`.

For the smoke run the checkpoint tree looks like:

```
experiments/results/training/qwen3-4b-grpo-smoke/<run-tag>/
│
├── latest_checkpointed_iteration.txt   # Contains the last saved global step number (e.g. "1").
│                                       # Used by VERL to find the latest checkpoint when resuming.
│
└── global_step_<N>/                    # One directory per saved step (save_freq: 1 → every step).
    │
    ├── data.pt                         # Dataloader state dict (StatefulDataLoader).
    │                                   # Stores the RNG state + sampler position so training can
    │                                   # resume mid-epoch without re-seeing the same batches.
    │
    └── actor/                          # Actor (policy) checkpoint — the model being trained.
        │                               # No critic/ directory: GRPO has no value network.
        │
        ├── model_world_size_1_rank_0.pt    # Full FSDP model state dict for rank 0 (~17 GB for Qwen3-4B).
        │                                   # Contains all trainable parameters (full weights when
        │                                   # USE_LORA=false; LoRA adapter + frozen base when true).
        │                                   # world_size and rank are part of the filename so multi-GPU
        │                                   # runs shard across multiple files (e.g. _rank_0, _rank_1…).
        │
        ├── optim_world_size_1_rank_0.pt    # Adam optimizer state for rank 0 (~30 GB — first & second
        │                                   # moment estimates, one tensor pair per parameter).
        │                                   # Required to resume training with identical behaviour.
        │                                   # Can be deleted if you only need inference.
        │
        ├── extra_state_world_size_1_rank_0.pt  # LR scheduler state + RNG state (~15 KB).
        │                                       # Needed for exact learning-rate resume.
        │
        ├── fsdp_config.json            # FSDP metadata: FSDP_version and world_size.
        │                               # Used by the checkpoint loader to validate shard count.
        │
        └── huggingface/                # HF-format tokenizer (always saved, even without hf_model).
            ├── config.json             # Model architecture config (vocab size, hidden dims, etc.)
            ├── generation_config.json  # Default generation parameters (temperature, top_p…)
            ├── tokenizer.json          # Fast tokenizer vocabulary + merge rules
            ├── tokenizer_config.json   # Tokenizer metadata (chat template path, special tokens…)
            ├── chat_template.jinja     # Qwen3 chat template (used by apply_chat_template)
            ├── vocab.json              # BPE vocabulary mapping token → id
            ├── merges.txt              # BPE merge rules
            ├── added_tokens.json       # Special tokens added on top of the base vocab
            └── special_tokens_map.json # Maps special token names (bos, eos…) to their strings
```

### What to keep vs. discard

| File | Keep for inference | Keep for resuming training |
|---|---|---|
| `model_world_size_*_rank_*.pt` | Yes | Yes |
| `optim_world_size_*_rank_*.pt` | No (large) | Yes |
| `extra_state_*_rank_*.pt` | No | Yes |
| `fsdp_config.json` | No | Yes |
| `huggingface/` | Yes (tokenizer) | Yes |
| `data.pt` | No | Yes |
| `latest_checkpointed_iteration.txt` | No | Yes |

### Resuming training

Set `trainer.resume_from_path` in the config to `global_step_<N>`, or leave it unset and VERL will
auto-resume from the step in `latest_checkpointed_iteration.txt`.

### Converting to a usable model (LoRA runs)

When `USE_LORA=true`, `model_world_size_1_rank_0.pt` contains only the LoRA adapter deltas — the
frozen base weights are not stored. Merge before inference:

```bash
# Smoke (USE_SCRATCH_CHECKPOINTS=false):
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-4B \
    --lora_path experiments/results/training/qwen3-4b-grpo-smoke/<run-tag>/global_step_<N>/actor/model_world_size_1_rank_0.pt \
    --output_dir experiments/results/training/qwen3-4b-grpo-smoke/<run-tag>/merged_model/

# Full training (USE_SCRATCH_CHECKPOINTS=true):
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path /scratch-shared/$USER/msc-thesis/training/qwen3-8b-grpo-search-math/<run-tag>/global_step_<N>/actor/model_world_size_1_rank_0.pt \
    --output_dir experiments/results/training/qwen3-8b-grpo-search-math/<run-tag>/merged_model/
```

When `USE_LORA=false` (current smoke config), `model_world_size_1_rank_0.pt` is the full model and
can be loaded directly with `from_pretrained`.

---

## Environment Variables Required at Runtime

| Variable | Where to set |
|---|---|
| `SERPER_API_KEY` or `TAVILY_API_KEY` | Snellius login script or `experiments/configs/train/config.yaml` env block |
| `SUBAGENT_ENDPOINT` | Set to `http://localhost:9998/v1` after starting the frozen sub-agent server |
| `WANDB_API_KEY` | Snellius login script |
| `HF_TOKEN` | Snellius login script (for gated datasets) |
