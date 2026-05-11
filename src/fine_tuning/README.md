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
sbatch jobs/train_orchestrator.sh
```

Or manually (two terminals):

```bash
# Terminal 1 — frozen sub-agent server (start first)
conda activate cosmas-train
vllm serve Qwen/Qwen3-1.7B --port 9998 --tensor-parallel-size 1 --gpu-memory-utilization 0.15
export SUBAGENT_ENDPOINT=http://localhost:9998/v1

# Terminal 2 — VERL server (after sub-agent server is up)
conda activate cosmas-train
python scripts/launch_verl.py --config experiments/configs/train/config.yaml

# Terminal 3 — rollout workers (after VERL vLLM is up, ~120s)
conda activate cosmas-train
python scripts/train_orchestrator.py --config experiments/configs/train/config.yaml
```

### 3. Merge LoRA and evaluate

```bash
conda activate cosmas-train
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path experiments/results/training/<run>/checkpoint_best/actor/lora_weights.pt \
    --output_dir experiments/results/training/<run>/merged_model/
```

Then update any experiment YAML:
```yaml
model:
  path_or_id: /path/to/experiments/results/training/<run>/merged_model/
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
| Thinking mode | `THINKING_MODE: ORCHESTRATOR_ONLY` | Matches the evaluation condition; exposes the "direct reasoning without action" failure to the gradient signal (model reasons internally, skips tool call, gets reward=0) |
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

Every episode is persisted to `rollout_dir/train|val/idx_N/rollout_<uuid8>.json`. With `rollout_n=8`, each training question produces 8 files (one per GRPO sample).

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

## Environment Variables Required at Runtime

| Variable | Where to set |
|---|---|
| `SERPER_API_KEY` or `TAVILY_API_KEY` | Snellius login script or `experiments/configs/train/config.yaml` env block |
| `SUBAGENT_ENDPOINT` | Set to `http://localhost:9998/v1` after starting the frozen sub-agent server |
| `WANDB_API_KEY` | Snellius login script |
| `HF_TOKEN` | Snellius login script (for gated datasets) |
