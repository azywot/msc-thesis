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

## W&B Metrics

| Metric | What it measures |
|---|---|
| `val_0/reward_mean` | Accuracy on held-out Search-R1 (NQ + HotpotQA) — confirms `web_search` tool use is improving |
| `val_1/reward_mean` | Accuracy on held-out DeepMath — confirms `code_generator` tool use is improving |
| `actor/reward_mean` | Average training reward across both domains |
| `actor/kl_divergence` | KL from reference policy — should stay low (~0.001 coef) |

VERL logs `val_0` and `val_1` separately because `data.val_files` is a list of two files.
`launch_verl.py` converts the YAML list to Hydra list syntax (`key=[elem1,elem2]`) automatically.
Per-domain breakdown is also available offline from `rollout_data/val/*.json`
(each record includes a `data_source` field).

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
