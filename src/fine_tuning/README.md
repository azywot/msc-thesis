# fine_tuning — RL Fine-Tuning Pipeline

GRPO-based reinforcement learning pipeline for fine-tuning the CoSMAS orchestrator (Qwen3-8B).
Only the orchestrator is trained; sub-agents (web search analyser, code generator) remain frozen.

See the design spec: `docs/superpowers/specs/2026-05-06-orchestrator-finetuning-design.md`
See the failure-mode rationale: `docs/failure_modes_fine_tuning_alignment.md`

---

## Structure

```
fine_tuning/
├── reward.py        # OrchestratorReward — binary exact-match via metrics.py
├── rollout.py       # OrchestratorRollout(LitAgent) — wraps AgenticOrchestrator for VERL
├── trainer.py       # Unused stub (training uses agentflow.Trainer directly)
└── data/
    └── prepare.py   # Download Search-R1 + DeepMath, write VERL parquet files
```

Training hyperparameters live in `experiments/configs/fine_tuning/config.yaml` and are
forwarded to verl by `scripts/launch_verl.py` (no Python config dataclass).

---

## Quick Start

### 1. Prepare training data

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

This writes:
- `data/training/train/combined_train.parquet` — 1800 mixed questions (900 Search-R1 + 900 DeepMath, shuffled)
- `data/training/val/val_search.parquet` — 20 held-out Search-R1 (NQ + HotpotQA) — offline reference
- `data/training/val/val_deepmath.parquet` — 10 held-out DeepMath (difficulty ≥ 3) — offline reference
- `data/training/val/val_aime.parquet` — 20 AIME problems sampled from `data/AIME/train.jsonl` (val-only signal) — offline reference
- `data/training/val/val_combined.parquet` — all three merged (50 rows) — **VERL reads this**
- `data/training/test/test_search.parquet` — 100 held-out Search-R1 (same proportions as train)
- `data/training/test/test_deepmath.parquet` — 100 held-out DeepMath (difficulty ≥ 3)
- `data/training/test/test_combined.parquet` — both merged (final reporting only, never used during training)

Search-R1/DeepMath proportions (85% HotpotQA / 15% NQ; 50/50 search/math) are identical across train and test splits.
Rows are carved in order — test first, then val, then train — so there is zero cross-split contamination.

**Note on validation:** VERL reads a single `val_combined.parquet` so W&B logs one `val/*` series. The per-domain
parquets are written for offline inspection (you can also reconstruct per-domain breakdowns from rollout JSONs
via the `data_source` field). The test split is held out entirely and used only once, for final metric reporting
after checkpoint selection — AIME held-out evaluation must remain disjoint from the val sample.

### 2. Start training (Snellius)

```bash
sbatch jobs/fine_tuning/005_train.job
```

Or manually (three terminals, after `conda activate cosmas-train` in each):

```bash
# Terminal 1 — frozen sub-agent server (start first, never needs restarting)
# util=0.12 (~11 GB) matches jobs/fine_tuning/005_train.job and gives ~7.5 GB KV cache,
# enough to absorb the concurrent tool-call traffic from N_WORKERS=4 × batch=32 × n=8.
vllm serve Qwen/Qwen3-1.7B --port 9998 --tensor-parallel-size 1 --gpu-memory-utilization 0.12

# Terminal 2 — VERL server (after sub-agent server is up)
python scripts/launch_verl.py --config experiments/configs/fine_tuning/config.yaml

# Terminal 3 — rollout workers (after VERL vLLM is up, ~120s)
# SUBAGENT_ENDPOINT is read from config.yaml env block (default: http://localhost:9998/v1)
python scripts/train_orchestrator.py --config experiments/configs/fine_tuning/config.yaml
```

### 3. Merge LoRA and evaluate

When `USE_LORA=true`, merge the adapter before inference:

```bash
conda activate cosmas-train
# Find the run tag printed by launch_verl.py at startup (also in the SLURM log)
RUN_TAG="<DD-MM-YYYY_HH-MM-JOBID>"
CKPT_STEP="experiments/results/fine_tuning/qwen3-8b-grpo-search-math/${RUN_TAG}/global_step_<N>"

python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path "${CKPT_STEP}/actor/model_world_size_1_rank_0.pt" \
    --output_dir "experiments/results/fine_tuning/qwen3-8b-grpo-search-math/${RUN_TAG}/merged_model/"
```

When `USE_LORA=false`, `model_world_size_1_rank_0.pt` is the full model — no merge needed, load directly with `from_pretrained`. (LoRA is the default; full-FT requires an explicit `USE_LORA: "false"` in `config.yaml`.)

Then update any experiment YAML:
```yaml
models:
  orchestrator:
    path_or_id: /path/to/experiments/results/fine_tuning/<experiment>/<run-tag>/merged_model/
    # all other fields (family, role, tensor_parallel_size, etc.) unchanged
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Algorithm | Flow GRPO, n=8 rollouts | No value network; same as AgentFlow. Final reward propagated to all turns so planning and tool-call steps receive gradient signal |
| Training data | Search-R1 (NQ+HotpotQA) + DeepMath-103K | Targets the two dominant failure modes: direct reasoning without action on retrieval tasks and math tasks |
| Training data mix | 85% HotpotQA / 15% NQ within Search-R1; 50/50 search/math overall | HotpotQA requires multi-hop evidence aggregation → stronger retrieval-policy signal than single-hop NQ; DeepMath difficulty ≥ 3 → medium-hard problems produce cleaner GRPO reward signal |
| Validation | 50 mixed rows: 20 Search-R1 + 10 DeepMath + 20 AIME (sampled from `data/AIME/train.jsonl`) | Single `val_combined.parquet` keeps W&B logging simple (one `val/*` series); AIME slice gives an early-warning signal for AIME-flavoured regressions during training. AIME val sample must remain disjoint from the held-out AIME eval set used for final reporting |
| Test split | 100 held-out Search-R1 + 100 held-out DeepMath | Held out entirely; used only for final reporting after checkpoint selection via val; same source proportions as train |
| Reward | Binary exact-match via `metrics.py` | Directly comparable to benchmark numbers |
| Model weights | LoRA rank-64, all-linear | ~130 MB checkpoints vs ~16 GB full fine-tune |
| Thinking mode | `THINKING_MODE: NO` (current config) | **Verify before training.** Config is set to `NO`. The recommended value is `ORCHESTRATOR_ONLY` — it matches the evaluation condition and exposes the "direct reasoning without action" failure to the gradient (model reasons in `<think>`, skips tool call, gets reward=0). Training with `NO` removes that signal. |
| Response budget | `max_prompt_length: 16384` / `max_response_length: 2048` | Sized from smoke-rollout token analysis (43 episodes, real Qwen3-8B tokenizer): assistant turn p95 = 992 tokens, tool response p95 = 4502 tokens, prompt_max already 6368 on AIME at just 2 turns. 16384 prompt fits typical 5-turn multi-hop HotpotQA with web responses; 2048 response is ~1.8× safety over observed max (1131). If first-epoch logs show `prompt_length/clip_ratio > 0` or `n_dropped_sample_because_of_prompt > 0`, bump prompt cap to 20480. With `THINKING_MODE=ORCHESTRATOR_ONLY`, raise response cap to 3072 (thinking adds ~500–1500). |

---

## Logging and Analysis

Training progress is captured in two places.

### W&B (live, step-based)

VERL logs automatically via `trainer.logger: ['console', 'wandb']` into the project set by `PROJECT_NAME`.
`val_before_train: true` runs a validation pass at step 0, giving a baseline before any gradient update.
`trainer.test_freq: 10` (production) / `1` (smoke) runs validation every N steps — ~6 evals per epoch for
the production run — so checkpoint selection does not have to wait for epoch boundaries.

| Metric | Source | What it tells you |
|---|---|---|
| `val/reward` | `val_combined.parquet` | Accuracy on the 50-row mixed val set (20 Search-R1 + 10 DeepMath + 20 AIME) — main checkpoint-selection signal and the key used by `_rotate_checkpoints` to update `best_checkpoint/`. Per-domain breakdown available offline from the per-source `val_*.parquet` files or from rollout JSONs via the `data_source` field |
| `actor/reward_mean` | training rollouts | Mean reward across both domains per step |
| `actor/reward_std` | training rollouts | Diversity signal — near-zero means all rollouts tied (bad) |
| `actor/kl_divergence` | GRPO | Should stay low; spike = policy drifting from reference |
| `actor/pg_loss` | GRPO | Policy gradient loss; should fall over epochs |

**Gap:** `actor/reward_mean` is the combined average — W&B does not get per-domain breakdown for training rollouts. That requires offline analysis of the rollout JSONs (see below).

### Rollout JSONs (disk, per episode)

Every episode is persisted to `experiments/results/fine_tuning/<experiment>/<run-tag>/rollout_data/train|val/idx_N/rollout_<uuid8>.json`. With `rollout_n=8`, each training question produces 8 files (one per GRPO sample).

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
`experiments/configs/fine_tuning/config.yaml` and relaunch.

---

## GPU Efficiency and Throughput Tuning

The training loop is rollout-bound: smoke-test profiling on Qwen3-8B (2× H100) showed
`timing_s/gen` consuming ~60 % of every step, with the actor update taking only ~6 s vs
60–70 s for generation. Memory utilisation was also low — 27 GB of 94 GB peak — because
`free_cache_engine=True` releases the vLLM KV pool between rollout and training. The knobs
below are tuned to fill that headroom and keep the GPUs fed.

| Knob | Value | Why |
|---|---|---|
| `N_WORKERS` | `4` (was 1) | Rollout loop is IO-bound (HTTP to VERL vLLM, HTTP to sub-agent, Serper). Parallel workers fill vLLM's continuous batcher; single-worker was serialising 256 in-flight episodes (`batch=32 × n=8`) through one Python client. |
| vLLM `gpu_memory_utilization` (LoRA) | `0.70` (was 0.60) | Set in `scripts/launch_verl.py` LoRA override (not the YAML). Smoke's 27 GB / 94 GB peak left ~67 GB of headroom; bumping util grows the KV pool to ≈ 50 GB. |
| `max_num_seqs` | `128` (was 64) | Decode parallelism cap. vLLM uses paged KV so this doesn't multiply by `max_model_len`; it just lets more in-flight sequences share the KV pool. |
| Sub-agent `--gpu-memory-utilization` | `0.12` (was 0.08) | ~7.5 GB KV for the frozen Qwen3-1.7B server (was ~4 GB). 256 concurrent rollouts dispatching tool calls used to queue here; doubling the KV pool removes that bottleneck while staying inside GPU 0's envelope alongside VERL's vLLM. |

**GPU 0 memory envelope** (the tight one — also hosts the sub-agent):
sub-agent 11 GB + vLLM 66 GB + FSDP shard 4 GB + activations 8 GB ≈ **89 GB / 94 GB**
(5 GB headroom). GPUs 1–3 carry only the VERL share at ≈ 82 GB. Drop sub-agent util back
to `0.10` if `out/fine_tuning/orchestrator_ft/ft_<jobid>_gpu.csv` shows GPU 0 near 94 GB.

### Ground-truth GPU monitor

All three fine-tuning jobs (`003_smoke_4b.job`, `004_smoke_8b.job`, `005_train.job`)
start an `nvidia-smi --query-gpu` sidecar (installed *after* the `trap cleanup EXIT`
line so it's always reaped) that samples GPU SM / memory / power every 10 s into a
CSV alongside the SLURM logs:

```
out/fine_tuning/smoke_test_4b/smoke4b_<jobid>_gpu.csv
out/fine_tuning/smoke_test_8b/smoke8b_<jobid>_gpu.csv
out/fine_tuning/orchestrator_ft/ft_<jobid>_gpu.csv
```

This is ground-truth (SM utilisation, real memory) to compare against VERL's own
`perf/mfu/actor` and `perf/max_memory_allocated_gb` metrics, which only see what
PyTorch allocated — they miss vLLM and the sub-agent server entirely.

### Health-check metrics to watch on the first prod run

Beyond `val/reward_mean`, the following should stay at zero. Any non-zero value means
the corresponding cap is too tight and is silently corrupting the GRPO signal:

| Metric (in `*_verl.log`) | If non-zero |
|---|---|
| `prompt_length/clip_ratio` | Bump `data.max_prompt_length` (16384 → 20480) |
| `response_length/clip_ratio` | Bump `data.max_response_length` (2048 → 3072) |
| `n_dropped_sample_because_of_prompt` | Same as above (prompt cap) |
| `n_trunc_sample_because_of_response` | Same as above (response cap) |
| `n_dropped_sample_because_of_mini_batch` | GRPO group is losing rollouts; tune `ppo_max_token_len_per_gpu` or `ppo_mini_batch_size` so the `n=8` fanout partitions cleanly across `N_GPUS=4` |

---

## Checkpoint Layout

VERL writes checkpoints to a unique run directory set by `trainer.default_local_dir` in `launch_verl.py`.
The run tag `<DD-MM-YYYY_HH-MM-JOBID>` is printed at startup and shared by checkpoints and rollout data.

| Config | Checkpoint base |
|---|---|
| `config_smoke.yaml`, `config_smoke8b.yaml` (`USE_SCRATCH_CHECKPOINTS: false`) | `experiments/results/fine_tuning/<experiment>/<run-tag>/` |
| `config.yaml` (`USE_SCRATCH_CHECKPOINTS: true`) | `/scratch-shared/$USER/msc-thesis/fine_tuning/<experiment>/<run-tag>/` |

Rollout JSONs always land in `experiments/results/fine_tuning/<experiment>/<run-tag>/rollout_data/`.

For the smoke run the checkpoint tree looks like:

```
experiments/results/fine_tuning/qwen3-4b-grpo-smoke/<run-tag>/
│
├── latest_checkpointed_iteration.txt   # Contains the last saved global step number (e.g. "2").
│                                       # VERL reads this to find the latest checkpoint on resume.
│
├── latest_checkpoint -> global_step_2/ # Symlink → most recently saved step dir.
│                                       # Updated by _rotate_checkpoints after every save.
│
├── best_checkpoint -> global_step_1/   # Symlink → step with highest val/reward_mean so far.
│                                       # Updated only when a fresh val beats the running best.
│                                       # If save_freq > test_freq it may lag one save behind.
│
└── global_step_<N>/                    # Concrete checkpoint dir for step N.
    │                                   # Rotation keeps at most 2 dirs: the one pointed to by
    │                                   # latest_checkpoint and the one pointed to by best_checkpoint.
    │                                   # All other dirs are deleted asynchronously after rotation.
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
        ├── optim_world_size_1_rank_0.pt    # Adam optimizer state for rank 0.
        │                                   # LoRA runs (~10s of MB, LoRA params only): saved by
        │                                   # all three configs (config.yaml, config_smoke.yaml,
        │                                   # config_smoke8b.yaml) — SAVE_OPTIMIZER=true.
        │                                   # Full-FT runs (~30 GB): set SAVE_OPTIMIZER=false if
        │                                   # disk is tight; optimizer restarts from scratch on resume.
        │                                   # Can be deleted from any checkpoint kept for inference only.
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

Leave `trainer.resume_from_path` unset (the default) and VERL will auto-resume from the step recorded
in `latest_checkpointed_iteration.txt`, which always points at the `latest_checkpoint/` symlink.
To resume from the best checkpoint instead, set `trainer.resume_from_path` to the resolved path of
`best_checkpoint/` (e.g. `/scratch-shared/$USER/msc-thesis/fine_tuning/<experiment>/<run-tag>/best_checkpoint`).

`SAVE_OPTIMIZER=true` (all three configs: config.yaml, config_smoke.yaml, config_smoke8b.yaml) stores
Adam moments and LR-scheduler state alongside the model weights, so the resumed run is byte-for-byte
identical to an uninterrupted run. For hypothetical full-FT runs, set `SAVE_OPTIMIZER=false` if disk
is tight — the optimizer will restart from scratch, which affects the first few gradient steps.

### Converting to a usable model (LoRA runs)

When `USE_LORA=true`, `model_world_size_1_rank_0.pt` contains only the LoRA adapter deltas — the
frozen base weights are not stored. Merge before inference:

```bash
# Smoke (USE_SCRATCH_CHECKPOINTS=false):
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-4B \
    --lora_path experiments/results/fine_tuning/qwen3-4b-grpo-smoke/<run-tag>/global_step_<N>/actor/model_world_size_1_rank_0.pt \
    --output_dir experiments/results/fine_tuning/qwen3-4b-grpo-smoke/<run-tag>/merged_model/

# Full training (USE_SCRATCH_CHECKPOINTS=true):
python $HOME/azywot/AgentFlow/util/model_merger.py \
    --base_model Qwen/Qwen3-8B \
    --lora_path /scratch-shared/$USER/msc-thesis/fine_tuning/qwen3-8b-grpo-search-math/<run-tag>/global_step_<N>/actor/model_world_size_1_rank_0.pt \
    --output_dir experiments/results/fine_tuning/qwen3-8b-grpo-search-math/<run-tag>/merged_model/
```

When `USE_LORA=false`, `model_world_size_1_rank_0.pt` is the full model and can be loaded
directly with `from_pretrained`. (All three configs — `config.yaml`, `config_smoke.yaml`,
`config_smoke8b.yaml` — use `USE_LORA: "true"` by default.)

---

## Environment Variables Required at Runtime

| Variable | Where to set |
|---|---|
| `SERPER_API_KEY` or `TAVILY_API_KEY` | Snellius login script or `experiments/configs/fine_tuning/config.yaml` env block |
| `SUBAGENT_ENDPOINT` | Set to `http://localhost:9998/v1` after starting the frozen sub-agent server |
| `WANDB_API_KEY` | Snellius login script |
| `HF_TOKEN` | Snellius login script (for gated datasets) |
