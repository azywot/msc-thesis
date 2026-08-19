# SFT pipeline - supervised fine-tuning of the orchestrator

Supervised fine-tuning of the Qwen3-8B orchestrator on trajectories from a
stronger teacher, with a rank-64 LoRA or full-parameter.

Unlike RL this needs **no rollouts**: training is static cross-entropy, so no
tools execute, no sub-agent server runs, and no `SERPER_API_KEY` /
`TAVILY_API_KEY` is needed. Only `WANDB_API_KEY`, for logging.

Historical status and evidence: [`../archive/sft_status.md`](../archive/sft_status.md)
(archived - read it before changing anything here, but check its paths against
the current tree).

---

## The format rule - the thing to get right

This is the part that has already cost this project a wrong result, so it comes
first.

**The orchestrator never sees a growing conversation at inference.** Every
non-baseline turn rebuilds a fresh two-message prompt via
`AgenticOrchestrator._build_memory_prompt`, with prior steps compressed into
`Action Step N: Tool / Sub-goal / Command / Result` prose inside the user turn.

Training on the stored multi-turn transcript therefore optimises a distribution
the model is never evaluated on. Loss falls while task performance drops, and
**validation loss cannot detect it** because the val split shares the defect.
The symptom is an SFT model that scores *below* its own base model.

So SFT rows are **memory-folded**: one row per orchestrator decision, each
reproducing exactly the prompt inference would have built at that step.

| | Trajectories | Rows | Supervised tokens |
|---|---|---|---|
| Train | 968 | 2995 | 589,805 |
| Val | 108 | 362 | 66,556 |

A folded row is one decision where a multi-turn row was a whole trajectory, so
the row count is just the assistant-turn count. The supervision is byte-for-byte
identical; only the conditioning changes.

Three components enforce this:

- **`scripts/build_sft_parquet.py --format folded`** builds the rows, importing
  the orchestrator's own `_format_action_history` / `_extract_sub_goal` helpers
  so the folded prompt cannot drift from the real one.
- **`src/verl_ext/folded_sft_dataset.py`** (`FoldedSFTDataset`, wired in via
  verl's `data.custom_cls`) renders with `add_generation_prompt=True` so it
  matches inference token-for-token - *including* Qwen3's empty
  `<think>\n\n</think>` block - and supervises only `target + <|im_end|>`.
- **`scripts/check_sft_folded_format.py`** is a pre-flight gate asserting prompt
  identity, span purity, no thinking, no tool output in the loss, and no
  truncation, on every row. The training job runs it and **refuses to start** if
  it fails.

If you change anything about how prompts are built, that gate is what tells you
the training data went stale.

---

## Running it

```bash
# 1. Collect teacher trajectories (once). Produces collected_<ts>.jsonl,
#    NOT sft_train.parquet.
sbatch jobs/fine_tuning/006_collect_sft_data.job

# 2. Build the multi-turn parquets from that jsonl.
#    --reference-parquet must be passed explicitly: the script's own default
#    points at an unreadable /projects/0/gusr0608 path.
python scripts/build_sft_parquet.py data/training/sft/collected_<ts>.jsonl \
    --output-dir data/training/sft \
    --reference-parquet data/training/train/combined_train.parquet

# 3. Fold them into the memory-folded rows training actually uses.
python scripts/build_sft_parquet.py \
    --from-parquet data/training/sft/sft_train.parquet \
    --output-dir data/training/sft --output-name sft_folded_train.parquet

# 4. Verify on the CPU partition - tests, gate, and gate trip-wire. No GPU cost.
sbatch jobs/fine_tuning/007_run_tests_for_sft_folded.job

# 5. Train. LoRA rank 64 (~187 steps, 2xH100, ~40 min):
sbatch jobs/fine_tuning/007_train_sft_folded.job
#    or full parameter (4xH100):
sbatch jobs/fine_tuning/007_train_sft_full.job
```

Step 4 is cheap and catches the expensive mistakes. Run it.

## Evaluating the result

**LoRA** - paste the run tag the job prints into `SFT_ADAPTER_PLACEHOLDER` in
`scripts/generate_configs.py`, then regenerate and run:

```bash
python scripts/generate_configs.py --suite sft_inference
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/sft_inference
```

**Full parameter** - point an inference config's `path_or_id` straight at the
archived `best_checkpoint/` directory the job prints. No `lora_adapter_path`, no
merge step.

Eval configs live in `experiments/configs/qwen3/sft_inference/` (five
benchmarks, `thinking_mode: NO`, orchestrator named `Qwen3-8B-SFT` so W&B keeps
it distinct from the GRPO rows).

> `max_lora_rank` must match the training rank. vLLM's default is 16, ours is
> 64, and a mismatch loads without complaint.

---

## Checkpoint handling

There is no manual post-training step. The job selects the best-val-loss and
last checkpoints, extracts them (PEFT adapters for LoRA, complete HuggingFace
model directories for full parameter), deletes the shards, and archives into
`data/adapters/<experiment>/<run-tag>/` or
`data/checkpoints/<experiment>/<run-tag>/`.

Why that machinery exists: `verl.trainer.sft_trainer` never writes a
`lora_adapter/` directory - only the RL path does. It writes the full FSDP state
dict, ~32 GB per checkpoint, of which ~350 MB is trained weight for a LoRA run.
The shards are **sharded DTensors**, so rank 0 holds only its own slice of each
tensor.

| Script | Role |
|---|---|
| `scripts/sft_checkpoint_janitor.py` | Runs alongside training; collapses each checkpoint as soon as verl's atomic tracker marks it complete, keeping peak disk at ~32–64 GB instead of ~256 GB. `--mode lora` (default) or `--mode full`. |
| `scripts/finalize_sft_run.py` | Selects best/last across whatever form each step is in, then cleans up. Same `--mode` flag. |
| `src/verl_ext/checkpoint_utils.py` | The shared DTensor-gathering logic both call. |

> **`scripts/merge_lora.py` must not be used on SFT checkpoints**, LoRA or
> full-parameter. It expects the RL path's `actor/` layout with a single
> consolidated shard, and exits with an error pointing at
> `finalize_sft_run.py --mode full` if it sees the SFT trainer's layout.

## Job files

| File | Purpose | Log |
|---|---|---|
| `jobs/fine_tuning/006_collect_sft_data.job` | Collect teacher trajectories | `out/fine_tuning/sft_collect/collect_<job_id>.log` |
| `jobs/fine_tuning/007_run_tests_for_sft_folded.job` | CPU verification (tests + gate + trip-wire) | `out/fine_tuning/tests/sft_folded_tests_<job_id>.log` |
| `jobs/fine_tuning/007_train_sft_folded.job` | Folded-format training, LoRA rank 64 | `out/fine_tuning/sft_train/sft_folded_<job_id>.log` |
| `jobs/fine_tuning/007_train_sft_full.job` | Same, full parameter | `out/fine_tuning/sft_train/sft_full_<job_id>.log` |
