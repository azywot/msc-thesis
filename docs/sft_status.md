# Orchestrator SFT: status and handover

**Last updated:** 2026-08-12
**Branch:** `feat/sft-folded-format`

The single source of truth for the orchestrator SFT work: the diagnosis, the fix, the data,
the run mechanics, and what is left. 

---

## 1. Status and next action

The SFT adapter underperformed the base model because it was trained on a conversation format
the orchestrator never sees at inference. That diagnosis is **verified by executing both code
paths**, not inferred. The fix is implemented, tested, and applied to the real data: folded
parquets are built and every pre-flight gate passes on all 3357 rows. Both research decisions
are resolved (§6).

A first training run went 50 steps and was **deliberately cancelled** (§12). It proved the
pipeline works end to end and that val loss falls (0.6025 → 0.4947), but it wrote 33 GB per
checkpoint, to the wrong directory, and kept all of them. All three are now fixed (§7).

A **pre-launch audit found one critical bug** (§9): the training run would have silently
reintroduced the very prompt gap this work exists to close. It is fixed, with regression tests,
and re-verified on the real data.

**Training and evaluation are now DONE.** Job run tag `06-08-2026_21-47-25300018` trained 186
steps, val loss falling monotonically 0.5404 → 0.4234 (`data/adapters/qwen3-8b-sft-folded-v1/
06-08-2026_21-47-25300018/selection.json`); best == last checkpoint (step 186). The adapter is
archived at `data/adapters/qwen3-8b-sft-folded-v1/06-08-2026_21-47-25300018/best_adapter/`, and
all five `sft_inference` configs were pointed at it and run on 2026-08-07 (§8 has the numbers).
**NEXT ACTION: fold the results into Ch 7** of the thesis — see §8 for the numbers and the one
open question (AIME) they raise. Test suite: **496 passed**.

**`007_train_sft.job` (the original training job) has been removed from the repo.** Its
training conversation format did not match the orchestrator's inference format, which is the
bug this document diagnoses, and it is superseded entirely by the folded pipeline below; it was
kept around for a while as a comparison artifact, but that risked reading as if the broken
pipeline were still current. References to it below (§2, §6, §12) are historical.

---

## 2. The bug, and the evidence for it

### What was wrong

SFT rows were stored as the multi-turn transcript the teacher produced:

```
[system, user, assistant(plan), assistant(tool_call), tool(result), ..., assistant(answer)]
```

built by `_build_sft_messages` (`scripts/collect_sft_data.py:70-91`).

But the orchestrator never sees that at inference. Every non-baseline turn rebuilds a fresh
two-message prompt from scratch via `_build_memory_prompt`
(`src/agent_engine/core/orchestrator.py:641-664`, called at `:207`, `:241`, `:323`, `:367`):

```
[system, user(question + **Query Analysis:** + **Previous Steps:**)]
```

Prior steps are compressed into `Action Step N: Tool / Sub-goal / Command / Result` prose, and
the tool result becomes `- Result:` text inside the user turn rather than its own turn.

So the adapter minimised cross-entropy honestly, on a distribution it is never evaluated on.
Loss went down; task performance went down. **Val loss could not have caught this**, because
the val split had exactly the same defect.

### How it was verified (not assumed)

Both sides were produced by running the production code:

- **Training side:** `verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`,
  constructed with the exact flags the (now-removed) `007_train_sft.job:135-142` passed, then decoding
  `input_ids` and the `loss_mask == 1` positions.
- **Inference side:** the orchestrator's own `_build_memory_prompt`, `_commit_tool_result`,
  `_format_action_history`, `_extract_sub_goal` and the real `parse_tool_call`, driven on a
  real `ExecutionState`, rendered through the Qwen3 branch of `VLLMProvider._render_messages`
  (`vllm_provider.py:338-343`).

The only function that could not be *invoked* is `_run_planning_turn` (it needs a live model);
its two prompt-shaping lines were mirrored with the suffix constant imported from the module.
The Qwen3 render branch was asserted programmatically, not assumed.

**Result over 60 randomly sampled trajectories (seed 0), 194 decision points:**

| Measurement | Result |
|---|---|
| Decision points where training context == inference context | **0 / 194** |
| Decision points that diverge | **194 / 194** |
| First differing token index | 751–897 (immediately after the shared system prompt) |
| Training contexts containing `**Query Analysis:**` | **0 / 968** (whole split) |
| Training contexts containing the planning suffix | **0 / 60** |
| Inference planning prompts containing the suffix | **60 / 60** |

The divergence is total, and it starts at the earliest possible content position.

### A second, independent defect found in the same audit

`MultiTurnSFTDataset` tokenises each turn in isolation, so a target always begins right after
`<|im_start|>assistant\n`. Qwen3's template rendering a whole conversation — and the
orchestrator at inference with thinking off — first emits `<think>\n\n</think>\n\n`. The
adapter was trained to start generating 4 tokens earlier than it is ever sampled from, on
**every** target.

VERL's own sanity check fails on **60/60** sampled rows, delta **exactly 4 tokens** every time.
The removed `007_train_sft.job:141` silenced it with `data.ignore_input_ids_mismatch=True`. Folding alone
does **not** fix this — measured on a real folded row:

| Path | Prompt tokens | Identical to inference? |
|---|---|---|
| Folded row through `MultiTurnSFTDataset` | 1374 | **No** — missing exactly `<think>\n\n</think>\n\n` |
| Folded row through `FoldedSFTDataset` (new) | 1378 | **Yes, token-for-token** |

This gap came back a second time through a different route; see §9.

---

## 3. Alternative causes, ruled in or out

Everything below was checked before accepting the format hypothesis.

| Candidate | Verdict | Evidence |
|---|---|---|
| **Loss-masking bug** | **Ruled out** | Over **all 968 train + 108 val** rows, the decoded `loss_mask==1` spans equal *exactly* the list of assistant message contents (each + `<\|im_end\|>`), with matching span counts. The mask was never the bug. |
| **Thinking leaking into targets** | **Ruled out** | 0 rows contain `<think>`/`</think>` in the supervised span or anywhere in context, both splits. `_strip_thinking` (`build_sft_parquet.py:106-122`) removed it at build time. |
| **Tool output in the loss** | **Ruled out** | Structural: tool turns get `loss_mask = zeros`. A substring probe first flagged 8 hits across 4 rows — all false positives from 1–4-character tool results (`"1"`, `"oo"`, `"[15]"`) trivially inside assistant prose. The span-equality check above is authoritative and clean. |
| **Chat-template / special-token mismatch** | **RULED IN** | The 4-token think-block gap above. Real, on every target. |
| **EOS handling** | **Ruled out** | Every supervised span ends with `<\|im_end\|>`, 968/968 and 108/108. |
| **Truncation at max_seq_len** | **Ruled out** | `max_length=16384`; measured max 7255 (train) / 9821 (val). 0 rows truncated. |
| **Overfitting** | **Not excluded, but orthogonal** | ~90 optimizer steps on a rank-64 LoRA. Val loss was computed in the same wrong format, so it could not have flagged the format gap either way. |
| **Eval harness change** | **Not checkable** | No SFT eval config or result existed in this checkout at diagnosis time. |
| **LR / schedule** | **Unlikely primary** | `lr=2e-5`, `warmup=20` of ~90 steps (~22%). Aggressive but cannot explain a context the model never sees. |
| **Sampling config differing** | **Ruled out as differential** | Qwen3 uses framework defaults, `temperature=0.0` greedy (`models/base.py:185-188`); base and adapter share the identical `VLLMProvider` path. |

**One thing that could not be verified: the symptom itself.** There is no SFT training log, no
SFT eval result anywhere in this checkout, and `/projects/0/gusr0608/msc-thesis` is
`Permission denied`. A session note from 2026-06-17 records **GAIA 16/165 base → 17 GRPO → 14
SFT** and **AIME 12/60 → 8 → 6**; consistent with the diagnosis, but treat as provisional
until the source-of-record run is readable.

---

## 4. What was built

| File | Status | Why |
|---|---|---|
| `scripts/build_sft_parquet.py` | modified | Adds `--format {native,folded}` (default `native`, so existing behaviour is untouched), `--from-parquet`, `--planning-suffix`, `--drop-planning-answers`. New functions: `_memory_user_content`, `_plan_query_analysis`, `_classify_turns`, `_fold_trajectory`, `_fold_records`, `_refold_parquet`, `_default_planning_suffix`. |
| `src/verl_ext/folded_sft_dataset.py` | new | `FoldedSFTDataset` for VERL's `data.custom_cls` hook. Renders the prompt with `add_generation_prompt=True`, byte-for-byte the inference string *including* the think block, and supervises only `target + eos`. Closes the 4-token gap and makes `ignore_input_ids_mismatch` unnecessary. Rejects non-single-turn rows loudly. |
| `scripts/check_sft_folded_format.py` | new | Pre-flight gate, run before training and by the job itself. Asserts prompt identity, span purity, no thinking, no tool output, tool calls retained, no degenerate rows, no truncation. |
| `scripts/finalize_sft_run.py` | new | Selects best-val-loss and last, extracts both as PEFT adapters (DTensor-correct), deletes the shards. |
| `scripts/sft_checkpoint_janitor.py` | new | Collapses checkpoints to adapters *during* training (§7.2). |
| `jobs/fine_tuning/007_train_sft_folded.job` | new | The training job. Supersedes `007_train_sft.job` (since removed — §12), whose training format did not match the orchestrator's inference format. |
| `jobs/fine_tuning/007_run_tests_for_sft_folded.job` | new | CPU-only verification suite: tests, gate, gate trip-wire, gate under the training env, one decoded example. |
| `experiments/configs/qwen3/sft_inference/` | new | Five eval configs (§8). |
| `tests/unit/test_sft_folded_format.py` | new | 32 tests. |

### Key design decisions

**The fold is offline, not in a collator.** It needs `parse_tool_call`, `_extract_sub_goal` and
`_format_action_history` from `agent_engine`. The training env (`cosmas-train`) is not the
inference env (`agent_engine`); importing the orchestrator inside a VERL dataloader would
couple every training run to the agent stack. Offline also makes the folded parquet a
reviewable artifact, which matters when the fold is the thing under suspicion.

**Refold the shipped parquet, don't rebuild from JSONL.** Rebuilding needs
`--reference-parquet data/training/train/combined_train.parquet` for the math:search ratio.
That path is unreadable, and without it the build yields 1224/136 trajectories at 1.53:1
instead of the shipped 968/108 at 1:1, plus a null `extra_info`. Refolding preserves the exact
trajectory set and inherits every control already applied (strip-thinking, strip-suffix,
one-per-question, balance, split). `collected_20260605_214650.jsonl` remains the untouched
source of truth.

**The fold imports the orchestrator's helpers rather than reimplementing them**, so the folded
prompt cannot drift from the real one. A test locks this
(`test_folded_user_turn_matches_the_orchestrators_own_formatter`).

**Turn classification is by position, never by content.** `_build_sft_messages` always emits
`[system, user] + [plan] + output_messages`, so `messages[2]` is the plan whenever one exists.
Content sniffing ("the leading assistant turn with no `<tool_call>` is the plan") silently
drops the plan on the 109 trajectories whose planning turn emitted a tool call.

### The masking spec

Each folded row is exactly `[system, user, assistant]`, one row per orchestrator decision.

| Span | Supervised? | Why |
|---|---|---|
| system turn | No | context |
| folded user turn (question, planning suffix or Query Analysis + Previous Steps) | No | context — **and this is where every simulated tool result lives**, as `- Result:` prose, so tool output cannot enter the loss by construction |
| `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | No | generation prompt; the model does not produce it |
| assistant target text | **Yes** | orchestrator-authored; for action rows this includes `<sub_goal>` and `<tool_call>`, because issuing the call is the decision being learned |
| trailing `<\|im_end\|>` | **Yes** | teaches the stop token |

**No tools execute.** Training is static cross-entropy: no sub-agent vLLM server, no AgentFlow
server, no rollout workers, no `SERPER_API_KEY`/`TAVILY_API_KEY`. `FoldedSFTDataset` only
tokenises.

**No thinking in targets.** Stripped at build time; re-asserted by tests and the pre-flight gate.

---

## 5. The folded data

Built with:

```bash
python scripts/build_sft_parquet.py \
    --from-parquet data/training/sft/sft_train.parquet \
    --output-dir data/training/sft --output-name sft_folded_train.parquet
```

| | Train | Val |
|---|---|---|
| Trajectories → rows | 968 → **2995** (3.09x) | 108 → **362** (3.35x) |
| Row kinds | 968 plan + 1234 action + 793 answer | 108 + 160 + 94 |
| plan-with-tool-call trajectories handled | 109 | 16 |
| planning-answered trajectories handled | 175 | 14 |
| Tokens: total / mean / p95 / max | 4,062,350 / 1356.4 / 2040 / 7300 | 500,014 / 1381.3 / 2032 / 9832 |
| (multi-turn, for comparison) | 1,587,150 / 1639.6 / 2338 / 7255 | 186,668 / 1728.4 / 2426 / 9821 |

### Why the row count multiplied (968 → 2995)

No data was added. **A multi-turn row is a whole trajectory; a folded row (consistent with
inference) is one orchestrator decision.** Those 968 trajectories contain exactly 2995 assistant turns, so 2995 is the
assistant-turn count, not a multiplier applied to anything:

| assistant turns in a trajectory | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| trajectories (train) | 175 | 48 | 383 | 280 | 57 | 15 | 4 | 2 | 4 |
| trajectories (val) | 14 | 4 | 43 | 33 | 10 | 1 | 2 | 0 | 1 |

The distribution sums to exactly 2995, which also proves the fold drops no assistant turn. The
175 one-turn trajectories are the turn-0 answers of decision 1 in §6. A typical 3-turn
trajectory is plan → one tool call → answer.

**The supervision is byte-for-byte identical.** Measured over every row:

| | multi-turn | folded | ratio |
|---|---|---|---|
| supervised tokens, train | 589,805 | **589,805** | 1.000 |
| supervised tokens, val | 66,556 | **66,556** | 1.000 |

The same assistant text is the target in both; it is only distributed one-per-row instead of
many-per-row.

**The token growth (1.59M → 4.06M) is prompt, not target:**

```
3,472,545 prompt  +  589,805 supervised  =  4,062,350
```

The system prompt and question are re-rendered in every row because that is literally what the
orchestrator does at inference: `_build_memory_prompt` rebuilds `[system, user]` from scratch
at every decision. The multi-turn format amortised that prefix across one long sequence; the
deployed system never does. The extra ~2.5M tokens are the real conditioning cost of the
format the model is evaluated in, not bloat added by the fold.

Two consequences, both already handled:

1. **Steps per epoch tripled**: 2995 / 32 ≈ 94 steps vs 968 / 32 ≈ 30. This is why job 008 sets
   `total_epochs=2` (~187 steps) where the removed `007_train_sft.job` used 3 (~90): the run was
   matched on optimizer steps, not on epochs.
2. **Long trajectories now contribute more rows.** A 9-turn trajectory yields 9 rows where a
   turn-0 answer yields 1. Search trajectories run longer than math ones, so the 484/484
   trajectory balance becomes 1172/1823 at row level. That is the mechanism behind decision 2
   in §6, not a separate problem.

### Pre-flight gate results

Run on every row, in both `agent_engine` and `cosmas-train` (different transformers versions,
identical results), and re-verified under verl's real config after the §9 fix:

| Gate | Train | Val |
|---|---|---|
| Prompt token-identical to the inference prompt | **2995 / 2995** | **362 / 362** |
| Supervised span == target + `<\|im_end\|>` | **2995 / 2995** | **362 / 362** |
| Thinking in supervised span | 0 | 0 |
| Tool result in supervised span | 0 | 0 |
| Degenerate rows (empty target) | 0 | 0 |
| Rows over `max_length=16384` | 0 | 0 |
| Targets retaining `<tool_call>` | 1343 (1234 actions + 109 plans) | 176 |
| `extra_info` populated | 2995/2995 | 362/362 |
| Every multi-turn question present | yes | yes |
| Columns identical to multi-turn | yes | yes |
| No question straddles train/val | yes | |

The gate is a real trip-wire, not a rubber stamp: it exits **1** on the multi-turn parquet and **0**
on the folded one, both confirmed.

Files on disk (gitignored via `.gitignore:48 data/*`):

```
data/training/sft/sft_folded_train.parquet   2.8 MB   2995 rows
data/training/sft/sft_folded_val.parquet     243 KB    362 rows
data/training/sft/sft_folded_{train,val}.jsonl        (human-readable mirrors)
```

---

## 6. Research decisions — both resolved

**1. Turn-0 answers (175 train rows, 18.1%): KEPT.** These teach answering at turn 0, which is
the *Premature direct answering* failure mode in the Ch 6 taxonomy. `--drop-planning-answers`
is implemented and tested but **not used**. Rationale: format is then the only variable that
changed against the (now-removed) 007 run, so the comparison stayed clean. If *Premature direct answering*
shows up in the eval, dropping them is a one-flag rebuild and a follow-up run.

**2. Loss-level composition shift: ACCEPTED, to be stated in Ch 7.** Trajectories stay 484 math
/ 484 search (1:1), but rows become **1172 math / 1823 search (1:1.56)**, because search
trajectories take more steps and expand into more rows. This is inherent to one-row-per-decision,
not a defect — but "differs in format only" is true at the *trajectory* level and **false at the
loss level**, so it is a confound that must be disclosed rather than left implicit. Subsampling
search rows back to 1:1 would discard ~651 real supervision rows to preserve a ratio that was
itself only a heuristic mirror of `combined_train`.

**Ch 7 needs one sentence to this effect:** the folded and multi-turn runs use the same trajectories
and the same balanced 1:1 math/search sampling, but because the folded format supervises one row
per orchestrator decision and search trajectories take more steps, the loss-level ratio is
1:1.56 rather than 1:1.

---

## 7. Training-run mechanics

### 7.1 What verl actually writes

Four facts about `verl.trainer.sft_trainer`, all established by inspecting a real checkpoint
from job 25268454. Each invalidates something previously assumed.

**1. The SFT trainer never writes a `lora_adapter/` directory.** The RL path does
(`fsdp_workers.py`, which the GRPO run depends on), but the SFT path only ever *reads*
`lora_adapter_path` (`verl/trainer/sft_trainer.py:101`). So an SFT checkpoint contains no PEFT
adapter, and `huggingface/` holds only tokenizer and config files, no weights.

**2. It writes the full FSDP state dict instead: ~33 GB per checkpoint.**

| file | size | needed? |
|---|---|---|
| `model_world_size_2_rank_{0,1}.pt` | 16 GB each | yes, but 99% is unchanged base weights |
| `optim_world_size_2_rank_{0,1}.pt` | 667 MB each | no (nothing resumes; dropped via `save_contents`) |
| `extra_state_*`, `data_*`, `*.json` | ~15 KB | trivial |

Only ~350 MB of that is trained LoRA weight.

**3. The shards are sharded DTensors, not replicas.** `rank_0` holds `(32, 4096)` of a
`(64, 4096)` `lora_A` — placement `Shard(dim=0)`. Reconstruction must concatenate **every** rank
along each tensor's own shard dimension.

> **`scripts/merge_lora.py` cannot be used on these checkpoints.** It reads
> `model_world_size_*_rank_0.pt` alone, documenting it as "the rank-0 consolidated shard
> regardless of training world size". That is false here. It also expects an `actor/`
> subdirectory, which the SFT trainer does not create.

**4. `lora_train_meta.json` records the true hyperparameters** (`{"r": 64, "lora_alpha": 64,
"task_type": "CAUSAL_LM"}`). `finalize_sft_run.py` prefers these over its CLI arguments, because
a mismatched alpha silently rescales every adapter weight and nothing downstream would flag it.

The adapter size is independently predictable from the model config: 7 target modules x 2
(A,B) x 36 layers = **504 tensors**, 174,587,904 params, **333 MiB** in bf16 — matching the 504
tensors and 334 MiB actually produced. A full model would be ~16 GB.

### 7.2 Checkpoints are collapsed to adapters during training

Converting only at the end peaks at ~8 x 32 GB = **~256 GB**, and a cancelled job leaves all of
it behind (this is what job 25268454 did). `scripts/sft_checkpoint_janitor.py` runs in the
background for the duration of training and collapses each checkpoint into
`step_adapters/step_<N>/` as soon as it is complete, deleting the shards. Peak becomes one or
two checkpoints (~32-64 GB) plus a few GB of adapters.

**Knowing a checkpoint is finished is not guesswork.** `fsdp_checkpoint_manager.py` ends
`save_checkpoint` with a `torch.distributed.barrier()` after every rank has written and closed
its shard; `CheckpointHandler.save_checkpoint` then has rank 0 write
`latest_checkpointed_iteration.txt` as write-temp-plus-`os.rename`, which is atomic. So the
tracker naming step N implies `global_step_N` is complete on all ranks. The janitor only ever
touches steps at or below the tracker value.

Four properties make deletion safe:

1. **Nothing reads these back.** `trainer.resume_mode=disable`.
2. **Shards are deleted only after the adapter exists** on disk.
3. **Extraction is atomic.** It writes to `.tmp_step_<N>/` and renames into `step_<N>/`. A
   janitor killed mid-write would otherwise leave a truncated `adapter_model.safetensors` that
   the next pass reads as finished, and the shards would be deleted against it. The staging name
   does not match the `step_*` glob, so nothing downstream can pick it up.
4. **Failure is local.** One bad checkpoint is logged and skipped with its shards intact;
   `finalize_sft_run.py` retries it at the end. If the janitor dies entirely, the run degrades
   to end-of-run conversion.

The job stops the janitor with a stop file and `wait`s for its final sweep **before**
`finalize_sft_run.py` runs, so the two never touch the same checkpoint. A `trap` on
`EXIT INT TERM` stops it on `scancel` or a crash. It runs with `CUDA_VISIBLE_DEVICES=""` so a
CPU-only process cannot take a CUDA context from the training GPUs.

### 7.3 Selection, extraction, archiving

`finalize_sft_run.py` takes the union of `step_adapters/step_<N>/` (already collapsed) and any
remaining `global_step_<N>/` (shards); if a step has both, the adapter wins. Best (lowest
val/loss) and last are selected across that union, a pre-extracted adapter is copied rather than
re-extracted, and everything else — including the whole `step_adapters/` tree — is deleted. It
derives `target_modules` from the checkpoint keys rather than saving the literal string
`"all-linear"`, so load-time re-resolution cannot change which modules the adapter claims to
cover. A created-but-unwritten step dir (left by a cancelled save) is never selected as "last".

The job then **archives the adapters off scratch** into
`data/adapters/<experiment>/<run-tag>/` inside the checkout (`data/*` is gitignored; ~350 MB
against a 200 GiB home quota at ~65% used).

> **Scratch is not durable storage.**
> `/scratch-shared/azywot/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math{,-v2}/` are
> **empty** — purged around 2026-08-03 — while all ten
> `experiments/configs/qwen3/lora_inference/*` configs still point at
> `…/global_step_40/actor/lora_adapter`. Any GRPO number already in the thesis stands (it came
> from a completed eval), but **the GRPO evaluation cannot currently be re-run or extended**
> unless a merged copy survives elsewhere. This is why SFT adapters are archived automatically.

### Verified

A simulated run wrote four checkpoints with the tracker updated exactly as verl does, plus a
fifth the tracker never reached, with the janitor live alongside:

| Property | Result |
|---|---|
| Checkpoints collapsed during the run | 4 of 4, each deleted right after conversion |
| Checkpoint above the tracker value | left untouched, as required |
| Janitor-made adapter vs direct extraction | **bit-identical** |
| Staging dir visible to discovery | no |
| Finalize over the mixed state | best = step 50 (reused adapter), last = step 125 (extracted from shards) |
| Survivors | `best_adapter/`, `last_adapter/`, `selection.json` only |

Reconstruction was separately verified bit-exact against synthetic 2-rank sharded DTensors,
including an uneven split (63 rows over 2 ranks) and a non-zero shard dim, with the shape guard
confirmed to reject a missing rank.

---

## 8. Evaluation

`experiments/configs/qwen3/sft_inference/{gaia,hle,gpqa,aime,musique}/qwen8B_sub1_7b_none.yaml`,
generated by the `sft_inference` suite in `scripts/generate_configs.py`.

- **`_none` only.** Thinking is stripped from the SFT data at build time and the folded prompt
  renders an empty `<think>\n\n</think>`, so an `ORCHESTRATOR_ONLY` eval would sample the adapter
  off the distribution it was trained on. This also keeps the run inside the Ch 7 rule that
  everything is compared no-thinking.
- The orchestrator is named **`Qwen3-8B-SFT`**, so the `model_name` column of the W&B export
  separates it from the GRPO rows (`Qwen3-8B-LoRA`) in `orchestrator_ft_results.csv`.
- `lora_adapter_path` points at the **archive** path with a `<run-tag>` placeholder, which must
  be filled in after training. The run tag is `$(date +%d-%m-%Y_%H-%M)-$SLURM_JOB_ID` and is
  printed by the job.
- `max_lora_rank` defaults to 64 in `models/base.py:205`, matching the trained rank, so no
  config override is needed.
- Nothing else needs changing: `run_all_in_folder.sh` → `launch_experiment.sh` →
  `generate_job.py` builds the SLURM script generically from each config's `slurm:` block.

**Caution.** The committed `lora_inference` configs were **hand-edited after generation** (a
`_v2` name suffix and the `…-v2/…/global_step_40` adapter path); regenerating that suite
silently overwrites those edits. `generate_configs.py` is not the source of truth for
`lora_inference`. Prefer editing `SFT_ADAPTER_PLACEHOLDER` and regenerating over hand-editing
five files.

### Results (run tag `06-08-2026_21-47-25300018`, evaluated 2026-08-07)

All five configs were run against `best_adapter` (== `last_adapter`, both step 186). Raw output
under `experiments/results/sft_inference/<dataset>/qwen8B_sub1_7b_none/*/metrics.json`. Compared
against the thesis's matched no-thinking baseline (Table `tab:adaptation_combined`, 8B
orchestrator + frozen 1.7B sub-agents, no adapter) and the existing GRPO-LoRA row from
`orchestrator_ft_results.csv`:

| Benchmark | Baseline | GRPO-LoRA | **SFT-folded** |
|---|---|---|---|
| GAIA | 7.9 (13/165) | 12.7 (21/165) | **12.7 (21/165)** |
| GPQA | 41.9 (83/198) | 46.5 | **41.4 (82/198)** |
| AIME | 23.3 | 18.3 | **10.0 (6/60)** |
| MuSiQue | 9.5 | 14.0 | **17.5 (35/200)** |
| HLE | 3.0 | 10.0 | **8.5 (17/200)** |

The format fix clears the bar it was built for on three of five benchmarks: SFT now beats the
pre-adaptation baseline on GAIA, MuSiQue and HLE, which the pre-fold multi-turn adapter never
did (§2 — every prior SFT number sat below base). GPQA is a wash (82 vs 83 correct, within noise).

**AIME is the open question.** SFT-folded scores 6/60 (10.0%), matching the *pre-fold* multi-turn
adapter's AIME number from the 2026-06-17 session note exactly (also 6/60) and well below both
the baseline (23.3%) and GRPO-LoRA (18.3%). Total tool calls on AIME collapsed to 20 (3
`web_search` + 17 `code_generator`) against the baseline run's 79 — the adapter is answering from
memory far more than the baseline does. A plausible mechanism is the row-level composition shift
from §6 (folding dilutes math to 1172/2995 rows against search's 1823/2995, versus the 1:1
trajectory-level balance), but that is a hypothesis, not yet established: the training-side
tool-call rate wasn't measured directly, and no ablation (e.g. re-running §10's
`--drop-planning-answers`, or rebalancing rows rather than trajectories) has been tried. Needs one
sentence in Ch 7 either way; do not claim the mechanism without further evidence.

---

## 9. Pre-launch audit: one critical bug

A line-by-line audit of the whole chain before the first real run.

### CRITICAL, fixed: the training run would have lost the empty think block

`FoldedSFTDataset` read `config.get("enable_thinking_default", False)`. That default only
applies when the key is *absent*, and it is not: verl's `sft_trainer_engine.yaml:29` ships
`enable_thinking_default: none`, which YAML parses as the **string** `"none"`, and
`sft_trainer.py:471-473` passes the entire `data` config into `data.custom_cls`. So the real
training run rendered prompts with `enable_thinking="none"` — truthy — and Qwen3's template took
its thinking branch:

| | prompt tokens | ends with |
|---|---|---|
| Inference (`thinking_mode: NO`) | 19 | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| Training, as configured | **15** | `<\|im_start\|>assistant\n` |

Measured on real rows: the training prompt was **4 tokens short on every row** — precisely the
gap §2 describes and this whole format change exists to close.

**Why nothing caught it.** `check_sft_folded_format.py` built its own small config dict that did
not contain the key, so it got `False` and the correct prompt. The gate passed while training
was wrong: it was validating a code path training never took.

Fixed in three independent places:

1. `FoldedSFTDataset` accepts only an unambiguous boolean (or `"true"`/`"false"`) via
   `_strict_bool`; anything else falls back to no-thinking **and logs a warning**. It also logs
   the resolved `enable_thinking` / `pad_mode` / `truncation` at construction.
2. `check_sft_folded_format.py` loads **verl's real config defaults** and builds the dataset from
   them, so the gate exercises the configuration training uses. It reproduces them hard-coded
   when verl is not importable, so it is never weaker than training.
3. `007_train_sft_folded.job` passes `data.enable_thinking_default=false` explicitly.

Regression tests parametrise over `"none"`, `None`, `"None"`, `""`, `0` (must stay no-thinking)
and over real booleans (must be honoured). Re-verified with the job's exact config on both real
splits: 2995 and 362 rows, prompt identical to inference, target ends in `<|im_end|>`, 0 rows
over `max_length`.

### Also changed by the audit

- **`data.truncation=right` → `error`.** The gate proves 0 of 3357 rows exceed `max_length`
  (observed max 9832), so this can only fire on a regression, and `right` would silently drop the
  tail of a target along with its `<|im_end|>`, training the model never to stop.
- **The janitor runs with `CUDA_VISIBLE_DEVICES=""`.**

### Checked and found correct

- `_memory_user_content` reproduces `_build_memory_prompt` (`orchestrator.py:641-664`) exactly,
  including the `"\n".join` structure.
- `_plan_query_analysis` matches `_run_planning_turn` (`orchestrator.py:758-772`) branch for
  branch, including the "tool call at position 0 keeps the whole text" edge case.
- The action-history entry matches `_commit_tool_result` (`orchestrator.py:681-686`) field for
  field: `tool_name` from the re-parsed call, `sub_goal` via the orchestrator's own extractor,
  `command` as `json.dumps(tool_call)`, `result` as the stored tool content.
- `_build_sft_messages` stores `state.messages[:2]`, so the folded row's `messages[1]` is the
  same string `_build_memory_prompt` reads at inference.
- Per-row system prompts are self-consistent with the data source (deepmath → math prompt,
  hotpotqa/nq → search prompt; 1172 / 1823 rows).
- Every assistant turn becomes exactly one row (§5).
- Qwen3's `eos_token_id` is `<|im_end|>` (151645), so supervising `target + eos` teaches the
  right stop token. `pad_token` is `<|endoftext|>` and differs, as it should.

### Open observations, not fixed (low impact)

- **`orchestrator.py:320,364` uses `states[0].messages[0]` as the system prompt for a whole
  batch.** With two distinct system prompts in the collection set, a mixed batch means some
  teacher trajectories were generated under the other one. The fold uses each trajectory's own
  system prompt, which is correct for the student; this is a provenance caveat about the teacher
  data and a pre-existing orchestrator quirk, not a fold bug.
- **`_strip_thinking` and `strip_thinking_tags` differ** on the orphaned-`</think>` case (OLMo
  style). Irrelevant for the Qwen3 teacher and empirically 0 rows contain think tags, but the two
  would diverge if the teacher family ever changed.
- `latest_checkpointed_iteration.txt` survives finalize and names a deleted step. Harmless with
  `resume_mode=disable`.

---

## 10. How to run it

```bash
cd /gpfs/home3/xchen1/azywot/msc-thesis

# 0. Data prep, only needed from a fresh checkout (§5 has the shipped-run numbers):
#    collect trajectories (produces collected_<ts>.jsonl, not a parquet), build the
#    multi-turn sft_train/val.parquet from it, then fold those into the rows
#    (consistent with inference) that training uses.
#    --reference-parquet must be passed explicitly on the middle command — the script's
#    own default points at an unreadable /projects/0/gusr0608 path.
sbatch jobs/fine_tuning/006_collect_sft_data.job
python scripts/build_sft_parquet.py data/training/sft/collected_<ts>.jsonl \
    --output-dir data/training/sft --reference-parquet data/training/train/combined_train.parquet
python scripts/build_sft_parquet.py --from-parquet data/training/sft/sft_train.parquet \
    --output-dir data/training/sft --output-name sft_folded_train.parquet

# 1. Full verification suite (CPU partition, no GPU cost)
sbatch jobs/fine_tuning/007_run_tests_for_sft_folded.job

# 2. Train — ~187 steps, 2x H100, ~40 min wall clock, schedules immediately
sbatch jobs/fine_tuning/007_train_sft_folded.job

# 3. After training: paste the run tag the job prints into SFT_ADAPTER_PLACEHOLDER
#    in scripts/generate_configs.py, then regenerate and evaluate
python scripts/generate_configs.py --suite sft_inference
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/sft_inference
```

The training job runs the pre-flight gate itself and refuses to start training if it fails. It
also extracts the adapters and archives them, so there is no manual post-training step.

Local checks, worth re-running after any change to `build_sft_parquet.py` or
`folded_sft_dataset.py`:

```bash
bash -n jobs/fine_tuning/007_train_sft_folded.job   # and 007_run_tests_for_sft_folded, 006
conda activate agent_engine
python -m pytest tests/ -q --ignore=tests/unit/test_fine_tuning_rollout.py   # expect 496
python scripts/check_sft_folded_format.py \
    --folded data/training/sft/sft_folded_train.parquet \
    --native data/training/sft/sft_train.parquet --max-length 16384
```

To redo checkpoint selection by hand (e.g. after a crash), `--dry-run` reports what it would
keep without writing or deleting:

```bash
python scripts/finalize_sft_run.py \
    --ckpt-dir /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-sft-folded-v1/<run-tag> \
    --log out/fine_tuning/sft_train/sft_folded_<jobid>_verl.log --dry-run
```

Optional: rebuild dropping the 175 turn-0 answers (decision 1 in §6):

```bash
python scripts/build_sft_parquet.py \
    --from-parquet data/training/sft/sft_train.parquet \
    --output-dir data/training/sft --output-name sft_folded_train.parquet \
    --drop-planning-answers
```

**Cluster facts:** `gpu_h100` is fully available (`sbatch --test-only --partition=gpu_h100
--gpus=2` schedules immediately), account `gusei18108`, budget `EINF-18108/L2` had ~536,000 SBU
left. Scratch quota is 8 TiB at ~0.2% used. `$USER` on the compute node is `azywot`, not
`xchen1`, so checkpoints land under `/scratch-shared/azywot/`.

---

## 11. What is not done

- **Training and eval are done** (§1, §8) — run tag `06-08-2026_21-47-25300018`, five-benchmark
  eval on 2026-08-07. **Not yet done (re-verified 2026-08-15):** folding the §8 numbers and the
  AIME regression into Ch 7 of the thesis — `sections/7_adaptation.tex` still has only the GEPA
  and LoRA/GRPO methods, no SFT section, none of the §8 numbers anywhere — and investigating
  *why* AIME regressed (hypothesis in §8, not yet tested; no newer analysis artifact exists).
- `/projects/0/gusr0608` is no longer referenced by any SFT code path, but still appears in
  **6** `jobs/grpo_inference/*.job` files (re-verified 2026-08-15, was previously miscounted as
  10): `GRPO_eval_{gaia,aime}`, `SFT_eval_{gaia,aime}`, `BASE_eval_{gaia,aime}`, all
  `qwen8B_sub1_7b_none.job`, hardcoding `/gpfs/work5/0/gusr0608/msc-thesis`. Those are historical
  eval jobs that already ran; worth cleaning up only if those evals are re-run.

---

## 12. History

**The fold implementation was lost once already.** An earlier version existed in an ephemeral
worktree and vanished: no `--format folded`, no `_fold_trajectory`, no tests, and nothing in any
commit (`git log --all -S"_fold_trajectory"` was empty), despite the docs of the time claiming
it was "implemented, tested, and run successfully". Its *numbers* all reproduced exactly, so
those docs were a reliable spec, but the code was rewritten from scratch. `select_best_sft_
checkpoint.py`, referenced by `007_train_sft.job:180`, likewise never existed in any commit
(`007_train_sft.job` has since been removed from the repo entirely — see §1).

**Job 25268454 — ran 8m43s, then cancelled on purpose.** The in-job pre-flight gate passed on
both splits. What it proved before being stopped:

| step | val/loss |
|---|---|
| 25 | 0.602511 |
| 50 | 0.494696 |

So the folded pipeline trains, and trains fast: 50 steps in ~8 minutes, meaning the full
~187-step run is roughly **40 minutes**, not the multi-hour job the old 8 h wall-clock limit
suggested. It was cancelled for three reasons, all about output rather than training:
checkpoints were 33 GB each and all were kept, and they went to `…/fine_tuning/sft/…` rather
than the GRPO layout. Cancelling cost ~9 minutes of 2xH100 and avoided writing ~230 GB that
would have had to be deleted anyway. The cancel landed mid-save, leaving an empty
`global_step_50/` — which is why `finalize_sft_run.py` skips checkpoint dirs with no model
shards instead of treating one as "last". Its 33 GB of leftovers were deleted on 2026-08-06.

**Two failed submissions (25268308, 25268330) before that.** Job 008 resolved `PROJECT_DIR` from
`${BASH_SOURCE[0]}`, which works under `bash jobs/...` but not under `sbatch` — SLURM copies the
batch script to `/var/spool/slurm` on the compute node. Fixed in both 008 jobs by trying
candidates in order and accepting only one that *validates* as the repo (`scripts/build_sft_
parquet.py` present): `$PROJECT_DIR` → `$SLURM_SUBMIT_DIR` → script dir →
`$HOME/azywot/msc-thesis`. **007 had the same class of fragility** (a hardcoded
`/projects/0/gusr0608`); moot now that it has been removed.

**Post-cancel fixes**, all since verified: checkpoints moved to the GRPO layout;
`checkpoint.save_contents=['model','extra']`; `trainer.resume_mode=disable`;
`finalize_sft_run.py` written; `build_sft_parquet.py` defaults moved in-repo; `PROJECT_DIR`
de-hardcoded in 006 and 007. Then: two unreachable error branches in 008 fixed (under
`set -euo pipefail` a bare failing command aborts before `$?` is read), `--time` 30h → 2h,
adapter archiving added, the eval-config suite added, the janitor added, and the §9 audit.

**Job 25299486 (verification suite) — stage 1 failed on a collection error, not a test.** The
job ran a bare `python -m pytest -q --ignore=...` with no path, so pytest collected the whole
repo. `data/training/{smoke,test,train,val}` carry the read bit but **not the execute bit**
(`drw-r-----`), so probing them for a `conftest.py` raises `PermissionError` and collection
aborts before any test runs. Every other stage passed, including all 32 folded-format tests,
both gates, the trip-wire, and the gate under `cosmas-train`.

Every local verification had used an explicit `tests/` path and so never walked `data/` —
the same shape of gap as the §9 bug: the checking command differed from the one that runs.
Fixed at both levels: `pyproject.toml` now sets `testpaths = ["tests"]` so *any* invocation is
scoped, and the job passes `tests/` explicitly. Verified by reproducing the four
`PermissionError`s with `--override-ini="testpaths=."` and then getting **496 passed** from the
job's exact bare command.

> **Worth acting on separately:** those four directories are unreadable *only* because they
> lack the execute bit, which is almost certainly accidental. That is why
> `data/training/train/combined_train.parquet` reads as inaccessible — the rationale recorded
> in §4 for refolding the shipped parquet rather than rebuilding from JSONL. A
> `chmod u+x data/training/{smoke,test,train,val}` would likely restore access and make the
> reference parquet usable again. Not applied here, since it changes data-directory
> permissions outside the scope of this work.

**Test suite: 496 passed.** One pre-existing collection error,
`tests/unit/test_fine_tuning_rollout.py` (`No module named 'agentops'`), unrelated to this work
and present before it.

---

## Related documents

- `docs/grpo_sft_walkthrough.md` — how to run the fine-tuning pipeline end to end (conda env, HF
  access, API keys, smoke tests, training, LoRA inference, merging). Covers GRPO as well.
- `docs/fine_tuning_v2/verl_documentation_lora.md` — vendored verl reference on LoRA config.
- `docs/failure_modes_fine_tuning_alignment.md` — links the Ch 6 failure modes to fine-tuning
  design.
- MALGAI workshop paper: https://openreview.net/forum?id=RLHUTcvsA9
