# Orchestrator SFT: status and handover

**Last updated:** 2026-08-06
**Branch:** `feat/sft-folded-format`
**Read this first.** It supersedes the status sections of the older docs listed at the end.

---

## 1. Where things stand, in five lines

The SFT adapter underperformed the base model because it was trained on a conversation
format the orchestrator never sees at inference. That diagnosis is **verified by executing
both code paths**, not inferred. The fix is **implemented, tested, and applied to the real
data**: folded parquets are built and every pre-flight gate passes on all 3357 rows. Both
research decisions are **resolved** (§8).

**A first training run went 50 steps and was deliberately cancelled** (§14). It proved the
pipeline works end to end and that val loss falls (0.6025 → 0.4947), but it was writing
33 GB per checkpoint, to the wrong directory, and keeping all of them. The job now writes to
the GRPO adapter layout and ends by keeping only two ~365 MB adapters (best-val-loss and
last).

**NEXT ACTION: submit** — see §12. The post-cancel edits have now been verified (syntax, 487
tests, both pre-flight gates), the two job-script defects found in that pass are fixed, and the
SFT eval configs exist. Nothing has been submitted to SLURM.

---

## 2. The bug, and the evidence for it

### What was wrong

SFT rows were stored as the native multi-turn transcript that the teacher produced:

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

Prior steps are compressed into `Action Step N: Tool / Sub-goal / Command / Result` prose,
and the tool result becomes `- Result:` text inside the user turn rather than its own turn.

So the adapter minimised cross-entropy honestly, on a distribution it is never evaluated on.
Loss went down; task performance went down. **Val loss could not have caught this**, because
the val split had exactly the same defect.

### How it was verified (not assumed)

Both sides were produced by running the production code:

- **Training side:** `verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`,
  constructed with the exact flags `007_train_sft.job:135-142` passes, then decoding
  `input_ids` and the `loss_mask == 1` positions.
- **Inference side:** the orchestrator's own `_build_memory_prompt`, `_commit_tool_result`,
  `_format_action_history`, `_extract_sub_goal` and the real `parse_tool_call`, driven on a
  real `ExecutionState`, rendered through the Qwen3 branch of
  `VLLMProvider._render_messages` (`vllm_provider.py:338-343`).

The only function that could not be *invoked* is `_run_planning_turn` (it needs a live
model); its two prompt-shaping lines were mirrored with the suffix constant imported from
the module. The Qwen3 render branch was asserted programmatically, not assumed.

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

`MultiTurnSFTDataset` tokenises each turn in isolation, so a target always begins right
after `<|im_start|>assistant\n`. Qwen3's template rendering a whole conversation — and the
orchestrator at inference with thinking off — first emits `<think>\n\n</think>\n\n`. The
adapter was trained to start generating 4 tokens earlier than it is ever sampled from, on
**every** target.

VERL's own sanity check fails on **60/60** sampled rows, delta **exactly 4 tokens** every
time. `007_train_sft.job:141` silenced it with `data.ignore_input_ids_mismatch=True`.
Folding alone does **not** fix this — measured on a real folded row:

| Path | Prompt tokens | Identical to inference? |
|---|---|---|
| Folded row through `MultiTurnSFTDataset` | 1374 | **No** — missing exactly `<think>\n\n</think>\n\n` |
| Folded row through `FoldedSFTDataset` (new) | 1378 | **Yes, token-for-token** |

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
| **Overfitting** | **Not excluded, but orthogonal** | ~90 optimizer steps on a rank-64 LoRA. Needs the training log to settle — and val loss was computed in the same wrong format, so it could not have flagged the format gap either way. |
| **Eval harness change** | **Not checkable** | No SFT eval config or result exists in this checkout. |
| **LR / schedule** | **Unlikely primary** | `lr=2e-5`, `warmup=20` of ~90 steps (~22%). Aggressive but cannot explain a context the model never sees. |
| **Sampling config differing** | **Ruled out as differential** | Qwen3 uses framework defaults, `temperature=0.0` greedy (`models/base.py:185-188`); base and adapter share the identical `VLLMProvider` path. |

**One thing I could not verify:** the symptom itself. There is no SFT training log, no SFT
eval config, and no SFT eval result anywhere in this checkout, and
`/projects/0/gusr0608/msc-thesis` is `Permission denied`. A session note from 2026-06-17
records **GAIA 16/165 base → 17 GRPO → 14 SFT** and **AIME 12/60 → 8 → 6**; consistent with
the diagnosis, but treat as provisional until the source-of-record run is readable.

---

## 4. Three things the older docs got wrong

1. **The fold implementation did not exist.** `docs/sft_status.md` §4 (old) and the design
   doc both said it was "implemented, tested, and run successfully". It was not: no
   `--format folded`, no `_fold_trajectory`, no `tests/unit/test_build_sft_folded*.py`, and
   nothing in any commit (`git log --all -S"_fold_trajectory"` was empty). It was lost with
   an ephemeral worktree. Its *numbers* all reproduced exactly, so the docs were a reliable
   spec — but the code had to be rewritten from scratch.
2. **"Blocked: training partition access" was wrong.** `gpu_h100` is fully available:
   `sbatch --test-only --partition=gpu_h100 --gpus=2` schedules immediately (it offered
   `gcn118`), account is `gusei18108`, and budget `EINF-18108/L2` has **~536,000 SBU** left.
   What is inaccessible is only the *directory* `/projects/0/gusr0608/msc-thesis`, which
   `007_train_sft.job` hardcoded as `PROJECT_DIR`. The data is in the home checkout and
   `data/training/sft` is writable.
3. **The assistant-side gap was listed as an open decision.** It is now closed (§5).

Two smaller gaps in the same class: `scripts/select_best_sft_checkpoint.py` is called by
`007_train_sft.job:180` but has never existed in any commit; and there is **no SFT eval
config at all** — every `experiments/configs/qwen3/lora_inference/*` points its
`lora_adapter_path` at the **GRPO** adapter.

---

## 5. What was built, and why each piece exists

| File | Status | Why |
|---|---|---|
| `scripts/build_sft_parquet.py` | modified | Adds `--format {native,folded}` (default `native`, so existing behaviour is untouched), `--from-parquet`, `--planning-suffix`, `--drop-planning-answers`. New functions: `_memory_user_content`, `_plan_query_analysis`, `_classify_turns`, `_fold_trajectory`, `_fold_records`, `_refold_parquet`, `_default_planning_suffix`. |
| `src/verl_ext/folded_sft_dataset.py` | new | `FoldedSFTDataset` for VERL's `data.custom_cls` hook. Renders the prompt with `add_generation_prompt=True`, which is byte-for-byte the inference string *including* the think block, and supervises only `target + eos`. Closes the 4-token gap and makes `ignore_input_ids_mismatch` unnecessary. Rejects non-single-turn rows loudly. |
| `scripts/check_sft_folded_format.py` | new | Pre-flight gate, run before training. Asserts prompt identity, span purity, no thinking, no tool output, tool calls retained, no degenerate rows, no truncation. |
| `jobs/fine_tuning/008_train_sft_folded.job` | new | The training job. 007 is left untouched so the native run stays reproducible for comparison. |
| `jobs/fine_tuning/008_test_sft_folded.job` | new | CPU-only verification suite: tests, gate, gate trip-wire, gate under the training env, and one decoded example. See §14. |
| `tests/unit/test_sft_folded_format.py` | new | 23 tests. |
| `docs/sft_folded_relaunch_plan.md` | new | The full plan, with deeper evidence tables than this doc. |

### Key design decisions

**The fold is offline, not in a collator.** It needs `parse_tool_call`, `_extract_sub_goal`
and `_format_action_history` from `agent_engine`. The training env (`cosmas-train`) is not
the inference env (`agent_engine`); importing the orchestrator inside a VERL dataloader
would couple every training run to the agent stack. Offline also makes the folded parquet a
reviewable artifact, which matters when the fold is the thing under suspicion.

**Refold the shipped parquet, don't rebuild from JSONL.** Rebuilding needs
`--reference-parquet data/training/train/combined_train.parquet` for the math:search ratio.
That path is unreadable, and without it the build yields 1224/136 trajectories at 1.53:1
instead of the shipped 968/108 at 1:1, plus a null `extra_info`. Refolding preserves the
exact trajectory set and inherits every control already applied (strip-thinking,
strip-suffix, one-per-question, balance, split). `collected_20260605_214650.jsonl` remains
the untouched source of truth.

**The fold imports the orchestrator's helpers rather than reimplementing them**, so the
folded prompt cannot drift from the real one. A test locks this
(`test_folded_user_turn_matches_the_orchestrators_own_formatter`).

**Turn classification is by position, never by content.** `_build_sft_messages` always emits
`[system, user] + [plan] + output_messages`, so `messages[2]` is the plan whenever one
exists. Content sniffing ("the leading assistant turn with no `<tool_call>` is the plan")
silently drops the plan on the 109 trajectories whose planning turn emitted a tool call.

### The masking spec (requirements 1–3 from the brief)

Each folded row is exactly `[system, user, assistant]`, one row per orchestrator decision.

| Span | Supervised? | Why |
|---|---|---|
| system turn | No | context |
| folded user turn (question, planning suffix or Query Analysis + Previous Steps) | No | context — **and this is where every simulated tool result lives**, as `- Result:` prose, so tool output cannot enter the loss by construction |
| `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | No | generation prompt; the model does not produce it |
| assistant target text | **Yes** | orchestrator-authored; for action rows this includes `<sub_goal>` and `<tool_call>`, because issuing the call is the decision being learned |
| trailing `<\|im_end\|>` | **Yes** | teaches the stop token |

**No tools execute.** Training is static cross-entropy: no sub-agent vLLM server, no
AgentFlow server, no rollout workers, no `SERPER_API_KEY`/`TAVILY_API_KEY`. Nothing in this
change alters that; `FoldedSFTDataset` only tokenises.

**No thinking in targets.** Already stripped at build time; re-asserted by tests and by the
pre-flight gate.

---

## 6. How it was built (TDD trail)

Every implementation stage was watched failing first:

| Cycle | RED (observed failure) | GREEN |
|---|---|---|
| 1 | `AttributeError: _fold_trajectory` | walker + `_memory_user_content` + `_plan_query_analysis` |
| 2 | `ModuleNotFoundError: No module named 'verl_ext'` | `FoldedSFTDataset` |
| 3 | `unrecognized arguments: --from-parquet --drop-planning-answers` | CLI wiring + `_refold_parquet` |
| 4 | `assert 1 == 4` (`--format folded` parsed but inert on the JSONL path) | fold inside the split loop |

Two later tests (`no_padding` mode, native-row rejection) went green immediately — the class
already handled both paths generically, so they are regression guards rather than RED-driven.
Noting that rather than claiming a clean cycle.

Also fixed a latent logging bug the fold exposed: the JSONL mirror logged the trajectory
count while writing the row count.

**Test suite: 487 passed.** One pre-existing collection error,
`tests/unit/test_fine_tuning_rollout.py` (`No module named 'agentops'`), unrelated to this
work and present before it.

---

## 7. Verification on the real data

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
| (native, for comparison) | 1,587,150 / 1639.6 / 2338 / 7255 | 186,668 / 1728.4 / 2426 / 9821 |

### Why the row count multiplied (968 → 2995)

No data was added. **A native row is a whole trajectory; a folded row is one orchestrator
decision.** Those 968 trajectories contain exactly 2995 assistant turns, so 2995 is the
assistant-turn count, not a multiplier applied to anything:

| assistant turns in a trajectory | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| trajectories (train) | 175 | 48 | 383 | 280 | 57 | 15 | 4 | 2 | 4 |
| trajectories (val) | 14 | 4 | 43 | 33 | 10 | 1 | 2 | 0 | 1 |

The 175 one-turn trajectories are the turn-0 answers of decision 1 below. A typical 3-turn
trajectory is plan → one tool call → answer.

**The supervision is byte-for-byte identical.** Measured over every row:

| | native | folded | ratio |
|---|---|---|---|
| supervised tokens, train | 589,805 | **589,805** | 1.000 |
| supervised tokens, val | 66,556 | **66,556** | 1.000 |

The same assistant text is the target in both; it is only distributed one-per-row instead
of many-per-row.

**The token growth (1.59M → 4.06M) is prompt, not target:**

```
3,472,545 prompt  +  589,805 supervised  =  4,062,350   ← the table above
```

The system prompt and question are re-rendered in every row because that is literally what
the orchestrator does at inference: `_build_memory_prompt` (`orchestrator.py:641-664`)
rebuilds `[system, user]` from scratch at every decision. The native format amortised that
prefix across one long sequence; the deployed system never does. The extra ~2.5M tokens are
the real conditioning cost of the format the model is evaluated in, not bloat added by the
fold.

Two consequences follow, and both are already handled:

1. **Steps per epoch tripled**: 2995 / 32 ≈ 94 steps vs 968 / 32 ≈ 30. This is the reason
   job 008 sets `total_epochs=2` (~187 steps) where 007 used 3 (~90): the run is matched on
   optimizer steps, not on epochs.
2. **Long trajectories now contribute more rows.** A 9-turn trajectory yields 9 rows where a
   turn-0 answer yields 1. Search trajectories run longer than math ones, so the 484/484
   trajectory balance becomes 1172/1823 at row level. That is the mechanism behind decision
   2 below, not a separate problem.

**Gates, run on every row in both `agent_engine` and `cosmas-train` (different transformers
versions, identical results):**

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
| Every native question present | yes | yes |
| Columns identical to native | yes | yes |
| No question straddles train/val | yes | |

The gate is a real trip-wire, not a rubber stamp: it exits **1** on the native parquet and
**0** on the folded one, both confirmed.

Files now on disk (gitignored via `.gitignore:48 data/*`):

```
data/training/sft/sft_folded_train.parquet   2.8 MB   2995 rows
data/training/sft/sft_folded_val.parquet     243 KB    362 rows
data/training/sft/sft_folded_{train,val}.jsonl        (human-readable mirrors)
```

---

## 8. Decisions — both RESOLVED

**1. Turn-0 answers (175 train rows, 18.1%): KEPT.** These teach answering at turn 0, which is
the *Premature direct answering* failure mode in the Ch 6 taxonomy. `--drop-planning-answers`
is implemented and tested but **not used**; the parquets in the running job keep them. Rationale
for keeping: format is then the only variable that changed against the 007 run, so the
comparison stays clean. If *Premature direct answering* shows up in the eval, dropping them is
a one-flag rebuild and a follow-up run.

**2. Loss-level composition shift: ACCEPTED, to be stated in Ch 7.** Trajectories stay 484 math
/ 484 search (1:1), but rows become **1172 math / 1823 search (1:1.56)**, because search
trajectories take more steps and therefore expand into more rows (the mechanism is spelled out
at the end of §7). This is inherent to one-row-per-decision, not a defect — but "differs in
format only" is true at the *trajectory* level and **false at the loss level**, so it is a
confound that must be disclosed rather than left implicit. Rationale for accepting: subsampling
search rows back to 1:1 would discard ~651 real supervision rows to preserve a ratio that was
itself only a heuristic mirror of `combined_train`.

**Ch 7 needs one sentence to this effect:** the folded and native runs use the same
trajectories and the same balanced 1:1 math/search sampling, but because the folded format
supervises one row per orchestrator decision and search trajectories take more steps, the
loss-level ratio is 1:1.56 rather than 1:1.

---

## 9. What is NOT done

- **No completed training run.** Job 25268454 reached step 50 of ~187 and was cancelled on
  purpose (§14). Nothing is running now. Relaunch per §12.
- ~~**The post-cancel edits are unverified.**~~ **Done (§15):** `bash -n` passes on all four
  job scripts, `pytest` gives 487 passed, and both pre-flight gates pass on the real splits
  (2995 / 362 rows, numbers identical to §7). Two defects found in that pass are fixed.
- ~~**No SFT eval config exists.**~~ **Done (§15):** `experiments/configs/qwen3/sft_inference/`,
  five configs, `thinking_mode: NO`. The `<run-tag>` in `lora_adapter_path` is a placeholder
  until a training run produces one.
- ~~**No checkpoint-selection script.**~~ **Done:** `scripts/finalize_sft_run.py` selects
  best-val-loss and last, extracts both as PEFT adapters, and deletes the FSDP shards.
  Verified on a real checkpoint. `select_best_sft_checkpoint.py` (referenced by 007) never
  existed and is not needed. **`scripts/merge_lora.py` must not be used on SFT
  checkpoints** — see §11 for why.
- **No in-training generation eval.** Deliberately: a format mismatch is visible at step 0,
  so the pre-flight gate in §5 catches this class of bug at a fraction of the cost. An
  in-training callback would catch *degradation over training*, a different failure. Worth
  adding only if you want that too.
- **`stash@{0}` on `main` is still uncommitted** — the `history_mode` work (3 files,
  +66/-4), which evaluates the *existing* adapter in native format with no retraining. It is
  a viable fallback and one `git stash drop` from being lost, exactly like the fold code was.
  **Recommend committing it to a branch.**

---

## 10. Commands to pick up tomorrow

```bash
cd /gpfs/home3/xchen1/azywot/msc-thesis

# (a) tests — expect 487 passed
conda activate agent_engine
python -m pytest tests/ -q --ignore=tests/unit/test_fine_tuning_rollout.py

# (b) re-verify the built data (fast, no GPU)
python scripts/check_sft_folded_format.py \
    --folded data/training/sft/sft_folded_train.parquet \
    --native data/training/sft/sft_train.parquet

# (c) OPTIONAL: rebuild dropping the 175 turn-0 answers (decision 1)
python scripts/build_sft_parquet.py \
    --from-parquet data/training/sft/sft_train.parquet \
    --output-dir data/training/sft --output-name sft_folded_train.parquet \
    --drop-planning-answers

# (d) verify everything first — CPU only, no GPU SBUs burned
sbatch jobs/fine_tuning/008_test_sft_folded.job

# (e) train — ~187 steps, 2x H100, ~40 min wall clock, schedules immediately
sbatch jobs/fine_tuning/008_train_sft_folded.job

# (f) after training: NOTHING. Job 008 now ends by extracting the best-val-loss and last
#     adapters and deleting the 33 GB shards. Point an eval config at:
#       /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-sft-folded-v1/<run-tag>/best_adapter
#     Do NOT run scripts/merge_lora.py on an SFT checkpoint — see §11.
#     To redo the selection by hand (e.g. after a crash):
python scripts/finalize_sft_run.py \
    --ckpt-dir /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-sft-folded-v1/<run-tag> \
    --log out/fine_tuning/sft_train/sft_folded_<jobid>_verl.log --dry-run
```

Job 008 runs the pre-flight gate itself and refuses to start training if it fails, so (b) is
belt-and-braces. `finalize_sft_run.py --dry-run` reports which checkpoint it would keep and
why, without writing or deleting anything.

---

## 11. Checkpoint and adapter mechanics (discovered 2026-08-06)

Four facts about how `verl.trainer.sft_trainer` checkpoints, all established by inspecting a
real checkpoint from job 25268454. Each one invalidates something that was previously assumed.

**1. The SFT trainer never writes a `lora_adapter/` directory.** The RL path does
(`fsdp_workers.py`, which the GRPO run depends on), but the SFT path only ever *reads*
`lora_adapter_path` (`verl/trainer/sft_trainer.py:101`). A grep of the SFT and engine
checkpoint code finds no write site. So an SFT checkpoint contains no PEFT adapter, and the
`huggingface/` subdirectory holds only tokenizer and config files, no weights.

**2. It writes the full FSDP state dict instead: 33 GB per checkpoint.**

| file | size | needed? |
|---|---|---|
| `model_world_size_2_rank_{0,1}.pt` | 16 GB each | yes, but 99% of it is unchanged base weights |
| `optim_world_size_2_rank_{0,1}.pt` | 667 MB each | no (nothing resumes) |
| `extra_state_*`, `data_*`, `*.json` | ~15 KB | trivial |

Only ~365 MB of that is trained LoRA weight. At `save_freq=25` over 187 steps that is ~8
checkpoints, ~264 GB, to store ~700 MB of useful output.

**3. The shards are sharded DTensors, not replicas.** This is the trap. `rank_0` holds
`(32, 4096)` of a `(64, 4096)` `lora_A` — placement `Shard(dim=0)`. Reconstruction must
concatenate **every** rank along each tensor's own shard dimension.

> **`scripts/merge_lora.py` cannot be used on these checkpoints.** It reads
> `model_world_size_*_rank_0.pt` alone, documenting it as "the rank-0 consolidated shard
> regardless of training world size". That is false here. It also expects an `actor/`
> subdirectory, which the SFT trainer does not create. Both the merge command printed by
> `007_train_sft.job` and the one 008 printed before today would have failed or silently
> produced a half-weight adapter.

**4. `lora_train_meta.json` records the true hyperparameters** (`{"r": 64, "lora_alpha": 64,
"task_type": "CAUSAL_LM"}`). `finalize_sft_run.py` prefers these over its CLI arguments,
because a mismatched alpha silently rescales every adapter weight and nothing downstream
would flag it.

`scripts/finalize_sft_run.py` handles all four. Verified on the real `global_step_25`:
504 LoRA tensors over 7 target modules (`q,k,v,o,gate,up,down_proj`), written as a 365 MB
PEFT adapter. It also derives `target_modules` from the checkpoint keys rather than saving
the literal string `"all-linear"`, so load-time re-resolution cannot change which modules
the adapter claims to cover.

### Unrelated but important: the GRPO adapter is gone

`/scratch-shared/azywot/fine_tuning/lora_adapters/qwen3-8b-grpo-search-math-v2/` **is
empty** (and so is `…-search-math/`). Every `experiments/configs/qwen3/lora_inference/*`
config points `lora_adapter_path` at
`…/qwen3-8b-grpo-search-math-v2/29-05-2026_11-36-23210365/global_step_40/actor/lora_adapter`,
which no longer exists. The directory mtimes are 2026-08-03, consistent with scratch
retention having purged the contents.

Consequences to think about:

- Any GRPO number already in the thesis stands (it came from a completed eval), but **the
  GRPO evaluation cannot currently be re-run or extended**.
- **Scratch is not durable storage.** The SFT adapters will be purged the same way. Once
  `best_adapter/` exists, copy it somewhere that persists. Home has room: 200 GiB quota,
  ~65% used, and an adapter is 365 MB.
- Worth checking whether a merged GRPO model survives anywhere else before assuming it is
  unrecoverable.

---

## 12. Resume here

Steps 1 and 2 are **done** (§15): syntax, 487 tests, both gates, and the CLI check all pass on
the current tree. Nothing has been submitted. Resume at step 3.

```bash
cd /gpfs/home3/xchen1/azywot/msc-thesis

# 3. Full verification suite (CPU partition, no GPU cost)
sbatch jobs/fine_tuning/008_test_sft_folded.job

# 4. Launch training (~40 min wall clock at the observed rate)
sbatch jobs/fine_tuning/008_train_sft_folded.job

# 5. (done) The cancelled run's 33 GB of leftovers were deleted on 2026-08-06.

# 6. After training: paste the run tag the job prints into the five eval configs
#    (or set SFT_ADAPTER_PLACEHOLDER in scripts/generate_configs.py and regenerate), then
python scripts/generate_configs.py --suite sft_inference
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/sft_inference
```

For reference, the checks already run in step 1/2 (re-run them after any change to
`build_sft_parquet.py` or `folded_sft_dataset.py`):

```bash
bash -n jobs/fine_tuning/008_train_sft_folded.job   # and 008_test, 006, 007
conda activate agent_engine
python -m pytest tests/ -q --ignore=tests/unit/test_fine_tuning_rollout.py
python scripts/build_sft_parquet.py --help
python scripts/check_sft_folded_format.py \
    --folded data/training/sft/sft_folded_train.parquet \
    --native data/training/sft/sft_train.parquet --max-length 16384
```

Then the still-open work from §9: **committing `stash@{0}`**.

---

## 13. Related documents (history, not current state)

- `docs/sft_folded_relaunch_plan.md` — the full plan: deeper evidence tables, the rendered
  side-by-side contexts, and the ordered change list.
- `docs/sft_format_mismatch.md` — the original 2026-06-17 diagnosis. Sound; §7's access
  claims and §1's "tool-role result" phrasing are stale (Qwen3 renders `role: tool` as
  `<|im_start|>user\n<tool_response>…`).
- `docs/superpowers/specs/2026-07-21-sft-folded-format-design.md` — the design. Its data
  numbers all reproduce exactly; its "implemented and tested" status claims do not (§4).
- `jobs/fine_tuning/007_train_sft.job` — the original native run, left untouched for
  comparison.

---

## 14. Run log

**2026-08-06 — decisions 1 and 2 resolved (keep turn-0 answers, accept the composition
shift). No rebuild needed: the parquets already on disk encode exactly that.**

**`jobs/fine_tuning/008_test_sft_folded.job` added.** A CPU-partition (`genoa`) verification
suite, so checking costs no GPU SBUs. Five independently fatal stages: (1) full unit-test
suite; (2) the pre-flight gate on the real folded splits; (3) a **trip-wire** — the gate must
*reject* the native parquets, because a gate that passes everything proves nothing; (4) the
same gate under `cosmas-train`, since `FoldedSFTDataset` is authored and tested in
`agent_engine` but imported by VERL in `cosmas-train` with a different transformers version,
and a chat-template difference between the two would otherwise surface only at training time;
(5) one decoded row printed with the supervised span marked, so the token-level claim can be
checked by reading it. Run it before any training relaunch and after any change to
`build_sft_parquet.py` or `folded_sft_dataset.py`.

**Two failed submissions (25268308, 25268330) before the run that stuck.** Cause: job 008
resolved `PROJECT_DIR` from `${BASH_SOURCE[0]}`, which works under `bash jobs/...` but not
under `sbatch` — SLURM copies the batch script to `/var/spool/slurm` on the compute node, so
the script "location" is the spool copy. The job `cd`'d into `/var/spool/slurm`, found no
`.env` (hence a spurious `WANDB_API_KEY is not set` warning), and died at
`mkdir: cannot create directory 'out': Permission denied` after ~10 s.

Fixed in both 008 jobs by trying candidates in order and accepting only one that *validates*
as the repo (`scripts/build_sft_parquet.py` present), rather than trusting any single
mechanism: `$PROJECT_DIR` → `$SLURM_SUBMIT_DIR` → script dir → `$HOME/azywot/msc-thesis`,
with a clear error and a `sbatch --export=ALL,PROJECT_DIR=...` hint if all four fail. Verified
against four cases, including a `/var/spool/slurm` submit-dir that must be rejected.
Worth knowing: **007 has the same class of fragility** (a hardcoded `/projects/0/gusr0608`),
so copy the resolution block if that job is ever reused.

**Job 25268454: ran 8m43s, then CANCELLED on purpose.** `PROJECT_DIR` resolved correctly and
the in-job pre-flight gate passed on both splits. Note the checkpoint path is
`/scratch-shared/azywot/`, not `/scratch-shared/xchen1/` — `$USER` on the compute node is
`azywot`.

What it proved before being stopped:

| step | val/loss |
|---|---|
| 25 | 0.602511 |
| 50 | 0.494696 |

So the folded pipeline trains, and it trains fast: 50 steps in ~8 minutes, meaning the full
~187-step run is roughly **40 minutes**, not the multi-hour job the 8 h wall-clock limit
suggests.

**Why it was cancelled** (three reasons, all about output rather than training):

1. Checkpoints were **33 GB each**. At `save_freq=25` that is ~264 GB for a run whose useful
   output is ~700 MB.
2. They went to `…/fine_tuning/sft/…`, not the GRPO layout
   `…/fine_tuning/lora_adapters/<experiment>/<run-tag>/`.
3. **All** of them were kept, rather than best and last.

Cancelling cost ~9 minutes of 2×H100 and avoided writing ~230 GB that would have had to be
deleted anyway. The cancel landed mid-save, leaving an **empty** `global_step_50/` — which is
why `finalize_sft_run.py` now skips checkpoint dirs with no model shards instead of treating
one as "last".

**Changes made after the cancel** (all since verified — see §15):

| File | Change |
|---|---|
| `008_train_sft_folded.job` | checkpoints → `lora_adapters/<exp>/<run-tag>/`, matching GRPO; `checkpoint.save_contents=['model','extra']` (drops the 1.3 GB optimizer state); `trainer.resume_mode=disable`; runs `finalize_sft_run.py` at the end; corrected next-step instructions (the old ones named `merge_lora.py`, which cannot work here) |
| `scripts/finalize_sft_run.py` | **new** — best/last selection, DTensor-correct adapter extraction, shard deletion |
| `scripts/build_sft_parquet.py` | dropped the `/projects/0/gusr0608/...` defaults for `jsonl_files` and `--output-dir`; `--output-dir` now defaults to the in-repo `data/training/sft` |
| `006_collect_sft_data.job`, `007_train_sft.job` | `PROJECT_DIR` no longer hardcodes `/projects/0/gusr0608/msc-thesis`; falls back to `$SLURM_SUBMIT_DIR` then `$HOME/azywot/msc-thesis` |

**`/projects/0/gusr0608` is no longer referenced by any code path** in the SFT pipeline. It
still appears in `jobs/grpo_inference/*.job` (10 files hardcode
`/gpfs/work5/0/gusr0608/msc-thesis`) — those are historical eval jobs that already ran, so
they were left alone rather than rewritten blind. Worth cleaning up if those evals are ever
re-run.

**Leftovers on scratch from the cancelled run: DELETED 2026-08-06.**
`/scratch-shared/azywot/fine_tuning/sft/` held one 33 GB `global_step_25/`, an empty
`global_step_50/`, and a 349 MB `best_adapter/` extracted from step 25 while testing
`finalize_sft_run.py`. All of it is gone, including that adapter: it came from an incomplete
run (25 of ~187 steps, val loss 0.6025) and a full run supersedes it in ~40 minutes.
`/scratch-shared/azywot/fine_tuning/` now holds only the two empty GRPO `lora_adapters/`
directories. The relaunched job writes to `lora_adapters/<exp>/<run-tag>/`, not `sft/`, so
nothing pointed at the deleted path.

---

## 15. Verification pass and eval configs (2026-08-06, after the cancel)

### The post-cancel edits are now verified

| Check | Result |
|---|---|
| `bash -n` on `008_train`, `008_test`, `006`, `007` | all pass |
| `pytest tests/ --ignore=tests/unit/test_fine_tuning_rollout.py` | **487 passed**, 53 s |
| Pre-flight gate, train split | pass — 2995 rows, 1343 tool-call targets, 4,062,350 tokens |
| Pre-flight gate, val split | pass — 362 rows, 176 tool-call targets, 500,014 tokens |
| `build_sft_parquet.py --help` | resolves; `--output-dir` defaults to `data/training/sft` |

Every gate number reproduces §7 exactly, so the parquets on disk are the ones §7 describes.

### Two defects found in that pass, both fixed

**1. The error handling in `008_train_sft_folded.job` was unreachable.** Under `set -euo
pipefail` a bare failing command aborts the script at once, so `SFT_EXIT=$?` and
`FINALIZE_EXIT=$?` on the following lines never ran and both diagnostic branches were dead
code. The practical cost: if `finalize_sft_run.py` failed, the job would die silently and the
message telling you the 33 GB shards were kept and how to re-run extraction would never print.
Fixed with `|| SFT_EXIT=$?` (and the same for finalize), which puts the command in a condition
context and suspends `set -e`. Verified by simulating a non-zero exit: the branch runs and the
code propagates.

**2. `--time=30:00:00` for a ~40 minute run.** Costs queue priority for nothing. Now
`02:00:00`, roughly 3x headroom over the observed rate.

### Adapters are now archived off scratch automatically

`008_train_sft_folded.job` ends by copying `best_adapter/`, `last_adapter/` and
`selection.json` from scratch to `data/adapters/<experiment>/<run-tag>/` inside the checkout
(`data/*` is gitignored; ~365 MB against a 200 GiB home quota). This is the direct lesson of
§11: the GRPO adapters were purged from `/scratch-shared` while ten configs still pointed at
them, which is why those evaluations can no longer be re-run. If extraction fails, the job says
so loudly instead of leaving the only copy on scratch.

### SFT eval configs exist

`experiments/configs/qwen3/sft_inference/{gaia,hle,gpqa,aime,musique}/qwen8B_sub1_7b_none.yaml`,
generated by a new `sft_inference` suite in `scripts/generate_configs.py`
(`python scripts/generate_configs.py --suite sft_inference`).

- **`_none` only.** Thinking is stripped from the SFT data at build time and the folded prompt
  renders an empty `<think>\n\n</think>`, so an `ORCHESTRATOR_ONLY` eval would sample the
  adapter off the distribution it was trained on: the exact mismatch this whole exercise fixes.
  This also keeps the run inside the Ch 7 rule that everything is compared no-thinking.
- The orchestrator is named **`Qwen3-8B-SFT`**, so the `model_name` column of the W&B export
  separates it from the GRPO rows (`Qwen3-8B-LoRA`) in `orchestrator_ft_results.csv`.
- `lora_adapter_path` points at the **archive** path with a `<run-tag>` placeholder, matching
  the existing `lora_inference` convention. It must be filled in after training.
- All five load through the real `load_experiment_config`; `max_lora_rank` defaults to 64 in
  `models/base.py:205`, which matches the trained rank, so no config override is needed.

`generate_configs.py` gained three optional suite keys (`adapter_label`, `adapter_desc`,
`train_job`) so the SFT suite reuses the LoRA builder rather than duplicating it. Defaults
preserve the LoRA output unchanged.

**Caution, learned the hard way here:** the committed `lora_inference` configs were
**hand-edited after generation** (a `_v2` name suffix and the `…-v2/…/global_step_40` adapter
path). Regenerating that suite silently overwrites those edits — it happened during this pass
and was reverted with `git checkout`. `generate_configs.py` is not the source of truth for
`lora_inference`. The same will be true of `sft_inference` once a real run tag is pasted in, so
prefer editing `SFT_ADAPTER_PLACEHOLDER` and regenerating over hand-editing five files.

### Scratch cleaned

`/scratch-shared/azywot/fine_tuning/sft/` deleted, reclaiming 33 GB (details in §14).

### Still not done

Nothing has been submitted to SLURM. `stash@{0}` on `main` is still uncommitted.
