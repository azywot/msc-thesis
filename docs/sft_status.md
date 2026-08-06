# Orchestrator SFT: status and handover

**Last updated:** 2026-08-05
**Branch:** `feat/sft-folded-format`
**Read this first.** It supersedes the status sections of the older docs listed at the end.

---

## 1. Where things stand, in five lines

The SFT adapter underperformed the base model because it was trained on a conversation
format the orchestrator never sees at inference. That diagnosis is now **verified by
executing both code paths**, not inferred. The fix is **implemented, tested (23 new tests,
487 passing overall), and applied to the real data**: folded parquets are built and every
pre-flight gate passes on all 3357 rows. The training job is written and ready.
**Nothing has been trained or evaluated yet**, and two research decisions are open (§8).

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
| `jobs/fine_tuning/008_test_sft_folded.job` | new | CPU-only verification suite: tests, gate, gate trip-wire, gate under the training env, and one decoded example. See §12. |
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

- **Training has not been run.** The job is ready; nothing submitted.
- **No SFT eval config exists.** All `lora_inference` configs point at the GRPO adapter. One
  must be added (`thinking_mode: NO`, normal folded inference path) before there is a number
  for the thesis.
- **No checkpoint-selection script.** `select_best_sft_checkpoint.py` never existed. Job 008
  prints the `val/loss` lines and the exact `scripts/merge_lora.py` command instead, so
  selection is a manual one-liner. Writing the script is optional.
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

# (e) train — ~187 steps, 2x H100, schedules immediately
sbatch jobs/fine_tuning/008_train_sft_folded.job

# (f) after training: merge the lowest-val-loss checkpoint
python scripts/merge_lora.py \
    --checkpoint /scratch-shared/$USER/fine_tuning/sft/qwen3-8b-sft-folded-v1/global_step_<N> \
    --base-model Qwen/Qwen3-8B \
    --output-dir /scratch-shared/$USER/fine_tuning/sft/qwen3-8b-sft-folded-v1/global_step_<N>/merged
```

Job 008 runs the pre-flight gate itself and refuses to start training if it fails, so (b) is
belt-and-braces.

---

## 11. Related documents (history, not current state)

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

## 12. Run log

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

**Job 25268454: RUNNING.** `PROJECT_DIR` resolved to the repo, the in-job pre-flight gate
passed on both splits, checkpoints going to
`/scratch-shared/azywot/fine_tuning/sft/qwen3-8b-sft-folded-v1`. Note the path is
`/scratch-shared/azywot/`, not `/scratch-shared/xchen1/` — `$USER` on the compute node is
`azywot`, so look there for checkpoints.
