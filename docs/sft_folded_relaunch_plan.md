# SFT folded-format: verification + relaunch plan

**Date:** 2026-08-05
**Branch:** `feat/sft-folded-format`
**Supersedes the status claims in:** `docs/sft_status.md` §4 (see Finding 0)
**Verified against:** `data/training/sft/sft_train.parquet` (968 rows), `sft_val.parquet` (108 rows),
installed `verl` in `cosmas-train`, `src/agent_engine` at HEAD `e6f97f9`.

---

## Finding 0 (blocking, and not in either doc): the fold implementation does not exist

`docs/sft_status.md` §4 says the fold is "implemented, tested, and run successfully over the
real parquets", with 24 + 12 tests and a 389-passing suite. **None of that code is in the
repository.**

| Claimed artifact | Actual state |
|---|---|
| `--format folded`, `--from-parquet`, `--planning-suffix`, `--drop-planning-answers` in `scripts/build_sft_parquet.py` | Absent. `grep -n "folded\|--format\|_fold_trajectory\|_memory_user_content"` returns nothing; the file is the pristine native builder (472 lines). |
| `tests/unit/test_build_sft_folded.py` (24 tests) | File does not exist. |
| `tests/unit/test_build_sft_folded_cli.py` (12 tests) | File does not exist. |

It is not stashed, not on another branch, and not in any commit:
`git log --all -S"_fold_trajectory"` → empty; `git log --all --diff-filter=A -- '*test_build_sft_folded*'` → empty.
The only surviving trace is the two docs. The work was presumably done in an ephemeral
worktree or scratch directory that has since been cleaned.

**Consequence for Task 2:** this is a *build-and-relaunch*, not a relaunch. The fold has to be
written from scratch. The good news (Finding 6) is that the docs' data numbers reproduce
exactly, so they are a reliable spec even though the code is gone.

Two smaller gaps in the same class:

- `jobs/fine_tuning/007_train_sft.job:180` calls `scripts/select_best_sft_checkpoint.py`, which
  has never existed in any commit. The job tolerates it (`|| echo WARNING`), so training would
  finish but leave no merged adapter, and no checkpoint would be selected by val loss.
- There is **no SFT evaluation config anywhere** in `experiments/configs/`. The only adapter
  eval configs are `experiments/configs/qwen3/lora_inference/*/*.yaml`, and every one of them
  points `lora_adapter_path` at the **GRPO** adapter
  (`qwen3-8b-grpo-search-math-v2/.../global_step_40/actor/lora_adapter`), not an SFT one.

---

## Task 1 — the diagnosis is confirmed, with two corrections

### 1.1 The two rendered contexts, from the real code paths

Both sides were produced by executing the production code, not by reconstruction:

- **Training side:** `verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset`, constructed
  with exactly the flags `007_train_sft.job:135-142` passes (`messages_key=messages`,
  `max_length=16384`, `truncation=right`, `ignore_input_ids_mismatch=True`), then
  `tokenizer.decode(input_ids)` and `decode` of the `loss_mask==1` positions.
- **Inference side:** `AgenticOrchestrator._build_memory_prompt` / `_commit_tool_result` /
  `_format_action_history` / `_extract_sub_goal` and the real `parse_tool_call`, imported from
  `agent_engine` and called directly on a real `ExecutionState`; rendered through the Qwen3
  branch of `VLLMProvider._render_messages` (`vllm_provider.py:338-343`,
  `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`).

The one place the real function could not be *invoked* is `_run_planning_turn`, because it
requires a live model. Its two prompt-shaping lines (`orchestrator.py:736-740`) were mirrored
literally, importing `_DEFAULT_PLANNING_SUFFIX_TOOLS` from the module rather than retyping it.
The Qwen3 branch selection was asserted programmatically
(`ModelFamily.QWEN3 in _ENABLE_THINKING_KWARG_FAMILIES` and
`not in _THINK_PREFIX_FAMILIES`), not assumed.

Example trace: `sft_train.parquet` row 0, `data_source=deepmath`, roles
`[system, user, assistant, assistant, tool, assistant]`.

**Training: one sequence, 2273 tokens, 3 supervised spans.**

```
<|im_start|>system
You are a mathematical reasoning assistant ...          ← identical on both sides
<|im_end|>
<|im_start|>user
Calculate \(\lim_{r\rightarrow 0^+}...\)<|im_end|>       ← RAW question, no planning suffix
<|im_start|>assistant
The problem requires evaluating the limit ...<|im_end|>  ← SUPERVISED (plan)
<|im_start|>assistant
<tool_call>
{"name": "code_generator", ...}
</tool_call><|im_end|>                                    ← SUPERVISED (action)
<|im_start|>user
<tool_response>
y**3/2 + y
</tool_response><|im_end|>                               ← tool result as a separate turn
<|im_start|>assistant
To evaluate the limit ... \boxed{\frac{5}{8}}<|im_end|>  ← SUPERVISED (answer)
```

**Inference: three separate prompts, 914 / 1378 / 1468 tokens.** The third one:

```
<|im_start|>system
You are a mathematical reasoning assistant ...          ← identical
<|im_end|>
<|im_start|>user
Calculate \(\lim_{r\rightarrow 0^+}...\)

**Query Analysis:**                                      ← NEVER appears in training
The problem requires evaluating the limit ...

**Previous Steps:**                                      ← NEVER appears in training
Action Step 1:
  - Tool: code_generator
  - Sub-goal:
  - Command: {"name": "code_generator", "arguments": {...}}
  - Result: y**3/2 + y                                   ← result as PROSE, not a tool turn
<|im_end|>
<|im_start|>assistant
<think>

</think>

                                                         ← generation starts HERE
```

Turn 0 differs too: the inference prompt appends `_DEFAULT_PLANNING_SUFFIX_TOOLS`
("Before using any tools, analyze this query…") to the user turn. The stored training user turn
never carries it, because `_run_planning_turn` shallow-copies the list and rebinds
`planning_messages[-1]` rather than mutating `state.messages[1]` (`orchestrator.py:736-740`).

### 1.2 The concrete diff, with file:line

| | Training | Inference |
|---|---|---|
| Built at | `verl/utils/dataset/multiturn_sft_dataset.py:187-241` (per-turn tokenize), driven by `jobs/fine_tuning/007_train_sft.job:134-160` | `src/agent_engine/core/orchestrator.py:641-664`, called at `:207`, `:241`, `:323`, `:367` whenever `not self.baseline` |
| Data shape | one growing conversation, `T+3` turns | `T+2` independent `[system, user]` prompts |
| Question turn | raw question | question **+ planning suffix** (turn 0) or **+ `**Query Analysis:**` + `**Previous Steps:**`** (turns 1..T+1) |
| Prior steps | replayed verbatim as assistant/tool turns | compressed to `Action Step N: Tool/Sub-goal/Command/Result` prose (`orchestrator.py:696-706`) |
| Tool result | own turn → renders `<\|im_start\|>user\n<tool_response>…` | inlined as `- Result:` inside the user turn |
| Generation starts after | `<\|im_start\|>assistant\n` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |
| Origin of the stored shape | `scripts/collect_sft_data.py:70-91` `_build_sft_messages` returns `state.messages[:2] + [plan] + state.output_messages` | n/a |

**Measured over 60 randomly sampled traces (seed 0), 194 decision points:**

| Metric | Result |
|---|---|
| Decision points where the training conditioning prefix == the inference prompt | **0 / 194** |
| Decision points that diverge | **194 / 194** |
| First differing token index | 751 – 897 (i.e. immediately after the shared system prompt) |
| Planning prompts carrying the suffix at inference | 60 / 60 |
| Training contexts containing that suffix | 0 / 60 |
| Training contexts containing `**Query Analysis:**` | 0 / 60 (0 / 968 over the full split) |
| Inference action/answer prompts containing `**Query Analysis:**` | 134 / 134 |

The divergence is total. Every gradient step the adapter took was on a conditioning context it
provably never sees at evaluation.

### 1.3 Which doc claims are confirmed, unconfirmed, or wrong

**Confirmed (re-measured independently):**

| Claim | My measurement |
|---|---|
| 968 train / 108 val rows | 968 / 108 |
| 0 of 1076 rows contain `**Query Analysis:**` or `**Previous Steps:**` | 0 and 0, both splits |
| 793/968 train rows multi-turn; 745 have `tool` turns | 793 and 745 |
| 175 rows are exactly `[system, user, assistant]` (planning-answered) | 175 (val: 14) |
| 109 trajectories whose plan emitted a tool call | 109 (val: 16) |
| Inference is unconditionally folded; no `history_mode` anywhere | confirmed; `_build_memory_prompt` is the only non-baseline path |
| VERL trains loss on assistant turns only (`multiturn_sft_dataset.py:231-236`) | confirmed at those exact lines in the installed verl |
| No thinking anywhere in the parquets | 0 `<think>`/`</think>` in 968 + 108 rows, in context *and* in loss spans |
| Tool results carry no loss | confirmed structurally, see 1.4 |
| VERL `sanity_check` fails on every row, by exactly 4 tokens | 60/60 rows, delta **exactly 4** on every one; the 4 tokens are `<think>\n\n</think>\n\n` |
| 179/1234 action turns have no `<sub_goal>` (14.5%) | 179/1234 |
| 968 → 2995 folded rows (3.09x), 108 → 362 (3.35x) | identical (Finding 6) |
| 0 rows exceed `max_length=16384` | 0 native, 0 folded |
| `--format folded` etc. are "live" in `build_sft_parquet.py` | **WRONG** — see Finding 0 |

**Wrong or imprecise:**

1. **`sft_status.md` §4 "Implemented" and the design doc's "Status (2026-08-04): the fold is
   implemented and tested".** The code does not exist (Finding 0). This is the single most
   important correction: it changes the plan from "point the job at a new parquet" to "write
   the fold, then point the job at a new parquet".
2. **`sft_status.md` §5.2 "Run and verified 2026-08-05".** Cannot have been produced by code in
   this repository. The *numbers* are nonetheless correct (Finding 6), so they were produced by
   a real implementation that was subsequently lost.
3. **`sft_format_mismatch.md` §1's "tool-role result"** is imprecise, as the docs themselves
   note: Qwen3 renders `role: tool` as `<|im_start|>user\n<tool_response>…`. Confirmed in the
   rendered training context above. The argument is unaffected.

**Could not confirm (needs your input or access):**

4. **The premise itself: "training loss decreased, inference performance regressed."** There is
   no SFT training log (`out/fine_tuning/sft_train/` does not exist), no SFT eval config, and no
   SFT eval result anywhere in this checkout. `/projects/0/gusr0608/msc-thesis/` — where job 007
   runs and writes — is `Permission denied`. I verified the *mechanism* thoroughly; I could not
   verify the *symptom* from artifacts. See "Decisions I need from you" #1.
5. **Whether the base-vs-SFT comparison used an identical eval harness.** Not checkable without
   the SFT eval config and its `config.json` output.

### 1.4 The other candidate causes, ruled in or out

| Candidate | Verdict | Evidence |
|---|---|---|
| **Loss-masking bug** | **Ruled out, definitively** | Over **all 968 train and all 108 val rows**: the decoded `loss_mask==1` spans equal exactly the list of assistant message contents (each + `<\|im_end\|>`), with matching span counts. `rows_spans_exactly_assistant_turns: 968/968` and `108/108`. Nothing else is supervised. |
| **Thinking content leaking into targets** | **Ruled out** | 0 rows contain `<think>`/`</think>` in the supervised span or anywhere in the context, both splits. `_strip_thinking` (`build_sft_parquet.py:106-122`) removed it at build time. Note the *inverse* problem is real: inference **prepends** an empty think block the model was never trained after (see below). |
| **Tool output in the loss** | **Ruled out** | Structural: tool turns get `loss_mask = zeros` (`multiturn_sft_dataset.py:235-239`). A substring probe initially flagged 8 occurrences across 4 rows; all are false positives from 1–4-character tool results (`"1"`, `"oo"`, `"[15]"`) trivially appearing inside assistant prose. The span-equality check above is the authoritative one and it is clean. |
| **Special-token / chat-template mismatch** | **RULED IN — second, independent defect** | VERL tokenizes each turn in isolation, so targets begin right after `<\|im_start\|>assistant\n`. The whole-conversation render (and inference with `enable_thinking=False`) inserts `<think>\n\n</think>\n\n` first. VERL's own sanity check fails **60/60** sampled rows by **exactly 4 tokens**, and `007_train_sft.job:141` silences it with `ignore_input_ids_mismatch=True`. This is a genuine train/inference gap on *every* target, and folding alone does **not** fix it (see 2.1). |
| **EOS handling** | **Ruled out** | Every supervised span ends with `<\|im_end\|>` in 968/968 and 108/108 rows. |
| **Truncation at `max_seq_len`** | **Ruled out** | `data.max_length=16384`; measured max is 7255 (train) / 9821 (val) tokens. 0 rows truncated. `truncation=right` never fires. |
| **Overfitting** | **Cannot rule out; low prior, and orthogonal** | 968 rows × 3 epochs at `train_batch_size=32` ≈ 90 optimizer steps on a rank-64 LoRA. Val loss was computed on val data *in the same wrong format*, so it would not have caught the format gap regardless. Needs the training log to settle. |
| **Eval harness change** | **Cannot rule out** | No SFT eval config or result exists here to diff against the base-model run. |
| **LR / schedule** | **Unlikely as primary; one thing to note** | `optim.lr=2e-5`, `lr_warmup_steps=20` (`007:148-149`). With ~90 total steps, warmup is ~22% of the run. Aggressive but not obviously destructive; it cannot explain a context format the model never sees. |
| **Generation / sampling config differing** | **Ruled out as a differential cause** | Qwen3 takes the framework defaults, `temperature=0.0` (greedy), `top_p=0.8`, `top_k=20` (`models/base.py:185-188`); Qwen3 has no family override. Base and adapter runs go through the identical `VLLMProvider` path, so sampling is not a differential between them. |

**Conclusion.** The format mismatch is real, total (194/194 decision points), and sufficient on
its own to explain "loss down, eval down": cross-entropy on native multi-turn contexts is a
well-posed objective that the model minimised honestly, while the evaluated distribution is a
different one. A **second** defect of the same kind — the missing `<think>\n\n</think>\n\n`
prefix on every target — sits underneath it and must be fixed in the same change, or the
relaunch will still carry an uncontrolled difference.

---

## Task 2 — the plan

### 2.1 Where the transform lives, and why

**Offline in `scripts/build_sft_parquet.py`, plus a small custom VERL dataset class. Not in a
collator.**

- **Offline, not load-time.** The fold needs `parse_tool_call`, `_extract_sub_goal`,
  `_build_memory_prompt` and `_format_action_history` from `agent_engine`. The training env
  (`cosmas-train`) is not the inference env (`agent_engine`); importing the orchestrator inside
  a VERL dataloader couples them and makes every training run depend on the agent stack. Doing
  it offline also makes the folded parquet a reviewable artifact, which matters because the
  fold is the thing under suspicion.
- **`--from-parquet`, not a rebuild from `collected_*.jsonl`.** Rebuilding needs
  `--reference-parquet data/training/train/combined_train.parquet` for the math:search ratio.
  That path is `Permission denied`, and without it the build yields 1224/136 trajectories at
  1.53:1 instead of the shipped 968/108 at 1:1, plus a null `extra_info`. Refolding the shipped
  parquets preserves the exact trajectory set and every control already applied
  (strip-thinking, strip-suffix, one-per-question, ratio balance, split). The raw
  `collected_20260605_214650.jsonl` stays the source of truth and is untouched; the shipped
  parquets are its deterministic derivative.
- **Why a custom dataset class is also needed.** Measured on a real folded row:

  | Path | Prompt tokens | == inference prompt? |
  |---|---|---|
  | Folded row through `MultiTurnSFTDataset` (current code) | 1374 | **No** — diverges at token 1374, missing exactly `<think>\n\n</think>\n\n` |
  | Folded row through the proposed custom class | 1378 | **Yes, token-for-token** |

  So folding alone closes the system/user side and leaves the assistant side open. The custom
  class closes both and lets `ignore_input_ids_mismatch=True` be dropped. It is small precisely
  because folded rows are single-turn: render
  `apply_chat_template([system, user], add_generation_prompt=True, enable_thinking=False)`,
  tokenize `target + <|im_end|>`, mask zeros over the prompt and ones over the response.
  `data.custom_cls` is a supported VERL hook.

### 2.2 Exact masking spec

Each folded row is exactly three messages: `[system, user, assistant]`. One supervised span per
row, one row per orchestrator decision.

| Span | Supervised? | Rationale |
|---|---|---|
| `<\|im_start\|>system … <\|im_end\|>` | **No** | Context. Identical to inference. |
| `<\|im_start\|>user …` — question, planning suffix (turn 0) or `**Query Analysis:**` + `**Previous Steps:**` (turns ≥1) | **No** | Context. **This is where all simulated tool results live**, as `- Result:` prose inside `Action Step N` blocks. Requirement 2 is satisfied structurally: no tool result can ever be supervised, because tool results only ever appear here. |
| `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | **No** | Part of the generation prompt at inference; the model does not produce it. |
| assistant target text | **Yes** | Orchestrator-authored. For an action row this **includes** the `<sub_goal>` and `<tool_call>` blocks: emitting the call is the orchestrator's decision and is the whole point of the run. |
| trailing `<\|im_end\|>` | **Yes** | Teaches the stop token. |

Per row kind:

- **Planning row** — user = question + `_DEFAULT_PLANNING_SUFFIX_TOOLS`; target = the plan text.
- **Action row `k`** — user = question + `**Query Analysis:**` + `**Previous Steps:**` for steps
  `1..k-1` (action row 1 has no `**Previous Steps:**` block); target = the assistant turn that
  emitted the call, verbatim, including `<sub_goal>`/`<tool_call>`.
- **Answer row** — user = question + query analysis + all `T` steps; target = the final answer.

**No tool ever executes.** Job 007 runs static cross-entropy: no sub-agent vLLM server, no
AgentFlow server, no rollout workers, no `SERPER_API_KEY`/`TAVILY_API_KEY`
(`007_train_sft.job:22-31`). Nothing in this change alters that, and the custom dataset class
touches only tokenization. Requirement 2's "nothing may issue real tool calls" holds by
construction.

**Thinking:** already stripped at build time and re-asserted by the tests below. Requirement 1
holds.

### 2.3 The walker (two edge cases that are not optional)

Classify by **position**, never by sniffing content. `_build_sft_messages`
(`collect_sft_data.py:70-91`) always emits `[system, user] + [plan] + output_messages`, so
`messages[2]` is the plan whenever one exists — true for all 968 rows. From `messages[3:]`: an
assistant turn followed by a `tool` turn is an action; a trailing assistant turn is the answer.

- **Plan emitted a tool call (109 train / 16 val).** `raw_query_analysis` stores the *full*
  generation, but `state.query_analysis` — what inference actually folds — is only
  `text[:idx].strip()` where `idx` is the start of the call (`orchestrator.py:757-776`). The
  fold must apply the same truncation, or a `<tool_call>` block leaks into `**Query Analysis:**`.
  Content-sniffing ("the leading assistant turn with no `<tool_call>` is the plan")
  misclassifies these and silently drops the plan.
- **Plan emitted the final answer (175 train / 14 val).** `finished = True` at turn 0
  (`orchestrator.py:777-783`), `output_messages` empty, record is exactly
  `[system, user, assistant]` with `\boxed{}`. Emit one planning row, no answer row.

Assert `emitted_rows == include_planning + T + has_answer` per trajectory so a misclassification
fails loudly instead of vanishing.

### 2.4 The test that fails now and passes after

`tests/unit/test_sft_folded_format.py`, written first and watched fail.

1. **Train/inference identity (the headline test).** For N sampled real trajectories, for every
   decision point: the token sequence the training dataset conditions on (`input_ids` up to the
   first `loss_mask==1`) equals
   `tokenizer.apply_chat_template(orchestrator_prompt, add_generation_prompt=True, enable_thinking=False)`
   token-for-token, where `orchestrator_prompt` comes from the real `_build_memory_prompt`.
   *Fails today at 0/194 passing; must reach 194/194.*
2. **Supervised span purity.** For every folded row, the decoded `loss_mask==1` span equals the
   assistant target + `<|im_end|>`, contains no `<think>`/`</think>`, and contains no substring
   of any tool result of length ≥ 20 characters (the length floor avoids the `"1"`/`"oo"` false
   positives found in 1.4).
3. **Tool call retained.** For every action row the supervised span *does* contain `<tool_call>`,
   so a future "mask more" change cannot silently kill the signal.
4. Walker fixtures: 2-tool trajectory → 4 rows; `**Previous Steps:**` on row `k` holds steps
   `1..k-1` and not `k`; plan-with-tool-call keeps the plan and truncates the query analysis at
   the call; planning-answered → exactly one row, and `--drop-planning-answers` drops it;
   tool-free-but-planned → plan + answer only; no-silent-drops arithmetic.
5. **Byte-identity guard:** the fold's user content equals a reference built from the real
   `AgenticOrchestrator._format_action_history`, so the two cannot drift apart.

**Eyeballing one example.** `scripts/inspect_sft_data.py --show-mask ROW_IDX` prints the decoded
row with supervised tokens wrapped in `«»` and context tokens plain, plus a header line of
counts. One decoded example, masked positions marked, readable in a terminal.

### 2.5 Sanity checks before committing the full run

Run on the folded parquets and compare against the native ones. Projected values below are
already measured (Finding 6), so these are regression gates, not unknowns.

| Check | Native | Folded (projected) | Gate |
|---|---|---|---|
| Trajectories | 968 / 108 | 968 / 108 | must be identical |
| Rows (gradient targets) | 968 / 108 | 2995 / 362 | 3.09x / 3.35x |
| Row kinds (train) | n/a | 968 plan + 1234 action + 793 answer | sums to 2995 |
| Total tokens (train) | 1,587,150 | 4,062,350 | 2.56x |
| Mean / p95 / max tokens | 1640 / 2338 / 7255 | 1356 / 2040 / 7300 | max must stay < 16384 |
| Rows over `max_length` | 0 | 0 | must be 0 |
| Degenerate rows (empty target) | n/a | **0** | must be 0; if any appear, drop and log them |
| Thinking in loss span | 0 | 0 | must be 0 |
| Tool result in loss span | 0 | 0 | must be 0 |
| Every native question present | — | 968/968, 108/108 | no question lost |
| No question straddles train/val | yes | yes | fold after the split |
| `extra_info` populated | yes | 2995/2995, 362/362 | carried through verbatim |

Degenerate-row policy: measured count is 0, so nothing to do; keep the assertion so a future
data change surfaces it rather than training on empty targets.

### 2.6 The composition shift you must decide about

Folding preserves the trajectory set exactly but **not** the balance of gradient targets:

| | Native | Folded |
|---|---|---|
| Trajectories | 484 math / 484 search (1:1) | same 968, same questions |
| Rows | 484 / 484 (**1:1**) | 1172 math / 1823 search (**1:1.56**) |

Search trajectories take more tool steps, so they expand further. The effective training mix
moves from 50/50 to 39/61 toward search. This is inherent to one-row-per-decision, not a defect,
but "differs in format only" is true at the trajectory level and **false at the loss level**. If
left unaddressed it is a confound in the folded-vs-native comparison. See decision #3.

### 2.7 Relaunch config

**Changes vs. the previous run:**

| Setting | Before | After | Why |
|---|---|---|---|
| `data.train_files` / `val_files` | `sft_train.parquet` / `sft_val.parquet` | `sft_folded_train.parquet` / `sft_folded_val.parquet` | the point of the change |
| `data.custom_cls` | unset | points at the new single-turn dataset class | closes the assistant-side 4-token gap |
| `data.ignore_input_ids_mismatch` | `True` | **removed** | with the custom class there is no mismatch to silence; leaving it would re-hide exactly this class of bug |
| `trainer.total_epochs` | 3 | **2** (proposed) | 2995 rows / batch 32 ≈ 94 steps/epoch; 3 epochs = ~280 steps vs the previous ~90. 2 epochs ≈ 187 steps keeps the run comparable in optimizer steps without over-training 3.1x more rows. |
| `trainer.save_freq` / `test_freq` | 20 / 20 | **25 / 25** | ~7 checkpoints over 187 steps, similar granularity to before |
| `optim.lr_warmup_steps` | 20 | **20** (unchanged) | now ~11% of the run instead of ~22%, which is healthier; leaving the number fixed is also one fewer variable |

**Held fixed for comparability:** `model.path=Qwen/Qwen3-8B`, `lora_rank=64`, `lora_alpha=64`,
`target_modules=all-linear`, `optim.lr=2e-5`, `train_batch_size=32`,
`micro_batch_size_per_gpu=2`, `max_length=16384`, `truncation=right`, 2× H100, the same
trajectory set, and the same `seed=0` for the build (fold is deterministic and runs *after* the
split, so train/val membership is bit-identical to the native run).

**Catching this class of failure early — the part that was missing.** Val loss could not have
caught the format gap, because val was in the same wrong format. Add a **generation-based eval
on a held-out slice, run at intervals during training**:

- Hold out ~40 folded rows from val (stratified math/search, and across the three row kinds).
- Every `test_freq` steps, greedily generate the target from the folded prompt with the current
  adapter and log, to W&B alongside `val/loss`:
  - **format validity:** fraction of *action*-row generations that emit a parseable
    `<tool_call>` (via the real `parse_tool_call`), and of *answer*-row generations that emit
    `\boxed{}`;
  - **exact/near match** against the teacher target;
  - **degenerate-output rate:** empty generations, or ones that run to `max_tokens`.
- Trip-wire: if `<tool_call>` validity on action rows drops below the **base model's** score on
  the same slice, the run is making the orchestrator worse at the thing it is being trained for.
  Measure the base model's score on that slice *once, before training*, so there is a reference
  line from step 0. That single number is what would have caught this in the first run.

This is a generation eval on static prompts. It issues no tool calls: the sub-goal/tool-call
text is scored as text, never executed.

**After training:** write `scripts/select_best_sft_checkpoint.py` (referenced by
`007_train_sft.job:180` but never written), or drop the call and merge manually with
`verl.model_merger`. Then add a real SFT eval config — the existing `lora_inference` configs all
point at the GRPO adapter — and evaluate through the normal folded inference path against base
and GRPO.

### 2.8 Order of work

| # | Step | Files | Depends on |
|---|---|---|---|
| 1 | Write the failing test for train/inference identity + span purity | `tests/unit/test_sft_folded_format.py` (new) | — |
| 2 | Implement the fold: `--format {native,folded}`, `--from-parquet`, `--planning-suffix`, `--drop-planning-answers`; `_memory_user_content` delegating to `_format_action_history`, `_plan_query_analysis`, `_classify_turns` (positional), `_fold_trajectory`. Hook in the per-split loop **after** the split (`build_sft_parquet.py:425`). Default stays `native`. | `scripts/build_sft_parquet.py` | 1 |
| 3 | Implement the single-turn dataset class (~40 lines) | `src/verl_ext/folded_sft_dataset.py` (new; path TBD, see decision #4) | 1 |
| 4 | Make test 1 pass token-for-token; then tests 2–5 | tests + both above | 2, 3 |
| 5 | Build the folded parquets into `data/training/sft/` (writable, confirmed) and run the §2.5 gates | `data/training/sft/sft_folded_{train,val}.parquet` | 4 |
| 6 | Add the `--show-mask` viewer | `scripts/inspect_sft_data.py` | 2 |
| 7 | Update job 007: file paths, `custom_cls`, drop `ignore_input_ids_mismatch`, epochs/save_freq | `jobs/fine_tuning/007_train_sft.job` | 5 |
| 8 | Add the periodic generation eval + base-model reference line | new callback/script + job wiring | 7 |
| 9 | Write `select_best_sft_checkpoint.py`, or remove the call | `scripts/` | 7 |
| 10 | Add an SFT eval config (currently none exists) | `experiments/configs/qwen3/sft_inference/…` | 9 |
| 11 | Correct `docs/sft_status.md` §4/§5.2 to match reality (Finding 0) | `docs/` | — (do early) |
| 12 | Train, then evaluate against base and GRPO | — | all |

---

## Related: two other pieces of at-risk uncommitted work

- **`stash@{0}` (on `main`) holds the `history_mode` flag**: 3 files, +66/-4, 14 mentions of
  `history_mode` across `schema.py`, `orchestrator.py`, `run_experiment.py`. This is the *other*
  fix for the same bug: instead of retraining on folded data, feed the existing SFT adapter the
  native multi-turn conversation at inference so it is evaluated in-distribution. It is not
  committed. Given that the fold implementation was already lost once (Finding 0), this stash is
  one `git stash drop` away from the same fate. **Recommend committing it to a branch now**,
  regardless of which path you take. Note the docs' claim "no `history_mode` exists anywhere in
  the tree" is correct for HEAD and the working tree, but not for the stash.
- It is also a legitimate **fallback**: if the H100 grant (decision #5) does not come through,
  `history_mode: native` gives a valid base-native-vs-SFT-native comparison with no retraining.
  Its caveat is that native mode turns structured memory off, so GRPO (trained in memory mode)
  is not comparable in that setting.

---

## Decisions I need from you

1. **Can you get me the SFT training log and eval numbers, or read access to
   `/projects/0/gusr0608/msc-thesis`?** I verified the mechanism exhaustively but could not
   verify the symptom ("loss down, eval down") from any artifact in this checkout. The plan does
   not depend on it — the format gap is independently disqualifying — but the writeup will want
   the actual before/after, and the training log would settle the overfitting question.
   A session note from 2026-06-17 records the regression as **GAIA 16/165 base → 17 GRPO → 14
   SFT** and **AIME 12/60 → 8 → 6**; those numbers are consistent with the diagnosis but I could
   **not** re-verify them against any file in this checkout, so treat them as provisional until
   the source-of-record run is readable.
2. **Turn-0 answers (175 train rows, 18.1%).** They teach answering at turn 0, which is the
   *Premature direct answering* failure mode in the Ch 6 taxonomy. `--drop-planning-answers`
   will exist; default is keep, matching the native build. **My recommendation: keep them for
   the first folded run.** Changing data composition and format at once makes the comparison
   uninterpretable; drop them in a follow-up run if the failure mode shows up.
3. **The 1:1 → 1:1.56 math:search shift at the loss level (§2.6).** Options: accept and state it;
   subsample search rows back to 1:1 (costs ~651 rows); or per-row weighting. **My
   recommendation: accept and state it.** Subsampling throws away real supervision to preserve a
   ratio that was itself only a heuristic mirror of `combined_train`, and per-row weighting adds
   a second uncontrolled variable. But it must be named explicitly in the writeup.
4. **Where the custom dataset class lives.** `src/verl_ext/` (new, importable by
   `data.custom_cls` without polluting `agent_engine`) vs. `scripts/`. Low stakes; I'll use
   `src/verl_ext/` unless you prefer otherwise.
5. **Training partition access (`007` runs on H100 owned by `gusr0608`/`xchen`; `xchen1` is not
   in that group).** The build and all tests run fine in `cosmas-train` on any node; the
   training run is still blocked. Worth chasing in parallel with steps 1–6 so it is not the
   thing that holds up the relaunch.

I have not changed any code.
