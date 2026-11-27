# Tool-call intervention: design

**Date:** 2026-11-17 (updated 2026-11-27)
**Status:** built and reviewed on branch `feat/tool-call-intervention`; collection started
2026-11-27. Section 12 records the decisions made while building it; where it is more
specific than Sections 1 to 11, Section 12 is what the code does. How to run it:
`docs/pipelines/rsi_intervention.md`.
**Builds on:** `docs/superpowers/specs/2026-11-04-recursive-self-improvement-design.md`
(the RSI spec: Phase 2 cycle, arms, evaluation), `docs/pipelines/rsi_overview.md`
(Phase 1 results).

## 1. Intent

Phase 1 improved the agent team by having a reflector rewrite prompts. Neither confirmed
gain was significant (search 42.0% vs 46.5%, p = 0.19; math 62.0% vs 63.5%, p = 0.78).
A prompt is a blunt lever: one edit has to help every question at once.

This design tries a sharper lever. Instead of changing the prompt, the reflector
**fixes one orchestrator tool call** in a failed trajectory. The trajectory is re-run from
that point; if it now reaches the gold answer, it becomes training data. Prompts stay at
the seed throughout, so the fine-tuned model learns the better call directly under the
prompts it will be evaluated with.

The question it answers: **does training on trajectories rescued by a single fixed tool
call teach the model more than training on its own correct trajectories alone?**

## 2. What the failures look like (why these interventions)

Seed team (Qwen3-8B in every role), 200 unused confirmation questions per domain, from
`experiments/results/rsi/phase1/search_reflector_qwen3_8b_slots_full/{search,math}/confirm/v0.json`:

- **Search, 116 wrong.** 97 made exactly one `web_search` call and answered. Of the 107
  failures with a non-yes/no gold, the gold string never appeared in the retrieved pages
  in 71 (a rough exact-match check). The main fixable problem is the query, or a missing
  second search.
- **Math, 76 wrong.** 56 made no tool call: the planning turn decided "no
  `code_generator` needed" and turn 1 answered directly.

So an intervention must be able to rewrite a call, insert a call where none was made, and
add a follow-up call after a weak result. Section 4 shows these are one operation.

## 3. Goals, non-goals, anti-goals

**Goals**
- A pipeline that takes the model's failed trajectories, gets one tool-call fix per
  trajectory from a reflector, re-runs, and keeps the rescued ones.
- A new Phase 2 arm, **arm I (call intervention)**, trained on arm C's data plus the
  rescued trajectories, compared with arm C after one cycle.
- **Arm C is built in this plan (decided).** None of the Phase 2 pieces it needs exist
  yet (Section 8). Both arms train the **orchestrator only** (decided): no sub-agent rows.
  The intervention only changes orchestrator calls, so sub-agent rows would add work
  without bearing on the question. This departs from the RSI spec's arm C, which trains
  sub-agent rows too; those stay in scope for the prompt-search arms (A, B) whenever they
  run, not here.

**Success (decided):** arm I beats arm C on the five benchmarks under the seed prompts.
Beating base Qwen3-8B by a set margin is not required.

**Non-goals**
- Changing prompts. No prompt search in this arm.
- Rewriting the planning text, the sub-agents' outputs, or the final answer. Only
  orchestrator tool calls are intervened on.
- More than one intervention per trajectory (decided).
- Sub-agent training rows, in either arm (decided).
- A second cycle, until the first one works.
- RL. This is SFT, like the other RSI arms.

**Anti-goals (a failure even if it runs)**
- Rescued calls that work because they contain the answer. Training on those teaches
  cheating, not better tool use.
- A gain that cannot be attributed to the rescued trajectories (for example, a different
  data pipeline, SFT method or setting between arm I and arm C).
- A score gain with no change in the targeted behaviour (Section 7). That result would
  not be trusted.

## 4. The intervention

### 4.1 One operation covers all three kinds

A trajectory is: a planning turn, then actions `a_1 ... a_n` (each a `<sub_goal>` plus a
`<tool_call>`, with its tool result), then a final answer. An intervention is a pair
`(p, call)` with `0 <= p <= n`:

1. keep the planning turn and the first `p` actions exactly as they were;
2. execute `call` for real (live tools, same sub-agent model and settings);
3. let the orchestrator continue on its own from there, under the seed prompt, until it
   answers or hits `max_turns`.

- `p < n`: **rewrite** (action `p+1` is replaced by `call`; everything after it is
  discarded and regenerated).
- `p = n`, `n >= 1`: **follow-up** (a new call before the answer).
- `p = n = 0`: **insert** (a call where the orchestrator went straight to an answer).

The kind is a label for analysis only; the code path is the same.

### 4.2 Which trajectories

- Collection follows the RSI spec's Step 3 under the seed prompts: `M_0` (Qwen3-8B) in
  every role, over all of `combined_train.parquet`, `--n-samples 3`, the existing
  row-selection rules. That run **is** arm C's data. It needs a new collection config
  (Section 8); the existing `collect_sft_qwen32b.yaml` uses a 32B orchestrator and 1.7B
  sub-agents.
- Candidates for intervention are the questions with **0 of 3 correct**. Questions the
  model solves at least once are already in arm C, and rescuing them would add nothing.
- The trajectory intervened on is the one from the first pass. Collection must save
  failed trajectories with the full state needed to resume (planning text,
  `query_analysis`, `action_history`, `output_messages`, sub-agent call logs); today only
  correct ones are kept.
- Skip trajectories whose planning turn produced the final answer (finished at turn 0),
  because the plan text that later turns see already contains the wrong answer. Count them.

### 4.3 The reflector

- **Model:** the same model as the team (`M_0`, Qwen3-8B), with thinking on, as in Phase 1.
  This keeps it self-improvement. If the go/no-go (Section 6) fails, rerun the rescue
  step with Qwen3-32B as reflector before giving up, which mirrors the RSI spec's arm B.
- **Input:** one failed trajectory, rendered step by step with sub-agent inputs and
  outputs (reuse `src/rsi/reflector_input.py`), the question and the gold answer.
- **Task:** find the first step that went wrong and return `p`, a sub-goal and the
  corrected tool call. It may only use the tools the orchestrator has (`web_search` with
  `query`, `code_generator` with `task`) and their existing argument schemas.
- **Output format:** plain text with tags, the tool call in exactly the format the
  orchestrator emits so `parse_tool_call` reads it. No JSON wrapper around free text
  (Phase 1 found JSON replies mangle `\boxed` and `\frac`).
- **Second attempt:** if the continuation is still wrong, the reflector gets one more try
  on the same trajectory, shown its first fix and what happened. At most two attempts per
  question.

### 4.4 Leakage guard

The reflector sees the gold answer, so a fix can smuggle it in. Reject a fix (it counts as
a failed attempt) if:
- the normalised gold or any alias appears as a whole token in the new call's arguments
  or sub-goal, **unless** it also appears in the question itself; or
- it copies an 8-word sequence from any other training question (the existing
  `src/rsi/leakage.py` check).

A fix can still help by pointing at the right bridging entity the model could not find.
That is hindsight, and it is the point of the method; it is reported, not blocked (see the
blind-reflector check in Section 7).

### 4.5 Resuming a trajectory

The orchestrator rebuilds its prompt every turn from `[system, user]` where the user turn
is the question, the query analysis and the formatted `action_history`
(`AgenticOrchestrator._build_memory_prompt`). So the state after `p` actions is fully
determined by the stored planning text and the first `p` actions. Resuming means building
an `ExecutionState` with those fields, `turn = p`, `tool_calls` and `tool_counts`
recomputed, then executing `call` through the normal tool path (`_execute_tool`,
`_commit_tool_result`) and entering the normal turn loop. The injected call uses one turn
of the `max_turns` budget. The turn loop in `run()` needs to accept a starting state; with
no starting state, behaviour and outputs must be unchanged.

Continuations use the same sampling settings and seeding as collection.

## 5. Training data for arm I

- Arm I's data = arm C's data **plus** the rescued trajectories. Same row-selection rules,
  same SFT method and hyperparameters, same seed-prompt folding
  (`scripts/build_sft_parquet.py`, `_fold_trajectory`), orchestrator rows only, and the
  same format check (`scripts/check_sft_folded_format.py`). With no prompt search, both
  arms collect under the seed prompts, so there is nothing to swap back; the check only
  has to confirm each row's `[system, user]` equals what `AgenticOrchestrator` builds under
  the seed prompt.
- **Order of operations (Claude's call):** build arm C's dataset exactly as the existing
  pipeline does (50/50 rebalancing, 10% validation split), then append the rescued rows to
  arm C's training split. Arm C's rows and its train/validation membership are therefore
  identical in both arms. Rescued rows are not rebalanced and do not go into validation,
  so arm I's search:math mix is no longer exactly 50/50; report the actual mix.
- **SFT method (Claude's call):** `lora` (rank 64, existing hyperparameters, 2×H100,
  `jobs/fine_tuning/007_train_sft_folded.job`), the same as the existing SFT direction, so
  both arms are cheaper and comparable with the earlier LoRA runs. The adapter is not
  merged: it is evaluated on the orchestrator only (Section 7).
- **Training budget (Claude's call):** both arms train for the same number of epochs. Arm I
  therefore takes a few more steps; this is accepted as part of the effect and the step
  counts are reported.
- **Rows from a rescued trajectory:** only the intervention row and the rows after it are
  trained. The planning row and the first `p` action rows are kept as context in later
  rows' prompts but are not targets, since they led to a failure.
- **The intervention row's target** is the reflector's sub-goal and tool call, formatted
  exactly like an orchestrator action turn, so it folds and checks like any other row.
- Record per arm: number of rows, number of questions covered, rescued rows by kind
  (rewrite / follow-up / insert) and domain, and the share of no-tool rows.

Arm I covers more questions than arm C by construction (the rescued ones). That is the
effect being tested, not a confound, but report row and question counts side by side.

## 6. Go/no-go before training

Run the rescue step on the 0-of-3 questions and measure the **rescue rate**: rescued
questions over 0-of-3 questions, per domain, after leakage rejection.

Some 0-of-3 questions will be solved by sampling the continuation again, with no fix.
**No-fix control (Claude's call):** for every question, resume from the same point `p`
the reflector chose, with the original next step left to the model (no injected call),
once per reflector attempt slot: two draws for every question, whether or not the fix
already rescued it, each with the same sampling seed as that attempt's fix. A question
is control-solved if any of its draws is correct. The **net rescue rate** is the rescue
rate minus the no-fix control's rate, both over the domain's 0-of-3 questions.

- **Net rescue rate of at least 10 points in at least one domain:** build arm I and train.
- **Below that in both:** rerun the rescue step with the 32B reflector. If that also fails
  the gate, stop and write up why the fixes fail.

All rescued trajectories, including ones the no-fix control would also have solved, go
into arm I's data; the control only decides the gate.

Also report attempts used, leakage rejections, and the kind distribution.

## 7. Evaluation

- **Headline:** arm I vs arm C after one cycle, both under seed prompts, on AIME, GAIA,
  GPQA, HLE and MuSiQue, with the RSI spec's splits and tool setup (`rsi_inference`
  configs). **The trained adapter is loaded on the orchestrator only; both sub-agents stay
  base Qwen3-8B (Claude's call),** because only orchestrator rows were trained, so the
  sub-agents should be the ones the trajectories were collected with. Base Qwen3-8B in
  every role is reported alongside as the reference.
  **Success = arm I's mean accuracy over the five benchmarks is higher than arm C's.**
  Per-benchmark accuracy and a paired exact McNemar p (reuse `src/rsi/confirm.py`) are
  reported; significance is not required for success.
- **Behaviour (must move in the targeted direction for the gain to be trusted):**
  share of math questions (AIME) where the orchestrator calls `code_generator`, and share
  of search questions (MuSiQue, GAIA) with two or more `web_search` calls, arm I vs arm C.
- **Blind-reflector check (report):** on a sample of 100 rescued questions, rerun the
  reflector without the gold answer and report how often it proposes a call that still
  rescues. A large gap means the fixes lean on hindsight.

## 8. What has to be built

Nothing from the RSI spec's Phase 2 exists yet. This plan builds, for both arms:

1. **Collection config:** Qwen3-8B in every role, seed prompts, the Phase 1 tool set and
   settings (Section 9), `--n-samples 3`. **Each pass needs its own sampling seed
   (Claude's call):** `VLLMProvider` passes the same `config.seed` to every request, so
   without this the three passes repeat each other and "0 of 3" means "0 of 1". The
   rescue step's attempts and its no-fix control get seeds of their own in the same way.
2. **Collection changes** (`scripts/collect_sft_data.py`): save failed trajectories with
   the state needed to resume (Section 4.2), and the sub-agent call logs that
   `core/batching.py` already puts in `state.metadata` (the reflector needs them as
   input; they are not trained on).
3. **The rescue step:** reflector, resume, leakage guard, no-fix control, go/no-go
   report (Sections 4 and 6).
4. **Dataset builds** for arm C and arm I (Section 5), with the format check.
5. **SFT runs** for both arms with the existing LoRA pipeline (no merge).
6. **Evaluation configs** under `experiments/configs/qwen3/rsi_inference/`, generated by
   `scripts/generate_configs.py`, with the adapter on the orchestrator and base Qwen3-8B
   sub-agents; base Qwen3-8B
   runs reused if their splits, tool set and thinking mode match (RSI spec, "Reference:
   homogeneous base 8B").

## 9. Constraints

- Same model, tool set, `max_turns: 7`, thinking settings and sub-agent setup as the RSI
  Phase 1 and Phase 2 configs. Team thinking is off (`thinking_mode: "NO"`), reflector
  thinking is on.
- The prompts the trained model sees are byte-identical to the seed prompts. The existing
  folding and check scripts enforce this; arm I must pass them.
- Live web search goes through the existing Serper cache.
- Nothing chains automatically: collection, rescue, go/no-go, training and evaluation are
  launched by hand, as in the RSI spec.

## 10. Prior art

- **Agent-R** (2025): finds the first error step in a failed agent trajectory and splices
  in a corrected continuation for self-training. Closest to this design; read before
  planning. Differences: here the fix is restricted to one tool call, and correctness comes
  from a gold-answer verifier.
- **STaR rationalization:** rescues failures by showing the model the answer. The same
  hindsight risk applies, hence Sections 4.4 and 7.

## 11. Left to the builder

- Module and script names and layout (for example under `src/rsi/`).
- The exact reflector prompt and tag format, within Section 4.3.
- How collection saves failed trajectories, within Section 4.2.
- Batching and job layout on the cluster.

Section 12 records how each of these was settled.

## 12. Decisions made during implementation

Code lives in `src/rsi/intervention/` (records, fix, rescue, report),
`scripts/run_call_intervention.py`, `scripts/build_arm_i_parquet.py`,
`scripts/compare_arms.py` and `scripts/report_intervention.py`. Each point below was
settled during the build or its reviews.

### 12.1 Collection

- **Seeds:** pass `k` (0-based) samples with seed `config.seed + k` (0, 1, 2), set on
  every model provider, sub-agents included. Without this the three passes would repeat
  each other.
- **Later passes:** as in the existing collection script, passes 2 and 3 only rerun the
  questions still unsolved. A question has three records only if it failed every pass.
- **Grading** accepts the gold answer's aliases (`extra_info["golden_answers"]`), the
  same yardstick the rescue step uses, so "0 of 3" and "rescued" are measured alike.
- **System prompts:** each question gets the seed prompt of its own domain: the
  `deepmath` prompt for math, the `gaia` prompt for `hotpotqa` and `nq`. A pre-existing
  orchestrator bug built every prompt in a batch from the first question's system
  prompt, so mixed batches ran about half their questions under the other domain's
  prompt. It is fixed: every state now uses its own system prompt. Batches with one
  prompt, which covers every benchmark run, are byte-identical to before. Collections
  made before the fix (including the earlier 32B SFT collection) were affected.
- **Records:** every run is saved, right or wrong, with the question, gold answer and
  aliases, seed, prediction, plan, `query_analysis`, `action_history`, sub-agent calls
  and full messages, which is enough to rebuild the run at any step.

### 12.2 Candidates

- Candidates are the pass-1 records of questions with exactly three records, none
  correct.
- A question with an error record in any pass is not a genuine 0 of 3: it is counted as
  `skipped_error`. Runs whose planning turn gave the final answer are counted as
  `skipped_planning_answer` (Section 4.2). Malformed or truncated records raise an error
  instead of being guessed at.

### 12.3 The fix

- **Reply format:** `<step>N</step>`, `<sub_goal>...</sub_goal>` and
  `<tool_call>{json}</tool_call>`, after the reflector's thinking. `N` is the number of
  original actions kept (the `p` of Section 4.1), capped at `min(n_actions,
  max_turns - 1)` so the injected call always has a turn. The last complete block of each
  tag is used. The call must be `web_search` with a non-empty `query` or `code_generator`
  with a non-empty `task`, nothing else; the sub-goal is at most 500 characters.
- **Round-trip check:** the fix is injected as text in the orchestrator's own action
  format. A fix is rejected as a parse error unless that text parses back to exactly the
  same call and sub-goal, so the arm I training target is always the fix itself.
- **Reflector instruction on leakage:** the corrected call may use the question's own
  entities, numbers and wording (a corrected search has to name them); it must not
  contain the gold answer or anything that reveals it. Phase 1's prompt-search
  instruction ("do not copy named entities or numbers") does not apply to fixes.
- **Leakage guard (Section 4.4):** whole-token match of the normalised gold and aliases
  in the sub-goal and argument, after removing line-start list numbering, with no length
  exemption, so `2` in "print x, which should be 2" is rejected but a numbered step "2."
  is not. Answers that occur in the question are exempt, and so are 8-word sequences
  shared with the question itself.
- **Second attempt:** the reflector is shown its first fix, the step it targeted and the
  wrong answer that followed. If the first fix was rejected for leakage it is told so; a
  parse error passes nothing forward.
- **Long trajectories:** if the reflector input exceeds 14000 tokens, per-call text is
  halved down to 200 characters, then middle actions are cut.

### 12.4 The rescue step

- **Pairing:** for each attempt `a`, the fix continuation and the no-fix control start
  from separately rebuilt states and use the same sampling seed, `1000 + a`.
- **Control step:** the fix's `p`. For a parse error it is `min(n_actions,
  max_turns - 1)`. For a question already rescued at attempt 1, attempt 2 runs the
  control only, at the rescuing fix's step, and writes a row with status
  `rescued_earlier`.
- **Row statuses:** `ok`, `parse_error`, `leakage`, `rescued_earlier`. Every row carries
  a control draw.
- **Robustness:** each attempt runs in chunks of 100 candidates and appends its rows
  after every chunk, so a killed job loses at most one chunk and a rerun skips what is on
  disk. If every reflector call in a chunk fails, the run stops with an error rather than
  using up an attempt for every question. A single failed call counts as a parse error.

### 12.5 The gate

- Rescue rate, control rate and net points are over the domain's 0-of-3 questions
  (`zero_of_n`), as in Section 6. The same rates over the candidates only are reported
  next to them but do not decide the gate.
- **Blind check:** one attempt with the gold answer hidden (seed `1000 + 99`) on up to 100
  rescued questions. It reports its own no-fix control as the chance baseline.

### 12.6 Arm I

- One rescued trajectory per question: the first correct `ok` attempt.
- Its folded rows are kept from the fix on (`rows[1 + p:]`) only if row `1 + p`'s target
  equals the fix text. Otherwise the trajectory is dropped and counted as misaligned.
- Over-length filtering (16384 tokens) applies to the new rows only. A rescued question
  already in arm C (matched by question text, or by domain plus index) is an error.
- Arm C's rows come first and unchanged, and arm C's validation file is copied byte for
  byte. Training shuffles rows, so their order does not matter.

### 12.7 Training and evaluation

- The SFT job takes the data files and run name from the environment, so both arms run
  the same job and settings.
- GPU assignment counts model instances (path plus adapter), not paths, so a LoRA
  orchestrator and base sub-agents on the same Qwen3-8B get one GPU each. Configs without
  an adapter are assigned exactly as before.
- GAIA and HLE evaluation configs keep the `text_inspector` sub-agent, like the base
  reference runs, so base, arm C and arm I see the same tools. The adapter was trained
  without that tool.
- The arm comparison reads only the named variant in each results folder and takes its
  newest run by timestamp. The base MuSiQue run predates the `validation_subset_200` split
  name, so McNemar is computed for arm I against arm C only.

### 12.8 Figures report and trial runs

- `scripts/report_intervention.py` turns the rescue output into `report.md`,
  `report.json` and figures:
  - rescue against the no-fix control, with the net interval from Newcombe's paired
    method on the same questions;
  - the ceiling on the training questions: pass@1, pass@3, and pass@3 plus net rescued;
  - the funnel from 0-of-3 questions to rescued;
  - fix kinds and steps;
  - tool use before and after the fix;
  - the blind check;
  - examples.

  The ceiling is an upper bound on what the decision changes could add on the training
  questions, not a benchmark gain. After evaluation it also plots the arm comparison.
- `SUBSET=N` runs a trial: the collection job collects only the first N training
  questions, and the rescue job rescues only the first N 0-of-3 questions in collected
  order. The gate counts are restricted to the questions up to the last of those, so
  subset rates are not diluted.
