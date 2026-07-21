# SFT format mismatch: why collection is wrong, and how to reuse the data

**Status:** diagnosis code-confirmed; fix specified, not yet run.
**Scope:** the orchestrator SFT path only (GRPO is unaffected, see §3).
**One line:** SFT trajectories are stored as a *native multi-turn* conversation, but at
inference (and during GRPO) the orchestrator feeds a single *memory-folded* user turn, so
the adapter is trained on a format it never sees at evaluation.

---

## 1. The two prompt formats

The orchestrator can present a turn's context to the model in two different shapes.

**(A) Memory-folded — used at inference and during GRPO.**
`AgenticOrchestrator._build_memory_prompt` (`src/agent_engine/core/orchestrator.py:653-664`)
returns exactly two messages. The plan and every prior step are folded into text inside a
single **user** turn:

```
system:  <system prompt>
user:    <question>
         **Query Analysis:**
         <plan>
         **Previous Steps:**
         Action Step 1:
           - Tool: code_generator
           - Sub-goal: ...
           - Command: {"name": "code_generator", "arguments": {...}}
           - Result: ...
         Action Step 2:
           ...
```

The model is then asked to emit the next assistant turn (an action, or the final answer).
`**Previous Steps:**` is rendered by `_format_action_history`
(`orchestrator.py:695-706`).

**(B) Native multi-turn — how SFT data is stored.**
`scripts/collect_sft_data.py::_build_sft_messages` (sft branch) returns
`state.messages[:2] + assistant(raw_query_analysis) + state.output_messages`, and
`output_messages` is the interleaved assistant(action) / tool(result) sequence appended in
`_commit_tool_result` (`orchestrator.py:687-688`). A stored trajectory therefore looks like:

```
system:     <system prompt>
user:       <question>
assistant:  <plan>
assistant:  <sub_goal + tool_call, step 1>
tool:       <result 1>
assistant:  <sub_goal + tool_call, step 2>
tool:       <result 2>
...
assistant:  <final answer, \boxed{...}>
```

Each step here is a **real** assistant/tool message turn, not folded text.

---

## 2. Why the collection is wrong

VERL's SFT trainer (`jobs/fine_tuning/007_train_sft.job:30,134-137`,
`data.messages_key=messages`) uses `MultiTurnSFTDataset`, which **masks loss to zero on
system / user / tool turns and computes loss only on assistant turns**. So the adapter is
trained to produce each assistant turn *conditioned on the native multi-turn history that
precedes it* (real prior assistant and tool message turns, format B).

At evaluation the orchestrator never builds format B. It calls `_build_memory_prompt`
(format A): the prior steps arrive as a `**Previous Steps:**` block inside one user turn,
not as assistant/tool turns. The adapter is asked to continue a conversation shape it was
never trained on. That is the mismatch: **train on B, test on A.**

### 2.1 The mismatch, one decision at a time

Lining up a single one-tool trajectory makes the gap concrete. VERL stores the trajectory as
one native conversation and puts loss on its three assistant turns; at inference the
orchestrator rebuilds a fresh folded prompt for each of those same three decisions. The
**target string is identical** in both worlds. Only the left-context the model conditions on
differs, and it differs on every turn.

Stored trajectory (loss on the assistant turns marked with an arrow):

```
[0] system     <system prompt>
[1] user       Q
[2] assistant  <plan>                     <- planning target
[3] assistant  <tool_call>{...}</tool_call> <- action target
[4] tool       <result 1>
[5] assistant  <final answer>             <- answer target
```

Prompt actually built for each target at inference (`history_mode: memory`, the default):

| Target | Trained left-context (native, B) | Inference left-context (folded, A) |
|--------|----------------------------------|------------------------------------|
| plan [2] | `system, Q` | `system, user(Q + planning_suffix)` |
| action [3] | `system, Q, assistant(plan)` | `system, user(Q + **Query Analysis:** plan)` |
| answer [5] | `system, Q, assistant(plan), assistant(tool_call), tool(result)` | `system, user(Q + **Query Analysis:** plan + **Previous Steps:** Action Step 1 ...)` |

Every row differs. The plan moves from a standalone **assistant** turn into **user**-turn
prose under `**Query Analysis:**`; the prior action moves from a `<tool_call>` XML assistant
turn plus a `tool`-role result into folded `Command:/Result:` prose inside the user turn. SFT
thus optimises `P(target | native history)` while evaluation samples
`P(target | folded prompt)`: same tokens to emit, different conditioning distribution.

**Even the planning turn mismatches.** `_run_planning_turn` (`orchestrator.py:721-743`)
shallow-copies the message list and replaces its last entry with a *new* dict whose content is
`question + suffix` (`planning_messages = list(s.messages); planning_messages[-1] = {..., "content": <last> + suffix}`);
it never mutates `state.messages[1]`. So the stored planning target [2] is conditioned on the bare
question, whereas at inference the planning turn is conditioned on question + suffix. This is
why the suffix reconstruction in §4.1 is required, not optional polish.

### 2.2 Confounds already controlled

Two differences that could have confounded this were already controlled, which is why the
conversation *structure* is the sole remaining cause:

- **Thinking.** `build_sft_parquet.py::_strip_thinking` removes `<think>...</think>` from
  every assistant turn (matching the no-thinking evaluation).
- **System-prompt suffix.** `build_sft_parquet.py::_strip_system_suffix` removes the
  collection-time `system_prompt_suffix` (a tool-use nudge the teacher saw but the
  GRPO/inference system prompt does not), so the SFT system message is byte-identical to the
  inference system message.

With thinking and the system prompt already matched, the only uncontrolled difference is
folded-user (A) versus native-turns (B).

---

## 3. Why GRPO is not affected

GRPO rolls out **through the orchestrator itself**. `OrchestratorRollout._run_episode`
(`src/fine_tuning/rollout.py:292-305`, feat/fine-tuning branch) constructs
`AgenticOrchestrator(...)` with the default history mode and calls `orchestrator.run(...)`,
which goes through `_build_memory_prompt`. GRPO therefore trains on format A, the same
format used at evaluation. This asymmetry is the signature of the bug: a generic cause
(LoRA rank, learning rate, dataset size) would degrade GRPO too, whereas a format gap that
exists *only on the SFT path* predicts exactly the observed pattern (GRPO holds, SFT drops
below base).

### Evidence trail

| Claim | Location |
|-------|----------|
| Inference builds folded `[system, user]` | `orchestrator.py:653-664` |
| `**Previous Steps:**` formatting | `orchestrator.py:695-706` |
| SFT rows are native multi-turn | `collect_sft_data.py::_build_sft_messages` (sft) |
| Interleaved assistant/tool turns accumulate in `output_messages` | `orchestrator.py:687-688` |
| GRPO rolls out through the orchestrator (format A) | `rollout.py:292-305` (feat/fine-tuning) |
| Trainer computes loss on assistant turns only | `007_train_sft.job:30,134-137` |
| Thinking stripped from SFT data | `build_sft_parquet.py::_strip_thinking` |
| System suffix stripped from SFT data | `build_sft_parquet.py::_strip_system_suffix` |

---

## 4. The fix: reuse the already-collected trajectories

**No re-collection is needed.** The expensive step (running the Qwen3-32B teacher through
the orchestrator to get correct trajectories) is done. Its output already stores everything
required to re-derive the folded format, because each `collected_<ts>.jsonl` record keeps
the full native `messages` list plus metadata
(`collect_sft_data.py`, record schema: `question_id, question, data_source, ground_truth,
prediction, correct, messages, turns, tool_counts`).

The fix is a pure **offline transform**: native multi-turn trajectory (B) into a set of
memory-folded **single-turn** rows (A), one per assistant decision point, each with loss on
exactly one assistant target. That is the shape the model sees at inference.

### 4.1 Row expansion

One trajectory with `T` tool-call steps expands into `T + 2` single-turn rows:

1. **Planning row.** Prompt = `[system, user(question + planning_suffix)]`, target = the
   plan. This matches `_run_planning_turn` (`orchestrator.py:721-743`), which appends the
   planning suffix to the user turn. (Include only if the adapter should also learn
   planning; see §5.)
2. **Action rows** (`k = 1..T`). Prompt = folded `[system, user(question + Query Analysis +
   Previous Steps 1..k-1)]`, target = the step-`k` assistant action.
3. **Answer row.** Prompt = folded `[system, user(question + Query Analysis + all steps)]`,
   target = the final answer.

Each emitted row is a 3-message list `[system, user_folded, assistant_target]`. Because
`MultiTurnSFTDataset` masks everything but assistant turns, loss falls on the single target,
exactly as at inference.

### 4.2 Reconstructing the folded prompt from stored `messages`

Given a stored (thinking-stripped, suffix-stripped) `messages` list
`[system, user, assistant(plan), assistant(a1), tool(r1), assistant(a2), tool(r2), ...,
assistant(answer)]`, rebuild each step's `action_history` entry from the (action, result)
pairs using helpers that already exist:

- `tool_call = parse_tool_call(a_i)` (`src/agent_engine/utils/parsing.py:58`)
- `tool_name = tool_call["name"]`, `command = json.dumps(tool_call)`
- `sub_goal = _extract_sub_goal(a_i)` (the `<sub_goal>` tag, `orchestrator.py:708-719`)
- `result = r_i` (the following tool message content)

Then build the folded user content with the **same** code path the orchestrator uses, not a
hand-rolled string: import and call `_format_action_history` and reproduce the `parts` join
from `_build_memory_prompt` (`orchestrator.py:653-664`). Byte-identity with inference is then
guaranteed by construction, which is the whole point.

### 4.3 Where to put the transform

Recommended: add a `--format {native,folded}` flag to `scripts/build_sft_parquet.py`
(default `native` to preserve current behaviour). Run it **after** the existing per-record
processing so the transform inherits every control already in that script (thinking strip,
suffix strip, correct-only filter, dedup, stratified train/val split). For `folded`, replace
each record's native `messages` with the `T + 2` folded single-turn message lists, carry the
same `data_source / question / result / extra_info`, then split and write
`sft_train.parquet` / `sft_val.parquet` unchanged. Job `007` needs no change: it already
consumes a `messages` column with assistant-only loss.

This means the end-to-end fix is: rerun `build_sft_parquet.py --format folded` over the
existing `collected_<ts>.jsonl`, then rerun `007_train_sft.job`. One build, one retrain, no
new collection.

---

## 5. Details to get right

- **Planning suffix.** The planning row's user turn must append the *same* planning suffix
  used at inference (`_DEFAULT_PLANNING_SUFFIX_TOOLS`, or the configured `planning_suffix`),
  not the system-prompt suffix that `_strip_system_suffix` removes. These are two different
  suffixes on two different turns.
- **Whether to train the planning turn.** If the goal is to adapt the action/tool policy
  only and leave planning to the base model, drop the planning row. If the collected plans
  are part of what makes trajectories succeed, keep it. Decide explicitly; do not leave it
  implicit.
- **Sequence length.** Folded action rows re-include all prior steps in the user turn, so
  tokens across the `T + 2` rows grow as O(T^2) per trajectory versus O(T) for one native
  row. With `data.max_length=16384` and short trajectories this is fine, but check the
  longest trajectories are not truncated (`data.truncation=right`).
- **Refold both splits.** Regenerate `sft_val.parquet` the same way; a native val set would
  reintroduce the mismatch in the validation loss and mislead early stopping.

---

## 6. Validation plan

1. **Cheap diagnostic (no retrain).** Evaluate the *existing* adapter with the inference-side
   `history_mode: native` flag (currently in `stash@{0}` on main: adds
   `_build_native_prompt`, off by default). If accuracy recovers toward or above base under
   native mode, the mismatch is confirmed as the operative cause, not merely a plausible one.
2. **The fix.** Rebuild with `--format folded`, retrain, evaluate in the normal (folded)
   `history_mode: memory` inference path, and compare against base and GRPO.

---

## 7. Data and access

Collected data lives in the shared `gusr0608` project space (symlink
`/projects/0/gusr0608` -> `/gpfs/work5/0/gusr0608`):

- Raw trajectories: `/projects/0/gusr0608/msc-thesis/data/training/sft/collected_<ts>.jsonl`
  (e.g. `collected_20260605_214650.jsonl`), all attempts plus yield stats.
- Training parquet: `/projects/0/gusr0608/msc-thesis/data/training/sft/sft_train.parquet`
  (and `sft_val.parquet`), the `messages` column consumed by job `007`.
- Source parquet it was built from: `data/training/train/combined_train.parquet` (readable
  in the home checkout).

**Access blocker.** The `gusr0608` project space and the SFT adapter at
`/scratch-shared/xchen/fine_tuning/sft/` are owned by the `xchen` / `gusr0608` account; the
current account (`xchen1`) is not in that group, so both are `Permission denied`. The single
access grant that unblocks the diagnostic (read the adapter) also unblocks the fix (read
`collected_<ts>.jsonl` to refold). Request read access to both paths together.
