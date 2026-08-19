> **HISTORICAL — not maintained.** Archived 2026-08-16 during the repository
> handover. Kept for the reasoning it records; paths, commands, and numbers in
> it may be stale. For current documentation see `docs/`.

# GEPA Integration Design
**Date:** 2026-05-15
**Thesis chapter:** System Adaptation
**Replaces:** GRPO fine-tuning plan

---

## 1. Overview and Motivation

GEPA (Genetic-Pareto) is a prompt optimizer for compound AI systems. Instead of updating model weights via RL (e.g. GRPO), it reads full execution traces and uses an LLM reflector to rewrite module prompts. The paper claims up to 19% gains over GRPO on Qwen3-8B while using up to 35× fewer rollouts.

The thesis failure-mode analysis already shows that most agent errors are prompt-level problems: poor planning, wrong tool selection, over-searching, and bad sub-goal decomposition. These failure patterns are visible in the execution traces that `AgenticOrchestrator` already captures. GEPA's reflector is specifically designed to read exactly these traces and propose targeted prompt fixes — making this a natural fit.

**Core thesis claim supported by this integration:**
*System-level prompt adaptation — targeting the orchestrator's instructions and planning turn — significantly improves agentic performance, and reflective trace-based optimization is a more efficient path to that adaptation than weight-level RL.*

---

## 2. Architecture and Module Structure

### New module: `src/gepa_integration/`

Sits alongside `agent_engine/` and `fine_tuning/` in `src/`:

```
src/gepa_integration/
    __init__.py
    seed.py          # build_seed_candidate(), build_splits()
    adapter.py       # AgentGEPAAdapter — implements GEPAAdapter protocol

scripts/
    run_gepa.py      # CLI entry: --benchmark gaia/gpqa/math --config ...

experiments/configs/gepa/
    gaia.yaml
    gpqa.yaml
    math.yaml
```

### Dependencies

- `gepa` package (already in repo at `../gepa/`)
- All existing `agent_engine` internals — no forks
- `Qwen3-32B` as reflector (separate vLLM instance on the same cluster node)

---

## 3. Candidate Schema

Each GEPA candidate is a `dict[str, str]` with exactly two keys:

```python
{
    "system_prompt":   "<full rendered system prompt string>",
    "planning_suffix": "<text appended to the user message on the planning turn>"
}
```

**`system_prompt`** is the complete output of `PromptBuilder.build_system_prompt()` for the benchmark, rendered once at startup to form the seed. It contains three logical regions:

| Region | Optimizable? | Notes |
|---|---|---|
| Preamble (`base_instruction_tools`) | Yes | "You are a reasoning assistant…" |
| Tool schemas (`<tools>…</tools>`) | **No** | Machine-readable API definitions — must stay verbatim |
| In-context example (`### EXAMPLE`) | Yes | Reasoning chain, sub-goals, tool call format — rich target for reflection |
| Final instructions | Yes | Answer format, reminders |

GEPA's reflector is instructed (via its own system prompt) to **never modify content between `<tools>` and `</tools>` tags**. Everything else in the string is fair game. In practice the most productive mutations will be to the in-context example (teaching better reasoning patterns) and the preamble/final instructions (adding or removing behavioural constraints).

**`planning_suffix`** is the text currently hardcoded in `AgenticOrchestrator._run_planning_turn()`. It is extracted into a configurable parameter (see §4) so GEPA can mutate it independently of the system prompt.

### Seed construction (`seed.py`)

```python
def build_seed_candidate(benchmark: str, config: ExperimentConfig) -> dict[str, str]:
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt(
        dataset_name=benchmark,
        tool_schemas=...,   # from config
        direct_tool_call=config.tools.direct_tool_call,
        baseline=False,
    )
    planning_suffix = DEFAULT_PLANNING_SUFFIX_WITH_TOOLS  # extracted from orchestrator
    return {"system_prompt": system_prompt, "planning_suffix": planning_suffix}
```

---

## 4. Required Code Changes (minimal)

### `src/agent_engine/core/orchestrator.py`

Add `planning_suffix: Optional[str] = None` to `AgenticOrchestrator.__init__`. In `_run_planning_turn()`, replace the two hardcoded suffix strings with:

```python
suffix = self.planning_suffix if self.planning_suffix is not None else (
    _DEFAULT_PLANNING_SUFFIX_TOOLS if len(self.tools) > 0
    else _DEFAULT_PLANNING_SUFFIX_NO_TOOLS
)
```

The two default strings are extracted as module-level constants so `seed.py` can import and use them to construct the seed candidate.

Also in `_run_planning_turn()`, store the raw (pre-strip) planning output on the state before stripping:
```python
s.raw_query_analysis = text          # new — full text including <think> blocks
s.query_analysis = strip_thinking_tags(text)
```

### `src/agent_engine/core/state.py`

Add one field to `ExecutionState`:
```python
raw_query_analysis: Optional[str] = None
```

No other files change.

---

## 5. Adapter: `AgentGEPAAdapter`

**File:** `src/gepa_integration/adapter.py`

### Types

```python
DataInst     = DatasetExample          # existing class from datasets/base.py
Trajectory   = ExecutionState          # existing class from core/state.py
RolloutOutput = str                    # predicted answer string
```

### `evaluate(batch, candidate, capture_traces)`

1. Build a fresh `AgenticOrchestrator` with:
   - The existing shared vLLM model provider (no re-allocation)
   - The existing tool registry
   - `planning_suffix=candidate["planning_suffix"]`
2. Call `orchestrator.run_batch(system_prompts=[candidate["system_prompt"]] * N, ...)`
3. Score each `ExecutionState` with `evaluate_answer(state.answer, example.answer, choices=...)`
4. Return `EvaluationBatch(outputs=[state.answer, ...], scores=[score, ...], trajectories=[state, ...] if capture_traces else None)`

**Thinking mode:** `ORCHESTRATOR_ONLY` — matches the main experimental condition and gives the reflector access to the orchestrator's `<think>` blocks.

### `make_reflective_dataset(candidate, eval_batch, components_to_update)`

Produces per-component lists of JSON-serialisable trace records. At most 12 examples per call (6 correct, 6 wrong — balanced).

**For `system_prompt`:**
```python
{
    "Inputs": {
        "question": state.question,
    },
    "Generated Outputs": {
        "predicted_answer": state.answer,
        # state.output_messages[0] is the first assistant turn (may be planning or first action);
        # extract <think>...</think> content if present
        "thinking_before_first_tool": _extract_thinking(state.output_messages[0]["content"]) if state.output_messages else "",
        "action_steps": [
            {
                "tool": a["tool_name"],
                "sub_goal": a["sub_goal"],
                "result_snippet": a["result"][:300],
            }
            for a in state.action_history
        ],
    },
    "Feedback": (
        "CORRECT"
        if score > 0
        else f"WRONG — ground truth: {gt}. Predicted: {state.answer}. "
             + ("Max turns reached without answer." if state.metadata.get("max_turns_reached") else "")
             + (f" Failure mode: {state.metadata.get('failure_mode', 'unknown')}." if "failure_mode" in state.metadata else "")
    )
}
```

**For `planning_suffix`:**
```python
{
    "Inputs": {
        "question": state.question,
    },
    "Generated Outputs": {
        # state.raw_query_analysis is the full planning turn output including <think> blocks
        # (added to ExecutionState in §4; falls back to stripped query_analysis if None)
        "raw_planning_output": state.raw_query_analysis or state.query_analysis or "",
        "tools_subsequently_used": [tc["name"] for tc in state.tool_calls],
        "num_turns_taken": state.turn,
    },
    "Feedback": (
        "CORRECT — the planning analysis led to a successful solution."
        if score > 0
        else f"WRONG — the planning analysis was: '{state.query_analysis}'. "
             "Consider whether the plan correctly identified the required steps and tools."
    )
}
```

**Note on thinking extraction:** The orchestrator's `<think>...</think>` blocks are present in `state.output_messages` (they are preserved in the assistant content, only stripped from tool responses). The adapter extracts the thinking section from the first assistant message for the `system_prompt` reflective records. This gives the Qwen3-32B reflector visibility into the orchestrator's internal reasoning at the point of failure.

---

## 6. Data Strategy

### Existing runs as baseline

You already have Qwen3-8B `ORCHESTRATOR_ONLY` runs on the full dataset for all three benchmarks. These results are the baseline. For the held-out test comparison, filter `raw_results.json` to held-out test question IDs — no re-running the baseline.

### Split construction (`seed.py: build_splits()`)

Splits are defined by **question IDs**, fixed before any GEPA run, and saved to `experiments/configs/gepa/splits/{benchmark}_splits.json`. The generator takes the existing `raw_results.json` as input.

**Allocation:**

| Benchmark | Total used | GEPA train | GEPA val (D_pareto) | Held-out test |
|---|---|---|---|---|
| GAIA val | ~165 | 80 | 45 | 40 |
| GPQA Diamond | ~198 | 100 | 48 | 50 |
| MATH500 subset | 200 | 100 | 50 | 50 |

### Failure-stratified training selection

The GEPA **train set** is not random — it is stratified by failure mode using the existing run results.

**Failure mode classification** is done by importing `classify_failure()` directly from `scripts/failure_modes/analyze_failure_modes.py`. No need to pre-run the analysis script; `build_splits()` calls it inline on each failed record from `raw_results.json`. The six failure mode labels used are:

- `modality_tool_gap`
- `tool_loop_or_empty_final`
- `direct_reasoning_no_action`
- `computational_subgoal_error`
- `retrieval_evidence_failure`
- `single_shot_tool_trust`

**Sampling rule:**
- ~65% of training examples are questions the current Qwen3-8B (AgentFlow, `ORCHESTRATOR_ONLY`) **fails** on, sampled proportionally across the six failure mode categories
- ~35% are questions it **gets right** (gives the reflector positive signal about what the current prompts already do well)

**Val (D_pareto)** is randomly sampled — unbiased Pareto selection.
**Held-out test** is randomly sampled — unbiased final reporting.

This stratification directly connects the system adaptation chapter to the failure-mode chapter: the GEPA training set is designed to cover the full spectrum of identified failure patterns.

**Optional ablation:** random training selection vs. failure-stratified selection on the same held-out test — directly shows whether failure-mode-aware construction matters.

---

## 7. GEPA Hyperparameters

```yaml
rollout_budget: 150          # per benchmark; converges earlier in practice
minibatch_size: 10           # examples per reflective mutation step
reflector_model: Qwen3-32B   # separate vLLM instance; thinking enabled
merge_proposer: true         # cross-pollinate system_prompt and planning_suffix improvements
frontier_type: pareto        # Pareto-based candidate selection
num_parallel_proposals: 1    # sequential (stable; parallel is an optional speedup)
acceptance_criterion: strict # new minibatch score must strictly improve
```

**Compute estimate:** 150 rollouts × ~15 turns × ~5s/turn on H100 ≈ 3h per benchmark. Three benchmarks ≈ 9h total. Reflector calls add ~15min per benchmark. Total: ~10h cluster time.

---

## 8. Experiment Configs

Each benchmark gets a YAML under `experiments/configs/gepa/`, e.g. `gaia.yaml`:

```yaml
name: "GEPA_gaia_qwen3_8b"
description: "GEPA two-component prompt optimization on GAIA, Qwen3-8B, ORCHESTRATOR_ONLY thinking"

benchmark: "gaia"
thinking_mode: "ORCHESTRATOR_ONLY"
seed: 1

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  role: "orchestrator"

reflector:
  name: "Qwen3-32B"
  path_or_id: "Qwen/Qwen3-32B"

tools:
  enabled_tools: [web_search, code_generator, text_inspector]
  direct_tool_call: true

splits_file: "experiments/configs/gepa/splits/gaia_splits.json"
# Exact paths from the failure-mode analysis inventory (qwen8B, orchestrator thinking):
# GAIA:  experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gaia/qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json
# GPQA:  experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gpqa/qwen8B_subagent_tools_orchestrator/diamond_2026-03-15-21-19-20_20752198/raw_results.json
# MATH:  experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/<math_dataset>/qwen8B_subagent_tools_orchestrator/<run>/raw_results.json
existing_results: "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gaia/qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json"

gepa:
  rollout_budget: 150
  minibatch_size: 10
  merge_proposer: true
  run_dir: "experiments/results/gepa/gaia"

slurm:
  partition: "gpu_h100"
  num_gpus: 4        # 2 for agent + 2 for reflector
  ntasks: 1
  cpus_per_task: 16
  time: "12:00:00"
```

---

## 9. `run_gepa.py` Script Interface

```bash
# Generate splits (once, before any GEPA run)
python scripts/run_gepa.py --mode splits --config experiments/configs/gepa/gaia.yaml

# Run GEPA optimization
python scripts/run_gepa.py --mode optimize --config experiments/configs/gepa/gaia.yaml

# Evaluate best candidate on held-out test
python scripts/run_gepa.py --mode evaluate --config experiments/configs/gepa/gaia.yaml

# Print optimized vs. seed prompt diff
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/gaia.yaml
```

The `evaluate` mode runs `orchestrator.run_batch()` with `candidate["system_prompt"]` and `candidate["planning_suffix"]` on the held-out test question IDs and writes `gepa_results.json` in the same format as `raw_results.json`.

---

## 10. Thesis Narrative

### Section structure

**10.1 Motivation (1–2 paragraphs)**
The failure-mode analysis in the previous chapter identifies six categories of agent failure. These are all traceable to the orchestrator's prompt: insufficient planning guidance leads to poor sub-goal decomposition; vague tool instructions lead to mis-sequenced tool calls; absent recovery instructions lead to max-turns failures. Weight-level RL (GRPO) would require thousands of rollouts to encode these lessons as gradient signal. Because agent traces are natural language, an LLM can read them and propose targeted fixes directly.

**10.2 Method (GEPA overview, ~1 page)**
Describe the two-component optimization (system prompt + planning suffix), the reflective mutation loop, Pareto-based candidate selection, and the failure-stratified training set construction.

**10.3 Experimental setup**
Qwen3-8B, `ORCHESTRATOR_ONLY`, three benchmarks, 150 rollouts each. Baseline scores from existing runs filtered to held-out test IDs. GEPA-optimized scores on the same held-out test IDs.

**10.4 Results**
Main table: baseline vs. GEPA-optimized accuracy on held-out test across three benchmarks.
Qualitative: show seed prompt vs. optimized prompt diff for one benchmark — what changed and why.
Optional ablation: random vs. failure-stratified training selection.

**10.5 Discussion**
Connect back to failure modes: does the optimized planning suffix address the planning-failure category? Does the optimized system prompt reduce max-turns failures? This closes the loop between the failure-mode chapter and the adaptation chapter.

---

## 11. Implementation Order

1. Add `raw_query_analysis` to `ExecutionState` (`state.py`) and extract `_DEFAULT_PLANNING_SUFFIX_*` constants, add `planning_suffix` param, and store `raw_query_analysis` in `AgenticOrchestrator` (`orchestrator.py`) — see §4
2. Write `seed.py`: `build_seed_candidate()` and `build_splits()` (reads existing `raw_results.json`, applies failure-stratified sampling, saves `splits.json`)
3. Write `adapter.py`: `AgentGEPAAdapter` with `evaluate()` and `make_reflective_dataset()`
4. Write `run_gepa.py` with the four modes (splits / optimize / evaluate / diff)
5. Write three GEPA experiment configs
6. Run split generation for all three benchmarks
7. Run GEPA optimization: GAIA → GPQA → MATH (sequentially or in parallel if cluster allows)
8. Evaluate on held-out test, compare to filtered baseline results
9. Write thesis section

---

## 12. Open Questions

- **Reflector thinking mode:** Should Qwen3-32B reflector use `enable_thinking=True`? Likely yes — richer reasoning about what went wrong. Add as a config flag.
- **Minibatch composition within GEPA:** GEPA samples its own minibatches from the train set each iteration. The failure-stratified construction of the train set ensures these minibatches will naturally contain a good failure/success ratio without further intervention.
- **GPQA choices field:** `DataInst` must carry the `choices` list for GPQA so `evaluate_answer()` routes to the MC scorer. Already handled via `example.metadata["choices"]`.
- **Attachment-bearing GAIA questions:** The orchestrator already handles attachment paths correctly via `_inject_attachment_path()`. No special handling needed in the adapter or splits — include attachment-bearing questions normally.
- **Math benchmark identity:** The failure-mode analysis inventory only includes `aime` (20 questions) for math tasks. The MATH500-200-subset run is not in the inventory. Confirm which math benchmark and result path to use for split generation. If MATH500 was run separately (not part of the milestone 1 inventory), add its path to the `existing_results` field in `math.yaml`. If only AIME is available, note that 20 questions is too small for a meaningful GEPA split — in that case MATH500 should be prioritised.

---

## Addendum 2026-05-18: Enriched feedback function (μ_f)

The §5 design shipped a minimal feedback string (`CORRECT` or
`WRONG — ground truth: X. Predicted: Y.`). Inspection of the GEPA paper and
the first set of reflective traces from `experiments/results/gepa/gaia/`
showed this was too sparse: the reflector had to re-derive every failure
mode from raw `<think>` content, which is slow and noisy and biases prompt
edits toward whichever failure mode happens to be most visible in any one
trace.

The paper's μ_f is the *environment-derived* signal produced while scoring
— compiler errors, failed rubrics, retrieval-hop diagnostics. For QA
benchmarks the environment is the answer scorer plus the tool stack, so the
analogous deterministic signal is already on the `ExecutionState`. The
addendum below specifies how it's surfaced.

### Where it lives

A new method `AgentGEPAAdapter._diagnose(state, score) -> str` builds the
`Feedback` string. Both `_system_prompt_records` and `_planning_suffix_records`
call it — the per-component differentiation lives in `Generated Outputs`,
which keeps the feedback itself component-agnostic.

### Required state mutations (in `evaluate()`)

```python
state.metadata["eval_result"] = result          # full dict from evaluate_answer
state.metadata["choices"]     = example.metadata.get("choices")
```

Adding to existing `state.metadata["ground_truth"]` so the reflective pass
can read em/f1 (not just accuracy) and route format heuristics around
multiple-choice questions.

### Feedback schema

**Correct (one line):**
```
CORRECT. Predicted '<pred>'; tools={tool_counts}; turns=<turn>.
```

**Wrong (multi-line, one signal per line):**
```
WRONG. Ground truth: '<gt>'. Predicted: '<pred>' | (empty).
  Scoring: em=<x>, f1=<y>.
  [Normalised: '<n_gt>' vs '<n_pred>'.]              # if non-empty & differ
  [No final answer was produced.]                    # if pred is empty
  [Format mismatch: gold is numeric/symbolic; …]     # numeric gt, prose pred (non-MC)
  [Prediction is much longer than the gold answer …] # pred >4× gt word count (non-MC)
  [Token overlap is high (f1 ≥ 0.5) …]               # partial credit hint (non-MC)
  Tools used: {…} | No tools called — … parametric memory only.
  [N/M tool calls returned an error.]                # action_history result startswith
                                                     #   error/exception/traceback/failed
  [Max turns (K) reached without a final answer.]
```

Lines wrapped in `[ ]` are conditional — they fire only when their
underlying signal is present.

### What each line is meant to teach the reflector

- **Scoring em+f1** distinguishes "right content, wrong format" from "wrong
  content" — a high-f1 wrong answer should prompt format-tightening, not
  reasoning changes.
- **Normalised forms** hide whitespace/punct/article differences so the
  reflector doesn't waste edits on issues that the scorer already
  forgives.
- **Format mismatch / verbosity** are dual checks for the same failure
  ("prose where a short token was expected") — verbosity catches cases
  where `is_math_answer(gt)` is False (text gold answers).
- **Tool errors** are detected via a prefix match on the action result
  string. The orchestrator already inlines `ToolResult.error` into the
  human-readable result, so no `success: bool` field has to be added to
  `action_history`. If false positives become an issue, the cleaner fix is
  to add an explicit `error: bool` to the entry written in
  `orchestrator._record_tool_call`.
- **Parametric memory** flags the case where the orchestrator never called
  any tool — by far the most common pattern in confidently-wrong
  hallucinations and a high-leverage target for prompt fixes.

### Why not an LLM judge

The natural alternative is to insert a Qwen3-32B pass between
`evaluate_answer` and `make_reflective_dataset` that writes a free-form
diagnosis. We deliberately did not:

1. GEPA's reflection LM already performs the paper's "implicit credit
   assignment" over the trajectory. A same-family judge duplicates that
   work and uses reflector context budget for no new information.
2. A same-family judge confabulates plausible-but-wrong failure stories
   for questions the orchestrator itself couldn't solve, especially when
   the *gold* answer is unfamiliar. Wrong diagnoses → wrong prompt edits.

If an open-ended benchmark without a deterministic scorer is added later
(e.g. HLE long-form), the right pattern is to *append* an LLM judge's
paragraph to the deterministic lines inside `_diagnose`, not to replace
them — keeping the structured signal as a sanity floor underneath the
judge.

### Token-budget impact

The wrong-case feedback grows from ~1 line to typically 5–8 lines (~150
extra tokens). With `_MAX_RECORDS = 8`, the per-reflective-call overhead
is at most ~1.2 K tokens — well inside the existing 32 K reflector budget.
No change to `_THINKING_SNIPPET_LEN` or `_RESULT_SNIPPET_LEN` is needed.

### Tests

`tests/gepa_integration/test_adapter.py` gains 14 new tests under the
`_diagnose` section covering each conditional line independently plus an
end-to-end check that the tool-error signal reaches both the
`system_prompt` and `planning_suffix` reflective records. Total
`tests/gepa_integration/` count is now 58.

---

## Addendum 2026-05-18 (Iteration 2): record shape + sample selection

Review of the Iteration 1 feedback design (see previous addendum) surfaced
three remaining gaps. None of them needed orchestrator changes — the data is
already on `ExecutionState`; only the adapter's record-building and sampling
logic change. Because Iteration 1 already requires a GEPA re-run for its
feedback to take effect, these refinements are batched into the same re-run
at no additional compute cost.

### Motivation by failure mode

| Gap | Failure modes affected | Why Iteration 1 missed it |
|---|---|---|
| `system_prompt` records only show *first-turn* thinking | `single_shot_tool_trust`, `retrieval_evidence_failure` — the stopping decision lives in the *last* turn, not the first | Iteration 1 mirrored where the planning thinking lives (turn 0); but most failure thinking is later |
| `planning_suffix` records bury the planning `<think>` block inside the raw blob | `direct_reasoning_no_action`, `computational_subgoal_error` — the reflector has to parse XML to see the plan's reasoning, asymmetric with system_prompt records | Iteration 1 specified `raw_planning_output` rather than extracting the two halves |
| `_balanced_sample` takes the first half of each bucket | Any failure mode that systematically appears later in a minibatch — the reflector never sees those examples | Deterministic head-of-list slicing was the simplest correct implementation; bias was acceptable for Iteration 1 |

### Unified thinking-snippet cap

The Iteration 1 spec used `_THINKING_SNIPPET_LEN = 1500` for the single
thinking field. With three thinking snippets now in play
(first-turn + last-turn for system_prompt, plan thinking for
planning_suffix), three different caps would be hard to reason about. The
constant is **unified at 800 characters** for every snippet. Rationale:

- 800 chars ≈ 200 tokens — enough to capture a coherent reasoning chain
  segment without dominating the reflective prompt
- At `_MAX_RECORDS = 8`, the worst case is system_prompt records with two
  snippets per record = 16 × 800 = ~12.8 K chars per reflective call,
  matching the Iteration 1 budget ceiling for thinking content
- The cap is applied identically by both record builders, removing a
  per-call dimension of variability the reflector would otherwise see

The class constant is renamed in spirit but kept as
`_THINKING_SNIPPET_LEN = 800` for the diff to be one number.

### Change 1 — last-turn thinking in `system_prompt` records

Add one field next to the existing `thinking_before_first_tool`:

```python
"Generated Outputs": {
    "predicted_answer": state.answer or "",
    "thinking_before_first_tool": <first-turn snippet, 800-char cap>,
    "thinking_at_last_turn":      <last-turn snippet, 800-char cap>,  # NEW
    "action_steps": [...],
}
```

- **Source:** the last *assistant* message in `state.output_messages`. The
  list interleaves assistant and tool roles, so filter first:
  ```python
  assistant_msgs = [m for m in state.output_messages if m["role"] == "assistant"]
  ```
  Then `_extract_thinking(assistant_msgs[-1]["content"])`.
- **Conditional inclusion:** if `len(assistant_msgs) <= 1`, omit the field
  entirely. The first and last snippets would be identical, and a duplicate
  field under a misleading name would mislead the reflector.
- **Same cap:** 800 chars, "…[truncated]" suffix on overflow.

### Change 2 — extract planning thinking into its own field

Replace the single `raw_planning_output` field in `planning_suffix` records
with the two extracted halves:

```python
"Generated Outputs": {
    "plan":             state.query_analysis,                          # stripped of <think>
    "thinking_in_plan": _extract_thinking(state.raw_query_analysis or ""),  # NEW
    "tools_subsequently_used": [...],
    "num_turns_taken":  state.turn,
}
```

- **No orchestrator change:** `state.raw_query_analysis` already contains the
  full planning output including `<think>` (set in
  `orchestrator._run_planning_turn`, see Iteration 1 spec §4).
- **No data loss:** `query_analysis` is `raw_query_analysis` with thinking
  stripped, so the two new fields together contain everything the old raw
  blob did, presented to the reflector in the same shape as the
  system_prompt records.
- **Same cap:** `thinking_in_plan` truncated at 800 chars.

### Change 3 — seeded random shuffle in `_balanced_sample`

Replace head-of-list slicing with a deterministic shuffle:

```python
def _balanced_sample(self, states, scores):
    correct = [(s, sc) for s, sc in zip(states, scores) if sc > 0]
    wrong   = [(s, sc) for s, sc in zip(states, scores) if sc == 0]
    rng = random.Random(self._sample_seed)
    rng.shuffle(correct); rng.shuffle(wrong)
    half = self._MAX_RECORDS // 2
    return correct[:half] + wrong[:half]
```

- **New constructor param** `sample_seed: int = 0`, stored as
  `self._sample_seed`. Fixed seed means re-runs select the same records,
  preserving reproducibility while removing the head-of-list bias.
- **Why not "prefer high-f1 wrongs":** scoring shape is binary (em/f1 only
  inform the *feedback string*, not selection). High-f1 wrongs are mostly
  format-error failures — selecting them preferentially over-edits the
  prompt for format issues and under-edits orchestration ones. Random
  preserves the failure-mode mix already present in the minibatch.
- **Why not unseeded random:** GEPA needs reproducible runs; an
  unseeded RNG makes two otherwise-identical GEPA runs diverge in which
  records the reflector sees.

### Token-budget verification

Iteration 1 ceiling for the reflective prompt was ~30 K tokens (instruction
template ~2 K + system_prompt ~3 K + 8 records × ~3 K each). Iteration 2
deltas per record, in tokens (1 token ≈ 4 chars):

| Record type | Per-record delta | At 8 records |
|---|---|---|
| `system_prompt`: first-turn snippet shrink (1500 → 800 chars) | −175 tok | −1.4 K tok |
| `system_prompt`: new `thinking_at_last_turn` field (≤800 chars, often omitted on single-turn rollouts) | +0–200 tok | +0–1.6 K tok |
| `planning_suffix`: drop unbounded `raw_planning_output`, add capped `plan` (≈ raw − thinking) + capped `thinking_in_plan` (≤800 chars) | ≈ neutral; net-negative when raw blob ran long, slightly positive when it was short | ≈ 0, bounded |

Worst case combined: roughly *neutral* to *slightly negative* — Iteration 2
fits inside the Iteration 1 budget without raising the ceiling. The
unification at 800 chars also makes the per-call budget much more
predictable than Iteration 1's mix of capped (1500) and uncapped
(`raw_planning_output`) thinking content.

### Tests

`tests/gepa_integration/test_adapter.py` gains ~6 tests:

- `_diagnose` is unchanged; existing tests continue to pass.
- New: `thinking_at_last_turn` is included when output_messages contains ≥2 *assistant* messages, with content from the last assistant turn
- New: `thinking_at_last_turn` is omitted when output_messages contains ≤1 assistant message (single-turn rollout)
- New: `plan` field equals stripped `query_analysis` (no `<think>` tags)
- New: `thinking_in_plan` field contains the `<think>` content from
  `raw_query_analysis`
- New: `_balanced_sample` returns the same records on two calls with the same
  `sample_seed` (determinism)
- New: `_balanced_sample` returns a different order than head-of-list for a
  minibatch large enough to make collision unlikely (seed=0, list of 20+
  states with distinguishable identifiers)

Estimated total for `tests/gepa_integration/` after Iteration 2: ~64.

### Out of scope

- **LLM judge augmentation** — same rationale as Iteration 1.
- **Cross-record correlation features** — e.g., "this failure mode appeared
  in 3/4 wrong examples." Useful in theory but requires per-batch
  aggregation that the reflector currently can't act on (it gets per-record
  feedback). Defer until the reflector's meta-prompt is changed to consume
  batch-level signal.
- **Adaptive snippet length** — e.g., 400 chars when the trace is dense
  with `<think>` blocks, 1200 when it's sparse. Adds complexity without
  clear benefit at current scale.
