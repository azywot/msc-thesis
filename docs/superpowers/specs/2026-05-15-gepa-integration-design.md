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
