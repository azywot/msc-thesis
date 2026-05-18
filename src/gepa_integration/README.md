# `src/gepa_integration` — GEPA prompt optimisation module

GEPA (Generative Prompt Adaptation) optimises the two text components of the
orchestrator's prompt using execution traces as feedback. No weight updates are
performed. A Qwen3-32B *reflector* reads full `<think>` blocks, action
histories, and failure labels from agent rollouts, then proposes prompt rewrites
that are evaluated and selected by the GEPA loop.

This module wires the `gepa` package into the CoSMAS inference stack.

---

## Module structure

```
src/gepa_integration/
├── __init__.py
├── seed.py      — build the starting candidate + generate data splits
└── adapter.py   — AgentGEPAAdapter (GEPAAdapter protocol implementation)
```

---

## What is being optimised

Every GEPA run operates on exactly two string components:

| Component | What it is | Where it lives in inference |
|---|---|---|
| `system_prompt` | Full system prompt — preamble, few-shot example, tool instructions. Tool schemas inside `<tools>…</tools>` are never touched. | Passed as `system_prompts=[...]` to `AgenticOrchestrator.run_batch()` |
| `planning_suffix` | Instruction block appended to the user query on Turn 0 (the planning turn). Tells the orchestrator how to analyse the question before using tools. | Passed as `planning_suffix=` to `AgenticOrchestrator.__init__()` |

The seed values are the unmodified prompts used in the milestone-1 AgentFlow
runs, rendered by `PromptBuilder` — so the baseline comparison is exact.

---

## `seed.py`

### `build_seed_candidate`

```python
from gepa_integration.seed import build_seed_candidate
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.base import get_tool_call_format, ModelFamily

tool_schemas = tool_registry.get_all_schemas()
seed = build_seed_candidate(
    benchmark="gaia",                              # or "gpqa"
    tool_schemas=tool_schemas,
    direct_tool_call=False,                        # must match your inference config
    tool_call_format=get_tool_call_format(ModelFamily.QWEN3),
    max_search_limit=10,
)
# seed == {"system_prompt": "...", "planning_suffix": "..."}
```

Calls `PromptBuilder.build_system_prompt()` with the same arguments used in
`run_experiment.py`, so the seed is byte-for-byte identical to the inference
system prompt. The `planning_suffix` is `_DEFAULT_PLANNING_SUFFIX_TOOLS` from
`agent_engine.core.orchestrator` — again the same constant used at inference
time.

**Parameters that must match your experiment config:**

| Parameter | Inference config key | Notes |
|---|---|---|
| `benchmark` | `dataset.name` | Controls which YAML template is loaded |
| `direct_tool_call` | `tools.direct_tool_call` | Must be `False` for sub-agent mode |
| `tool_call_format` | derived from model family | `JSON` for Qwen3, `PYTHONIC` for OLMo |
| `max_search_limit` | `tools.max_search_limit` | Embedded in tool instructions |

### `build_splits`

Partitions an existing `raw_results.json` into three stratified splits that
share the same class distribution (correct + each of the six failure modes),
so GEPA-vs-seed comparisons on any split are apples-to-apples.

```python
from gepa_integration.seed import build_splits
from pathlib import Path

splits = build_splits(
    raw_results_path=Path("experiments/results/.../raw_results.json"),
    train_n=80,
    val_n=45,
    seed=1,
    output_path=Path("experiments/configs/gepa/splits/gaia_splits.json"),
)
# splits == {"train": [qid, ...], "val": [...], "test": [...]}
```

**Split construction:**
- Each record is labelled by class — `CORRECT` for solved questions, or
  one of the six failure modes from
  `scripts/failure_modes/analyze_failure_modes.py` for failed ones.
- Within each class, records are allocated to train/val/test in proportion
  to the requested split sizes (relative to the full dataset), so each
  split preserves the natural class distribution.
- Test is never seen during optimisation.

Pre-generated splits are committed to the repo at
`experiments/configs/gepa/splits/`. Regenerate with:
```bash
sbatch jobs/gepa/000_prep_gepa_data.job
# or directly:
python scripts/run_gepa.py --mode splits --config experiments/configs/gepa/gaia.yaml
```

---

## `adapter.py` — `AgentGEPAAdapter`

Implements the `GEPAAdapter` protocol from `gepa.core.adapter`. GEPA calls
`evaluate()` and `make_reflective_dataset()` during the optimisation loop.

### Construction

```python
from gepa_integration.adapter import AgentGEPAAdapter
from agent_engine.models.vllm_provider import VLLMProvider
from agent_engine.core.tool import ToolRegistry

# Build these exactly as you would for a normal inference run:
model_provider = VLLMProvider(model_cfg)         # Qwen3-8B
tool_registry  = _build_tool_registry(cfg, model_provider=model_provider)

adapter = AgentGEPAAdapter(
    model_provider=model_provider,
    tool_registry=tool_registry,
    use_thinking=True,    # ORCHESTRATOR_ONLY — orchestrator thinks, sub-agents don't
    max_turns=15,
    tool_limits={"web_search": 10},
)
```

The same `VLLMProvider` instance is used for both the orchestrator and the
tool sub-agents. Sub-agents call `model_provider.generate(...)` with
`use_thinking=False` (controlled by the tool's own `use_thinking` field,
set from `thinking_mode` in the config).

### `evaluate(batch, candidate, capture_traces=False)`

Runs `AgenticOrchestrator.run_batch()` on `batch` using the prompt strings
from `candidate`, then scores each prediction with `evaluate_answer()` —
the same function used in the main inference pipeline.

```python
from gepa.core.adapter import EvaluationBatch

batch: list[DatasetExample] = train_examples[:10]
result: EvaluationBatch = adapter.evaluate(
    batch=batch,
    candidate={"system_prompt": "...", "planning_suffix": "..."},
    capture_traces=True,   # needed for make_reflective_dataset
)
# result.outputs  — list of prediction strings
# result.scores   — list of float (1.0 = correct, 0.0 = wrong)
# result.trajectories — list of ExecutionState (only when capture_traces=True)
```

Ground truth is stored in `state.metadata["ground_truth"]` so
`make_reflective_dataset` can access it without the original examples.

### `make_reflective_dataset(candidate, eval_batch, components_to_update)`

Builds per-component feedback records from execution traces. Returns at most
12 records per component (6 correct + 6 wrong) to keep reflector context
manageable.

```python
dataset = adapter.make_reflective_dataset(
    candidate=current_candidate,
    eval_batch=result,                     # must have capture_traces=True
    components_to_update=["system_prompt", "planning_suffix"],
)
# dataset["system_prompt"] — list of {"Inputs", "Generated Outputs", "Feedback"} dicts
# dataset["planning_suffix"] — same structure
```

**`system_prompt` records** include:
- The question
- The orchestrator's predicted answer
- `thinking_before_first_tool` — the `<think>` block from the first assistant turn (capped at 800 chars)
- `thinking_at_last_turn` — the `<think>` block from the last assistant turn (capped at 800 chars); **omitted when there is only one distinct assistant turn** to avoid duplicating the first-turn field
- All action steps (tool name, sub-goal, result snippet)
- The enriched `Feedback` string (see [§ Feedback design](#feedback-design-μf-for-cosmas) below)

**`planning_suffix` records** include:
- The question
- `plan` — the planning turn output with `<think>` blocks stripped (i.e. `state.query_analysis`)
- `thinking_in_plan` — the `<think>` block extracted from the raw planning output (capped at 800 chars)
- The list of tools subsequently used
- Number of turns taken
- The enriched `Feedback` string — same `_diagnose()` output as the system-prompt records

### Feedback design (μ_f for CoSMAS)

The GEPA paper describes a *feedback function* (μ_f) that returns more than a
scalar — it surfaces the natural-language signal the environment produced
while scoring (compiler errors, failed rubrics, retrieval gaps). For
benchmarks like GAIA/GPQA where the "environment" is just an answer scorer,
we approximate μ_f deterministically: we read everything the orchestrator and
`evaluate_answer` already produced, and turn it into a per-failure-mode
diagnosis. **No second LLM judge is involved** — that was deliberately ruled
out (see [§ Why not an LLM judge?](#why-not-an-llm-judge)).

The feedback is assembled by `AgentGEPAAdapter._diagnose(state, score)`.

**Correct cases** (one line, kept short so they don't crowd the reflector):
```
CORRECT. Predicted '42'; tools={'web_search': 2}; turns=4.
```

**Wrong cases** (multi-line; each line is an independent signal the
reflector can credit-assign to a prompt change):

| Line | When it fires | Signal for the reflector |
|---|---|---|
| `WRONG. Ground truth: '…'. Predicted: '…'.` | always | basic disagreement |
| `Scoring: em=…, f1=…` | always | partial-credit shape |
| `Normalised: '…' vs '…'.` | normalised forms differ but are non-empty | hides irrelevant whitespace/punct/article differences so the reflector sees the real mismatch |
| `No final answer was produced.` | prediction is empty | extraction or convergence failure |
| `Format mismatch: gold is numeric/symbolic; prediction is prose.` | `is_math_answer(gt)` and not `is_math_answer(pred)` (skipped for MC) | needs a stricter answer-format instruction |
| `Prediction is much longer than the gold answer …` | pred word count > 4 × gt word count (skipped for MC) | gold answers are short — the model is over-explaining |
| `Token overlap is high (f1 ≥ 0.5) …` | f1 ≥ 0.5 (skipped for MC) | right *content*, wrong *precision/format* — typically a unit or rounding fix |
| `Tools used: {…}` / `No tools called — … parametric memory only.` | always | flags hallucination-from-memory and over/under-tooling |
| `N/M tool calls returned an error.` | ≥ 1 action-history result starts with `error/exception/traceback/failed` | infrastructure or query-phrasing problem |
| `Max turns (K) reached without a final answer.` | `state.metadata["max_turns_reached"]` is True | planning didn't converge — usually a `planning_suffix` target |

Multiple-choice questions (GPQA, with `example.metadata["choices"]` set)
skip the format/verbosity/f1 lines: gold is one letter and the heuristics
would misfire.

Both record types use the same `_diagnose` output, by design. The
`system_prompt` and `planning_suffix` records differ in their
`Generated Outputs` payload (one shows first + last assistant `<think>`
blocks + action steps, the other shows the planning `<think>` block +
the stripped plan + the downstream tool list) — the reflector already
has the per-component context it needs, so the *feedback* itself can be
component-agnostic.

**Iteration 2 (2026-05-18)** unified the thinking-snippet cap at 800 chars
across every field, added `thinking_at_last_turn` to `system_prompt`
records so the stopping decision is visible to the reflector, and split
the planning record's raw blob into `plan` + `thinking_in_plan` for
parity with the system-prompt record shape. `_balanced_sample` now
shuffles each bucket with `random.Random(self._sample_seed)` (default
seed `0`) before slicing, so the reflector's records are not biased by
the minibatch's arrival order while remaining reproducible across runs.
See `docs/superpowers/specs/2026-05-15-gepa-integration-design.md`
Iteration 2 addendum for the full rationale.

#### Why not an LLM judge?

The natural next step would be an extra Qwen3-32B pass that writes a free-form
"why was this wrong" paragraph and prepends it to the feedback. We chose not
to, for two reasons:

1. **Redundancy with the reflector.** GEPA's reflection LM already performs
   what the paper calls "implicit credit assignment" over the full trajectory.
   A same-family judge would duplicate that work and consume reflector
   context budget for no new signal.
2. **Confabulation risk.** A judge LM trained on similar data to the
   orchestrator confidently invents failure stories for hard questions
   (especially questions the orchestrator itself couldn't solve). Wrong
   diagnoses become wrong prompt edits.

If a future setting really needs LLM-judged feedback (e.g. open-ended
benchmarks without a deterministic scorer), the right place to plug it in is
a new branch inside `_diagnose` — keep the deterministic lines and *append*
the judge's paragraph after them, so the reflector still has the structured
signal as a sanity floor.

---

## Alignment with the inference pipeline

The GEPA pipeline is designed so that every component matches inference exactly:

| Aspect | GEPA | Inference (`run_experiment.py`) |
|---|---|---|
| System prompt construction | `build_seed_candidate` → `PromptBuilder.build_system_prompt(...)` | `PromptBuilder.build_system_prompt(...)` with same args |
| Planning suffix seed | `_DEFAULT_PLANNING_SUFFIX_TOOLS` constant | Same constant (when no custom suffix set) |
| Tool mode | `direct_tool_call: false` (sub-agent) | Matches milestone-1 `qwen8B_subagent_tools_orchestrator` |
| Sub-agent model | Same `VLLMProvider` as orchestrator (Qwen3-8B) | Sub-agent model role in config |
| Thinking mode | `ORCHESTRATOR_ONLY` — orchestrator on, sub-agents off | Same |
| Scoring | `evaluate_answer(prediction, ground_truth, choices=choices)` | Same function |
| GPQA choices | `example.metadata.get("choices")` | Same |

---

## Running the full pipeline

### Step 0 — prerequisites

The GAIA and GPQA milestone-1 AgentFlow results must exist:
```
experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/
  gaia/qwen8B_subagent_tools_orchestrator/.../raw_results.json
  gpqa/qwen8B_subagent_tools_orchestrator/.../raw_results.json
```
These paths are hardcoded in `experiments/configs/gepa/gaia.yaml` and
`gpqa.yaml`. Update them if you re-run the milestone experiments.

### Step 1 — install the `gepa` package

```bash
sbatch jobs/gepa/001_install_gepa_deps.job
# or: pip install gepa==0.0.22
```

### Step 2 — generate splits

```bash
sbatch jobs/gepa/000_prep_gepa_data.job
```

Writes (or overwrites) `experiments/configs/gepa/splits/{gaia,gpqa}_splits.json`.
Pre-generated splits are already committed — only re-run if you change the
source `raw_results.json` or split sizes.

### Step 3 — smoke tests

```bash
sbatch jobs/gepa/002_smoke_gepa.job        # CPU: imports, splits, evaluator
sbatch jobs/gepa/003_smoke_gepa_gpu.job    # GPU: 1 step, 2 examples, 3×H100
```

The GPU smoke test runs the full pipeline (optimize → evaluate → diff) on a
2-example subset with the real Qwen3-32B reflector. Only proceed to the full
run once this passes.

### Step 4 — full optimisation

```bash
sbatch jobs/gepa/004_run_gepa.job
```

Runs sequentially: GAIA optimise (3h) → GAIA evaluate → GPQA optimise (3h) →
GPQA evaluate. Supports env-var overrides:

```bash
sbatch --export=ALL,REGEN_SPLITS=1 jobs/gepa/004_run_gepa.job   # regenerate splits first
sbatch --export=ALL,SKIP_GAIA=1    jobs/gepa/004_run_gepa.job   # GPQA only
sbatch --export=ALL,SKIP_GPQA=1    jobs/gepa/004_run_gepa.job   # GAIA only
```

### Step 5 — analyse results

Each job run writes to `experiments/results/gepa/<benchmark>/<TIMESTAMP>_<JOB_ID>/`;
replace `<run>` below with the specific subdirectory you want to analyse.

```bash
# Accuracy + tool stats on the held-out test set
python scripts/analyze_results.py experiments/results/gepa/gaia/<run>/gepa_results.json --by-level --tools
python scripts/analyze_results.py experiments/results/gepa/gpqa/<run>/gepa_results.json --tools

# Diff between seed and optimised prompts (pass the same --run-dir used by the job)
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/gaia.yaml --run-dir experiments/results/gepa/gaia/<run>
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/gpqa.yaml --run-dir experiments/results/gepa/gpqa/<run>
```

---

## Using the optimised prompts in inference

After optimisation, `best_candidate.json` contains the two improved strings.
To run a standard inference experiment with them, pass them directly to the
orchestrator:

```python
import json
from agent_engine.core.orchestrator import AgenticOrchestrator

best = json.load(open("experiments/results/gepa/gaia/<TIMESTAMP>_<JOB_ID>/best_candidate.json"))

orchestrator = AgenticOrchestrator(
    model_provider=model_provider,
    tool_registry=tool_registry,
    planning_suffix=best["planning_suffix"],   # replaces the default constant
    ...
)

states = orchestrator.run_batch(
    questions=questions,
    question_ids=question_ids,
    system_prompts=[best["system_prompt"]] * len(questions),  # replaces default
    attachments=attachments,
)
```

Or, to run a full experiment via `run_experiment.py`, create a new YAML config
that points to the dataset + model you want and manually set the system prompt
template to the optimised text (edit the relevant YAML template under
`src/agent_engine/prompts/templates/system/`) — or patch `PromptBuilder` to
load from the `best_candidate.json` directly.

---

## Config reference (`experiments/configs/gepa/*.yaml`)

```yaml
benchmark: "gaia"           # dataset name — must match DatasetRegistry key
dataset_split: "all_validation"
data_dir: "./data"
thinking_mode: "ORCHESTRATOR_ONLY"
seed: 1
max_turns: 15

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  role: "orchestrator"

reflector:                  # Qwen3-32B served via vllm serve on port 8001
  path_or_id: "Qwen/Qwen3-32B"
  host: "localhost"
  port: 8001

tools:
  enabled_tools: [web_search, code_generator, text_inspector]
  direct_tool_call: false   # sub-agent mode — must match the baseline run
  web_tool_provider: "serper"
  max_search_limit: 10

splits_file: "experiments/configs/gepa/splits/gaia_splits.json"
existing_results: "experiments/results/.../raw_results.json"

splits:
  train_n: 80
  val_n: 45                 # remaining ~40 become held-out test

gepa:
  rollout_budget: 150       # total agent rollouts during optimisation
  minibatch_size: 10        # examples per reflector call
  merge_proposer: true      # whether to use GEPA's merge proposer
  run_dir: "experiments/results/gepa/gaia"   # base dir; jobs append <TIMESTAMP>_<JOB_ID>
```

---

## Outputs

Each invocation of `004_run_gepa.job` (or `003_smoke_gepa_gpu.job`) writes to a
timestamped + job-id subdirectory so successive runs never overwrite each
other:

```
experiments/results/gepa/<benchmark>/<YYYY-MM-DD-HH-MM-SS>_<SLURM_JOB_ID>/
```

The job script computes the path once and passes it via `--run-dir` to every
mode (`optimize`, `evaluate`, `diff`). When invoking `run_gepa.py` manually
without `--run-dir`, the base path from `gepa.run_dir` in the YAML is used as-is.

Files in each run directory:

| File | Contents |
|---|---|
| `seed_candidate.json` | `{"system_prompt": ..., "planning_suffix": ...}` — the starting point (written by `run_gepa.py`) |
| `best_candidate.json` | Same schema — the best candidate found by GEPA (written by `run_gepa.py`) |
| `gepa_results.json` | Held-out test evaluation; same schema as `raw_results.json` — readable by `analyze_results.py` |
| `gepa_state.bin` | Full GEPA optimisation state (pickled `GEPAState`); written by the GEPA library after each step. Can be used to resume an interrupted run — GEPA automatically reads it when `run_dir` already contains this file. |
| `generated_best_outputs_valset/` | Per-task best rollout outputs on the validation set (written by GEPA when `track_best_outputs=True`). |

---

## Tests

```bash
pytest tests/gepa_integration/ -v
```

76 unit tests covering: `ExecutionState.raw_query_analysis`,
`_DEFAULT_PLANNING_SUFFIX_TOOLS` constant, `build_seed_candidate` (structure,
planning suffix match, tool schema embedding), `build_splits` (sizes,
no-overlap, failure ratio, JSON output), `_extract_thinking`, all
`AgentGEPAAdapter` methods (`evaluate`, `make_reflective_dataset`, balanced
sampling cap), the `_diagnose` feedback function (score breakdown,
normalised-form line, empty-prediction, format-mismatch, verbosity, high-f1,
parametric-memory, tool-error counting, max-turns, multiple-choice skip),
and Iteration 2 additions (unified 800-char thinking cap, last-turn
thinking inclusion/skip, planning record `plan`/`thinking_in_plan` split,
seeded `_balanced_sample` shuffle determinism).
