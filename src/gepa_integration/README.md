# `src/gepa_integration` — GEPA prompt optimisation module

GEPA (Generative Prompt Adaptation) optimises the two text components of the
orchestrator's prompt using execution traces as feedback. No weight updates are
performed. A Qwen3-32B *reflector* reads action histories and failure labels
from agent rollouts, then proposes prompt rewrites that are evaluated and
selected by the GEPA loop. Thinking mode is config-driven (`thinking_mode`
field) — when enabled, the reflector also sees `<think>` traces.

This module wires the `gepa` package into the CoSMAS inference stack.

---

## Module structure

```
src/gepa_integration/
├── __init__.py
├── seed.py        — build_seed_candidate(); build_splits() (legacy path)
├── adapter.py     — AgentGEPAAdapter (GEPAAdapter protocol implementation)
├── reflection.py  — trim_prompt() — reflector context budget management
└── data/
    ├── prepare.py — Download Search-R1 + DeepMath → GEPA DatasetExamples
    └── loader.py  — load_gepa_examples() — JSON → DatasetExample
```

---

## What is being optimised

Every GEPA run operates on exactly two string components:

| Component | What it is | Where it lives in inference |
|---|---|---|
| `system_prompt` | Full system prompt — preamble, few-shot example, tool instructions. Tool schemas inside `<tools>…</tools>` are never touched. | Passed as `system_prompts=[...]` to `AgenticOrchestrator.run_batch()` |
| `planning_suffix` | Instruction block appended to the user query on Turn 0 (the planning turn). Tells the orchestrator how to analyse the question before using tools. | Passed as `planning_suffix=` to `AgenticOrchestrator.__init__()` |

The seed values are rendered by `PromptBuilder` — byte-for-byte identical to
the prompts used in the milestone-1 AgentFlow runs, so the baseline comparison
is exact.

---

## `data/` — training data preparation

Training data comes from open datasets that do not overlap with any benchmark
held-out test set:

| Preset | Composition | Purpose |
|---|---|---|
| `gaia` | 75 % Search-R1 (85/15 HotpotQA/NQ) + 25 % DeepMath (no difficulty filter) | Targets GAIA/HLE/MuSiQue failure modes: retrieval chaining, single-shot tool trust, evidence gaps |
| `math` | 75 % DeepMath (difficulty ≥ 5) + 25 % Search-R1 | Targets AIME failure mode: direct reasoning without code delegation |

Both presets produce 300 examples: **150 D_feedback (train) / 50 D_pareto (val) / 100 test**.

### `prepare.py`

```bash
python src/gepa_integration/data/prepare.py \
    --preset gaia \
    --output-dir data/gepa/gaia \
    --splits-out experiments/configs/gepa/splits/gaia_gepa_splits.json \
    --seed 1

python src/gepa_integration/data/prepare.py \
    --preset math \
    --output-dir data/gepa/math \
    --splits-out experiments/configs/gepa/splits/math_gepa_splits.json \
    --seed 1
```

Writes `data/gepa/<preset>/all_examples.json` (300 examples) and the splits
JSON. Each example carries:
- `question`, `answer` — the canonical question and primary answer
- `answer_aliases` — all valid answer strings from the source dataset (e.g.
  `["New York City", "New York", "NYC"]`); used by `evaluate_answer` so
  Search-R1 multi-answer examples score correctly
- `data_source` — `"hotpotqa"`, `"nq"`, or `"deepmath"`; used internally by
  the adapter to gate feedback hints (see [§ Feedback design](#feedback-design-μf-for-cosmas))

### `loader.py`

```python
from gepa_integration.data.loader import load_gepa_examples

examples = load_gepa_examples(
    data_file=Path("data/gepa/gaia/all_examples.json"),
    question_ids=splits["train"],   # list[int] from splits JSON
)
# → list[DatasetExample] with metadata["answer_aliases"] and metadata["data_source"]
```

All 300 examples are loaded into memory at once (negligible footprint — plain
strings). GEPA then samples minibatches of `minibatch_size` (3) from the
150-item train list during optimisation.

---

## `seed.py`

### `build_seed_candidate`

```python
from gepa_integration.seed import build_seed_candidate
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.base import get_tool_call_format, ModelFamily

tool_schemas = tool_registry.get_all_schemas()
seed = build_seed_candidate(
    benchmark="gaia",                              # or "math"
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

### `build_splits` (legacy)

Partitions an existing `raw_results.json` into stratified splits. This is the
*legacy* path used when `gepa_data_file` is not set in the config (i.e. when
the benchmark's own results are used as training data). For the current GAIA
and MATH runs, `prepare.py` generates the data and splits instead.

---

## `adapter.py` — `AgentGEPAAdapter`

Implements the `GEPAAdapter` protocol from `gepa.core.adapter`. GEPA calls
`evaluate()` and `make_reflective_dataset()` during the optimisation loop.

### Construction

```python
from gepa_integration.adapter import AgentGEPAAdapter
from agent_engine.models.vllm_provider import VLLMProvider
from agent_engine.models.api_provider import OpenAIProvider
from agent_engine.core.tool import ToolRegistry

# Orchestrator: in-process vLLM on GPU 0
model_provider = VLLMProvider(model_cfg)         # Qwen3-8B

# Sub-agent: separate vLLM serve on port 9998 (optional — omit for same-model sub-agents)
sub_agent_provider = OpenAIProvider(sa_model_cfg, api_key="EMPTY",
                                    base_url="http://localhost:9998/v1")

tool_registry = _build_tool_registry(cfg, model_provider=model_provider,
                                      sub_agent_provider=sub_agent_provider)

# use_thinking is config-driven: True for ORCHESTRATOR_ONLY/ALL, False for NO
adapter = AgentGEPAAdapter(
    model_provider=model_provider,
    tool_registry=tool_registry,
    use_thinking=False,   # thinking_mode: "NO" — matches LoRA fine-tuning setup
    max_turns=15,
    tool_limits={"web_search": 10},
)
```

When `sub_agent` is configured in the YAML, `run_gepa.py` creates a separate
`OpenAIProvider` pointing at the sub-agent vLLM serve endpoint. When omitted,
the orchestrator model is reused for sub-agents (previous default behavior).
Sub-agent thinking is controlled by `thinking_mode` — only `SUBAGENTS_ONLY`
and `ALL` enable it.

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

For each example, the following are stashed in `state.metadata` so
`make_reflective_dataset` can access them without the original examples:

| Key | Value |
|---|---|
| `ground_truth` | canonical answer string |
| `eval_result` | `{"correct", "accuracy", "em", "f1"}` from `evaluate_answer` |
| `choices` | MC answer choices, or `None` |
| `data_source` | `"hotpotqa"` / `"nq"` / `"deepmath"` / `""` for real benchmark examples |

`evaluate_answer` is called with both `choices` and `answer_aliases` from
`example.metadata`, so Search-R1 multi-answer examples score correctly without
false negatives feeding the reflector.

### `make_reflective_dataset(candidate, eval_batch, components_to_update)`

Builds per-component feedback records from execution traces. Returns at most
**8 records per component** (4 correct + 4 wrong), balanced by
`_balanced_sample` which shuffles each bucket with a fixed seed before
slicing (deterministic, not biased by minibatch arrival order).

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

### Thinking tags

When `use_thinking=True` (e.g. `thinking_mode: "ORCHESTRATOR_ONLY"`), the
model emits one `<think>…</think>` block per turn. When `use_thinking=False`
(e.g. `thinking_mode: "NO"`), thinking fields in the reflective records are
empty strings. The pipeline is designed so the reflector always sees clean
content regardless of thinking mode:

| Field | Raw source | What the reflector sees |
|---|---|---|
| `predicted_answer` | `extract_answer(gen_result.text)` — strips tags | clean answer string |
| `thinking_before_first_tool` / `thinking_at_last_turn` | `_extract_thinking(output_messages[n]["content"])` | content *inside* the think block only |
| `thinking_in_plan` | `_extract_thinking(state.raw_query_analysis)` | content inside the planning think block |
| `plan` | `state.query_analysis` — stripped by `strip_thinking_tags` in orchestrator | clean plan text |
| `action_steps[*].result_snippet` | `action_history[*]["result"]` — stripped by `strip_thinking_tags` before storage | clean tool output |

---

## Feedback design (μ_f for CoSMAS)

The GEPA paper describes a *feedback function* (μ_f) that returns more than a
scalar — it surfaces the natural-language signal the environment produced
while scoring (compiler errors, failed rubrics, retrieval gaps). For
benchmarks like GAIA/MATH where the "environment" is just an answer scorer,
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
| `Normalised: '…' vs '…'.` | normalised forms differ and are non-empty | hides irrelevant whitespace/punct/article differences |
| `No final answer was produced.` | prediction is empty | extraction or convergence failure |
| `Format mismatch: gold is numeric/symbolic; prediction is prose.` | `is_math_answer(gt)` and not `is_math_answer(pred)` — **suppressed when `data_source` is `"hotpotqa"` or `"nq"`** (skipped for MC) | needs stricter answer-format instruction; suppressed for Search-R1 examples whose year-like answers would misfire |
| `Prediction is much longer than the gold answer …` | pred word count > 4 × gt word count (skipped for MC) | gold answers are short — the model is over-explaining |
| `Token overlap is high (f1 ≥ 0.5) …` | f1 ≥ 0.5 (skipped for MC) | right *content*, wrong *precision/format* |
| `Tools used: {…}` / `No tools called — … parametric memory only.` | always | flags hallucination-from-memory and over/under-tooling |
| `N/M tool calls returned an error.` | ≥ 1 action-history result starts with `error/exception/traceback/failed` | infrastructure or query-phrasing problem |
| `Max turns (K) reached without a final answer.` | `state.metadata["max_turns_reached"]` is True | planning didn't converge — usually a `planning_suffix` target |

Multiple-choice questions (with `example.metadata["choices"]` set)
skip the format/verbosity/f1 lines: gold is one letter and the heuristics
would misfire.

Both record types use the same `_diagnose` output, by design. The
`system_prompt` and `planning_suffix` records differ in their
`Generated Outputs` payload — the reflector already has the per-component
context it needs, so the *feedback* itself can be component-agnostic.

**Note on mixed training data:** With 25 % of training examples from the
minority source (e.g. DeepMath examples in the GAIA optimizer), the format
mismatch hint could mislead the reflector about what answer format the target
benchmark requires. This is mitigated by suppressing the hint for
`_FACTUAL_QA_SOURCES = {"hotpotqa", "nq"}`. The `data_source` field is stored
in `state.metadata` but is *not* passed to the reflector — the reflector sees
only the question, outputs, and feedback string.

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

---

## Alignment with the inference pipeline

| Aspect | GEPA | Inference (`run_experiment.py`) |
|---|---|---|
| System prompt construction | `build_seed_candidate` → `PromptBuilder.build_system_prompt(...)` | `PromptBuilder.build_system_prompt(...)` with same args |
| Planning suffix seed | `_DEFAULT_PLANNING_SUFFIX_TOOLS` constant | Same constant (when no custom suffix set) |
| Tool mode | `direct_tool_call: false` (sub-agent) | Matches milestone-1 `qwen8B_subagent_tools_orchestrator` |
| Sub-agent model | Qwen3-1.7B via `OpenAIProvider` (port 9998) — configurable via `sub_agent` YAML section; falls back to orchestrator model when omitted | Sub-agent model role in config |
| Thinking mode | Config-driven (`thinking_mode` field); currently `NO` for fine-tuning comparability | Same |
| Scoring | `evaluate_answer(prediction, ground_truth, choices=choices, answer_aliases=aliases)` | Same function |
| Answer aliases | `example.metadata.get("answer_aliases")` | Same (where available) |

---

## Running the full pipeline

### Step 0 — prerequisites

Install the `gepa` package and prepare the training data:

```bash
# Install gepa into the conda env
sbatch jobs/gepa/001_install_gepa_deps.job
# or: pip install gepa==0.0.22

# Download Search-R1 + DeepMath and write all_examples.json + splits JSON
sbatch jobs/gepa/000_prep_gepa_data.job
# or directly:
python src/gepa_integration/data/prepare.py \
    --preset gaia --output-dir data/gepa/gaia \
    --splits-out experiments/configs/gepa/splits/gaia_gepa_splits.json --seed 1
python src/gepa_integration/data/prepare.py \
    --preset math --output-dir data/gepa/math \
    --splits-out experiments/configs/gepa/splits/math_gepa_splits.json --seed 1
```

Pre-generated splits are committed to the repo at
`experiments/configs/gepa/splits/`. Only re-run `prepare.py` if you change
the data composition or split sizes.

### Step 1 — smoke tests

```bash
sbatch jobs/gepa/002_smoke_gepa.job        # CPU: imports, splits, evaluator
sbatch jobs/gepa/003_smoke_gepa_gpu.job    # GPU: 1 step, 2 examples, 3×H100
```

The GPU smoke test runs the full pipeline (optimize → evaluate → diff) on a
2-example subset with the real Qwen3-32B reflector. Only proceed to the full
run once this passes.

### Step 2 — full optimisation

GAIA and MATH are independent jobs — submit them separately or together:

```bash
sbatch jobs/gepa/006_run_gepa_gaia.job   # ~24h, 3×H100
sbatch jobs/gepa/007_run_gepa_math.job   # ~24h, 3×H100
```

Or step-by-step manually (Qwen3-32B reflector must be running on port 8001):

```bash
python scripts/run_gepa.py --mode optimize --config experiments/configs/gepa/gaia.yaml
python scripts/run_gepa.py --mode evaluate --config experiments/configs/gepa/gaia.yaml
python scripts/run_gepa.py --mode diff     --config experiments/configs/gepa/gaia.yaml
```

### Step 3 — analyse results

Each job writes to `experiments/results/gepa/<benchmark>/<TIMESTAMP>_<JOB_ID>/`;
replace `<run>` below with the specific subdirectory.

```bash
# Accuracy + tool stats on the held-out test set
python scripts/analyze_results.py experiments/results/gepa/gaia/<run>/gepa_results.json --by-level --tools
python scripts/analyze_results.py experiments/results/gepa/math/<run>/gepa_results.json --tools

# Diff between seed and optimised prompts
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/gaia.yaml \
    --run-dir experiments/results/gepa/gaia/<run>
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/math.yaml \
    --run-dir experiments/results/gepa/math/<run>
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

---

## Config reference (`experiments/configs/gepa/*.yaml`)

```yaml
benchmark: "gaia"           # dataset name — must match DatasetRegistry key
thinking_mode: "NO"         # NO / ORCHESTRATOR_ONLY / SUBAGENTS_ONLY / ALL
seed: 1
max_turns: 15
cache_dir: "./cache"

# GEPA data — generated by src/gepa_integration/data/prepare.py
gepa_data_file: "data/gepa/gaia/all_examples.json"
splits_file: "experiments/configs/gepa/splits/gaia_gepa_splits.json"

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  family: "qwen3"
  role: "orchestrator"
  gpu_memory_utilization: 0.80  # leave room for sub-agent on same GPU

# Optional: separate sub-agent model (omit to reuse orchestrator model)
sub_agent:
  name: "Qwen3-1.7B"
  path_or_id: "Qwen/Qwen3-1.7B"
  family: "qwen3"
  host: "localhost"
  port: 9998

reflector:                  # Qwen3-32B served via vllm serve on port 8001
  path_or_id: "Qwen/Qwen3-32B"
  host: "localhost"
  port: 8001

tools:
  enabled_tools: [web_search, code_generator, text_inspector]
  direct_tool_call: false   # sub-agent mode — must match the baseline run
  web_tool_provider: "serper"
  max_search_limit: 10

gepa:
  rollout_budget: 750       # total agent rollouts (≥ 15 × |D_pareto| = 15 × 50)
  minibatch_size: 3         # examples per reflector call (GEPA paper default)
  merge_proposer: true
  track_best_outputs: true
  run_dir: "experiments/results/gepa/gaia"   # base dir; jobs append <TIMESTAMP>_<JOB_ID>

wandb:
  enabled: true
  project: "gepa"
  name: "gepa_gaia_qwen3_8b_1.7b_no_think"
  tags: ["gaia", "gepa", "qwen3-8b", "qwen3-1.7b", "no-think", "search-r1", "deepmath"]
```

---

## Outputs

Each job run writes to a timestamped + job-id subdirectory:

```
experiments/results/gepa/<benchmark>/<YYYY-MM-DD-HH-MM-SS>_<SLURM_JOB_ID>/
```

| File | Contents |
|---|---|
| `seed_candidate.json` | `{"system_prompt": ..., "planning_suffix": ...}` — the starting point |
| `best_candidate.json` | Same schema — the best candidate found by GEPA |
| `gepa_results.json` | Held-out test evaluation; same schema as `raw_results.json` — readable by `analyze_results.py` |
| `gepa_state.bin` | Full GEPA optimisation state (pickled); written after each step. Can resume an interrupted run — GEPA reads it automatically when `run_dir` already contains this file. |
| `generated_best_outputs_valset/` | Per-task best rollout outputs on the validation set (written when `track_best_outputs=True`). |
| `optimize.stderr` / `evaluate.stderr` | Per-step stderr logs (vLLM tqdm/INFO); replayed to stderr on failure. |

---

## Tests

```bash
pytest tests/gepa_integration/ -v
```

Covers: `ExecutionState.raw_query_analysis`, `_DEFAULT_PLANNING_SUFFIX_TOOLS`
constant, `build_seed_candidate` (structure, planning suffix match, tool schema
embedding), `build_splits` (sizes, no-overlap, class distribution, JSON output),
GEPA data preparation (`_norm_to_example`, `make_gepa_splits`, `build_gepa_examples`,
full pipeline round-trip, `load_gepa_examples`), `_extract_thinking`, all
`AgentGEPAAdapter` methods (`evaluate` with alias + data_source stashing,
`make_reflective_dataset`, balanced sampling cap), the `_diagnose` feedback
function (score breakdown, normalised-form line, empty-prediction,
format-mismatch with `_FACTUAL_QA_SOURCES` gate, verbosity, high-f1,
parametric-memory, tool-error counting, max-turns, multiple-choice skip),
and reflective record structure (unified 800-char thinking cap, last-turn
thinking inclusion/skip, planning record `plan`/`thinking_in_plan` split,
seeded `_balanced_sample` shuffle determinism).
