# Architecture

CoSMAS answers one research question: **does structured multi-agent
collaboration help a small LLM, or does it just add machinery?** Everything in
the design exists to make that comparison fair, so the two execution modes -
AgentFlow and baseline - share every component except the ones under test.

This page describes how a question becomes a result. For the knobs, see
[configuration.md](configuration.md); for how to extend any of it, see
[guides/](guides/).

---

## The path a question takes

```
YAML config
    ↓  load_experiment_config()            config/loader.py
ExperimentConfig
    ↓  build providers, tools, dataset     runner/
AgenticOrchestrator.run_batch(questions)   core/orchestrator.py
    ↓  planning turn (AgentFlow only)
    ↓  reasoning turns, until answered or max_turns
List[ExecutionState]
    ↓  score each answer                   datasets/evaluators/
raw_results.json + metrics.json
```

The runner is the composition root: it reads the config, builds model providers
(`runner/providers.py`), builds the tool registry (`runner/tools.py`), loads the
dataset, drives the orchestrator, and writes the outputs. The orchestrator knows
nothing about YAML, benchmarks, or files.

---

## Everything is batched

The unit of execution is a **batch of questions**, not a question.
`run_batch()` advances every unfinished question by one turn, and each turn is
one `model.generate()` call over all of their prompts at once.

```
turn N:  [Q1, Q2, Q3, Q4]  →  one generate() call  →  4 outputs
         Q2 answers and drops out
turn N+1: [Q1, Q3, Q4]     →  one generate() call  →  3 outputs
```

This is why a 200-question run is feasible at all: the GPU sees a handful of
large batches rather than thousands of single-sequence calls. It is also why
several things in the code look more complicated than a single-question agent
would - the complexity buys throughput.

`batch_size: 1` disables cross-question batching, which is the setting to use
when debugging one question.

---

## The turn loop

`_process_batch_turn` (`core/orchestrator.py`) is the heart of the system:

1. **Increment** every active state's turn counter.
2. **Build one prompt per state** - the memory prompt in AgentFlow mode, the
   growing conversation in baseline mode.
3. **One batched `generate()`.** If it raises, every state in the batch is
   marked failed and finished; one bad batch does not take down the run.
4. **Parse each output for a tool call** (`utils/parsing.py`).
   - No tool call → the model answered. Extract the answer, mark finished.
   - Tool call → classify it (below).
5. **Apply immediate results**, then **flush the batched ones**.

When a question hits `max_turns` without answering, it is not simply dropped:
the orchestrator sends one final "you are out of tool calls, answer now" nudge
and records `max_turns_reached: true` in its metadata. Those rows are still
scored, which matters when reading the metrics - a low score can mean bad
answers or exhausted turns, and only the metadata distinguishes them.

### Two kinds of tool call

`_classify_tool_call` routes each call down one of two paths:

- **Immediate** - the tool runs inline and returns a `ToolResult` right away.
  This is every tool in `direct_tool_call: true` mode, and always the case for
  `mind_map`, `text_inspector` and `image_inspector`.
- **Batched** - the tool defers. It returns a `BatchJob` instead of a result,
  and all jobs for that tool are executed together after the turn's generation.

A tool is batched when it satisfies the `BatchedTool` protocol *and* is not in
direct mode:

```python
@staticmethod
def _is_batched(tool):
    return isinstance(tool, BatchedTool) and not getattr(tool, "direct_mode", True)
```

`BatchedTool` is a `runtime_checkable` `Protocol`, so a new tool opts in simply
by implementing the methods - there is no registry of batched tools to update
and nothing to remember to edit.

---

## Deferred tools and batching

A sub-agent tool does two things: real work (an HTTP search, building a prompt)
and an **LLM call** to analyse the result. The LLM call is the expensive part,
and it is the part worth batching across questions.

So batched tools split into three phases (`core/batching.py`):

| Phase | Runs | Does |
|---|---|---|
| `prepare(state, tool_call, args)` | per call, inline | The non-LLM work. Returns a `BatchJob` to defer, **or** a `ToolResult` to short-circuit (missing arguments, cache hit, failure). |
| `pre_batch(jobs)` | once per tool per turn | Cross-job work - `web_search` fetches all URLs for all jobs here, in one parallel pass. |
| `batch_prompt(job)` → `finalize(job, generation)` | once per job | Build the prompt; turn the generation into a `ToolResult`. |

`flush_batches` then:

1. sorts tools by `batch_priority` ascending - `web_search` is 10,
   `code_generator` is 20, and the order is load-bearing because a web analysis
   populates a cache a code job in the same turn may read;
2. calls `pre_batch` for each tool;
3. groups jobs by `id(model_provider)` so tools sharing a provider share a call;
4. runs one `generate()` per group and calls `finalize` for each job.

> **`finalize` owns output cleaning; `flush_batches` commits `result.output`
> untouched.** This is not stylistic. `strip_thinking_tags` is not idempotent
> on text containing two orphaned `</think>` markers, so a "tidy" central strip
> in `flush_batches` silently corrupts the web path. If you are adding a batched
> tool, clean inside `finalize`.

---

## State

One `ExecutionState` per question (`core/state.py`) carries both conversations
at once - which is what makes the two modes comparable:

| Field | Used by | Holds |
|---|---|---|
| `messages` | baseline | The growing chat history. |
| `query_analysis` | AgentFlow | The planning turn's output. |
| `action_history` | AgentFlow | One record per tool call: tool, sub-goal, command, result. |
| `output_messages` | both | Assistant + tool turns, for the trace in `raw_results.json`. |
| `tool_calls`, `tool_counts` | both | Bookkeeping and per-tool limits. |
| `metadata` | both | `max_turns_reached`, errors, token usage, dataset fields. |

`_commit_tool_result` writes the bookkeeping fields in **both** modes and
appends to `action_history` only in AgentFlow mode and to `messages` only in
baseline mode. Keeping the shared bookkeeping outside the branch is what lets
the same analysis scripts read runs from either mode.

---

## Baseline vs AgentFlow

Both modes use the same model, the same tools, the same batching, the same
scorer. Three things differ.

### 1. The planning turn

AgentFlow runs one extra generation before turn 1 (`_run_planning_turn`) whose
output is stored as `query_analysis` and included in every later prompt.
Baseline skips it.

### 2. What the model sees each turn

**Baseline** - the conversation grows:

```
[system, user, assistant, tool, assistant, tool, ...]
```

**AgentFlow** - the prompt is rebuilt from structured memory every turn, and is
always exactly two messages:

```
[system, user]
```

where the user message is the original question plus:

```
**Query Analysis:**
<the planning turn's output>

**Previous Steps:**
Action Step 1:
  - Tool: web_search
  - Sub-goal: Find the population of Lyon in 2020
  - Command: {"name": "web_search", "arguments": {...}}
  - Result: ...
```

The consequence worth internalising: **in AgentFlow the model never sees its own
raw previous outputs**, only the distilled record. Nothing accumulates except
`action_history`, so context growth is linear in tool calls rather than in
tokens emitted.

### 3. Explicit sub-goals

AgentFlow prompts instruct the model to emit `<sub_goal>...</sub_goal>` before
each `<tool_call>`. `_extract_sub_goal` parses it (truncating at 500 chars) into
`action_history`. An empty string means the model did not comply - worth
checking when a run underperforms, since it degrades the memory record itself.

Note that these prompts come from different template files entirely: AgentFlow
uses `*_dataset*.yaml`, baseline uses `*_baseline*.yaml`
(`prompts/builder.py`). A change to one does not affect the other, which is a
frequent source of "why did only half my runs change?".

---

## Extension seams

Four places are designed to be extended without touching the orchestrator. Each
has a guide:

| To add | Seam | Guide |
|---|---|---|
| A benchmark | `DatasetSpec` row + loader + template | [add-a-benchmark](guides/add-a-benchmark.md) |
| A tool or sub-agent | `@register_tool` factory, optionally `BatchedTool` | [add-a-tool-or-subagent](guides/add-a-tool-or-subagent.md) |
| A model family | `ModelFamily` enum + the `_*_FAMILIES` frozensets | [add-a-model-family](guides/add-a-model-family.md) |
| An adaptation method | GEPA / SFT / RL hook points | [add-an-adaptation-method](guides/add-an-adaptation-method.md) |

A fifth thing people commonly want to change is not a clean seam and is called
out as such: the RL/GEPA **training data**. Reweighting or filtering the
existing sources is a CLI flag; adding a genuinely different source is real
code (a normaliser + a downloader, no table to add a row to). See
[change-training-data](guides/change-training-data.md).

The shared principle: **the orchestrator dispatches on capability, not on
name.** There is no `if tool_name == "web_search"` anywhere in the turn loop,
and adding a tool does not mean editing an if/elif chain. When you find
yourself wanting to add a name check to `core/`, that is a sign the capability
belongs in a protocol or a spec table instead.

---

## Package map

```
src/agent_engine/
  config/      YAML schema (schema.py) + loader
  core/        orchestrator.py, state.py, tool.py, batching.py
  models/      base.py (families, ModelConfig), vllm/mlx/api providers
  tools/       web_search, code_generator, mind_map, text/image inspector,
               registry.py (the @register_tool factory table)
  datasets/    loaders, spec.py (DATASET_SPECS), evaluators/
  prompts/     templates/*.yaml + builder.py
  runner/      experiment.py (the run loop), providers.py, tools.py, metrics.py
  external/    serper, tavily, url_fetcher
  caching/     manager.py - the search/URL cache
  analysis/    failure-mode classifier and the analyses over recorded runs
  utils/       parsing (tool-call formats), logging
```

Two packages sit alongside it: `src/fine_tuning/` (RL and SFT, including
vendored AgentFlow) and `src/gepa_integration/` (prompt optimisation). Both are
covered in [pipelines/](pipelines/).
