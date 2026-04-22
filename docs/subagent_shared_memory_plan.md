# Sub-agent Shared Memory — Implementation Plan

Ablation: **give task-directed sub-agents the same structured context the orchestrator
already has**, on top of their existing per-call task argument. Hypothesis: extra
context can improve sub-agent quality (better-targeted web analyses, better-scoped
generated code, better-focused file/image reasoning) at the cost of more input tokens.

Status: **planning only — nothing implemented yet**. Default: **disabled**.

---

## 1. Does the approach make sense here?

Yes — the repository is unusually well-suited to this ablation:

- Structured memory already exists on `ExecutionState`
  (`query_analysis` + `action_history`) and is currently consumed only by
  `AgenticOrchestrator._build_memory_prompt`. No new plumbing on the state side.
- Every task-directed sub-agent has exactly **one** prompt-construction entry
  point, which makes the injection surface tiny:

  | sub-agent         | prompt builder / single entry point                                           |
  |-------------------|-------------------------------------------------------------------------------|
  | `web_search`      | `WebSearchTool.build_analysis_prompt(query, formatted_results)`               |
  | `code_generator`  | `CodeGeneratorTool.build_task_prompt(task, context="")`                       |
  | `text_inspector`  | `TextInspectorTool._analyze_with_llm(file_content, question)`                 |
  | `image_inspector` | `ImageInspectorTool._analyze_with_vlm(image, question)`                       |
  | `mind_map`        | GraphRAG entity extraction — **out of scope** (internal, not task-directed)   |

- Token accounting already runs through `_accumulate_usage` — the *total* extra
  prompt tokens land in `metadata.token_usage.prompt_tokens` automatically. We
  only need an additional **explicit** counter for the ablation write-up.
- There is already precedent for this kind of structural ablation
  (`structured_memory_ablation` suite), so we can mirror its YAML/suite
  conventions.

Risks / things that are genuinely new work:
- The `web_search` per-run `_analysis_cache` is keyed by `query` only. With
  shared memory, the same query at two different turns has different correct
  outputs → the cache becomes unsafe and must be bypassed.
- Sub-agent prompts can balloon in size if `action_history` grows. We need a
  truncation policy.
- Interaction with `baseline: true` (no planning turn, no `query_analysis`, no
  `action_history`) is undefined — we will treat it as a config error.

---

## 2. Design decisions (agreed with user)

| Decision                   | Choice                                                                                                              |
|----------------------------|---------------------------------------------------------------------------------------------------------------------|
| Scope                      | All task-directed sub-agents: `web_search`, `code_generator`, `text_inspector`, `image_inspector`                    |
| Context payload            | **Lightweight**: original question (+ attachment note) + `query_analysis` + `action_history` with **results AND commands stripped** (`tool_name` + `sub_goal` only — the raw JSON tool_call is noisy, and sub_goal already captures intent) |
| Current-turn reasoning     | **Include** the current turn's `<sub_goal>` (the one that triggered *this* tool call) as a separate "Current sub-goal" field |
| `web_search` analysis cache| **Bypass** when the flag is ON (simplest + correct)                                                                  |
| Interaction with `baseline`| **Config error** — `tools.subagent_shared_memory=true` with `baseline: true` is rejected by the config loader        |
| Token accounting           | **Explicit** counter `metadata.subagent_shared_memory_tokens` (accumulated across all sub-agent calls), in addition to the existing automatic `metadata.token_usage` |
| Truncation                 | Keep **last K action steps** (default `K = 5`), configurable via `tools.subagent_shared_memory_last_k`               |
| Config location            | `tools.subagent_shared_memory: bool = False` (+ `tools.subagent_shared_memory_last_k: int = 5`)                      |
| Default                    | **Disabled** everywhere unless explicitly turned on                                                                  |

---

## 3. Shared-context block (single source of truth)

Rendered exactly once per sub-agent call, as a plain-text preface. Format
chosen to be close to the orchestrator's existing memory prompt so the
sub-agent sees something familiar and so we don't fork formatting logic.

```
**Original Question:**
<state.messages[1]["content"]>   # includes the "[Attachment]" note if any

**Query Analysis:**
<state.query_analysis>           # omitted if empty

**Previous Steps (last K):**
# Step numbering is ABSOLUTE — matches the orchestrator's own memory prompt.
# Example with 8 committed steps and K=5 the sub-agent sees steps 4..8:
Action Step 4:
  - Tool: <tool_name>
  - Sub-goal: <sub_goal>
Action Step 5:
  ...
Action Step 8:
  ...
# whole section omitted if action_history is empty
# We intentionally drop the raw JSON `command` too: `sub_goal` conveys
# intent at a higher level of abstraction and the command adds tokens
# without adding signal at the sub-agent level.

**Current Sub-goal:**
<sub_goal extracted from state.current_output via AgenticOrchestrator._extract_sub_goal>
# omitted if empty
# Placed LAST so it sits closest to the sub-agent's task/query payload — the
# immediately relevant intent is the last thing the model reads before the task.
```

Missing fields (e.g. empty `query_analysis`, empty `action_history`) collapse
cleanly — no empty headings are rendered.

The block is produced by a new helper in `src/agent_engine/core/orchestrator.py`:

```python
def _build_subagent_shared_context(
    self,
    state: ExecutionState,
    current_output: str | None = None,
    last_k: int = 5,
) -> tuple[str, int]:
    """Return (rendered_context, approx_char_len). Returns ("", 0) if disabled."""
```

…exposed via a new `AgenticOrchestrator` attribute `self.subagent_shared_memory`
(bool) and `self.subagent_shared_memory_last_k` (int), set in `__init__`.

---

## 4. Per-file implementation plan

### 4.1 `src/agent_engine/config/schema.py`

Add to `ToolsConfig`:

```python
subagent_shared_memory: bool = False
subagent_shared_memory_last_k: int = 5
```

with docstring describing behaviour + default = off.

Add validation in `ExperimentConfig` (pydantic `model_validator`):

```python
@model_validator(mode="after")
def _validate_shared_memory(self):
    if self.tools.subagent_shared_memory and self.baseline:
        raise ValueError(
            "tools.subagent_shared_memory=true is incompatible with baseline=true "
            "(baseline has no query_analysis or action_history to share)."
        )
    if self.tools.subagent_shared_memory and self.tools.direct_tool_call:
        raise ValueError(
            "tools.subagent_shared_memory=true requires direct_tool_call=false "
            "(no sub-agents run in direct mode; the flag would be a no-op)."
        )
    if self.tools.subagent_shared_memory_last_k < 0:
        raise ValueError("tools.subagent_shared_memory_last_k must be >= 0")
    return self
```

### 4.2 `src/agent_engine/core/state.py`

Add one metadata-style field to `ExecutionState`:

```python
# Incremental counter of *extra* prompt tokens introduced by the shared-memory
# block in sub-agent calls (approximate: measured via tokenizer on the rendered
# context). 0 when the feature is disabled.
subagent_shared_memory_tokens: int = 0
```

(We keep it as a first-class field rather than burying it in `metadata` so
downstream analysis scripts can discover it without digging.)

### 4.3 `src/agent_engine/core/orchestrator.py`

New constructor params + attributes:

```python
def __init__(self, ..., subagent_shared_memory: bool = False,
             subagent_shared_memory_last_k: int = 5):
    ...
    self.subagent_shared_memory = subagent_shared_memory
    self.subagent_shared_memory_last_k = subagent_shared_memory_last_k
```

New helper methods (private):

- `_build_subagent_shared_context(state, current_output=None, last_k=None) -> str`
  - Returns `""` when `self.subagent_shared_memory` is `False`.
  - Reconstructs the block exactly as in §3.
  - Uses `AgenticOrchestrator._format_action_history_lightweight(truncated)` —
    a new static method that renders `action_history` *without the `result`
    field*.
  - Slices `state.action_history[-last_k:]`.
  - Extracts current-turn sub-goal via existing `_extract_sub_goal(current_output)`.

- `_accumulate_shared_memory_tokens(state, context_str)` — counts tokens using
  `self.model.tokenizer.encode(context_str)` (fallback: `len(context_str) // 4`
  if tokenizer unavailable) and adds to `state.subagent_shared_memory_tokens`.
  We tokenize with the *orchestrator* model's tokenizer for consistency; the
  actual sub-agent may use a slightly different tokenizer but the magnitude is
  what matters.

Call-site changes — **four** places only:

1. **`_schedule_web_job`** (single sub-agent query build):
   Currently `tool.build_analysis_prompt(query, ...)` is called later in
   `_run_web_analysis_batch`. Best to attach the pre-built context onto the
   `_WebJob` namedtuple so it flows through batching untouched:

   ```python
   class _WebJob(NamedTuple):
       state: ExecutionState
       tool_call: Dict[str, Any]
       tool: Any
       query: str
       payload: Dict[str, Any]
       shared_context: str   # NEW, "" when disabled
   ```

   `_schedule_web_job` builds the context once from the current state + output,
   accumulates tokens, and puts it on the job.

2. **`_schedule_code_job`**: same pattern — add `shared_context` to `_CodeJob`,
   build from current state/output, accumulate tokens.

3. **`_run_web_analysis_batch`**: when calling
   `job.tool.build_analysis_prompt(...)`, pass the extra context:

   ```python
   job.tool.build_analysis_prompt(job.query, <formatted>, shared_context=job.shared_context)
   ```

   Also: when `self.subagent_shared_memory` is True, **bypass** the
   `analysis_cache` read/write (early return check and final insert are both
   skipped). The dedup cache hit path in `_schedule_web_job` must also check
   this flag and *not* serve cached analyses.

4. **`_run_code_generation_batch`**: pass `shared_context` through
   `job.tool.build_task_prompt(task=..., attachment_context=att_ctx, shared_context=...)`.
   (Note: the former `context=` kwarg was renamed to `attachment_context=` to
   make the MAT-style attachment block clearly orthogonal from the new
   orchestrator `shared_context=` block.)

5. **`_execute_tool`** path (non-batched / single-call, used for
   `text_inspector`, `image_inspector`, and legacy `web_search`/`code_generator`
   in direct mode — but direct mode doesn't apply here):
   - Text/image inspectors don't go through the batched path. They also don't
     currently have a prompt-builder abstraction; their prompts are built
     inside `_analyze_with_llm` / `_analyze_with_vlm`. Two options:
     - (a) Add `shared_context` parameter to those methods + thread through
       `tool.execute(..., shared_context=...)` — requires widening the tool
       signatures and `_execute_tool` threading.
     - (b) Store `shared_context` on the tool instance transiently per call
       (like `mind_map.set_current_question`): `tool.set_shared_context(str)`.
   - **Chosen**: (a), the explicit kwarg. It's consistent with
     `build_analysis_prompt`/`build_task_prompt` and avoids hidden instance
     state that is error-prone across parallel/batched execution.
   - `_execute_tool` becomes aware of the orchestrator's current state and
     current_output for the call and, when `self.subagent_shared_memory` is on
     and the tool is one of the task-directed sub-agents, passes the context
     through `**arguments`. (Helper: `self._maybe_inject_shared_context(
     tool_name, state, arguments)`.)

New static:

```python
@staticmethod
def _format_action_history_lightweight(actions: list[dict]) -> str:
    """Like _format_action_history but drops the `result` field."""
```

**Logging of what each sub-agent sees.** Every time a shared-memory block is
built and attached to a sub-agent call, the orchestrator logs a single
structured record so we can inspect exactly what each sub-agent received.

New helper:

```python
def _log_subagent_shared_context(
    self,
    *,
    tool_name: str,             # "web_search" | "code_generator" | "text_inspector" | "image_inspector"
    state: ExecutionState,
    shared_context: str,
    task_payload: str,          # the tool's primary argument: query / task / question
    added_tokens: int,
) -> None:
    """Emit a single log line describing what this sub-agent will see."""
```

Log format (both the metadata line **and** the full rendered block are emitted
at INFO by default — the whole point of the ablation is being able to inspect
exactly what each sub-agent received, so it should be visible in normal runs
without needing `--log-level DEBUG`):

```
INFO  Sub-agent shared-memory injection | tool=web_search q_id=42 turn=3
      shared_ctx_chars=1847 shared_ctx_tokens≈412 task_payload_chars=58
INFO  Sub-agent shared-memory block (tool=web_search q_id=42 turn=3):
      --- BEGIN shared_context ---
      **Original Question:**
      ...
      **Query Analysis:**
      ...
      **Previous Steps (last 5):**
      Action Step 4:
        - Tool: code_generator
        - Sub-goal: ...
      ...
      **Current Sub-goal:**
      ...
      --- END shared_context ---
      task_payload: <the query/task/question as the model will see it>
```

Also emitted even when `self.subagent_shared_memory is False` would be noise,
so the logger is a no-op in that case (`if not self.subagent_shared_memory: return`).

Call sites (one each, right after the context is built):
- `_schedule_web_job` → logs `tool="web_search"`, payload = `query`.
- `_schedule_code_job` → logs `tool="code_generator"`, payload = `task`.
- `_maybe_inject_shared_context` (text/image inspector single-call path) →
  logs `tool=<tool_name>`, payload = `arguments.get("question", "")`.

This makes the ablation auditable: grep `experiment.log` for
`Sub-agent shared-memory injection | tool=` to see every injection, and flip
to DEBUG when you want to eyeball actual prompts.

### 4.4 `src/agent_engine/tools/web_search.py`

Extend `build_analysis_prompt` signature:

```python
def build_analysis_prompt(
    self,
    query: str,
    formatted_results: str,
    shared_context: str = "",
) -> str:
```

Prepend a clearly delimited section at the top of the existing instruction,
only when `shared_context` is non-empty:

```
**Shared context:**

{shared_context}

---

**Task Instruction:**
...
```

`_analyze_with_llm` also gets `shared_context=""` and forwards it.
`execute(...)` remains unchanged for non-batched direct-mode calls (the
orchestrator threads context in via the single-call `_execute_tool` path when
applicable — see §4.3 item 5).

### 4.5 `src/agent_engine/tools/code_generator.py`

Extend `build_task_prompt`:

```python
def build_task_prompt(
    self,
    task: str,
    attachment_context: str = "",
    shared_context: str = "",
) -> str:
```

When `shared_context` is non-empty, prepend the same delimited block *before*
the existing `context_block` (attachment context). Keep `attachment_context`
and `shared_context` orthogonal: the former carries `[ATTACHED_FILE_PATH] ...`
and prior `text_inspector` output, the latter carries orchestrator memory.

`generate_code(...)` gets the same kwarg and forwards it.

### 4.6 `src/agent_engine/tools/text_inspector.py`

Extend `execute(..., shared_context: str = "")` and `_analyze_with_llm(..., shared_context="")`.
Prepend the shared context to the user prompt (between a "Context:" header and
the existing "File content:" block). System prompt unchanged.

### 4.7 `src/agent_engine/tools/image_inspector.py`

Extend `execute(..., shared_context: str = "")` and
`_analyze_with_vlm(..., shared_context="")`. Prepend the shared context as an
extra `{"type": "text", "text": ...}` entry inside the user multimodal message,
before the `"Question:\n..."` entry.

### 4.8 `scripts/run_experiment.py`

In `build_orchestrator(...)` / wherever `AgenticOrchestrator(...)` is
constructed, thread through:

```python
AgenticOrchestrator(
    ...,
    subagent_shared_memory=config.tools.subagent_shared_memory,
    subagent_shared_memory_last_k=config.tools.subagent_shared_memory_last_k,
)
```

No tool-registration changes required.

### 4.9 Analysis scripts & output schema

- `metadata.token_usage.prompt_tokens` — unchanged, already captures the delta.
- `state.subagent_shared_memory_tokens` — **new** — surface it in `raw_results.json`.
  Confirm it's included via the default pydantic serialisation of
  `ExecutionState` (it is — we're adding a top-level field).
- `scripts/analyze_results.py`: add an optional breakdown column
  `shared_mem_tokens` (mean/sum per run) when the field is present.

### 4.10 Config generation — `scripts/generate_configs.py`

Add a new suite for the ablation, mirroring `structured_memory_ablation`:

```python
VARIANTS_SUBAGENT_SHARED_MEMORY_ABLATION = [
    ("qwen8B_subagent_tools_shared_mem_on",  "8B", False, "tools", "ORCHESTRATOR_ONLY"),
    # Control run ("off") is already covered by the existing AgentFlow suite,
    # but we include it here as well for direct 1:1 comparison under the same folder:
    ("qwen8B_subagent_tools_shared_mem_off", "8B", False, "tools", "ORCHESTRATOR_ONLY"),
]

SUITES["subagent_shared_memory_ablation"] = {
    "description_tag": (
        "[Sub-agent shared-memory ablation: AgentFlow + subagent tools + orch thinking, "
        "sub-agents receive question + query_analysis + last-K stripped action_history; "
        "NO image_inspector, NO mindmap]"
    ),
    "name_prefix":     "AF_subagent_shared_mem",
    "output_dir_root": "./experiments/results/subagent_shared_memory_ablation",
    "config_subdir":   "qwen3/subagent_shared_memory_ablation",
    "baseline":        False,
    "variants":        VARIANTS_SUBAGENT_SHARED_MEMORY_ABLATION,
    "num_gpus":        2,
    "wandb_project":   "benchmarks",
    "split_overrides": {},
    # New: per-variant YAML overlay for the ON/OFF switch
    "variant_overrides": {
        "qwen8B_subagent_tools_shared_mem_on":  {"tools.subagent_shared_memory": True},
        "qwen8B_subagent_tools_shared_mem_off": {"tools.subagent_shared_memory": False},
    },
}
```

If `variant_overrides` is not already supported by the existing
`make_config(...)` path, either (a) extend it (small change — it only needs a
dotted-key assignment into the dict just before `yaml.safe_dump`), or (b)
write two tiny post-process passes that patch each generated YAML. (a) is
cleaner.

### 4.11 Documentation

- Update `CLAUDE.md` "Config essentials" section with a line:
  `tools.subagent_shared_memory: true` — pass the orchestrator's structured
  memory (question + query_analysis + last-K action steps w/o results +
  current sub-goal) to every task-directed sub-agent. Requires
  `direct_tool_call: false`. Incompatible with `baseline: true`.
- This document (`docs/subagent_shared_memory_plan.md`) stays as the design
  record.

---

## 5. Tests

New unit tests under `tests/`:

1. `tests/core/test_shared_memory_context.py`
   - `_build_subagent_shared_context` returns `""` when flag is off.
   - With `query_analysis` + 3-step `action_history` + current sub-goal, the
     rendered block contains all four sections and no `- Result:` / `- Command:` lines.
   - `last_k=2` truncates to the last 2 steps.
   - `last_k=0` drops the "Previous Steps" block entirely but keeps the others.
   - Empty `action_history` / empty `query_analysis` / missing sub-goal each
     collapse cleanly.

2. `tests/tools/test_subagent_shared_context_injection.py`
   - `WebSearchTool.build_analysis_prompt(..., shared_context="CTX")` contains
     `"CTX"` above the `**Task Instruction:**` marker.
   - Same for `CodeGeneratorTool.build_task_prompt(...)`, with `shared_context`
     appearing before the attachment `context_block`.
   - Empty `shared_context` produces byte-identical output to the pre-change
     prompt (regression guard).

3. `tests/core/test_shared_memory_cache_bypass.py`
   - With flag OFF, `_run_web_analysis_batch` stores the LLM output in
     `_analysis_cache`.
   - With flag ON, it does not — a second call with the same query re-runs the
     sub-agent.

4. `tests/config/test_shared_memory_validation.py`
   - `tools.subagent_shared_memory=true` + `baseline=true` → `ValueError`.
   - `tools.subagent_shared_memory=true` + `direct_tool_call=true` → `ValueError`.
   - `subagent_shared_memory_last_k=-1` → `ValueError`.

5. `tests/core/test_shared_memory_token_accounting.py`
   - Mock orchestrator run with 2 web_search calls, flag ON: verify
     `state.subagent_shared_memory_tokens` > 0 and matches
     `tokenizer.encode(context).__len__()` summed over calls.
   - Flag OFF: counter stays at 0.

All tests use mocks for model providers (no GPU needed) — consistent with the
existing test style.

---

## 6. Rollout / run plan

1. Implement §4.1–§4.3 + §4.4–§4.7 behind the flag (default off). **No
   behaviour change when the flag is off** — verify via the existing test
   suite.
2. Add tests §5.1–§5.5, all green.
3. Add §4.10 suite; regenerate configs:
   `python scripts/generate_configs.py --suite subagent_shared_memory_ablation`.
4. Run sanity check locally on a tiny subset (e.g. GAIA level-1, n=5) with
   MLX/OpenAI backend to confirm sub-agent prompts look right in the
   `experiment.log` (sub-agents' prompts are already logged under
   `_analyze_with_llm` paths via `logger.info` when DEBUG is on — may need a
   one-line `logger.debug("sub-agent prompt: %.500s...", prompt)` in each
   builder for easier inspection).
5. Launch ON vs. OFF on 8B Qwen3 + subagent tools across GAIA / HLE / GPQA /
   AIME / MuSiQue / BigCodeBench (mirroring the existing
   `structured_memory_ablation` matrix).
6. Analyse deltas: accuracy, `prompt_tokens` total, `subagent_shared_memory_tokens`,
   per-tool call counts. Expected signals:
   - `prompt_tokens` ↑ roughly by `subagent_shared_memory_tokens` (×
     number_of_subagent_calls).
   - Accuracy: direction unknown — that's the ablation.

---

## 7. Edge cases / things to double-check during implementation

- **Repeated tool-call dedup**: `_classify_tool_call` and `_execute_tool` both
  key dedup on `(tool_name, arguments)` — unchanged; shared context is *not*
  part of the dedup key (it shouldn't be — the point of dedup is "don't
  re-issue the same action").
- **Current-turn sub-goal extraction**: `state.current_output` for batched
  paths is the orchestrator's just-generated output; `_extract_sub_goal`
  already tolerates missing tags and returns `""`. Good.
- **Text/image inspector attachment injection**: `_execute_tool` injects
  `full_file_path` into `arguments` before calling `tool.execute(**arguments)`.
  We add `shared_context` *after* attachment injection in the same place so
  tools with strict signatures (filtered by `_sanitize_tool_arguments`) will
  accept it — need to make sure the new `shared_context` kwarg is part of
  `tool.execute`'s formal signature so it survives `_sanitize_tool_arguments`
  filtering.
- **Qwen3 / DeepSeek chat-template wrapping**: all sub-agent prompts go
  through `model_provider.apply_chat_template(...)`. The shared context is
  prepended *inside* the user message content, so the template handling is
  unchanged. No new family-specific logic needed.
- **OLMo `tool` role rewrite**: only applies to orchestrator-level messages,
  not to sub-agent single-turn prompts (which are `[{"role": "user", ...}]`).
  Unaffected.
- **Parallel / batched safety**: everything added is pure function of
  `(state, current_output)` → no shared mutable state across concurrent
  questions. Token accumulation is per-state.
- **Serialization**: `subagent_shared_memory_tokens` is a plain `int` on the
  pydantic model — serialized automatically into `raw_results.json`.

---

## 8. Out of scope (explicit non-goals)

- **`mind_map` GraphRAG entity-extraction sub-agent**: internal to indexing,
  not a task-directed action; leaving it alone.
- **Varying the context payload shape** (e.g. per-tool different subsets):
  the design leaves room (`_build_subagent_shared_context` is the single
  choke-point) but we ship one payload definition for the first pass.
- **Carrying sub-agent outputs back into the orchestrator's memory
  differently**: this ablation only changes what the sub-agents *see*, not
  what the orchestrator records after they finish.
- **Persistent cross-question memory**: irrelevant to this ablation; cache
  strategy is per-run in-memory only.

---

## 9. Rendered-prompt examples (flag ON)

All examples use this running scenario: a GAIA-like question where the
orchestrator has already completed a planning turn and two action steps.

**State snapshot:**

- `question`: *"In the 2008 paper by Smith & Lee, what is the Hall coefficient reported for sample A3 at 77 K? Report the answer in SI units."*
- `query_analysis`: *"Identify the Smith & Lee 2008 paper, locate the Hall-coefficient table, read the value for sample A3 at 77 K, and convert to SI units (m³/C)."*
- `action_history` (2 entries, full):
  1. `web_search` — sub-goal "locate the Smith & Lee 2008 paper"
  2. `code_generator` — sub-goal "fetch and parse the PDF metadata"
- `current_output` (this turn, not yet committed): `"<think>...</think>\n<sub_goal>read the Hall-coefficient value for sample A3 at 77 K</sub_goal>\n<tool_call>..."`
- Tool being called this turn: `web_search` with `query="Smith Lee 2008 Hall coefficient sample A3 77 K"`.

The shared-context block (identical for every sub-agent, since it's built once
per turn) is:

```
**Original Question:**
In the 2008 paper by Smith & Lee, what is the Hall coefficient reported for sample A3 at 77 K? Report the answer in SI units.

**Query Analysis:**
Identify the Smith & Lee 2008 paper, locate the Hall-coefficient table, read the value for sample A3 at 77 K, and convert to SI units (m³/C).

**Previous Steps (last 2):**
Action Step 1:
  - Tool: web_search
  - Sub-goal: locate the Smith & Lee 2008 paper
Action Step 2:
  - Tool: code_generator
  - Sub-goal: fetch and parse the PDF metadata

**Current Sub-goal:**
read the Hall-coefficient value for sample A3 at 77 K
```

### 9.1 `web_search` sub-agent prompt (flag ON)

```
**Shared context:**

**Original Question:**
In the 2008 paper by Smith & Lee, what is the Hall coefficient reported for sample A3 at 77 K? Report the answer in SI units.

**Query Analysis:**
Identify the Smith & Lee 2008 paper, locate the Hall-coefficient table, read the value for sample A3 at 77 K, and convert to SI units (m³/C).

**Previous Steps (last 2):**
Action Step 1:
  - Tool: web_search
  - Sub-goal: locate the Smith & Lee 2008 paper
Action Step 2:
  - Tool: code_generator
  - Sub-goal: fetch and parse the PDF metadata

**Current Sub-goal:**
read the Hall-coefficient value for sample A3 at 77 K

---

**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Current Search Query** and **Searched Web Pages**. Your objective is to extract relevant and helpful information for the **Current Search Query** from the **Searched Web Pages**.

**Guidelines:**
... (original web_search instruction unchanged) ...

**Inputs:**
- **Current Search Query:**
Smith Lee 2008 Hall coefficient sample A3 77 K

- **Searched Web Pages:**
**Web Page 1:**
{
  "title": "...",
  "url": "...",
  "content": "..."
}
**Web Page 2:**
...

Now you should analyze each web page and find helpful information based on the current search query "Smith Lee 2008 Hall coefficient sample A3 77 K".
```

### 9.2 `code_generator` sub-agent prompt (flag ON, with attachment context)

```
You are a code generator. Generate ONLY executable Python code, with NO explanations, NO comments about what the code does, and NO additional text.

Shared context:

**Original Question:**
In the 2008 paper by Smith & Lee, what is the Hall coefficient reported for sample A3 at 77 K? Report the answer in SI units.

**Query Analysis:**
Identify the Smith & Lee 2008 paper, locate the Hall-coefficient table, read the value for sample A3 at 77 K, and convert to SI units (m³/C).

**Previous Steps (last 2):**
Action Step 1:
  - Tool: web_search
  - Sub-goal: locate the Smith & Lee 2008 paper
Action Step 2:
  - Tool: code_generator
  - Sub-goal: fetch and parse the PDF metadata

**Current Sub-goal:**
read the Hall-coefficient value for sample A3 at 77 K

---

Context:

[Attachment file: smith_lee_2008.pdf — use text_inspector for full content]

Problem: Convert the reported Hall coefficient from cgs (cm^3/C) to SI (m^3/C), assuming the input value is 3.1e-3 cm^3/C.

Requirements:
- Output ONLY the Python code
- The code must be executable as a standalone script
- The code must print its output directly
- NO explanatory text before or after the code
- NO comments like "Here's the code" or "This will output"

Python code:
```

### 9.3 `text_inspector` sub-agent prompt (flag ON)

System prompt is unchanged. User prompt:

```
Shared context:

**Original Question:**
In the 2008 paper by Smith & Lee, what is the Hall coefficient reported for sample A3 at 77 K? Report the answer in SI units.

**Query Analysis:**
Identify the Smith & Lee 2008 paper, locate the Hall-coefficient table, read the value for sample A3 at 77 K, and convert to SI units (m³/C).

**Previous Steps (last 2):**
Action Step 1:
  - Tool: web_search
  - Sub-goal: locate the Smith & Lee 2008 paper
Action Step 2:
  - Tool: code_generator
  - Sub-goal: fetch and parse the PDF metadata

**Current Sub-goal:**
read the Hall-coefficient value for sample A3 at 77 K

---

File content:

<extracted text of smith_lee_2008.pdf, possibly truncated at max_chars>

Question:
What is the Hall coefficient reported for sample A3 at 77 K in Smith & Lee 2008?
```

### 9.4 `image_inspector` sub-agent prompt (flag ON)

System prompt unchanged. The user turn is multimodal; the shared-context block
is injected as an extra text element *before* the image, so the VLM reads the
context first, then looks at the picture, then reads the question:

```
[user turn]
  text: "Shared context:\n\n
         <the same block as above>\n\n---\n\n"
  image: <attached PNG of Figure 3>
  text: "Question:\nWhich bar in the chart corresponds to sample A3?"
```

### 9.5 Flag OFF — baseline reference

For comparison, when `tools.subagent_shared_memory: false` (the default), the
prompts are byte-identical to the pre-ablation behaviour:

- `web_search`: starts directly at `**Task Instruction:**`.
- `code_generator`: no `Shared context` block; attachment `Context:` block
  stays where it was.
- `text_inspector`: user prompt is `File content:\n\n...\n\nQuestion:\n...`.
- `image_inspector`: user turn has only `[image, question text]`.

The `state.subagent_shared_memory_tokens` counter remains at 0 and the
`web_search._analysis_cache` is populated normally. This is guarded by unit
tests (`test_*_empty_shared_context_is_regression_safe`).

### 9.6 What you'll see in `experiment.log` when the flag is ON

Two lines per sub-agent call (both at INFO):

```
Sub-agent shared-memory injection | tool=web_search q_id=42 turn=3 shared_ctx_chars=1427 shared_ctx_tokens~356 task_payload_chars=54
Sub-agent shared-memory block (tool=web_search q_id=42 turn=3):
--- BEGIN shared_context ---
**Original Question:**
...
**Current Sub-goal:**
read the Hall-coefficient value for sample A3 at 77 K
--- END shared_context ---
task_payload: Smith Lee 2008 Hall coefficient sample A3 77 K
```

Grep recipe:

```bash
# Every injection event in a run
rg "Sub-agent shared-memory injection" experiments/results/<run>/experiment.log

# Count injections per tool
rg -o "tool=\w+" experiments/results/<run>/experiment.log | sort | uniq -c
```
