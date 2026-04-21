# DeepSeek R1 Distill Integration — As-Built Record

## Overview

Integration of `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and
`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` into CoSMAS, with the following
model-specific behaviours:

| Requirement | Implementation |
|---|---|
| Sampling defaults (T=0.6, top_p=0.95, max_tokens=32768) | `_FAMILY_DEFAULTS["deepseek"]` in `base.py` |
| No system prompt (merge into user turn) | `_NO_SYSTEM_PROMPT_FAMILIES` + `merge_system_into_user()` |
| `<think>` prefix forced when thinking on / off | `_THINK_PREFIX_FAMILIES` in providers |
| Tool-call format: single JSON object `{"tool_call": {…}}` | `ToolCallFormat.JSON_SINGLE` in `_TOOL_CALL_FORMAT` |
| Stop generation on hallucinated `<tool_response>` | `_TOOL_CALL_STOP_TOKEN[JSON_SINGLE] = "<tool_response>"` |
| Force-tool-call prefix on turn 1 (AgentFlow only) | `AgenticOrchestrator._force_tool_call` + `_render_messages(force_tool_call=…)` |

---

## Files Changed

| File | Change |
|---|---|
| `src/agent_engine/models/base.py` | `ModelFamily.DEEPSEEK`, `_FAMILY_DEFAULTS`, `_NO_SYSTEM_PROMPT_FAMILIES`, `_THINK_PREFIX_FAMILIES`, `ToolCallFormat.JSON_SINGLE`, `merge_system_into_user()` |
| `src/agent_engine/models/vllm_provider.py` | `_render_messages`: system→user merge + think/force prefix injection; `_TOOL_CALL_STOP_TOKEN[JSON_SINGLE]` |
| `src/agent_engine/models/mlx_provider.py` | Same as vllm |
| `src/agent_engine/utils/parsing.py` | `parse_tool_call` accepts JSON_SINGLE, code-fenced JSON, and bare `{"name": …, "arguments": …}` fallback |
| `src/agent_engine/prompts/builder.py` | `_CALL_PLACEHOLDER[JSON_SINGLE]`, sub-goal kept for AF but JSON replaces `<tool_call>`; baseline variant omits sub-goal |
| `src/agent_engine/core/orchestrator.py` | `_force_tool_call` gated on family ∈ `_THINK_PREFIX_FAMILIES` **and** `not baseline` |
| `scripts/generate_configs.py` | DeepSeek models, variant grid, suites `deepseek-baseline` / `deepseek-agentflow` |

---

## 1. `src/agent_engine/models/base.py`

### Family defaults
```python
_FAMILY_DEFAULTS = {
    "deepseek": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    ...
}
```

### DeepSeek-specific constants
```python
_NO_SYSTEM_PROMPT_FAMILIES = frozenset({ModelFamily.DEEPSEEK})
_THINK_PREFIX_FAMILIES     = frozenset({ModelFamily.DEEPSEEK})
_TOOL_CALL_FORMAT[ModelFamily.DEEPSEEK] = ToolCallFormat.JSON_SINGLE
```

### `merge_system_into_user(msgs)`
Prepends the system message content to the first user turn (separator = blank
line) and drops the system message. Used by both providers before
`apply_chat_template`, because DeepSeek R1's chat template has no system slot.

---

## 2. `src/agent_engine/models/vllm_provider.py` / `mlx_provider.py`

### `_render_messages(msgs, use_thinking, force_tool_call)`

**System→user merge (DeepSeek):**
```python
if self.config.family in _NO_SYSTEM_PROMPT_FAMILIES:
    msgs = merge_system_into_user(msgs)
```

**Think-prefix injection (DeepSeek) — three suffixes after the generation prompt:**

| Condition | Suffix |
|---|---|
| `force_tool_call=True` | `<think>\nI need to call a tool to answer this question.\n</think>\n<sub_goal>` |
| `use_thinking=True` (no force) | `<think>\n` |
| otherwise | `<think>\n\n</think>\n` |

### Stop tokens

```python
_TOOL_CALL_STOP_TOKEN = {
    ToolCallFormat.JSON:        "</tool_call>",
    ToolCallFormat.PYTHONIC:    "</function_calls>",
    ToolCallFormat.JSON_SINGLE: "<tool_response>",
}
```

JSON_SINGLE has no natural closing tag; we stop on `<tool_response>` to
prevent the model from continuing past its own tool call and hallucinating a
fake tool response. The real tool result is appended by the orchestrator.

---

## 3. `src/agent_engine/utils/parsing.py`

`parse_tool_call` tries formats in order:

1. `<tool_call>{"name": …, "arguments": …}</tool_call>` (Qwen3 / default, last wins)
2. `<function_calls>pythonic</function_calls>` (OLMo 3)
3. `{"tool_call": {"name": …, "arguments": …}}` (DeepSeek JSON_SINGLE, first wins)
4. Code-fenced JSON (```…```)
5. Bare `{"name": …, "arguments": {…}}` (DS fallback)

Stages 3–5 strip `<think>…</think>` blocks first to avoid matching
hallucinated calls inside the reasoning block.

---

## 4. `src/agent_engine/prompts/builder.py`

`tool_call_format` drives three per-format surface elements:

| | JSON | PYTHONIC | JSON_SINGLE |
|---|---|---|---|
| Open/close tags | `<tool_call>…</tool_call>` | `<function_calls>…</function_calls>` | (none) |
| Placeholder | `{"name": …, "arguments": …}` | `function_name(arg=value)` | `{"tool_call": {"name": …, "arguments": …}}` |

For **AgentFlow** runs the system prompt instructs the model to emit a
`<sub_goal>…</sub_goal>` tag before each tool call. For **baseline** runs
the sub-goal instruction is omitted (matches the other baseline templates).

---

## 5. `src/agent_engine/core/orchestrator.py`

```python
self._force_tool_call = (
    bool(tool_registry)
    and not baseline
    and model_provider.config.family in _THINK_PREFIX_FAMILIES
)
```

The force-prefix is only applied on turn 1 in AgentFlow mode. In baseline
mode the model is expected to emit tool calls from the plain system prompt
alone, because the forced prefix injects a `<sub_goal>` tag that the baseline
prompt never teaches.

---

## 6. `scripts/generate_configs.py`

### Models
```python
"deepseek-7b":  {"family": "deepseek", "path_or_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  "tp": None, "gpus": 1}
"deepseek-32b": {"family": "deepseek", "path_or_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "tp": 2,    "gpus": 2}
```

### Variant grid (`VARIANTS_DEEPSEEK_ALL`)

Mirrors the Qwen3 grid (4 thinking modes × tool configurations):

```
# 7B — no tools
ds7b_no_tools_none               baseline=True   tools=none   thinking=NO
ds7b_no_tools_orchestrator       baseline=True   tools=none   thinking=ORCHESTRATOR_ONLY

# 7B — direct tools
ds7b_direct_tools_none           baseline=True   tools=tools  thinking=NO
ds7b_direct_tools_orchestrator   baseline=True   tools=tools  thinking=ORCHESTRATOR_ONLY

# 7B — sub-agent tools (agentflow)
ds7b_subagent_tools_none         baseline=False  tools=tools  thinking=NO
ds7b_subagent_tools_orchestrator baseline=False  tools=tools  thinking=ORCHESTRATOR_ONLY
ds7b_subagent_tools_subagents    baseline=False  tools=tools  thinking=SUBAGENTS_ONLY
ds7b_subagent_tools_all          baseline=False  tools=tools  thinking=ALL

# 32B — no tools
ds32b_no_tools_none              baseline=True   tools=none   thinking=NO
ds32b_no_tools_orchestrator      baseline=True   tools=none   thinking=ORCHESTRATOR_ONLY

# 32B — direct tools
ds32b_direct_tools_none          baseline=True   tools=tools  thinking=NO
ds32b_direct_tools_orchestrator  baseline=True   tools=tools  thinking=ORCHESTRATOR_ONLY
```

### Suite splits
- `VARIANTS_DEEPSEEK_BASELINE` — `baseline=True` variants
- `VARIANTS_DEEPSEEK_AGENTFLOW` — `ds7b_subagent_tools_*` (AF, 7B only)

### Config layout
```
experiments/configs/deepseek/baseline/<dataset>/<variant>.yaml
experiments/configs/deepseek/agentflow/<dataset>/<variant>.yaml
```

### CLI
```bash
python scripts/generate_configs.py --suite deepseek-baseline
python scripts/generate_configs.py --suite deepseek-agentflow
```

---

## 7. GPU sizing

`_is_large` matches `"32b"` → DeepSeek 32B automatically gets 2 GPUs for
tensor parallelism (both in auto-detect and multi-model pinning paths). No
change needed.

---

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Tool-call format | `JSON_SINGLE` (bare `{"tool_call": {…}}`, no XML wrap) | Matches the DeepSeek R1 template's native single-call-per-turn contract; parser also accepts bare `{"name":…}` as safety net |
| Stop token for JSON_SINGLE | `<tool_response>` | No natural closing tag; stops the model from fabricating tool results past its own call |
| Force-prefix scope | AgentFlow only | Baseline prompt doesn't teach `<sub_goal>`; keeping the pure-baseline contract |
| Thinking-mode variants | Full Qwen3 grid (NO / ORCHESTRATOR_ONLY / SUBAGENTS_ONLY / ALL) | Fair comparison |
| Sub-agent model | Same checkpoint as orchestrator (self-as-sub-agent) | Consistent with Qwen3 / OLMo setups |
| System-prompt merge | Inside provider `_render_messages`, transparent to orchestrator | Keeps orchestrator/tool code family-agnostic |
