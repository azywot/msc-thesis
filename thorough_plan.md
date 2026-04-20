# DeepSeek R1 Distill Integration — As-Built Record

## Overview

Integration of `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` and `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
into CoSMAS, with the following model-specific behaviours:

| Requirement | Implementation |
|---|---|
| Temperature 0.5–0.7 (0.6 default) | `_FAMILY_DEFAULTS["deepseek"]` in `base.py` |
| No system prompt (merge into user turn) | `_NO_SYSTEM_PROMPT_FAMILIES` + `merge_system_into_user()` |
| Enforce `Think step by step.<think>\n` prefix when thinking on | `_THINK_PREFIX_FAMILIES` in providers |
| Suppress thinking via `<think>\n\n</think>\n` prefix when thinking off | Same |
| Tool call format: `<tool_call>…</tool_call>` JSON | Default `ToolCallFormat.JSON` (unlisted in `_TOOL_CALL_FORMAT`) |
| Tool-use encouragement nudge in system prompt | `_TOOL_ENCOURAGEMENT_FAMILIES` + `encourage_tool_use` param |

---

## Files Changed

| File | Change |
|---|---|
| `src/agent_engine/models/base.py` | `ModelFamily.DEEPSEEK`, `_FAMILY_DEFAULTS`, `_NO_SYSTEM_PROMPT_FAMILIES`, `_THINK_PREFIX_FAMILIES`, `_TOOL_ENCOURAGEMENT_FAMILIES`, `merge_system_into_user()` |
| `src/agent_engine/models/vllm_provider.py` | `_render_messages`: system→user merge + think prefix injection |
| `src/agent_engine/models/mlx_provider.py` | Same as vllm |
| `src/agent_engine/prompts/builder.py` | `encourage_tool_use` param on `build_system_prompt` and `_format_tool_schemas` |
| `scripts/run_experiment.py` | Import `_TOOL_ENCOURAGEMENT_FAMILIES`; pass `encourage_tool_use` to `build_system_prompt` |
| `scripts/generate_configs.py` | New models, variant grid, suites `deepseek-baseline` / `deepseek-agentflow` |

---

## 1. `src/agent_engine/models/base.py`

### Family defaults
```python
_FAMILY_DEFAULTS = {
    "deepseek": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    ...
}
```

### New constants (all DeepSeek-only)
```python
_NO_SYSTEM_PROMPT_FAMILIES   = frozenset({ModelFamily.DEEPSEEK})
_THINK_PREFIX_FAMILIES       = frozenset({ModelFamily.DEEPSEEK})
_TOOL_ENCOURAGEMENT_FAMILIES = frozenset({ModelFamily.DEEPSEEK})
```

### `merge_system_into_user(msgs)`
Helper that prepends the system message content to the first user turn, then removes the system
message. Used by both providers before calling `apply_chat_template`.

---

## 2. `src/agent_engine/models/vllm_provider.py` and `mlx_provider.py`

### `_render_messages` — two behaviours (DeepSeek only)

**System→user merge:**
```python
if self.config.family in _NO_SYSTEM_PROMPT_FAMILIES:
    msgs = merge_system_into_user(msgs)
```

**Think prefix injection (after `apply_chat_template`):**
```python
if self.config.family in _THINK_PREFIX_FAMILIES:
    if use_thinking and self.config.supports_thinking:
        rendered += "<think>\nThink step by step.\n"   # prime reasoning
    else:
        rendered += "<think>\n\n</think>\n"             # suppress reasoning
```

- `"Think step by step.\n"` is injected as the first tokens of every reasoning block.
  This applies to **both orchestrator and sub-agents** whenever their `use_thinking=True`.
- vLLM returns only newly generated tokens, so the output continues from after the injected prefix.
- MLX behaves identically: `mlx_lm.generate` returns only new tokens.
- `strip_thinking_tags` in the orchestrator loop handles the `</think>` closing tag in the output.

---

## 3. `src/agent_engine/prompts/builder.py`

`build_system_prompt` and `_format_tool_schemas` both accept `encourage_tool_use: bool = False`.

When `True`, the following paragraph is appended to the tool-schema section:
```
IMPORTANT: You are expected to use the tools above to answer the question.
If the question requires factual information, current data, or computation,
call a tool rather than guessing. Do not skip tool calls when they would improve your answer.
```

Default is `False` — fully backwards-compatible with all existing callers.

---

## 4. `scripts/run_experiment.py`

```python
from agent_engine.models.base import ModelFamily, get_tool_call_format, _TOOL_ENCOURAGEMENT_FAMILIES
...
system_prompt_for_config = prompt_builder.build_system_prompt(
    ...
    encourage_tool_use=(bool(tool_schemas) and orch_family in _TOOL_ENCOURAGEMENT_FAMILIES),
)
```

---

## 5. `scripts/generate_configs.py`

### Models
```python
"deepseek-7b":  {"family": "deepseek", "path_or_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  "tp": None, "gpus": 1}
"deepseek-32b": {"family": "deepseek", "path_or_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "tp": 2,    "gpus": 2}
```

### Variant grid (`VARIANTS_DEEPSEEK_ALL`)

Mirrors the Qwen3 grid exactly (4 thinking modes × tool configurations):

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
- `VARIANTS_DEEPSEEK_BASELINE` — all variants with `baseline=True` (8 variants × 7 datasets = 56 configs)
- `VARIANTS_DEEPSEEK_AGENTFLOW` — `ds7b_subagent_tools_*` only (4 variants × 7 datasets = 28 configs)

### Config output layout
```
experiments/configs/deepseek/baseline/<dataset>/<variant>.yaml
experiments/configs/deepseek/agentflow/<dataset>/<variant>.yaml
```

### CLI flags
```bash
python scripts/generate_configs.py --suite deepseek-baseline
python scripts/generate_configs.py --suite deepseek-agentflow
```

---

## 6. GPU sizing

`_is_large` in `vllm_provider.py` already matches `"32b"` → DeepSeek 32B automatically gets 2 GPUs
for tensor parallelism. No change needed.

---

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Thinking-mode variants | Full grid (NO / ORCHESTRATOR_ONLY / SUBAGENTS_ONLY / ALL) | Matches Qwen3 grid for fair comparison |
| Sub-agent model | Same model as orchestrator (self-as-sub-agent) | Consistent with OLMo approach; no mixing |
| System prompt merge | Inside provider `_render_messages`, transparent to orchestrator | Keeps orchestrator code clean |
| `encourage_tool_use` scope | Family-derived (`_TOOL_ENCOURAGEMENT_FAMILIES`), not per-YAML | Simpler; per-run ablations possible via code |
| Think step encouragement | `"Think step by step.\n"` injected in `<think>` prefix | Applied to both orchestrator and sub-agents whenever `use_thinking=True` |
