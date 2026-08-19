# Add a model family

Onboarding a family is mostly a matter of answering questions about its chat
template and its tool-call syntax, then recording each answer in the right
table. All the tables are in `src/agent_engine/models/base.py`, and each is a
`frozenset` keyed by `ModelFamily`, so a family opts in to a quirk by appearing
in a set - there is no per-family `if` chain anywhere.

The families already handled - Qwen3, DeepSeek-R1-Distill, OLMo 3 Think and
Instruct - between them cover most of the ways a template can misbehave, so the
existing entries double as worked examples.

---

## 1. Add the enum member

```python
class ModelFamily(Enum):
    ...
    MY_FAMILY = "my-family"
```

The string is what a config's `family:` key must say. Adding the member is
enough to make `family: my-family` parse; everything below is about making it
behave.

## 2. Answer six questions

### Does it think?

```python
_THINKING_FAMILIES = frozenset({QWEN3, QWQ, DEEPSEEK, OLMO_THINK})
```

Membership sets `supports_thinking` when the config leaves it unset, which in
turn gates `thinking_mode`.

### How is thinking switched on?

There are two mechanisms and a family uses exactly one:

- **A template kwarg** - `_ENABLE_THINKING_KWARG_FAMILIES` (Qwen3, QwQ). The
  provider passes `enable_thinking=True/False` to the chat template.
- **Prefix forcing** - `_THINK_PREFIX_FAMILIES` (DeepSeek). The provider appends
  `"<think>\n"` to prime reasoning, or `"<think>\n\n</think>\n"` to suppress it.

A family that always thinks and offers no switch (OLMo 3 Think) appears in
neither set.

### Does the template have a system role?

```python
_NO_SYSTEM_PROMPT_FAMILIES = frozenset({DEEPSEEK})
```

Listed families get the system message merged into the first user turn
(`merge_system_into_user`). Without this the system prompt is dropped and the
model never learns its tools exist.

### Does the template understand `role: tool`?

```python
_TOOL_ROLE_AS_ENVIRONMENT_FAMILIES = frozenset({OLMO_THINK, OLMO_INSTRUCT})
```

The orchestrator emits `role: tool` turns. OLMo 3 Think's template has no `tool`
branch and **silently drops them** - the model simply never sees any tool
output, and nothing errors. Listed families get those turns rewritten to
`role: environment` before rendering.

This is the failure mode most worth checking by hand on a new family, because
it produces a plausible-looking run in which the agent appears to ignore
everything its tools return.

### Does the template inject anything unwanted?

```python
_SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES = frozenset({OLMO_THINK})
```

OLMo 3 Think appends *"You do not currently have access to any functions."* to
any system message lacking a `functions` key - directly contradicting our system
prompt, which lists the tools. Listed families get `functions=""` injected to
neutralise it.

The general lesson: **render one prompt and read it** before trusting a new
family. Templates add text you did not write.

### What tool-call syntax does it emit?

```python
_TOOL_CALL_FORMAT: Dict[ModelFamily, ToolCallFormat] = {
    OLMO_THINK:    ToolCallFormat.PYTHONIC,
    OLMO_INSTRUCT: ToolCallFormat.PYTHONIC,
    DEEPSEEK:      ToolCallFormat.JSON_SINGLE,
}
```

> **This table is sparse on purpose.** Unlisted families default to
> `ToolCallFormat.JSON`, so a family that emits standard
> `<tool_call>{...}</tool_call>` needs no entry at all. Do not "complete" the
> table - `get_tool_call_format(family)` is the accessor, and it resolves every
> family, listed or not.

The three formats:

| Format | Syntax |
|---|---|
| `JSON` | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` - last one wins |
| `PYTHONIC` | `<function_calls>\ntool(arg=val)\n</function_calls>` - newline-delimited |
| `JSON_SINGLE` | `{"tool_call": {"name": ..., "arguments": {...}}}` - no XML wrapper, first occurrence wins |

The format also feeds `PromptBuilder`, which documents the expected syntax in
the system prompt, so the model is *told* to emit what the parser expects.

## 3. Sampling defaults

If the model card specifies sampling parameters, record them rather than
inheriting the Qwen-shaped defaults:

```python
_FAMILY_DEFAULTS: ClassVar[Dict[str, Dict[str, Any]]] = {
    "deepseek":      {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    "olmo-think":    {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768,
                      "top_k": -1, "repetition_penalty": 1.0},
    ...
}
```

These are applied by a `mode="before"` validator using `setdefault`, so an
explicit value in the YAML always wins. `top_k: -1` disables top-k in vLLM and
`repetition_penalty: 1.0` is the no-op - that is how you express "the card does
not set this".

## 4. Teach the parser, if needed

Only if the family emits a syntax none of the three formats covers. Add a branch
to `parse_tool_call` in `src/agent_engine/utils/parsing.py`, which tries formats
in a fixed order and returns the first success.

Be careful where you put it: the order is a precedence decision. The DeepSeek
branch strips thinking tags *before* matching, specifically so a tool call
hallucinated inside a `<think>` block is not mistaken for a real one.

## 5. Check it before spending a night of GPU time

```python
from agent_engine.models.base import ModelConfig, ModelFamily, get_tool_call_format

cfg = ModelConfig(name="x", family=ModelFamily.MY_FAMILY,
                  path_or_id="org/model", role="orchestrator")
print(cfg.supports_thinking, cfg.temperature, get_tool_call_format(cfg.family))
```

Then render one real prompt through the provider and **read it**:

```python
provider._render_messages([
    {"role": "system", "content": "SYSTEM PROMPT"},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "<tool_call>...</tool_call>"},
    {"role": "tool", "tool_name": "web_search", "content": "TOOL OUTPUT"},
])
```

Confirm three things in the output: the system prompt is present, `TOOL OUTPUT`
is present, and no text you did not write has been appended. Those correspond to
the three failure modes above, and each one produces a run that *looks* fine.

Run the family tables' own tests:

```bash
pytest tests/unit/test_base.py tests/unit/test_parsing.py -q
```

`test_family_tables_contain_only_real_families` will catch a stale entry left
behind by a rename.

## 6. Add configs

Families get their own config directory - `experiments/configs/qwen3/`,
`deepseek/`, `olmo3/`. If the family should appear across a whole suite, add it
to `scripts/generate_configs.py` rather than hand-writing files; see
[configuration.md](../configuration.md#where-configs-live).

---

## Checklist

| Question | Table |
|---|---|
| Does it think? | `_THINKING_FAMILIES` |
| Thinking via kwarg? | `_ENABLE_THINKING_KWARG_FAMILIES` |
| Thinking via prefix? | `_THINK_PREFIX_FAMILIES` |
| No system role? | `_NO_SYSTEM_PROMPT_FAMILIES` |
| Drops `role: tool`? | `_TOOL_ROLE_AS_ENVIRONMENT_FAMILIES` |
| Injects unwanted text? | `_SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES` |
| Non-JSON tool calls? | `_TOOL_CALL_FORMAT` (leave out if JSON) |
| Card-specified sampling? | `ModelConfig._FAMILY_DEFAULTS` |
