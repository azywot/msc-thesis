# DeepSeek R1 & OLMo 3 Integration — As-Built Record

Consolidated summary of every model-specific trick and integration hook needed
to run **DeepSeek-R1-Distill-Qwen-{7B,32B}** and **Olmo-3-{7B,32B}-{Instruct,Think}**
through CoSMAS (AgentFlow + baseline). Complements `docs/DS_integration_plan.md`
(DeepSeek-only deep dive) and is the canonical reference for future families.

---

## 1. Per-family requirements at a glance

| Requirement | DeepSeek R1 Distill | OLMo 3 Think | OLMo 3 Instruct |
|---|---|---|---|
| Sampling defaults | `T=0.6, top_p=0.95, max_tokens=32768` | `T=0.6, top_p=0.95, max_tokens=32768, top_k=-1, rep_pen=1.0` | same as Think |
| `supports_thinking` | always (forced `<think>` prefix) | always | off |
| System-role slot in template | **none** — merge into first user turn | present | present |
| Tool-call format | `JSON_SINGLE` → `{"tool_call": {...}}` | `PYTHONIC` → `<function_calls>\nfn(arg=val)\n</function_calls>` | same as Think |
| Tool-result role | `tool` (template supports it) | **no `tool` branch** — rename to `environment` | aliases `tool`→`environment` (rename is a no-op) |
| "No functions" suffix | n/a | template auto-appends if `functions` key missing — inject `functions=""` | not added to user-supplied system prompts (no-op) |
| Stop token on tool call | `<tool_response>` (hallucination guard) | `</function_calls>` | `</function_calls>` |
| Force-tool-call prefix | AgentFlow only | n/a | n/a |

All of these are driven from `src/agent_engine/models/base.py` frozensets and
applied inside `vllm_provider._render_messages` / `mlx_provider._render_messages`
so the orchestrator and prompt builder stay family-agnostic.

---

## 2. Wiring in `src/agent_engine/models/base.py`

```python
_FAMILY_DEFAULTS = {
    "deepseek":      {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    "olmo-think":    {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768,
                       "top_k": -1, "repetition_penalty": 1.0},
    "olmo-instruct": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768,
                       "top_k": -1, "repetition_penalty": 1.0},
    ...
}

_NO_SYSTEM_PROMPT_FAMILIES          = {DEEPSEEK}
_THINK_PREFIX_FAMILIES              = {DEEPSEEK}
_TOOL_ROLE_AS_ENVIRONMENT_FAMILIES  = {OLMO_THINK, OLMO_INSTRUCT}
_SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES = {OLMO_THINK}

_TOOL_CALL_FORMAT = {
    DEEPSEEK:      ToolCallFormat.JSON_SINGLE,
    OLMO_THINK:    ToolCallFormat.PYTHONIC,
    OLMO_INSTRUCT: ToolCallFormat.PYTHONIC,
}
```

### Message-preprocessing helpers

| Helper | Applied for | Purpose |
|---|---|---|
| `merge_system_into_user(msgs)` | DeepSeek | Prepends system content to first user turn (blank-line separator); template has no system slot. |
| `rewrite_tool_role_to_environment(msgs)` | OLMo Think + Instruct | Renames `role: tool` → `role: environment` (shallow-copy, no mutation). Without it OLMo Think silently drops tool results. |
| `suppress_no_functions_suffix(msgs)` | OLMo Think only | Injects `functions: ""` on the system message so the template doesn't append `"You do not currently have access to any functions."` (our prompt already documents every tool). |

---

## 3. Provider-level hooks (`vllm_provider.py` / `mlx_provider.py`)

### `_render_messages(msgs, use_thinking, force_tool_call)`

Applied in this order, guarded by the frozenset membership checks above:

1. `merge_system_into_user` — DeepSeek.
2. `rewrite_tool_role_to_environment` — OLMo 3.
3. `suppress_no_functions_suffix` — OLMo 3 Think.
4. `apply_chat_template(..., add_generation_prompt=True)`.
5. **Think-prefix injection** (DeepSeek only — three cases):

   | Condition | Suffix appended after the generation prompt |
   |---|---|
   | `force_tool_call=True` | `<think>\nI need to call a tool to answer this question.\n</think>\n<sub_goal>` |
   | `use_thinking=True`, no force | `<think>\n` |
   | otherwise | `<think>\n\n</think>\n` |

### Stop tokens

```python
_TOOL_CALL_STOP_TOKEN = {
    ToolCallFormat.JSON:        "</tool_call>",
    ToolCallFormat.PYTHONIC:    "</function_calls>",   # OLMo 3
    ToolCallFormat.JSON_SINGLE: "<tool_response>",     # DeepSeek (hallucination guard)
}
```

DeepSeek's JSON_SINGLE has no natural closing tag, so we stop on
`<tool_response>` — the marker the model tends to hallucinate after its own
tool call. The real tool result is always appended by the orchestrator, so
stopping there is safe and prevents fabricated tool responses from leaking
into subsequent turns (critical for baseline mode).

---

## 4. Parsing (`src/agent_engine/utils/parsing.py`)

`parse_tool_call` tries formats in order:

1. `<tool_call>{...}</tool_call>` — Qwen3/default (last wins).
2. `<function_calls>\nfn(arg=val)\n</function_calls>` — OLMo 3. Parser accepts
   **both** Python (`True/False/None`) and JSON (`true/false/null`) literals;
   if multiple calls are emitted only the first is dispatched (the
   orchestrator runs one tool per turn). Matches `--tool-call-parser olmo3`
   in vLLM and the Allen AI model cards.
3. `{"tool_call": {"name": ..., "arguments": {...}}}` — DeepSeek JSON_SINGLE
   (first wins).
4. Code-fenced JSON ``` ```…``` ``` — DS fallback.
5. Bare `{"name": ..., "arguments": {...}}` — DS fallback.

Stages 3–5 strip `<think>…</think>` blocks first so hallucinated calls inside
the reasoning trace don't get picked up.

---

## 5. Prompt builder (`src/agent_engine/prompts/builder.py`)

`tool_call_format` drives the surface syntax shown in the system prompt:

|  | JSON (Qwen3) | PYTHONIC (OLMo 3) | JSON_SINGLE (DeepSeek) |
|---|---|---|---|
| Open/close tags | `<tool_call>…</tool_call>` | `<function_calls>…</function_calls>` | *(none)* |
| Placeholder | `{"name": …, "arguments": …}` | `function_name(arg=value)` | `{"tool_call": {"name": …, "arguments": …}}` |

AgentFlow prompts additionally teach the `<sub_goal>…</sub_goal>` tag before
each tool call; baseline prompts omit it.

---

## 6. Orchestrator (`src/agent_engine/core/orchestrator.py`)

```python
self._force_tool_call = (
    bool(tool_registry)
    and not baseline
    and model_provider.config.family in _THINK_PREFIX_FAMILIES
)
```

Force-prefix is DeepSeek-only, AgentFlow-only. Disabled in baseline so the
pure-baseline contract isn't contaminated by an injected `<sub_goal>` tag the
baseline prompt never teaches. OLMo 3 does not need this hook — its pythonic
format and chat template already make tool calls the natural completion.

---

## 7. Config generation (`scripts/generate_configs.py`)

### Models

```python
"deepseek-7b":     family="deepseek",      tp=None, gpus=1
"deepseek-32b":    family="deepseek",      tp=2,    gpus=2
"olmo-7b-instruct":family="olmo-instruct", tp=None, gpus=1
"olmo-7b-think":   family="olmo-think",    tp=None, gpus=1
"olmo-32b-instruct":family="olmo-instruct",tp=2,    gpus=2
"olmo-32b-think":  family="olmo-think",    tp=2,    gpus=2
```

`_is_large` matches `"32b"` → all 32 B checkpoints get 2-GPU tensor parallelism
automatically.

### Suite layout

```
experiments/configs/deepseek/{baseline,agentflow}/<dataset>/<variant>.yaml
experiments/configs/olmo3/{think,instruct}/<dataset>/<variant>.yaml
```

### CLI

```bash
python scripts/generate_configs.py --suite deepseek-baseline
python scripts/generate_configs.py --suite deepseek-agentflow
python scripts/generate_configs.py --suite olmo3-think
python scripts/generate_configs.py --suite olmo3-instruct
```

---

## 8. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| DS tool-call format | `JSON_SINGLE` (bare `{"tool_call": {…}}`) | Matches DeepSeek R1 template's native single-call-per-turn contract. Parser accepts bare `{"name":…}` as safety net. |
| DS stop token | `<tool_response>` | No natural closing tag for `JSON_SINGLE`; prevents fabricated tool results past the model's own call. |
| DS force-prefix scope | AgentFlow only | Baseline prompt doesn't teach `<sub_goal>`; keeps pure-baseline comparison intact. |
| OLMo `tool`→`environment` rewrite | Applied to **both** Think and Instruct | Safety-first: Think silently drops `tool` messages; Instruct already aliases, so rewrite is a harmless no-op. |
| OLMo `functions=""` injection | Think only | Instruct's template doesn't add the suffix to user-supplied system messages — no injection needed. |
| OLMo sampling defaults | `top_k=-1, repetition_penalty=1.0` hard-coded | Model card specifies only `T=0.6, top_p=0.95` explicitly; setting the other two to no-op values locks generation to the authoritative recipe byte-for-byte. |
| System-prompt merge / role rewrite / suffix suppression | All inside provider `_render_messages` | Keeps orchestrator, prompt builder, and tools completely family-agnostic. |
| Thinking-mode grid | Full Qwen3 grid mirrored (`NO / ORCHESTRATOR_ONLY / SUBAGENTS_ONLY / ALL`) for DS 7B AF | Fair comparison across families. |
| Sub-agent model | Same checkpoint as orchestrator | Consistent with Qwen3 and OLMo configurations. |

---

## 9. Verification

- `tests/unit/test_base.py` covers:
  - `_FAMILY_DEFAULTS` for DeepSeek + both OLMo variants.
  - Frozenset membership (`_NO_SYSTEM_PROMPT_FAMILIES`, `_THINK_PREFIX_FAMILIES`,
    `_TOOL_ROLE_AS_ENVIRONMENT_FAMILIES`, `_SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES`).
  - `merge_system_into_user`, `rewrite_tool_role_to_environment`, and
    `suppress_no_functions_suffix` — including no-op paths and non-mutation of inputs.
- `tests/unit/test_parsing.py` covers DS JSON_SINGLE (with code-fence and bare-JSON
  fallbacks) and OLMo pythonic parsing (Python + JSON literals, parallel calls).
- `pytest tests/unit -q` → **129 passed**.

## 10. References

- DeepSeek R1 Distill — https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- OLMo 3 model cards — https://huggingface.co/allenai/Olmo-3-7B-Think, `.../Olmo-3-7B-Instruct`
- vLLM OLMo 3 tool parser — `--tool-call-parser olmo3` in vLLM ≥ 0.7.
