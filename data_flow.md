# Agent Engine — Prompt, State & Message Flow

This document describes the complete data flow: how messages are built, how
prompts are formatted, what each model sees, and how tool results return to the
orchestrator.

---

## Table of contents

1. [High-level picture](#1-high-level-picture)
2. [ExecutionState](#2-executionstate)
3. [Initial messages](#3-initial-messages)
4. [System prompt format](#4-system-prompt-format)
5. [Prompt rendering](#5-prompt-rendering-apply_chat_template)
6. [Orchestrator loop](#6-orchestrator-loop-per-turn)
7. [Model output format](#7-model-output-format-qwen3)
8. [Tool call format and injection](#8-tool-call-format-and-injection)
9. [Sub-agent: web_search](#9-sub-agent-web_search)
10. [Sub-agent: code_generator](#10-sub-agent-code_generator)
11. [Sub-agent: text_inspector](#11-sub-agent-text_inspector)
12. [Sub-agent: image_inspector](#12-sub-agent-image_inspector)
13. [Tool: context_manager](#13-tool-context_manager)
14. [Message accumulation across turns](#14-message-accumulation-across-turns)
15. [Token usage tracking](#15-token-usage-tracking)
16. [Results stored in raw_results.json](#16-results-stored-in-raw_resultsjson)
17. [GPU / model assignment summary](#17-gpu--model-assignment-summary)
18. [Experiment runner flow](#18-experiment-runner-flow)
19. [Reasoning context module](#19-reasoning-context-module)

---

## 1. High-level picture

### Batch path (experiments)

```
run_experiment.py
  ├─ load_experiment_config(config_path)
  ├─ DatasetRegistry.get(config.dataset)  → BaseDataset
  ├─ dataset.get_subset(subset_num)  → List[DatasetExample]
  ├─ PromptBuilder.build_system_prompt(dataset_name, tool_schemas, ...)
  ├─ setup_tools(config, cache_manager, ...)  → ToolRegistry
  ├─ setup_model_provider(...)  → orchestrator model (cached)
  └─ AgenticOrchestrator.run_batch(questions, question_ids, system_prompts, attachments)
       ├─ [Turn 1..N] _process_batch_turn(active_states)
       │     ├─ apply_chat_template(state.messages)  →  JSON payload
       │     ├─ VLLMProvider.generate(prompts)  →  GenerationResult.text
       │     ├─ parse_tool_call(text)
       │     └─ _classify_tool_call → dispatch:
       │           ├─ web_search (sub-agent)  → _schedule_web_job → _flush_web_batch
       │           │     ├─ search_and_format(query)
       │           │     ├─ _fetch_urls_for_web_jobs (Serper only)
       │           │     ├─ _format_results(results)
       │           │     └─ _run_web_analysis_batch: LLM.generate(analysis_prompt)
       │           ├─ code_generator (sub-agent)  → _schedule_code_job → _flush_code_batch
       │           │     └─ _run_code_generation_batch: LLM.generate(task_prompt) → execute_code()
       │           ├─ text_inspector  → _execute_tool (immediate)
       │           ├─ image_inspector  → _execute_tool (immediate)
       │           └─ context_manager  → _execute_tool (immediate)
       └─ extract_answer(state.current_output)
```

### Single path (examples)

```
orchestrator.run(question, question_id, system_prompt, attachments)
  └─ [Turn 1..N] loop:
        ├─ apply_chat_template(state.messages) → generate([prompt])
        ├─ parse_tool_call(gen_result.text)
        └─ _execute_tool(tool_call, state)  → tool.execute(**arguments)
              (all tools go through _execute_tool; web_search/code_generator
               call their own model_provider.generate internally in sub-agent mode)
```

---

## 2. ExecutionState

`src/agent_engine/core/state.py`

```python
@dataclass
class ExecutionState:
    question_id: int
    question:    str
    attachments: Optional[List[str]]   # file paths (prefer absolute) for tools

    # Conversation history — the only mutable part that grows each turn
    messages: List[Dict[str, str]]     # [{"role": "system"|"user"|"assistant"|"tool", "content": str}]
    current_output: str                # last raw model output (including <think> tags)

    # Execution tracking
    turn:     int                      # current turn number (starts at 0)
    finished: bool
    answer:   Optional[str]            # final extracted answer

    # Tool usage tracking
    tool_calls:  List[Dict]            # all tool_calls made (name + arguments)
    tool_counts: Dict[str, int]        # per-tool call count

    metadata: Dict[str, Any]           # error, max_turns_reached, token_usage, etc.
```

`messages` starts as `[system, user]` and grows by two entries per turn:
`assistant` (model output) then `tool` (tool response).

---

## 3. Initial messages

Built by `_build_initial_messages(question, system_prompt, attachments)`:

```
messages = [
  {"role": "system", "content": <system_prompt>},
  {"role": "user",   "content": <question> [+ attachment note]},
]
```

**Attachment note** (appended to user message when a file is attached):

```
[Attachment]
- There is an attached file for this question: <filename>
- To inspect the image, call the tool `image_inspector` ...   # for images
  OR
- To read the file, call the tool `text_inspector` ...        # for text files
- Important: do NOT guess or provide file paths; inspectors use the attached file automatically.
```

---

## 4. System prompt format

Built by `PromptBuilder.build_system_prompt(dataset_name, tool_schemas, max_search_limit, direct_tool_call)`.

Templates loaded from `src/agent_engine/prompts/templates/system/<name>.yaml`.
GAIA and HLE share the `gaia` template.

Sections are joined with `\n\n`:

```
<base_instruction>          # from YAML template (gaia / gpqa / hle / base)

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "web_search", ...}}
{"type": "function", "function": {"name": "code_generator", ...}}
...
</tools>

For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Important: ...

<example>                   # from YAML template (search+code, search-only, etc.)

<final_instructions>        # from YAML template
```

---

## 5. Prompt rendering (`apply_chat_template`)

### VLLMProvider

`apply_chat_template(messages, use_thinking)` does **not** call the tokenizer
directly. It returns a JSON envelope:

```json
{"messages": [...], "use_thinking": true|false}
```

Inside `_generate_text`, this envelope is unpacked and the tokenizer template
is applied:

```python
# Qwen3 with thinking=True
tokenizer.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=True, enable_thinking=True)
# Qwen3 without thinking / all other models
tokenizer.apply_chat_template(msgs, tokenize=False,
                               add_generation_prompt=True, enable_thinking=False)
```

The rendered string is tokenized, truncated if `prompt_tokens > max_model_len`,
then passed to `llm.generate()`.

---

## 6. Orchestrator loop (per turn)

### Batch path (`_process_batch_turn`)

```
for each active state s:
    s.turn += 1
    prompt = apply_chat_template(s.messages, use_thinking)
    gen_result = model.generate([prompt])[0]
    s.current_output = gen_result.text          # may include <think>...</think>

    tool_call = parse_tool_call(gen_result.text)

    if tool_call:
        s.add_message("assistant", gen_result.text)
        _classify_tool_call(s, tool_call, gen_result.text, web_jobs, code_jobs, immediate_results)
    else:
        s.add_message("assistant", gen_result.text)
        s.finished = True
        s.answer = extract_answer(gen_result.text)

# After all states processed:
_apply_immediate_results(immediate_results)
_flush_web_batch(web_jobs)
_flush_code_batch(code_jobs)
```

### Single path (`run`)

```
while state.turn < max_turns and not state.finished:
    state.turn += 1
    prompt = apply_chat_template(state.messages, use_thinking)
    gen_result = model.generate([prompt])[0]
    state.current_output = gen_result.text

    tool_call = parse_tool_call(gen_result.text)

    if tool_call:
        state.add_message("assistant", gen_result.text)
        _index_reasoning_in_context_manager(gen_result.text, tool_call["name"], state)
        tool_result = _execute_tool(tool_call, state)
        state.add_message("tool", "<tool_response>\n{output}\n</tool_response>")
        state.tool_calls.append(tool_call)
        state.increment_tool_count(tool_call["name"])
    else:
        state.add_message("assistant", gen_result.text)
        state.finished = True
        state.answer = extract_answer(gen_result.text)
```

Tool results are committed with `strip_thinking_tags` applied to the output.

---

## 7. Model output format (Qwen3)

The model emits one of two patterns per turn:

**Tool call turn:**
```
<think>
[Extended reasoning, stripped before storing to sub-agents and results]
</think>
I need to search for ...
<tool_call>
{"name": "web_search", "arguments": {"query": "..."}}
</tool_call>
```

**Final answer turn:**
```
<think>
[Extended reasoning]
</think>
Based on the search results, the answer is \boxed{42}.
```

Parsing:
- `parse_tool_call` extracts the **last** `<tool_call>…</tool_call>` block.
- `extract_answer` looks for `\boxed{…}`, `Final Answer: …`, `Answer: …`, or `The answer is …`.
- `strip_thinking_tags` removes `<think>…</think>` before any text is shown to a sub-agent or stored in tool responses.

---

## 8. Tool call format and injection

The orchestrator passes to each tool:

```python
tool.execute(**arguments)
```

where `arguments` is built from `tool_call["arguments"]` and then modified by two injection steps.

### 8.1 Attachment path injection (`_inject_attachment_path`)

For `image_inspector` and `text_inspector`, the orchestrator injects
`full_file_path` from `state.attachments` before calling `execute`:

- **image_inspector**: first attachment with `.jpg`, `.jpeg`, `.png`
- **text_inspector**: first attachment with `.txt`, `.md`, `.log`, `.json`, `.jsonl`, `.xml`, `.csv`, `.tsv`, `.yaml`, `.yml`, `.docx`, `.xlsx`, `.jsonld`, `.parquet`, `.pdf`, `.pdb`, `.pptx`, `.py`

If no matching attachment exists, returns an error and does not call the tool.

### 8.2 Reasoning context injection (`_inject_reasoning_context`)

For `web_search` (sub-agent mode) and `code_generator` (sub-agent mode):

- **web_search**: injects `prev_reasoning` = `get_reasoning_context_for_state(state)`
- **code_generator**: injects `context` = reasoning context + attachment context (path always when text attachment exists; `<FILE_CONTENT>` when `text_inspector` was called)

See [§19. Reasoning context module](#19-reasoning-context-module).

### 8.3 Tool routing (batch path)

| Tool            | Condition                          | Path                    |
|-----------------|------------------------------------|-------------------------|
| web_search      | sub-agent mode, has build_analysis_prompt | _schedule_web_job → _flush_web_batch |
| code_generator  | sub-agent mode, has build_task_prompt     | _schedule_code_job → _flush_code_batch |
| context_manager| always                             | _execute_tool (immediate) |
| text_inspector  | always                             | _execute_tool (immediate) |
| image_inspector | always                             | _execute_tool (immediate) |

---

## 9. Sub-agent: `web_search`

**Direct mode** (`model_provider=None`): returns formatted results string directly, no LLM call.

**Sub-agent mode** (batched path, used in experiments):

### Step A — search_and_format (before batched URL fetch)

```
SerperRM.forward(query)  →  [{title, url, content (snippet)}, ...]
  OR
TavilyRM.forward(query)  →  [{title, url, content (cleaned text)}, ...]
```

Returns payload:
```python
{
  "results": [...],
  "urls_to_fetch": ["https://..."],   # Serper only, uncached
  "url_snippets":  {"url": "snippet"},
  "cached": bool,
  "query": str,
}
```

### Step B — batch URL fetch (Serper only)

`fetch_page_content(urls, snippets)` fetches all URLs across all pending web
jobs in one batch using `ThreadPoolExecutor`. Content is stored in
`tool.url_cache[url]`. Tavily does not require URL fetching (content in results).

### Step C — _format_results

Produces a string of web page blocks:

```
**Web Page 1:**
{
  "title": "...",
  "url": "https://...",
  "content": "...up to max_doc_len chars extracted around best-matching sentence..."
}
**Web Page 2:**
...
```

For Serper, `content` is extracted from the full cached page via
`extract_snippet_with_context` (F1-based sentence matching, returns
`context_chars=max_doc_len` chars around the best-matching sentence).
For Tavily, `content` is the raw Tavily content field.

### Step D — sub-agent LLM call

Prompt built by `build_analysis_prompt(query, formatted_results, prev_reasoning)`:

```
**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following
inputs: Previous Reasoning Steps, Current Search Query, and Searched Web
Pages. Your objective is to extract relevant and helpful information ...

**Inputs:**
- **Previous Reasoning Steps:**
  <truncated reasoning from state.messages — see §19>

- **Current Search Query:**
  <query>

- **Searched Web Pages:**
  <formatted_results from Step C>

Now you should analyze each web page ...
```

Messages: `[{"role": "user", "content": <above>}]`

Output format expected:
```
**Final Information**

[Helpful information extracted from web pages]
```
or
```
**Final Information**

No helpful information found.
```

`<think>…</think>` is stripped from the output before it is returned.

The cleaned text is stored in `analysis_cache[query]` (per-tool in-memory) and
returned as `ToolResult.output`, which the orchestrator wraps:

```
<tool_response>
**Final Information**
[summary text]
</tool_response>
```

---

## 10. Sub-agent: `code_generator`

**Direct mode**: `execute(code=...)` — runs the Python code directly, no LLM call.

**Sub-agent mode** (batched path):

Prompt built by `build_task_prompt(task, context)`:

```
You are a code generator. Generate ONLY executable Python code, with NO
explanations, NO comments about what the code does, and NO additional text.

Context:
<truncated reasoning from state.messages — see §19>
<optional: [ATTACHED_FILE_PATH] {path} when text attachment exists; <FILE_CONTENT> when text_inspector was called>

Problem: <task description>

Requirements:
- Output ONLY the Python code
- The code must be executable as a standalone script
- The code must print its output directly
- ...

Python code:
```

Messages: `[{"role": "user", "content": <above>}]`

The LLM output is stripped of `<think>` tags and markdown fences, then executed
in a sandboxed subprocess with `timeout_seconds=60`. `stdout` + `stderr` is
returned as `ToolResult.output`.

**Attachment context** (MAT-style): `get_attachment_context_for_code(state)` appends:
- `[ATTACHED_FILE_PATH] {path}` — always when a text attachment exists (full absolute path via `os.path.abspath`), so the code generator knows where to read the file even if `text_inspector` was not called
- `<FILE_CONTENT>` — only when `text_inspector` was called, with its response (truncated to 4000 chars)

---

## 11. Sub-agent: `text_inspector`

**Direct mode**: returns raw file content (no LLM call).

**Sub-agent mode with question**:

Messages:
```python
[
  {"role": "system", "content":
      "You are given the content of a plain-text file attached to the user's question. "
      "Answer the question using only the file content. If the file does not contain "
      "the answer, say so."},
  {"role": "user", "content":
      f"File content:\n\n{file_content}\n\nQuestion:\n{question}\n"},
]
```

`<think>` stripped from output before returning.

Supported formats: `.txt`, `.md`, `.log`, `.json`, `.jsonl`, `.xml`, `.csv`,
`.tsv`, `.yaml`, `.yml`, `.docx`, `.xlsx`, `.jsonld`, `.parquet`, `.pdf`,
`.pdb`, `.pptx`, `.py`.

---

## 12. Sub-agent: `image_inspector`

Always requires a model provider (VLM). Multimodal prompt:

```python
[
  {"role": "system", "content":
      "You are given an image attached to the user's question. "
      "Answer the question using only the image content. "
      "If the image does not contain enough information, say so."},
  {"role": "user", "content": [
      {"type": "image"},
      {"type": "text", "text": f"Question:\n{question}\n"},
  ]},
]
```

`apply_chat_template` is called on these multimodal messages. The generate
call passes both the rendered prompt and `multi_modal_data={"image": pil_image}`
to vLLM.

---

## 13. Tool: `context_manager`

**Non-direct mode (sub-agent)**:

- Pre-tool reasoning from the orchestrator is indexed into a GraphRAG knowledge
  base before each `web_search`, `code_generator`, or `context_manager` call.
- Tool call arguments: `{"query": str}`.
- Returns `ToolResult.output` = retrieved context passages (up to 2000 chars).
- No LLM call inside the tool itself; GraphRAG handles retrieval.

**Direct mode**: persistent text file with `op: write|read` interface.

### How reasoning gets into the knowledge base (non-direct mode)

Before executing any `web_search`, `code_generator`, or `context_manager` call,
the orchestrator calls `_index_reasoning_in_context_manager`. It strips the
`<tool_call>…</tool_call>` block from the model output and feeds the remaining
reasoning text into `ContextManagerTool.add_entry(reasoning, question_id)`.
This means the GraphRAG graph is built incrementally from the model's own chain
of thought, so a later `context_manager` query can retrieve conclusions reached
in earlier turns.

---

## 14. Message accumulation across turns

After turn *k* the messages list looks like:

```
[
  {"role": "system",    "content": "<system_prompt>"},
  {"role": "user",      "content": "<question> [+ attachment note]"},

  # Turn 1
  {"role": "assistant", "content": "<think>...</think>\nI need to search...\n<tool_call>...</tool_call>"},
  {"role": "tool",      "content": "<tool_response>\n**Final Information**\n...\n</tool_response>"},

  # Turn 2
  {"role": "assistant", "content": "<think>...</think>\nBased on...\n<tool_call>...</tool_call>"},
  {"role": "tool",      "content": "<tool_response>\n...code output...\n</tool_response>"},

  # Final turn
  {"role": "assistant", "content": "<think>...</think>\nThe answer is \\boxed{42}."},
]
```

Notes:
- `<think>…</think>` tags **are kept** in `assistant` messages stored in
  `state.messages` (so subsequent turns can see the full reasoning).
- `<think>…</think>` tags **are stripped** (`strip_thinking_tags`) from tool
  responses before they are stored in `tool` messages and before any text is
  shown to a sub-agent.
- The full `state.messages` list is what gets passed to `apply_chat_template`
  for each subsequent turn — the model always sees its full history.

---

## 15. Token usage tracking

Token counts are accumulated via `_accumulate_usage(state, gen_result.usage)` in
`orchestrator.py` after every `model.generate()` call. The function writes into
`state.metadata["token_usage"]` — a plain dict that exists entirely within
the already-present `metadata` field of `ExecutionState`.

### Which generation calls are tracked

| Call site | What it tracks |
|-----------|----------------|
| `run()` — single-turn loop | Each orchestrator turn (prompt + completion) |
| `run()` — tool execution | Any tool that returns `ToolResult(usage=...)` (web_search, text_inspector, image_inspector in single-run) |
| `_process_batch_turn()` — batched loop | All orchestrator turns in a batch |
| `_apply_immediate_results()` | Tools returning `ToolResult.usage` (text_inspector, image_inspector, context_manager, web_search in direct/single path) |
| `_run_web_analysis_batch()` | Web-search sub-agent LLM analysis call |
| `_run_code_generation_batch()` | Code-generator sub-agent LLM call |

**ToolResult.usage**: Tools that call `model_provider.generate()` internally
(text_inspector, image_inspector, web_search in single-run) return
`ToolResult(..., usage=result.usage)`. The orchestrator accumulates this in
`_apply_immediate_results()` (batched path) or immediately after `_execute_tool()`
(single-run path).

### Accumulator structure

```python
state.metadata["token_usage"] = {
    "prompt_tokens":     int,   # cumulative across all tracked calls
    "completion_tokens": int,
    "total_tokens":      int,
}
```

Counts start at zero for every new question and grow turn by turn.

### Where token usage is stored and computed

| Location | What is stored | How it is produced |
|----------|----------------|--------------------|
| **In-memory (runtime)** | `state.metadata["token_usage"]` | `_accumulate_usage()` adds each `gen_result.usage` (or `ToolResult.usage`) into this dict. One per `ExecutionState` (per question). |
| **raw_results.json** | `"token_usage": {...}` in each result record | Copied from `state.metadata.get("token_usage", {})` when building the record after each question completes. |
| **metrics.json** | `overall.token_usage` | Sum of `token_usage` from all result records. Computed in `_compute_metrics()`. |
| **metrics.json** | `per_level[level].token_usage` | Same sum, but only over results for that level (e.g. GAIA level 2, 3). |

---

## 16. Results stored in raw_results.json

After processing, each example produces one record:

```python
{
  "question_id":  int,
  "question":     str,
  "prediction":   str,             # extracted answer
  "ground_truth": str,
  "correct":      bool,
  "evaluation":   dict,            # dataset-specific eval result
  "output_text":  str,             # all assistant + tool messages joined by \n
  "tool_calls":   [                # one record per tool call
    {
      "name":      str,
      "arguments": dict,
      "response":  str,            # content of <tool_response>...</tool_response>
    }
  ],
  "turns":       int,
  "tool_counts": {"web_search": int, ...},
  "token_usage": {
      "prompt_tokens":     int,
      "completion_tokens": int,
      "total_tokens":      int,
  },
  "metadata":    dict,             # from dataset example (level, file_name, etc.)
}
```

`output_text` is produced by `_state_to_output_text`: concatenates `.content`
of all `assistant` and `tool` messages (excluding `system` and `user`).

---

## 17. GPU / model assignment summary

| Role              | Model (test config)      | Thinking | Notes                             |
|-------------------|--------------------------|----------|-----------------------------------|
| orchestrator      | Qwen3-4B                 | Yes      | Drives the main loop              |
| web_search        | Qwen3-4B (shared)        | No       | Analyzes fetched pages            |
| code_generator    | Qwen3-4B (shared)        | No       | Generates Python code             |
| text_inspector    | Qwen3-4B (shared)        | No       | Reads/answers about text files    |
| context_manager   | Qwen3-4B (shared)        | No       | GraphRAG entity extraction        |
| image_inspector   | Qwen2.5-VL-3B-Instruct   | No       | Multimodal VLM                    |

Model instances sharing the same `path_or_id` are reused (single vLLM engine
for all Qwen3-4B roles). GPU assignment is computed by
`resolve_gpu_assignments`: each distinct model path gets its own GPU(s) with
memory utilization = `0.9 / num_distinct_models_on_same_gpu`.

---

## 18. Experiment runner flow

`scripts/run_experiment.py`

### Input

- Config file (YAML) via `--config`
- Dataset: `dataset.name`, `dataset.split`, `dataset.subset_num`

### Flow

1. **Load config** — `load_experiment_config(config_path)`
2. **Load dataset** — `DatasetRegistry.load(config.dataset)` → `List[DatasetExample]`
3. **Build system prompt** — `PromptBuilder.build_system_prompt(dataset_name, tool_schemas, max_search_limit, direct_tool_call)`
4. **Setup tools** — `setup_tools(config, cache_manager, api_keys, model_providers, ...)` → `ToolRegistry`
5. **Setup orchestrator model** — `setup_model_provider(orchestrator_config)` (cached)
6. **Create orchestrator** — `AgenticOrchestrator(model_provider, tools, max_turns, tool_limits, use_thinking, cache_manager)`
7. **Process in batches** — `orchestrator.run_batch(questions, question_ids, system_prompts, attachments)`
8. **Evaluate** — `dataset.evaluate(prediction, ground_truth, metadata)` per example
9. **Write outputs** — `raw_results.json`, `metrics.json`, `config.json`

### Run directory

`{output_dir}/{split}_{YYYY-MM-DD-HH-MM-SS}_{job_id}/`

- `raw_results.json` — per-example results
- `raw_results.partial.json` — intermediate flush (every 10 batches)
- `metrics.json` — aggregated accuracy, token usage, per-level stats
- `config.json` — experiment config + system prompt

### Partial flush

Every 10 examples (when `(base_idx + len(batch)) % 10 == 0`), `raw_results.partial.json` is written and `cache_manager.save_caches()` is called.

---

## 19. Reasoning context module

`src/agent_engine/utils/reasoning_context.py`

Provides previous reasoning to `web_search` and `code_generator` sub-agents (MAT-style).

### Functions

| Function | Purpose |
|----------|---------|
| `get_accumulated_output_from_state(state)` | Concatenate all assistant + tool messages |
| `extract_reasoning_context(steps, mind_map, tool_markers)` | Truncate or summarize reasoning |
| `get_reasoning_context_for_state(state, mind_map)` | Convenience: extract from state |
| `get_attachment_context_for_code(state)` | `[ATTACHED_FILE_PATH] {path}` always when text attachment; `<FILE_CONTENT>` when text_inspector was called (MAT-style) |

### Truncation logic (no mind_map)

1. Build accumulated output from `state.messages` (assistant + tool)
2. Split by newlines into steps
3. Keep: first step, last 4 steps, steps containing `<tool_call>` or `<tool_response>`
4. Replace middle steps with `...`

### Mind map (optional)

If `mind_map` (GraphRAG) is provided and content ≥ 100 chars, query:
`"Summarize the reasoning process, be short and clear. Keep the summary under 500 words."`
Result truncated to 2000 chars.

### Attachment context for code (MAT-style)

- **Path**: always when a text attachment exists (`.txt`, `.md`, `.json`, `.csv`, `.py`, `.yaml`, `.yml`, `.jsonl`, `.xml`, `.log`) — so the code generator knows where to read the file even if `text_inspector` was not called
- **Format**: `[ATTACHED_FILE_PATH] {path}` (space after bracket, MAT format)
- **Path resolution**: full absolute path via `os.path.abspath(p)`
- **Content**: only when `text_inspector` was called — last tool response, truncated to 4000 chars, wrapped in `<FILE_CONTENT>`

### Injection points

- **Single run**: `_inject_reasoning_context` in `_execute_tool` (before `tool.execute(**arguments)`)
- **Batch run**: `get_reasoning_context_for_state(job.state)` passed to `build_analysis_prompt` / `build_task_prompt` in `_run_web_analysis_batch` and `_schedule_code_job`
