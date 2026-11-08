# RSI Phase 1: Prompt Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone, resumable MASS-style prompt search in which a reflector model (Qwen3-8B by default, Qwen3-32B as control) improves the CoSMAS prompt bundle (orchestrator parts plus web_search and code_generator sub-agent prompts) per domain, with base Qwen3-8B in every executor role, and writes an inspectable report. No collection, no training.

**Architecture:** A prompt bundle is plain data (`src/rsi/bundle.py`) rendered through small, seed-preserving override hooks added to `PromptBuilder`, the two sub-agent tools and the orchestrator. The search loop (`src/rsi/search.py`) runs the champion bundle, asks the reflector for K one-unit replacements (8B: the executor's own in-process vLLM engine; 32B control: a separate OpenAI-compatible vLLM server), validates them (placeholders, example, schema, leakage), screens them on unseen questions and keeps a new champion only on a strict held-out improvement. Every round is written to disk, so a killed job resumes and a markdown report can be regenerated at any time.

**Tech Stack:** Python 3.11, pydantic, vLLM (in-process executor + `vllm serve` reflector), `openai` client, pandas/parquet, pytest, SLURM on Snellius.

**Spec:** `docs/superpowers/specs/2026-11-04-recursive-self-improvement-design.md` (sections "The prompt bundle", "Leakage guards", "Data", "The search loop", "Reflector input", "Phases / Phase 1", "Thinking mode").

## Global Constraints

- Python 3.11+, `black` line length 100, `isort` black profile; no type annotations required.
- Tests run from the repo root with `pytest`; new tests live under `tests/rsi/` (plus the touched `tests/unit/` files).
- **Never run `git commit`.** Each task ends by staging its files with `git add`; the author reviews and commits.
- No em dashes in any prose (docstrings, comments, report text, docs).
- Seed byte-identity: with no overrides (or the seed bundle), every rendered prompt (system prompt, planning user turn, memory user turns, web_search and code_generator sub-agent prompts) is byte-identical to today's. `tests/characterization/test_orchestrator_trace.py` must stay green **without** `--update-fixtures`.
- Data: only `data/training/train/combined_train.parquet` (1,800 rows: hotpotqa 765, nq 135, deepmath 900). Domains: `search` = hotpotqa + nq rows, rendered with the `gaia` template; `math` = deepmath rows, rendered with the `deepmath` template (`collect_sft_data.py:_data_source_to_prompt_dataset`). No benchmark data anywhere in the loop.
- Executor defaults (match the existing collection setup): tools `web_search` + `code_generator`, `direct_tool_call: false`, `max_turns: 7`, `thinking_mode: "NO"`, Qwen3-8B in every role (no `models.<tool>` entries, so the runner's fallback gives every tool the orchestrator model).
- Search defaults: 10 rounds, K = 3 candidates per round, shown minibatch 16, screen minibatch 16, held-out batch 50 per domain, leakage n-gram length 8, reflector input budget 14,000 tokens and 16,384 output tokens (fits the executor's unchanged `max_model_len` of 32,768), reflector thinking on, T 0.6 / top_p 0.95 / top_k 20.
- Serving (decided): for the homogeneous 8B run, **one** vLLM engine serves every role, reflector included. The reflector calls the executor's in-process `VLLMProvider.llm` with its own `SamplingParams` (per-candidate seed, reflector temperature and max tokens, no tool-call stop token), so the executor's settings are untouched. Only the 32B-reflector control uses a separate `vllm serve`. The 8B job needs 1 GPU; the 32B job needs 3.
- Editable units: `orchestrator` (guidance opening, guidance closing, worked example, turn-0 prompt, turn-1..N prompt, tool descriptions, tool-use note), `web_search` (analysis template), `code_generator` (task template, `return_code: false` branch only). `text_inspector` stays at seed.
- Protected: tool names, parameter names, types, `required`; the `<tools>` header and `<sub_goal>`/`<tool_call>` format paragraph; `\boxed{}` rule (`final_instructions`); attachment notes; memory-turn structure.
- The reflector is shown the gold answer.
- Outputs under `experiments/results/rsi/phase1/<run_name>/<domain>/`.

## Review Focus

1. **Reflector returns prose, a truncated JSON (finish by length) or an unknown unit.** Expected: the candidate is re-sampled up to `max_parse_retries` times, each failure is logged in the round record, and the round still completes with fewer (possibly zero) candidates and an unchanged champion. Test: `test_round_with_only_unparseable_proposals_keeps_champion` (Task 11).
2. **Job killed mid-search.** Expected: rerunning the same command continues from the last fully written round, does not redo completed rounds, and the rollout counter continues from the saved value. Test: `test_resume_skips_completed_rounds` (Task 11).
3. **A candidate changes the web_search prompt but the analysis cache returns seed-prompt analyses for repeated queries.** Expected: analyses are cached per template, so a new template never reuses another template's analysis. Test: `test_analysis_cache_is_per_template` (Task 2).
4. **One trajectory alone exceeds the reflector input budget** (very long plan or many tool calls). Expected: the request still fits; per-call excerpts and then the trajectory's action list are cut harder, and the cut is logged; no over-budget request is sent. Test: `test_single_oversized_trajectory_is_cut_to_budget` (Task 9).
5. **A candidate's worked example calls a disabled tool, uses `code` in sub-agent mode, or drops the `\boxed{}` final answer.** Expected: rejected with a named reason before any rollout. Tests in Task 5.

---

### Task 1: PromptBuilder overrides for the editable orchestrator parts

**Files:**
- Modify: `src/agent_engine/prompts/builder.py` (`build_system_prompt` at :64, `_format_tool_schemas` at :183, `_select_and_format_example` at :265)
- Test: `tests/unit/test_builder_overrides.py`

**Interfaces:**
- Produces:
  - `DEFAULT_TOOL_USE_NOTE: str` module constant in `builder.py`, equal to today's `tail` string (:206-211).
  - `PromptBuilder.resolve_orchestrator_parts(dataset_name: str, tool_schemas: list[dict], direct_tool_call: bool) -> dict` with keys `guidance_opening: str` (the stripped `base_instruction_tools`), `guidance_closing: str` (the stripped `final_instructions_tools*` value chosen by today's `mode_key`/`ft_key` logic, `""` if absent), `example: dict | None` (the template's example dict under the key today's selector picks), `tool_use_note: str` (`DEFAULT_TOOL_USE_NOTE`).
  - `PromptBuilder.build_system_prompt(..., overrides: Optional[dict] = None)`; `overrides` may hold any subset of `guidance_opening`, `guidance_closing`, `example`, `tool_use_note`, each replacing exactly that section. Only applies when `tool_schemas` is non-empty and `baseline` is False.

- [ ] **Step 1: Write the failing tests**

```python
SCHEMAS = [WebSearchTool(...).get_schema(), CodeGeneratorTool(model_provider=FakeProvider()).get_schema()]
# use the sub-agent-mode schemas (model_provider set) as in collection

@pytest.mark.parametrize("dataset", ["gaia", "deepmath"])
def test_resolved_parts_as_overrides_render_identically(dataset):
    pb = PromptBuilder()
    plain = pb.build_system_prompt(dataset, SCHEMAS, direct_tool_call=False)
    parts = pb.resolve_orchestrator_parts(dataset, SCHEMAS, direct_tool_call=False)
    assert pb.build_system_prompt(dataset, SCHEMAS, direct_tool_call=False, overrides=parts) == plain

def test_guidance_opening_override_replaces_only_first_section():
    pb = PromptBuilder()
    plain = pb.build_system_prompt("gaia", SCHEMAS, direct_tool_call=False)
    out = pb.build_system_prompt("gaia", SCHEMAS, direct_tool_call=False,
                                 overrides={"guidance_opening": "You are a research coordinator."})
    assert out.startswith("You are a research coordinator.\n\n# Tools")
    assert out.split("# Tools", 1)[1] == plain.split("# Tools", 1)[1]

def test_tool_use_note_override():
    out = PromptBuilder().build_system_prompt("gaia", SCHEMAS, direct_tool_call=False,
                                              overrides={"tool_use_note": "Delegate every look-up."})
    assert "Delegate every look-up." in out
    assert DEFAULT_TOOL_USE_NOTE not in out
    assert "<sub_goal>A specific, actionable goal for this step</sub_goal>" in out

def test_example_override_renders_through_builder():
    ex = {"question": "Q?", "steps": [
        {"step": 1, "reasoning": "r", "sub_goal": "g",
         "tool_call": '{"name": "web_search", "arguments": {"query": "x"}}', "tool_response": "y"},
        {"step": "final", "answer": "\\boxed{y}"}]}
    out = PromptBuilder().build_system_prompt("gaia", SCHEMAS, direct_tool_call=False,
                                              overrides={"example": ex})
    assert 'Question: "Q?"' in out and "<tool_call>\n" in out and "**Answer:** \\boxed{y}" in out
    assert "atomic number of Oxygen" not in out

def test_boxed_rule_is_not_overridable():
    out = PromptBuilder().build_system_prompt("gaia", SCHEMAS, direct_tool_call=False,
                                              overrides={"guidance_closing": "- Be brief."})
    assert "Provide your final answer in the format \\boxed{YOUR_ANSWER}" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_builder_overrides.py -v`
Expected: FAIL (`resolve_orchestrator_parts` / `overrides` / `DEFAULT_TOOL_USE_NOTE` not defined).

- [ ] **Step 3: Implement**

Refactor `build_system_prompt` so template-key selection (base instruction, example key, final-instructions key) happens once in a private helper used by both `build_system_prompt` and `resolve_orchestrator_parts`. Thread `tool_use_note` into `_format_tool_schemas` (replaces `tail` in every format branch) and `example_override` into `_select_and_format_example` (formats the given dict with the existing step loop). Leave `baseline=True` and no-tools paths untouched.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_builder_overrides.py tests/unit/test_builder.py tests/characterization -v`
Expected: all PASS, no fixture updated.

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/prompts/builder.py tests/unit/test_builder_overrides.py
```

---

### Task 2: Overridable sub-agent templates and per-template analysis cache

**Files:**
- Create: `src/agent_engine/prompts/templating.py`
- Modify: `src/agent_engine/tools/web_search.py` (`build_analysis_prompt` :279, `analysis_cache` :446, `finalize` :528)
- Modify: `src/agent_engine/tools/code_generator.py` (`build_task_prompt` :234, `prepare` ~:470)
- Test: `tests/unit/test_subagent_templates.py`

**Interfaces:**
- Produces:
  - `render_placeholders(template: str, values: dict[str, str]) -> str`: single-pass `re.sub(r"\{([a-z_]+)\}", ...)` replacing only names present in `values`; other braces untouched, substituted values never re-scanned.
  - `find_placeholders(template: str) -> list[str]`: names matching `\{([a-z_]+)\}`, in order, with repeats.
  - `WEB_SEARCH_ANALYSIS_TEMPLATE: str`, `WEB_SEARCH_PLACEHOLDERS = ("query", "formatted_results")` in `web_search.py`; `WebSearchTool.analysis_template: str` (default the constant), `WebSearchTool.set_analysis_template(template: str) -> None`.
  - `CODE_GENERATOR_TASK_TEMPLATE: str`, `CODE_GENERATOR_PLACEHOLDERS = ("context_block", "task")` in `code_generator.py`; `CodeGeneratorTool.task_template: str`, `CodeGeneratorTool.set_task_template(template: str) -> None`. Applies only to the `return_code=False` branch; `context_block` is rendered as today (`f"Context:\n\n{context}\n\n"` or `""`).
  - `CodeGeneratorTool.prepare` payload additionally carries `"task"` and `"context"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_placeholders_single_pass():
    assert render_placeholders("{a} {b} {a}", {"a": "{b}", "b": "x"}) == "{b} x {b}"
    assert render_placeholders("keep {unknown} and {", {"a": "1"}) == "keep {unknown} and {"

def test_web_seed_template_reproduces_todays_prompt(web):
    # literal pieces of today's f-string, start, middle and end
    msgs = captured_messages(web.build_analysis_prompt("q1", "PAGES"))
    text = msgs[0]["content"]
    assert text.startswith("**Task Instruction:**\n\nYou are tasked with reading and analyzing web pages")
    assert "- **Current Search Query:**\nq1\n\n- **Searched Web Pages:**\nPAGES\n" in text
    assert text.endswith('based on the current search query "q1".\n')
    assert find_placeholders(WEB_SEARCH_ANALYSIS_TEMPLATE) == ["query", "formatted_results", "query"]

def test_web_override_template_is_used(web):
    web.set_analysis_template("Extract facts for {query} from {formatted_results}.")
    assert captured_messages(web.build_analysis_prompt("q", "P"))[0]["content"] == "Extract facts for q from P."

def test_analysis_cache_is_per_template(web):
    web.analysis_cache["q"] = "seed analysis"
    web.set_analysis_template("T {query} {formatted_results}")
    assert "q" not in web.analysis_cache
    web.set_analysis_template(WEB_SEARCH_ANALYSIS_TEMPLATE)
    assert web.analysis_cache["q"] == "seed analysis"

def test_code_seed_template_reproduces_todays_prompt(code):
    text = captured_messages(code.build_task_prompt("add 1+1"))[0]["content"]
    assert text.startswith("You are a code generator. Generate ONLY executable Python code")
    assert "\n\nProblem: add 1+1\n\nRequirements:\n- Output ONLY the Python code\n" in text
    assert text.endswith("Python code:")
    with_ctx = captured_messages(code.build_task_prompt("t", context="C"))[0]["content"]
    assert "\n\nContext:\n\nC\n\nProblem: t\n\n" in with_ctx

def test_code_prepare_payload_keeps_task_and_context(code, state):
    job = code.prepare(state, {"name": "code_generator", "arguments": {"task": "t"}}, {"task": "t"})
    assert job.payload["task"] == "t" and "context" in job.payload
```

(`captured_messages` is a fake provider's `apply_chat_template` that returns `json.dumps(messages)`; reuse the fixtures pattern in `tests/unit/test_batched_tools.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_subagent_templates.py -v`
Expected: FAIL (names not defined).

- [ ] **Step 3: Implement**

Move each f-string body verbatim into its module constant with `{query}`/`{formatted_results}` and `{context_block}`/`{task}` placeholders, and render with `render_placeholders`. Replace the single `_analysis_cache` dict by `_analysis_caches: dict[str, dict]` keyed by the template string; `analysis_cache` returns the current template's dict, so `prepare`/`finalize` code is unchanged.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_subagent_templates.py tests/unit/test_batched_tools.py tests/characterization -v`
Expected: all PASS, no fixture updated.

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/prompts/templating.py src/agent_engine/tools/web_search.py \
  src/agent_engine/tools/code_generator.py tests/unit/test_subagent_templates.py
```

---

### Task 3: Acting suffix and sub-agent call logging

**Files:**
- Modify: `src/agent_engine/core/orchestrator.py` (`__init__` :99, `_build_memory_prompt` :489)
- Modify: `src/agent_engine/core/batching.py` (`flush_batches`)
- Modify: `src/agent_engine/tools/web_search.py`, `src/agent_engine/tools/code_generator.py` (add `subagent_record`)
- Test: `tests/unit/test_acting_suffix_and_logging.py`

**Interfaces:**
- Consumes: Task 2 payload fields (`task`, `context`).
- Produces:
  - `AgenticOrchestrator(..., acting_suffix: str = "")`. `_build_memory_prompt` user content is `"\n".join(parts) + self.acting_suffix` (raw append; callers pass the leading newlines).
  - Tool method `subagent_record(job, generation, result) -> dict`:
    - web_search: `{"tool": "web_search", "inputs": {"query": str, "formatted_results": str}, "response": str, "output": str}`; `formatted_results` recomputed with `self._format_results(job.payload["payload"].get("results", []), query)`; `response` = `strip_thinking_tags(generation.text)`.
    - code_generator: `{"tool": "code_generator", "inputs": {"task": str, "context": str}, "response": str, "output": str}`; `output` is the executed result text.
  - `flush_batches`: when a job's tool has `subagent_record`, append its dict plus `"action_index": len(job.state.action_history)` (taken before `commit`) to `job.state.metadata.setdefault("subagent_calls", [])`.

- [ ] **Step 1: Write the failing tests**

```python
def test_empty_acting_suffix_leaves_memory_prompt_unchanged(orch_factory, state_with_history):
    a = orch_factory()._build_memory_prompt(state_with_history, "SYS")
    b = orch_factory(acting_suffix="")._build_memory_prompt(state_with_history, "SYS")
    assert a == b

def test_acting_suffix_is_appended_after_previous_steps(orch_factory, state_with_history):
    msgs = orch_factory(acting_suffix="\n\nCheck every link.")._build_memory_prompt(state_with_history, "SYS")
    assert msgs[1]["content"].endswith("**Previous Steps:**\n" + ANY_HISTORY_TEXT + "\n\nCheck every link.")

def test_flush_records_subagent_calls_with_action_index(code, state):
    job = code.prepare(state, {"name": "code_generator", "arguments": {"task": "t"}}, {"task": "t"})
    flush_batches({"code_generator": [job]}, commit=_commit_appending_history, accumulate_usage=lambda *a: None)
    rec = state.metadata["subagent_calls"][0]
    assert rec["tool"] == "code_generator" and rec["inputs"]["task"] == "t"
    assert rec["action_index"] == 0 and "response" in rec and "output" in rec
```

(`_commit_appending_history` appends a dict to `state.action_history`, mirroring `_commit_tool_result`; `ANY_HISTORY_TEXT` is `AgenticOrchestrator._format_action_history(state_with_history.action_history)`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_acting_suffix_and_logging.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** the three changes above. Note in the `flush_batches` docstring that logging never alters `commit` or usage accounting.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_acting_suffix_and_logging.py tests/unit/test_batched_tools.py tests/characterization -v`
Expected: all PASS, no fixture updated.

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/core/orchestrator.py src/agent_engine/core/batching.py \
  src/agent_engine/tools/web_search.py src/agent_engine/tools/code_generator.py \
  tests/unit/test_acting_suffix_and_logging.py
```

---

### Task 4: The prompt bundle

**Files:**
- Create: `src/rsi/__init__.py`, `src/rsi/bundle.py`
- Test: `tests/rsi/__init__.py`, `tests/rsi/conftest.py` (shared `SCHEMAS` fixture: sub-agent-mode web_search + code_generator schemas), `tests/rsi/test_bundle.py`

**Interfaces:**
- Consumes: Task 1 `resolve_orchestrator_parts`, `overrides`; Task 2 template constants and setters; Task 3 `acting_suffix`; `_DEFAULT_PLANNING_SUFFIX_TOOLS` from `orchestrator.py`.
- Produces (all in `rsi.bundle`):
  - `DOMAINS = {"search": "gaia", "math": "deepmath"}`; `UNITS = ("orchestrator", "web_search", "code_generator")`.
  - `@dataclass PromptBundle(domain: str, orchestrator: dict, subagents: dict)` with `to_dict() -> dict`, `classmethod from_dict(d) -> PromptBundle`.
    - `orchestrator` keys: `guidance_opening`, `guidance_closing`, `example` (template-shaped dict: `question`, `steps` with `step`, `reasoning`, `sub_goal`, `tool_call` JSON string, `tool_response`, final step `{"step": "final", "answer": ...}`), `turn0_prompt`, `turn1n_prompt`, `tool_descriptions` (`{tool: {"description": str, "parameters": {param: str}}}`), `tool_use_note`.
    - `subagents` keys: `web_search`, `code_generator` (template strings).
  - `seed_bundle(domain: str, tool_schemas: list[dict], direct_tool_call: bool = False) -> PromptBundle`; `turn0_prompt` = `_DEFAULT_PLANNING_SUFFIX_TOOLS` without its leading `"\n\n"`; `turn1n_prompt` = `""`.
  - `save_bundle(bundle, path: Path) -> None`, `load_bundle(path: Path) -> PromptBundle` (JSON, `indent=2`, `ensure_ascii=False`).
  - `apply_tool_descriptions(schemas: list[dict], tool_descriptions: dict) -> list[dict]` (deep copy, replaces only `description` strings, key order preserved).
  - `render_system_prompt(bundle, tool_schemas, tool_call_format) -> str`.
  - `planning_suffix(bundle) -> str` = `"\n\n" + turn0_prompt.strip()`; `acting_suffix(bundle) -> str` = `""` if blank else `"\n\n" + turn1n_prompt.strip()`.
  - `apply_subagent_templates(bundle, registry: ToolRegistry) -> None` (calls the Task 2 setters on the tools present).
  - `replace_unit(bundle, unit: str, payload) -> PromptBundle` (new object; `orchestrator` payload is a partial dict merged over the current parts; sub-agent payload is the full template string).
  - `unit_text(bundle, unit: str) -> str` (orchestrator: `json.dumps(parts, indent=2, ensure_ascii=False, sort_keys=True)`; sub-agent: the template) used for diffs.
  - `editable_texts(bundle) -> list[str]` (every editable string, including example fields and tool descriptions) used by the leakage check.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("domain,dataset", [("search", "gaia"), ("math", "deepmath")])
def test_seed_bundle_renders_todays_system_prompt(domain, dataset, SCHEMAS):
    b = seed_bundle(domain, SCHEMAS)
    assert render_system_prompt(b, SCHEMAS, ToolCallFormat.JSON) == \
        PromptBuilder().build_system_prompt(dataset, SCHEMAS, direct_tool_call=False)

def test_seed_suffixes_match_todays_defaults(SCHEMAS):
    b = seed_bundle("search", SCHEMAS)
    assert planning_suffix(b) == _DEFAULT_PLANNING_SUFFIX_TOOLS
    assert acting_suffix(b) == ""

def test_seed_subagent_templates_are_todays(SCHEMAS):
    b = seed_bundle("math", SCHEMAS)
    assert b.subagents == {"web_search": WEB_SEARCH_ANALYSIS_TEMPLATE,
                           "code_generator": CODE_GENERATOR_TASK_TEMPLATE}

def test_apply_tool_descriptions_changes_only_descriptions(SCHEMAS):
    out = apply_tool_descriptions(SCHEMAS, {"web_search": {"description": "D", "parameters": {"query": "P"}}})
    ws = next(s for s in out if s["function"]["name"] == "web_search")
    assert ws["function"]["description"] == "D"
    assert ws["function"]["parameters"]["properties"]["query"] == {"type": "string", "description": "P"}
    assert SCHEMAS[0]["function"]["description"] != "D"  # input not mutated

def test_round_trip_and_replace_unit(tmp_path, SCHEMAS):
    b = seed_bundle("search", SCHEMAS)
    save_bundle(b, tmp_path / "b.json")
    assert load_bundle(tmp_path / "b.json") == b
    b2 = replace_unit(b, "orchestrator", {"turn1n_prompt": "Check links."})
    assert acting_suffix(b2) == "\n\nCheck links." and b2.orchestrator["turn0_prompt"] == b.orchestrator["turn0_prompt"]
    assert b.orchestrator["turn1n_prompt"] == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_bundle.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `src/rsi/bundle.py` per the interface.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_bundle.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/__init__.py src/rsi/bundle.py tests/rsi/__init__.py tests/rsi/conftest.py tests/rsi/test_bundle.py
```

---

### Task 5: Candidate validation

**Files:**
- Create: `src/rsi/validate.py`
- Test: `tests/rsi/test_validate.py`

**Interfaces:**
- Consumes: Task 4 `PromptBundle`, `UNITS`; Task 2 `find_placeholders`, placeholder tuples.
- Produces:
  - `normalize_example(example: dict) -> dict`: accepts `tool_call` as dict or JSON string and returns a JSON string (`json.dumps(obj, ensure_ascii=False)`); renumbers non-final steps 1..n.
  - `validate_unit(unit: str, payload, seed: PromptBundle, tool_schemas: list[dict]) -> list[str]` returning human-readable error strings (empty list = valid). Rules:
    - `unit` in `UNITS`, and the sub-agent's tool is enabled (present in `tool_schemas`).
    - orchestrator payload: dict whose keys are a subset of the seven part names; string parts non-empty except `turn1n_prompt` and `guidance_closing`.
    - example: non-empty `question`; at least one non-final step; every non-final `tool_call` parses as `{"name", "arguments"}`, `name` is an enabled tool, argument keys are a subset of that tool's schema properties and include its `required`; last step is `{"step": "final"}` with an `answer` containing `\boxed{`.
    - tool descriptions: same tool set as `seed.orchestrator["tool_descriptions"]`, same parameter names per tool, every string non-empty.
    - sub-agent template: contains each of that tool's placeholders at least once; no other `{name}` placeholder.

- [ ] **Step 1: Write the failing tests**

```python
def test_seed_parts_are_valid(seed_b, SCHEMAS):
    assert validate_unit("orchestrator", seed_b.orchestrator, seed_b, SCHEMAS) == []
    assert validate_unit("web_search", seed_b.subagents["web_search"], seed_b, SCHEMAS) == []
    assert validate_unit("code_generator", seed_b.subagents["code_generator"], seed_b, SCHEMAS) == []

def test_unknown_unit_rejected(seed_b, SCHEMAS):
    assert any("unknown unit" in e for e in validate_unit("planner", {}, seed_b, SCHEMAS))

def test_missing_placeholder_rejected(seed_b, SCHEMAS):
    errs = validate_unit("web_search", "Analyse {query}.", seed_b, SCHEMAS)
    assert any("formatted_results" in e for e in errs)

def test_unknown_placeholder_rejected(seed_b, SCHEMAS):
    errs = validate_unit("code_generator", "{context_block}{task}{language}", seed_b, SCHEMAS)
    assert any("language" in e for e in errs)

def test_example_with_code_arg_in_subagent_mode_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "code_generator", "arguments": {"code": "print(1)"}})
    assert any("code" in e for e in validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS))

def test_example_calling_disabled_tool_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "text_inspector", "arguments": {"question": "x"}})
    assert any("text_inspector" in e for e in validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS))

def test_example_without_boxed_answer_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "web_search", "arguments": {"query": "x"}}, answer="It is x.")
    assert any("boxed" in e for e in validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS))

def test_renamed_tool_parameter_rejected(seed_b, SCHEMAS):
    td = copy.deepcopy(seed_b.orchestrator["tool_descriptions"])
    td["web_search"]["parameters"] = {"q": "x"}
    assert validate_unit("orchestrator", {"tool_descriptions": td}, seed_b, SCHEMAS)

def test_normalize_example_accepts_object_tool_calls():
    ex = normalize_example(example_with_call({"name": "web_search", "arguments": {"query": "x"}}))
    assert ex["steps"][0]["tool_call"] == '{"name": "web_search", "arguments": {"query": "x"}}'
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_validate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/validate.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_validate.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/validate.py tests/rsi/test_validate.py
```

---

### Task 6: Leakage guard

**Files:**
- Create: `src/rsi/leakage.py`
- Test: `tests/rsi/test_leakage.py`

**Interfaces:**
- Produces: `class LeakageIndex(questions: list[str], n: int = 8)` with `check(texts: list[str], shown_answers: list[str]) -> list[str]` (error strings, each naming the matching span).
  - Tokens: lowercase `re.findall(r"\w+", text)`. N-gram rule: any word n-gram of length `n` from any text that occurs in any indexed question.
  - Answer rule: answers normalized with `agent_engine.datasets.evaluators.metrics.normalize_answer`; empty answers ignored; an answer shorter than 4 characters or purely numeric matches only as a standalone token sequence; longer answers match as a token sequence anywhere in the normalized text.

- [ ] **Step 1: Write the failing tests**

```python
Q = ["Who was the editor of the book Grand Canyon A Different View that Kurt Wise contributed to?"]

def test_long_ngram_from_any_training_question_is_rejected():
    errs = LeakageIndex(Q, n=8).check(["Example: the editor of the book Grand Canyon A Different View"], [])
    assert errs and "grand canyon" in errs[0]

def test_seven_word_overlap_passes_at_n8():
    assert LeakageIndex(Q, n=8).check(["the editor of the book Grand Canyon A"], []) == []

def test_shown_answer_is_rejected():
    assert LeakageIndex([], n=8).check(["The answer is Tom Vail, remember it."], ["Tom Vail"])

def test_short_and_numeric_answers_match_only_as_tokens():
    idx = LeakageIndex([], n=8)
    assert idx.check(["Say yes or no."], ["no"])          # standalone token: rejected
    assert idx.check(["Note the notation."], ["no"]) == []  # substring only: passes
    assert idx.check(["Use 2 searches."], ["2"])
    assert idx.check(["Use 12 searches."], ["2"]) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_leakage.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** with a `set` of n-gram tuples built once from all 1,800 questions.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_leakage.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/leakage.py tests/rsi/test_leakage.py
```

---

### Task 7: Domain data and pools

**Files:**
- Create: `src/rsi/data.py`
- Test: `tests/rsi/test_data.py`

**Interfaces:**
- Produces:
  - `load_domain_examples(parquet_path: Path, domain: str) -> list[DatasetExample]`: `question_id` = the row's position in the parquet (stable across domains), `answer` = `str(row["result"])`, `metadata = {"data_source": ..., "answer_aliases": list(extra_info["golden_answers"] or [])}`. `search` keeps `hotpotqa` + `nq`; `math` keeps `deepmath`.
  - `@dataclass PoolSizes(shown: int = 300, screen: int = 200, heldout: int = 50)`.
  - `@dataclass Pools(shown: list[int], screen: list[int], heldout: list[int])` with `to_dict()` / `from_dict()`.
  - `make_pools(examples: list[DatasetExample], sizes: PoolSizes, seed: int) -> Pools`: disjoint, deterministic for a seed; each pool keeps the `data_source` proportions of `examples` (largest-remainder rounding); raises `ValueError` if sizes exceed the domain.
  - `sample_minibatch(pool: list[int], size: int, rng: random.Random) -> list[int]` (without replacement).

- [ ] **Step 1: Write the failing tests**

```python
def test_search_domain_rows_and_aliases(tiny_parquet):
    ex = load_domain_examples(tiny_parquet, "search")
    assert {e.metadata["data_source"] for e in ex} == {"hotpotqa", "nq"}
    assert ex[0].metadata["answer_aliases"]  # golden_answers carried

def test_pools_are_disjoint_deterministic_and_stratified(search_examples):
    p1 = make_pools(search_examples, PoolSizes(30, 20, 10), seed=0)
    p2 = make_pools(search_examples, PoolSizes(30, 20, 10), seed=0)
    assert p1 == p2
    assert not (set(p1.shown) & set(p1.screen) or set(p1.shown) & set(p1.heldout) or set(p1.screen) & set(p1.heldout))
    assert [len(p1.shown), len(p1.screen), len(p1.heldout)] == [30, 20, 10]
    nq_share = share_of(search_examples, "nq")
    assert abs(share_of_ids(p1.shown, "nq") - nq_share) < 0.05

def test_pools_too_large_raise(search_examples):
    with pytest.raises(ValueError):
        make_pools(search_examples, PoolSizes(10_000, 1, 1), seed=0)
```

(`tiny_parquet` writes a 60-row parquet with the real column layout: 40 hotpotqa, 8 nq, 12 deepmath, `extra_info` dicts with `golden_answers`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_data.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/data.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_data.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/data.py tests/rsi/test_data.py
```

---

### Task 8: Bundle executor

**Files:**
- Create: `src/rsi/executor.py`
- Test: `tests/rsi/test_executor.py`

**Interfaces:**
- Consumes: Task 4 `render_system_prompt`, `planning_suffix`, `acting_suffix`, `apply_tool_descriptions`, `apply_subagent_templates`; Task 3 `acting_suffix` param and `metadata["subagent_calls"]`; `evaluate_answer`; `agent_engine.analysis.failure_modes.classify_failure`.
- Produces:
  - `@dataclass TrajectoryRecord(question_id: int, question: str, gold: str, aliases: list[str], prediction: str, correct: bool, failure_label: str | None, plan: str, action_history: list[dict], subagent_calls: list[dict], turns: int, tool_counts: dict)` with `to_dict()` / `from_dict()`.
  - `class BundleExecutor(model_provider, tool_registry, tool_call_format, max_turns: int = 7, tool_limits: dict | None = None, use_thinking: bool = False)`:
    - `run(bundle: PromptBundle, examples: list[DatasetExample]) -> list[TrajectoryRecord]`: applies sub-agent templates to the registry, renders the system prompt from the registry schemas with the bundle's descriptions, builds an `AgenticOrchestrator` with the bundle's suffixes, calls `run_batch`, scores with `evaluate_answer(prediction, gold, answer_aliases=aliases)["correct"]`, and sets `failure_label = classify_failure({"action_history", "prediction", "turns", "question"})` for wrong answers only.
    - `rollouts: int` (incremented by `len(examples)` per `run`).
  - `accuracy(records: list[TrajectoryRecord]) -> float` (0.0 on empty).

- [ ] **Step 1: Write the failing tests**

Use `tests/characterization/scripted_provider.ScriptedProvider` with an empty `ToolRegistry` (no tool calls needed) for scoring, and with a registry holding a fake `web_search` for the bundle-application test.

```python
def test_scores_and_labels(seed_b):
    prov = ScriptedProvider(["plan", "\\boxed{Paris}", "plan", "\\boxed{Rome}"])
    ex = [mk_example(1, "Capital of France?", "Paris"), mk_example(2, "Capital of Spain?", "Madrid")]
    recs = BundleExecutor(prov, ToolRegistry(), ToolCallFormat.JSON).run(seed_b, ex)
    assert [r.correct for r in recs] == [True, False]
    assert recs[0].failure_label is None and recs[1].failure_label == "direct_reasoning_no_action"

def test_bundle_parts_reach_the_model(seed_b, SCHEMAS):
    b = replace_unit(seed_b, "orchestrator", {"guidance_opening": "You are a coordinator.",
                                               "turn0_prompt": "Plan the links."})
    prov = ScriptedProvider(["plan", "\\boxed{x}"])
    BundleExecutor(prov, registry_with_fake_tools(), ToolCallFormat.JSON).run(b, [mk_example(1, "Q?", "x")])
    first = prov.prompts_seen[0]  # the planning turn
    assert "You are a coordinator." in first
    assert "Q?\n\nPlan the links." in first

def test_rollout_counter(seed_b):
    ex_ = BundleExecutor(ScriptedProvider(["p", "\\boxed{a}"] * 3), ToolRegistry(), ToolCallFormat.JSON)
    ex_.run(seed_b, [mk_example(i, "q", "a") for i in range(3)])
    assert ex_.rollouts == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_executor.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/executor.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_executor.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/executor.py tests/rsi/test_executor.py
```

---

### Task 9: Reflector input

**Files:**
- Create: `src/rsi/reflector_input.py`
- Test: `tests/rsi/test_reflector_input.py`

**Interfaces:**
- Consumes: Task 4 `PromptBundle`, `unit_text`; Task 8 `TrajectoryRecord`.
- Produces:
  - `@dataclass HistoryEntry(label: str, unit: str, diff: str, rationale: str, screen_score: float | None, heldout_score: float | None, accepted: bool)`.
  - `@dataclass ReflectorContext(domain: str, champion: PromptBundle, seed: PromptBundle, best: PromptBundle | None, history: list[HistoryEntry], trajectories: list[TrajectoryRecord], rendered_system_prompt: str, units: list[str])`.
  - `OUTPUT_SCHEMA_TEXT: str`: the JSON contract `{"unit": "...", "rationale": "...", "expected_effect": "...", "replacement": ...}` where `replacement` is a partial orchestrator dict (any of the seven part names; `example` in the template shape, `tool_call` may be an object) or a full sub-agent template string.
  - `build_reflector_messages(ctx, count_tokens, budget_tokens: int, per_call_chars: int) -> tuple[list[dict], dict]`; returns `[{"role": "system", ...}, {"role": "user", ...}]` and stats `{"trajectories_total", "trajectories_shown", "per_call_chars_used", "actions_cut", "input_tokens"}`.
    - System message: role and task (improve one unit of this team's prompts for the `<domain>` domain), the leakage instruction verbatim: "Write general strategies only. Do not copy question text, named entities, numbers or answers from the trajectories below, and do not build a worked example from any of these questions.", the protected-parts statement, and `OUTPUT_SCHEMA_TEXT`.
    - User message sections in spec order: current champion (rendered system prompt read-only, then every unit's `unit_text`), seed (v0) and best if different, history (last entries with diff, scores, rationale) plus a "Recurring problems" line counting `failure_label` over the shown trajectories, then trajectories.
    - Per trajectory: question, gold answer, prediction, CORRECT/WRONG, failure label, plan, every action (tool, sub_goal, command, result), and for each `subagent_calls` entry matched by `action_index`: the inputs (web_search: query plus a head-and-tail excerpt of `formatted_results` at `per_call_chars`; code_generator: task), the response, and for code_generator the output excerpt.
    - Fitting order: keep all, else drop trajectories alternately from the larger of the correct/wrong groups until it fits; if one trajectory alone is too long, halve `per_call_chars` (floor 200), then cut the action list to its first and last actions with a marker; log every cut in stats. Never return messages over `budget_tokens`.

- [ ] **Step 1: Write the failing tests**

```python
def test_contains_gold_failure_label_and_subagent_inputs(ctx):
    msgs, _ = build_reflector_messages(ctx, char_count, budget_tokens=10_000, per_call_chars=400)
    user = msgs[1]["content"]
    assert "Gold answer: Tom Vail" in user and "direct_reasoning_no_action" in user
    assert "Query: grand canyon different view editor" in user and "[...]" in user  # head/tail excerpt

def test_leakage_instruction_present(ctx):
    msgs, _ = build_reflector_messages(ctx, char_count, 10_000, 400)
    assert "Do not copy question text, named entities, numbers or answers" in msgs[0]["content"]

def test_drops_trajectories_before_cutting_harder(ctx_many):
    _, stats = build_reflector_messages(ctx_many, char_count, budget_tokens=3_000, per_call_chars=400)
    assert stats["trajectories_shown"] < stats["trajectories_total"]
    assert stats["per_call_chars_used"] == 400

def test_single_oversized_trajectory_is_cut_to_budget(ctx_one_huge):
    msgs, stats = build_reflector_messages(ctx_one_huge, char_count, budget_tokens=2_000, per_call_chars=4_000)
    assert char_count(msgs[0]["content"] + msgs[1]["content"]) <= 2_000
    assert stats["trajectories_shown"] == 1 and (stats["per_call_chars_used"] < 4_000 or stats["actions_cut"])

def test_history_shows_diff_and_scores(ctx):
    user = build_reflector_messages(ctx, char_count, 10_000, 400)[0][1]["content"]
    assert "held-out 0.42" in user and "+  - Find each link" in user
```

(`char_count = lambda s: len(s) // 4`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_reflector_input.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/reflector_input.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_reflector_input.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/reflector_input.py tests/rsi/test_reflector_input.py
```

---

### Task 10: Reflector clients and proposal parsing

**Files:**
- Create: `src/rsi/reflector.py`
- Test: `tests/rsi/test_reflector.py`

**Interfaces:**
- Produces two clients with the same method, `sample(messages: list[dict], seeds: list[int]) -> list[dict]`, each item `{"text": str, "finish_reason": str, "seed": int}`; exceptions become `{"text": "", "finish_reason": "error: <msg>", "seed": s}`.
  - `class InProcessReflector(provider, max_tokens: int, temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, enable_thinking: bool = True)` (default for the homogeneous run). `provider` is the executor's `VLLMProvider`. Renders the prompt with `provider.apply_chat_template(messages, use_thinking=enable_thinking)` and makes **one** batched call `provider.llm.generate([prompt] * len(seeds), sampling_params=[SamplingParams(max_tokens=max_tokens, temperature=..., top_p=..., top_k=..., seed=s) for s in seeds])` inside `with provider._lock:`. No `stop` list, no LoRA request beyond the provider's own `lora_request`. It must not mutate `provider.config`.
  - `class ReflectorClient(base_url: str, model: str, max_tokens: int, temperature: float = 0.6, top_p: float = 0.95, top_k: int = 20, enable_thinking: bool = True, timeout_s: int = 900)` (32B control) using `openai.OpenAI(base_url=..., api_key="EMPTY")`: one `chat.completions.create` per seed with `seed=s`, `extra_body={"top_k": top_k, "chat_template_kwargs": {"enable_thinking": enable_thinking}}`.
  - `class ReflectorSettings(BaseModel)` is defined here in `src/rsi/reflector.py` (fields listed in Task 11) and imported by `src/rsi/search.py`.
  - `build_reflector(settings: ReflectorSettings, executor_provider)` returns `InProcessReflector` when `settings.backend == "in_process"`, else `ReflectorClient`; raises `ValueError` for `openai` without `base_url`.
  - `@dataclass Proposal(unit: str, replacement: object, rationale: str, expected_effect: str)`.
  - `class ProposalParseError(ValueError)`.
  - `parse_proposal(text: str) -> Proposal`: strip `<think>...</think>` with `strip_thinking_tags`; take the content of the last ```json fenced block if present, else the outermost `{...}` span; `json.loads`; require `unit` and `replacement`; raise `ProposalParseError` with the reason otherwise.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_fenced_json_after_thinking():
    t = '<think>long</think>Here:\n```json\n{"unit": "web_search", "replacement": "{query} {formatted_results}", "rationale": "r", "expected_effect": "e"}\n```'
    p = parse_proposal(t)
    assert p.unit == "web_search" and p.replacement == "{query} {formatted_results}"

def test_parse_bare_object():
    assert parse_proposal('ok {"unit": "orchestrator", "replacement": {"turn1n_prompt": "x"}}').unit == "orchestrator"

@pytest.mark.parametrize("bad", ["no json here", '{"unit": "orchestrator"}', '{"unit": "orchestrator", "replacement": '])
def test_parse_errors(bad):
    with pytest.raises(ProposalParseError):
        parse_proposal(bad)

def test_in_process_reflector_uses_own_sampling_on_the_shared_engine():
    prov = FakeVLLMProvider()  # records llm.generate calls; config.temperature=0.7, role="orchestrator"
    out = InProcessReflector(prov, max_tokens=99).sample([{"role": "user", "content": "u"}], [5, 6, 7])
    (prompts, params), = prov.llm.calls  # exactly one batched call
    assert len(prompts) == 3 and [p.seed for p in params] == [5, 6, 7]
    assert all(p.temperature == 0.6 and p.max_tokens == 99 and not p.stop for p in params)
    assert prov.config.temperature == 0.7 and prov.rendered_with_thinking == [True]
    assert [o["seed"] for o in out] == [5, 6, 7]

def test_sample_passes_seed_and_thinking(monkeypatch):
    calls = install_fake_openai(monkeypatch, reply="x")
    out = ReflectorClient("http://h/v1", "m", max_tokens=10).sample([{"role": "user", "content": "u"}], [5, 6])
    assert [c["seed"] for c in calls] == [5, 6]
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert [o["seed"] for o in out] == [5, 6]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_reflector.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/reflector.py`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi/test_reflector.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/reflector.py tests/rsi/test_reflector.py
```

---

### Task 11: Search loop with on-disk state and resume

**Files:**
- Create: `src/rsi/search.py`
- Test: `tests/rsi/test_search.py`

**Interfaces:**
- Consumes: Tasks 4 to 10.
- Produces:
  - `ReflectorSettings` (defined in Task 10's `src/rsi/reflector.py`), fields: `backend: Literal["in_process", "openai"] = "in_process"`, `model: str` (for the tokenizer, and the served model name for `openai`), `base_url: Optional[str] = None` (required when `backend == "openai"`), `max_model_len: int = 32768`, `max_tokens: int = 16384`, `input_budget_tokens: int = 14000`, `per_call_chars: int = 1500`, `temperature: float = 0.6`, `top_p: float = 0.95`, `top_k: int = 20`, `enable_thinking: bool = True`, `timeout_s: int = 900`.
  - `class SearchConfig(BaseModel)`: `data_path: Path`, `pools: PoolSizes` fields (`shown: 300`, `screen: 200`, `heldout: 50`), `pool_seed: int = 0`, `rounds: int = 10`, `candidates_per_round: int = 3`, `shown_minibatch: int = 16`, `screen_minibatch: int = 16`, `max_parse_retries: int = 2`, `history_max: int = 8`, `ngram: int = 8`, `seed: int = 0`, `output_root: Path = Path("experiments/results/rsi/phase1")`, `reflector: ReflectorSettings`.
  - `run_search(cfg: SearchConfig, domain: str, executor, reflector, examples: dict[int, DatasetExample], pools: Pools, seed: PromptBundle, leakage: LeakageIndex, count_tokens, tool_schemas: list[dict], tool_call_format, out_dir: Path) -> dict` returning the final state dict.
  - On-disk layout in `out_dir`: `config.json`, `pools.json`, `bundles/v0.json` (seed), `bundles/v<N>.json` (each held-out-evaluated candidate), `rounds/round_<RR>.json`, `heldout/v<N>.json` (held-out `TrajectoryRecord`s per evaluated version), `state.json`:
    `{"rounds_done": int, "champion": "v<N>", "champion_heldout": float, "seed_heldout": float, "history": [...HistoryEntry dicts...], "versions": [{"label", "round", "unit", "screen", "heldout", "accepted", "prompt_chars"}], "rollouts": int, "beats_seed": bool}`. `prompt_chars` is the length of the rendered system prompt plus planning suffix plus sub-agent templates, so prompt growth is visible.
  - Round algorithm (round r, rng seeded with `cfg.seed * 1000 + r`):
    1. Shown minibatch from `pools.shown`; run champion.
    2. Build reflector messages (Task 9) with history = last `history_max` entries.
    3. Sample `candidates_per_round` proposals with seeds `cfg.seed * 100000 + r * 100 + k`; on parse failure, empty text or `finish_reason == "length"`, re-sample that slot with seed `+ 10 * attempt`, up to `max_parse_retries` times.
    4. Validate (Task 5, after `normalize_example` for orchestrator payloads) and leakage-check (`leakage.check(editable_texts(candidate_bundle), shown_answers)` where `shown_answers` = gold + aliases of every trajectory shown to the reflector in any round so far).
    5. If no candidate survives, skip to 7 (no screen rollouts). Otherwise draw a screen minibatch from `pools.screen` and run the champion and every surviving candidate on it.
    6. If the best screened candidate's screen accuracy is at least the champion's, run it on the full held-out pool; it becomes champion only if held-out accuracy is strictly greater than `champion_heldout`. Ties keep the champion.
    7. Write `rounds/round_<RR>.json` (every proposal: seed, raw text, finish reason, parsed unit, errors, leak errors, screen score; chosen candidate; held-out score; accepted; reflector stats; shown question ids) and then `state.json` (written last, atomically via temp file + rename).
  - Before round 1: run the seed on the held-out pool (`seed_heldout`, `champion_heldout`), unless `state.json` exists. `beats_seed = champion_heldout > seed_heldout`.
  - Resume: if `state.json` exists, load it and continue at `rounds_done + 1`; `executor.rollouts` is restored from `state["rollouts"]` before continuing.

- [ ] **Step 1: Write the failing tests**

Use a `FakeExecutor` whose accuracy is a deterministic function of the bundle (for example, the fraction of correct answers equals `0.2 + 0.1 * bundle.orchestrator["turn1n_prompt"].count("check")`, capped at 1.0), and a `FakeReflector` that returns scripted texts per call.

```python
def test_strictly_better_candidate_becomes_champion(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check check"})] * 3)
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    assert state["champion"] != "v0" and state["champion_heldout"] > state["seed_heldout"]
    assert state["beats_seed"] is True

def test_tie_keeps_champion(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "no change in score"})] * 3)
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    assert state["champion"] == "v0" and state["beats_seed"] is False

def test_round_with_only_unparseable_proposals_keeps_champion(tmp_path, env):
    refl = FakeReflector(["not json"] * 20)
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert state["champion"] == "v0" and all(p["errors"] for p in rec["proposals"])
    assert len(rec["proposals"]) == 3 * (1 + env.cfg().max_parse_retries)

def test_leaking_candidate_is_rejected_without_rollouts(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": env.a_shown_gold_answer})] * 3)
    before = env.executor.rollouts
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert all(p["leak_errors"] for p in rec["proposals"] if not p["errors"])
    # only seed held-out + champion on the shown minibatch were run; no screen without survivors
    assert env.executor.rollouts - before == env.heldout + env.cfg().shown_minibatch

def test_resume_skips_completed_rounds(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check"})] * 30)
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    calls_after_first = refl.calls
    state = run_search(env.cfg(rounds=2), "search", env.fresh_executor(), refl, *env.args(tmp_path))
    assert state["rounds_done"] == 2
    assert refl.calls - calls_after_first == 3          # only round 2 sampled
    assert (tmp_path / "rounds" / "round_01.json").exists()
    assert state["rollouts"] > env.heldout              # counter continued, seed held-out not rerun

def test_reflector_never_sees_screen_or_heldout_questions(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check"})] * 9)
    run_search(env.cfg(rounds=3), "search", env.executor, refl, *env.args(tmp_path))
    unseen = {env.examples[i].question for i in env.pools.screen + env.pools.heldout}
    assert not any(q in m["content"] for msgs in refl.messages for m in msgs for q in unseen)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_search.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/search.py`. Keep `run_search` a thin loop over a `_run_round(...)` helper; all file writes go through one `_write_json_atomic(path, data)` helper.

- [ ] **Step 4: Run tests**

Run: `pytest tests/rsi -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/search.py tests/rsi/test_search.py
```

---

### Task 12: Markdown report

**Files:**
- Create: `src/rsi/report.py`
- Test: `tests/rsi/test_report.py`

**Interfaces:**
- Consumes: Task 11 on-disk layout; Task 4 `load_bundle`, `unit_text`, `render_system_prompt`, `planning_suffix`, `acting_suffix`.
- Produces: `write_report(run_dir: Path, tool_schemas: list[dict], tool_call_format) -> Path` writing `run_dir/report.md` with these sections, in order:
  1. Summary: domain, reflector model, rounds done, rollouts, seed vs champion held-out accuracy, `beats_seed`.
  2. Search curve: a table of round, rollouts so far, champion held-out, candidate held-out (if run), accepted, champion `prompt_chars`.
  3. Rejections: counts by reason (parse error, validation error, leakage, screened out).
  4. Units edited: count per unit over all proposals and over accepted ones.
  5. Final prompt versus seed: a unified diff per unit (`difflib.unified_diff` on `unit_text`), "unchanged" when identical.
  6. Rendered prompts under the champion: system prompt, the planning user turn for a placeholder question `<question>`, one acting turn with a placeholder memory block, and each sub-agent prompt rendered with placeholder values.
  7. Paired trajectories: up to 3 held-out questions where seed and champion outcomes differ (from `heldout/v0.json` and `heldout/<champion>.json`), each showing both plans, actions and answers.

- [ ] **Step 1: Write the failing test**

```python
def test_report_has_all_sections(tmp_path, finished_run):  # finished_run: Task 11 fake run with an accepted candidate
    path = write_report(finished_run, SCHEMAS, ToolCallFormat.JSON)
    text = path.read_text()
    for heading in ["## Summary", "## Search curve", "## Rejections", "## Units edited",
                    "## Final prompt versus seed", "## Rendered prompts", "## Paired trajectories"]:
        assert heading in text
    assert "+check" in text or "+  \"turn1n_prompt\": \"check" in text
    assert "\\u2014" not in text.encode("unicode_escape").decode()  # no em dashes
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `src/rsi/report.py`.

- [ ] **Step 4: Run test**

Run: `pytest tests/rsi/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/rsi/report.py tests/rsi/test_report.py
```

---

### Task 13: Entry point, configs and SLURM jobs

**Files:**
- Create: `scripts/run_prompt_search.py`
- Create: `experiments/configs/rsi/phase1/search_reflector_qwen3_8b.yaml`, `experiments/configs/rsi/phase1/search_reflector_qwen3_32b.yaml`
- Create: `jobs/rsi/_prompt_search_body.sh`, `jobs/rsi/001_prompt_search_8b.job`, `jobs/rsi/002_prompt_search_32b.job`
- Test: `tests/rsi/test_run_prompt_search_cli.py`

**Interfaces:**
- Consumes: everything above; `agent_engine.config.loader.load_experiment_config`; `agent_engine.runner.providers.setup_model_provider`; `agent_engine.runner.tools.build_tool_registry` with `ToolDeps`; `agent_engine.caching.CacheManager`; `gepa_integration.reflection.make_token_counter`; `transformers.AutoTokenizer`.
- Produces:
  - CLI: `python scripts/run_prompt_search.py --config <yaml> --domain search|math [--run-name NAME] [--dry-run] [--report-only]`.
    - The YAML is an `ExperimentConfig` (loaded with `load_experiment_config`, which ignores the extra key) plus a `search:` block parsed into `SearchConfig`.
    - `out_dir = search.output_root / run_name / domain`; `run_name` defaults to the config file stem.
    - `--dry-run`: load data, build pools, write `config.json`, `pools.json`, `bundles/v0.json`, print `shown=<n> screen=<n> heldout=<n>`, exit 0. No model, no server: the tools are built with a stub provider object that is never called, so the sub-agent-mode schemas (which depend on a provider being set) are produced.
    - `--report-only`: call `write_report` on an existing `out_dir`.
    - Normal run: build providers and tools exactly like `run_experiment` (orchestrator model; every enabled tool falls back to it), `CacheManager(dataset_name="rsi_" + domain)`, `BundleExecutor`, `build_reflector(search.reflector, orchestrator_provider)`, token counter from the reflector model's tokenizer, then `run_search`, then `write_report`. For `backend: in_process`, assert `search.reflector.model == models.orchestrator.path_or_id` so the reflector really is the executor model.
  - Config `search_reflector_qwen3_8b.yaml` below. The 32B file differs only in `name` and the reflector block: `backend: "openai"`, `model: "Qwen/Qwen3-32B"`, `base_url: "http://localhost:8001/v1"`, same budgets (so the two reflectors are compared at equal input and output budgets).

```yaml
name: "rsi_phase1_qwen3_8b_reflector_qwen3_8b"
models:
  orchestrator:
    name: "Qwen3-8B"
    family: "qwen3"
    path_or_id: "Qwen/Qwen3-8B"
    role: "orchestrator"
    gpu_memory_utilization: 0.70
tools:
  enabled_tools: [web_search, code_generator]
  direct_tool_call: false
  web_tool_provider: "serper"
  max_search_limit: 10
max_turns: 7
batch_size: -1
seed: 0
thinking_mode: "NO"
cache_dir: "./cache"
search:
  data_path: "data/training/train/combined_train.parquet"
  pools: {shown: 300, screen: 200, heldout: 50}
  pool_seed: 0
  rounds: 10
  candidates_per_round: 3
  shown_minibatch: 16
  screen_minibatch: 16
  max_parse_retries: 2
  history_max: 8
  ngram: 8
  seed: 0
  output_root: "experiments/results/rsi/phase1"
  reflector:
    backend: "in_process"
    model: "Qwen/Qwen3-8B"
    max_model_len: 32768
    max_tokens: 16384
    input_budget_tokens: 14000
    per_call_chars: 1500
    enable_thinking: true
```

  - Jobs: `_prompt_search_body.sh` takes `CONFIG`, `DOMAIN`, `EXTRA_ARGS` and optional `REFLECTOR_SERVER_MODEL`, `REFLECTOR_GPUS`, `REFLECTOR_TP`; loads modules/env/`.env` like `jobs/gepa/006_run_gepa_gaia.job`. Only when `REFLECTOR_SERVER_MODEL` is set it starts `vllm serve $REFLECTOR_SERVER_MODEL --port 8001 --tensor-parallel-size $REFLECTOR_TP --max-model-len 40960 --enable-prefix-caching` on `CUDA_VISIBLE_DEVICES=$REFLECTOR_GPUS`, waits for `/v1/models`, and a cleanup trap kills it. It then runs the CLI with `CUDA_VISIBLE_DEVICES=0`.
    - `001_prompt_search_8b.job`: `--gpus=1`, no server (in-process reflector).
    - `002_prompt_search_32b.job`: `--gpus=3`, `REFLECTOR_SERVER_MODEL=Qwen/Qwen3-32B` on GPUs 1,2, TP 2.
    - Both: `--time=24:00:00`, logs under `out/rsi/`, `DOMAIN` from `sbatch --export=ALL,DOMAIN=search|math`, default `search`.

- [ ] **Step 1: Write the failing test**

```python
def test_dry_run_writes_pools_and_seed_bundle(tmp_path, tiny_parquet, monkeypatch):
    cfg = write_config(tmp_path, data_path=tiny_parquet, output_root=tmp_path / "out",
                       pools={"shown": 10, "screen": 6, "heldout": 4})
    monkeypatch.setenv("SERPER_API_KEY", "x")
    rc = main(["--config", str(cfg), "--domain", "search", "--dry-run"])
    out = tmp_path / "out" / cfg.stem / "search"
    assert rc == 0
    pools = json.loads((out / "pools.json").read_text())
    assert [len(pools[k]) for k in ("shown", "screen", "heldout")] == [10, 6, 4]
    assert load_bundle(out / "bundles" / "v0.json").domain == "search"

def test_shipped_configs_parse():
    for p in Path("experiments/configs/rsi/phase1").glob("*.yaml"):
        exp, search = load_search_config(p)
        assert exp.thinking_mode.value == "NO" and exp.max_turns == 7
        assert search.reflector.input_budget_tokens + search.reflector.max_tokens <= search.reflector.max_model_len
```

(`main(argv) -> int` and `load_search_config(path) -> tuple[ExperimentConfig, SearchConfig]` live in `scripts/run_prompt_search.py`; import via `importlib` from the script path as other script tests do.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/rsi/test_run_prompt_search_cli.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** the CLI, the two configs and the job files.

- [ ] **Step 4: Run tests and a local dry run**

Run: `pytest tests/rsi -v`
Expected: PASS.
Run: `python scripts/run_prompt_search.py --config experiments/configs/rsi/phase1/search_reflector_qwen3_8b.yaml --domain math --dry-run --run-name dryrun`
Expected: prints `shown=300 screen=200 heldout=50`, writes `experiments/results/rsi/phase1/dryrun/math/{config.json,pools.json,bundles/v0.json}`. Delete that `dryrun` directory afterwards.
Run: `bash -n jobs/rsi/_prompt_search_body.sh jobs/rsi/001_prompt_search_8b.job jobs/rsi/002_prompt_search_32b.job`
Expected: no output.

- [ ] **Step 5: Stage**

```bash
git add scripts/run_prompt_search.py experiments/configs/rsi/phase1 jobs/rsi tests/rsi/test_run_prompt_search_cli.py
```

---

### Task 14: Optional diagnostic: benchmarks under a bundle

**Files:**
- Modify: `src/agent_engine/config/schema.py` (`ExperimentConfig`, :127-133)
- Modify: `src/agent_engine/runner/experiment.py` (system prompt and orchestrator construction, :186-215)
- Test: `tests/unit/test_prompt_bundle_inference.py` (pattern: `tests/unit/test_gepa_inference.py`)

**Interfaces:**
- Consumes: Task 4 `load_bundle`, `render_system_prompt`, `planning_suffix`, `acting_suffix`, `apply_subagent_templates`.
- Produces: `ExperimentConfig.prompt_bundle_path: Optional[Path] = None` (path-coerced like `gepa_prompt_path`). In the runner, when set: `apply_subagent_templates(bundle, tools)`, `system_prompt_for_config = render_system_prompt(bundle, tool_schemas, tcf)`, orchestrator gets `planning_suffix(bundle)` and `acting_suffix(bundle)`. Setting both `prompt_bundle_path` and `gepa_prompt_path` raises `ValueError` at load time. The bundle file is copied into the run directory as `prompt_bundle.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prompt_bundle_path_parses(tmp_path):
    cfg = load_from_yaml(_BASE_YAML + "prompt_bundle_path: some/bundle.json\n", tmp_path)
    assert cfg.prompt_bundle_path == Path("some/bundle.json")

def test_both_prompt_sources_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_from_yaml(_BASE_YAML + "prompt_bundle_path: a.json\ngepa_prompt_path: b.json\n", tmp_path)

def test_runner_uses_bundle_prompts(monkeypatch, tmp_path):
    bundle = replace_unit(seed_bundle("search", SCHEMAS), "orchestrator",
                          {"guidance_opening": "You are a coordinator.", "turn1n_prompt": "Check links."})
    captured = run_runner_with_bundle(monkeypatch, tmp_path, bundle)  # patches providers/tools like test_gepa_inference
    assert captured["system_prompt"].startswith("You are a coordinator.")
    assert captured["orchestrator_kwargs"]["acting_suffix"] == "\n\nCheck links."
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/test_prompt_bundle_inference.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement** the schema field, the mutual-exclusion validator and the runner branch.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_prompt_bundle_inference.py tests/unit/test_gepa_inference.py -v`
Expected: PASS.

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/config/schema.py src/agent_engine/runner/experiment.py tests/unit/test_prompt_bundle_inference.py
```

---

## Final verification

- [ ] Run: `pytest tests -q`
  Expected: all PASS; `git status` shows no change under `tests/characterization/fixtures/`.
- [ ] Run: `black --check -l 100 src/rsi scripts/run_prompt_search.py && isort --check --profile black src/rsi scripts/run_prompt_search.py`
  Expected: no changes needed.
- [ ] Hand-off note for the author: submit `sbatch --export=ALL,DOMAIN=search jobs/rsi/001_prompt_search_8b.job` (then `DOMAIN=math`); read `experiments/results/rsi/phase1/search_reflector_qwen3_8b/<domain>/report.md`; Phase 2 is not started by anything in this plan.
