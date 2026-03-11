# AgentFlow Alignment

How the msc-thesis structured memory orchestrator maps to each AgentFlow module.

## Architecture Mapping

| AgentFlow Module | AgentFlow File | msc-thesis Equivalent | msc-thesis File |
|---|---|---|---|
| **Planner** | `models/planner.py` | Orchestrator LLM (planning turn + action turns) | `core/orchestrator.py` |
| **Executor** | `models/executor.py` | Tool sub-agents (web_search, code_generator) | `tools/web_search.py`, `tools/code_generator.py` |
| **Memory** | `models/memory.py` | `ExecutionState.query_analysis` + `action_history` | `core/state.py` |
| **Verifier** | `models/verifier.py` | Implicit: model decides to stop (no tool call = finished) | `core/orchestrator.py` |

## Solver Loop Comparison

### AgentFlow (`solver.py`)

```
1. query_analysis = planner.analyze_query(question)
2. loop step_count = 1..max_steps:
   a. next_step = planner.generate_next_step(question, query_analysis, memory)
   b. context, sub_goal, tool_name = extract(next_step)
   c. command = executor.generate_tool_command(context, sub_goal, tool_name)
   d. result  = executor.execute_tool_command(tool_name, command)
   e. memory.add_action(step_count, tool_name, sub_goal, command, result)
   f. conclusion = verifier.verificate_context(question, query_analysis, memory)
   g. if conclusion == STOP: break
3. final_output = planner.generate_final_output(question, memory)
```

### msc-thesis (`orchestrator.py`)

```
1. _run_planning_turn(states)       → state.query_analysis
2. loop turn = 1..max_turns:
   a. prompt = _build_memory_prompt(state, system_prompt)
      contains: question + query_analysis + previous_steps
   b. output = model.generate(prompt)
   c. tool_call = parse_tool_call(output)
   d. if no tool_call: finished (implicit verifier → STOP)
   e. result = execute_tool(tool_call)
   f. state.action_history.append({tool_name, sub_goal, command, result})
3. answer = extract_answer(last output)
```

## Key Differences

### Planner + Verifier in one model

AgentFlow uses separate LLM calls for planning and verification. We use a single orchestrator model that decides both what tool to call next AND when to stop (by not producing a `<tool_call>`). This is the standard agentic reasoning pattern for tool-calling LLMs.

### Executor: sub-agents vs. direct dispatch

AgentFlow's Executor generates a command string via LLM, then executes it. In msc-thesis:
- **Sub-agent mode**: web_search and code_generator have their own LLM that generates analysis/code (matches AgentFlow's Executor pattern).
- **Direct mode**: the orchestrator emits the tool call arguments directly (no separate Executor LLM).

In both modes, sub-agents no longer receive `prev_reasoning` — they operate independently, like AgentFlow's Executor which only receives `context` and `sub_goal` from the Planner.

### Tool schemas in system prompt vs. repeated in user message

AgentFlow injects `Available Tools` and `Toolbox Metadata` into every planner prompt because it uses stateless single-turn API calls. In msc-thesis, tool schemas are in the system prompt (via chat template), so they don't need repeating in the user message.

## Memory Structure

### AgentFlow (`memory.py`)

```python
memory.add_action(step_count, tool_name, sub_goal, command, result)
# Stored as:
# {"Action Step 1": {"tool_name": ..., "sub_goal": ..., "command": ..., "result": ...}}
```

Retrieved via `memory.get_actions()` and injected as `**Previous Steps:**` in the planner prompt.

### msc-thesis (`state.py`)

```python
state.action_history.append({
    "tool_name": "web_search",
    "sub_goal": "Search for the capital of France",
    "command": '{"query": "capital of France"}',
    "result": "Paris is the capital..."
})
```

Formatted by `_format_action_history()` into:
```
Action Step 1:
  - Tool: web_search
  - Sub-goal: Search for the capital of France
  - Command: {"query": "capital of France"}
  - Result: Paris is the capital...
```

Injected as `**Previous Steps:**` in the user message via `_build_memory_prompt()`.

Same four fields, same `Action Step N` naming, same prompt label.

## Planning Turn

### AgentFlow (`planner.analyze_query`)

```
Task: Analyze the given query to determine necessary skills and tools.
Inputs:
- Query: {question}
- Available tools: {available_tools}
- Metadata for tools: {toolbox_metadata}
Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.
```

### msc-thesis (`_run_planning_turn`)

Appends to the user message:
```
Before using any tools, analyze this query to determine the approach needed.
Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each tool, explain how it helps address the query.
4. Note any additional considerations.
Be brief and precise. Do NOT call any tools yet.
```

Same 4-point structure. Tool information is already in the system prompt.

## Action Turn Prompt

### AgentFlow (`planner.generate_next_step`)

```
Context:
- **Query:** {question}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {available_tools}
- **Toolbox Metadata:** {toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}
Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the single best tool for the next step.
3. Formulate a specific, achievable sub-goal for that tool.
4. Provide all necessary context for the tool to function.
```

### msc-thesis (`_build_memory_prompt`)

```
[system: tool schemas + instructions]
[user:
  <original question + attachments>

  **Query Analysis:**
  <planning turn output>

  **Previous Steps:**
  Action Step 1:
    - Tool: ...
    - Sub-goal: ...
    - Command: ...
    - Result: ...
]
```

Same content: query, query analysis, previous steps. Tool information lives in the system prompt instead of repeated in the user message.

## Files Changed

| File | Change |
|---|---|
| `core/state.py` | Added `query_analysis: str` and `action_history: List[Dict]` |
| `core/orchestrator.py` | Added `_build_memory_prompt`, `_format_action_history`, `_extract_sub_goal`, `_run_planning_turn`; removed `_inject_reasoning_context`; action turns use memory prompts instead of full conversation |
| `tools/web_search.py` | Removed `prev_reasoning` from `execute()`, `build_analysis_prompt()`, `_analyze_with_llm()` |
| `tools/code_generator.py` | Removed `context` from `execute()` and `generate_code()` |
| `utils/reasoning_context.py` | Deleted `get_reasoning_context_for_state` and helpers; kept `get_attachment_context_for_code` |
| `scripts/run_experiment.py` | Added `query_analysis` and `action_history` to results output |
