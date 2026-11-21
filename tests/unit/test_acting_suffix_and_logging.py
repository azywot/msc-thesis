"""Acting suffix on memory turns and sub-agent call logging in flush_batches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_engine.core.batching import flush_batches
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry
from agent_engine.tools.code_generator import CodeGeneratorTool
from agent_engine.tools.web_search import WebSearchTool


@pytest.fixture
def orch_factory():
    def make(**kwargs):
        model = MagicMock()
        model.config = MagicMock()
        model.config.supports_thinking = False
        model.config.family = "qwen3"
        return AgenticOrchestrator(model_provider=model, tool_registry=ToolRegistry(), **kwargs)

    return make


@pytest.fixture
def state_with_history():
    return ExecutionState(
        question_id=0,
        question="Q?",
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q?"}],
        query_analysis="analysis",
        action_history=[
            {"tool_name": "web_search", "sub_goal": "g", "command": "{}", "result": "R"}
        ],
    )


def test_empty_acting_suffix_leaves_memory_prompt_unchanged(orch_factory, state_with_history):
    a = orch_factory()._build_memory_prompt(state_with_history, "SYS")
    b = orch_factory(acting_suffix="")._build_memory_prompt(state_with_history, "SYS")
    assert a == b


def test_acting_suffix_is_appended_after_previous_steps(orch_factory, state_with_history):
    msgs = orch_factory(acting_suffix="\n\nCheck every link.")._build_memory_prompt(
        state_with_history, "SYS"
    )
    history = AgenticOrchestrator._format_action_history(state_with_history.action_history)
    assert msgs[1]["content"].endswith("**Previous Steps:**\n" + history + "\n\nCheck every link.")


class _Provider:
    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return "".join(m["content"] for m in messages)

    def generate(self, prompts):
        return [
            SimpleNamespace(text="<think>x</think>```python\nprint(1)\n```", usage={})
            for _ in prompts
        ]


def _commit_appending_history(state, tool_call, text):
    state.action_history.append({"tool_name": tool_call["name"], "result": text})


def _state():
    return SimpleNamespace(question_id=0, attachments=None, metadata={}, action_history=[])


def test_flush_records_subagent_calls_with_action_index():
    code = CodeGeneratorTool(model_provider=_Provider(), return_code=True)
    state = _state()
    job = code.prepare(state, {"name": "code_generator", "arguments": {"task": "t"}}, {"task": "t"})
    flush_batches(
        {"code_generator": [job]},
        commit=_commit_appending_history,
        accumulate_usage=lambda *a: None,
    )
    rec = state.metadata["subagent_calls"][0]
    assert rec["tool"] == "code_generator" and rec["inputs"]["task"] == "t"
    assert rec["action_index"] == 0 and "response" in rec and "output" in rec
    assert rec["response"] == "```python\nprint(1)\n```"
    assert rec["output"] == state.action_history[0]["result"]


def test_web_subagent_record_shape():
    web = WebSearchTool(api_key="k", search_cache={}, url_cache={}, model_provider=_Provider())
    job = SimpleNamespace(payload={"query": "q", "payload": {"results": []}})
    gen = SimpleNamespace(text="<think>t</think>analysis")
    result = SimpleNamespace(output="analysis")
    rec = web.subagent_record(job, gen, result)
    assert rec == {
        "tool": "web_search",
        "inputs": {"query": "q", "formatted_results": web._format_results([], "q")},
        "response": "analysis",
        "output": "analysis",
    }


def test_action_index_counts_prior_history():
    code = CodeGeneratorTool(model_provider=_Provider(), return_code=True)
    state = _state()
    state.action_history.append({"tool_name": "x"})
    job = code.prepare(state, {"name": "code_generator", "arguments": {"task": "t"}}, {"task": "t"})
    flush_batches(
        {"code_generator": [job]},
        commit=_commit_appending_history,
        accumulate_usage=lambda *a: None,
    )
    assert state.metadata["subagent_calls"][0]["action_index"] == 1
