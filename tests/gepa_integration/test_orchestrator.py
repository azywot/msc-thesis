import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.orchestrator import (
    AgenticOrchestrator,
    _DEFAULT_PLANNING_SUFFIX_NO_TOOLS,
    _DEFAULT_PLANNING_SUFFIX_TOOLS,
)
from agent_engine.core.tool import ToolRegistry


def _make_orchestrator(planning_suffix=None, with_tools=True):
    model = MagicMock()
    model.config = MagicMock()
    model.config.supports_thinking = False
    model.config.family = "qwen3"
    tools = ToolRegistry()
    if with_tools:
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.get_schema.return_value = {"function": {"name": "web_search"}}
        tools.register(mock_tool)
    return AgenticOrchestrator(
        model_provider=model,
        tool_registry=tools,
        planning_suffix=planning_suffix,
    )


def test_default_planning_suffix_constants_exist():
    assert isinstance(_DEFAULT_PLANNING_SUFFIX_NO_TOOLS, str)
    assert isinstance(_DEFAULT_PLANNING_SUFFIX_TOOLS, str)
    assert "tools" in _DEFAULT_PLANNING_SUFFIX_TOOLS.lower()
    assert len(_DEFAULT_PLANNING_SUFFIX_NO_TOOLS) > 20


def test_orchestrator_stores_planning_suffix():
    orch = _make_orchestrator(planning_suffix="custom suffix")
    assert orch.planning_suffix == "custom suffix"


def test_orchestrator_planning_suffix_defaults_none():
    orch = _make_orchestrator(planning_suffix=None)
    assert orch.planning_suffix is None


def test_raw_query_analysis_stored_on_state():
    """planning turn stores raw output (with thinking) before stripping."""
    orch = _make_orchestrator()

    raw_text = "<think>internal</think>analysis"
    gen_result = MagicMock()
    gen_result.text = raw_text
    gen_result.usage = {}
    orch.model.generate.return_value = [gen_result]
    orch.model.apply_chat_template.return_value = "prompt"

    from agent_engine.core.state import ExecutionState
    state = ExecutionState(
        question_id=1,
        question="Q",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "Q"}],
    )
    orch._run_planning_turn([state])

    assert state.raw_query_analysis == raw_text
    assert "<think>" not in state.query_analysis
