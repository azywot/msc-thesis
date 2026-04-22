"""Unit tests for the sub-agent shared-memory ablation.

Covers:
1. `_build_subagent_shared_context` rendering
2. Prompt-builder injection (web_search, code_generator, text_inspector)
3. web_search analysis-cache bypass
4. Config validation (baseline/direct_tool_call incompatibilities, negative K)
5. Token accounting (state.subagent_shared_memory_tokens)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_engine.config.schema import ExperimentConfig, ToolsConfig
from agent_engine.core.orchestrator import AgenticOrchestrator, _SHARED_MEMORY_TOOLS
from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.base import ModelConfig, ModelFamily
from agent_engine.tools.code_generator import CodeGeneratorTool
from agent_engine.tools.text_inspector import TextInspectorTool
from agent_engine.tools.web_search import WebSearchTool


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(enabled: bool = True, last_k: int = 5) -> AgenticOrchestrator:
    """Build an AgenticOrchestrator with a fully-mocked model provider."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.supports_thinking = False
    model.config.family = ModelFamily.QWEN3
    model.tokenizer = None  # force char-based fallback for deterministic counts
    return AgenticOrchestrator(
        model_provider=model,
        tool_registry=ToolRegistry(),
        max_turns=5,
        subagent_shared_memory=enabled,
        subagent_shared_memory_last_k=last_k,
    )


def _make_state(
    *,
    question: str = "What is 2+2?",
    query_analysis: str = "",
    action_history=None,
    current_output: str = "",
) -> ExecutionState:
    state = ExecutionState(
        question_id=42,
        question=question,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": question},
        ],
        query_analysis=query_analysis,
        action_history=action_history or [],
        current_output=current_output,
    )
    state.turn = len(state.action_history)
    return state


# ---------------------------------------------------------------------------
# 1. Shared-context renderer
# ---------------------------------------------------------------------------

class TestBuildSubagentSharedContext:
    def test_returns_empty_when_disabled(self):
        orch = _make_orchestrator(enabled=False)
        state = _make_state(query_analysis="foo", action_history=[
            {"tool_name": "web_search", "sub_goal": "sg", "command": "{}", "result": "r"},
        ])
        assert orch._build_subagent_shared_context(state, current_output="<sub_goal>now</sub_goal>") == ""

    def test_renders_all_sections_and_strips_results(self):
        orch = _make_orchestrator(enabled=True, last_k=5)
        state = _make_state(
            question="Solve X.",
            query_analysis="Decompose into A and B.",
            action_history=[
                {"tool_name": "web_search", "sub_goal": "look up A", "command": '{"name":"web_search"}', "result": "HUGE_RESULT_A"},
                {"tool_name": "code_generator", "sub_goal": "compute B", "command": '{"name":"code_generator"}', "result": "HUGE_RESULT_B"},
            ],
        )
        ctx = orch._build_subagent_shared_context(state, current_output="text <sub_goal>final check</sub_goal> more")

        assert "**Original Question:**" in ctx
        assert "Solve X." in ctx
        # query_analysis is intentionally *not* rendered (stale + redundant
        # with action_history; see orchestrator._build_subagent_shared_context
        # docstring and docs/subagent_shared_memory_plan.md §2).
        assert "**Query Analysis:**" not in ctx
        assert "Decompose into A and B." not in ctx
        assert "**Previous Steps (last 2):**" in ctx
        assert "Action Step 1:" in ctx
        assert "Action Step 2:" in ctx
        assert "look up A" in ctx
        assert "**Current Sub-goal:**" in ctx
        assert "final check" in ctx
        # Results must be stripped
        assert "HUGE_RESULT_A" not in ctx
        assert "HUGE_RESULT_B" not in ctx
        assert "- Result:" not in ctx
        # Commands are also stripped (noise; sub-goal already conveys intent)
        assert "- Command:" not in ctx
        assert '{"name":"web_search"}' not in ctx

    def test_absolute_numbering_preserved_on_tail_slice(self):
        orch = _make_orchestrator(enabled=True, last_k=2)
        history = [
            {"tool_name": "web_search", "sub_goal": f"sg{i}", "command": "{}", "result": "r"}
            for i in range(5)
        ]
        state = _make_state(action_history=history)
        ctx = orch._build_subagent_shared_context(state, current_output="")

        # Last 2 of 5 → absolute steps 4 and 5.
        assert "Action Step 4:" in ctx
        assert "Action Step 5:" in ctx
        assert "Action Step 1:" not in ctx
        assert "Action Step 3:" not in ctx
        assert "sg3" in ctx and "sg4" in ctx
        assert "sg0" not in ctx

    def test_last_k_zero_drops_previous_steps_block(self):
        orch = _make_orchestrator(enabled=True, last_k=0)
        state = _make_state(
            query_analysis="qa",  # populated but intentionally not rendered
            action_history=[{"tool_name": "x", "sub_goal": "y", "command": "{}", "result": ""}],
            current_output="<sub_goal>cur</sub_goal>",
        )
        ctx = orch._build_subagent_shared_context(state, current_output=state.current_output)
        assert "**Previous Steps" not in ctx
        assert "**Query Analysis:**" not in ctx
        assert "**Original Question:**" in ctx
        assert "**Current Sub-goal:**" in ctx

    def test_missing_optional_sections_collapse_cleanly(self):
        orch = _make_orchestrator(enabled=True)
        state = _make_state(question="Q only.")  # no qa, no history, no current_output
        ctx = orch._build_subagent_shared_context(state, current_output="")
        assert ctx.startswith("**Original Question:**")
        assert "Q only." in ctx
        assert "**Query Analysis:**" not in ctx
        assert "**Previous Steps" not in ctx
        assert "**Current Sub-goal:**" not in ctx


# ---------------------------------------------------------------------------
# 2. Prompt-builder injection
# ---------------------------------------------------------------------------

class TestPromptInjection:
    def _mock_model(self):
        """Minimal model-provider mock for tools that call apply_chat_template."""
        model = MagicMock()
        model.apply_chat_template = lambda messages, use_thinking=False, **kw: "\n".join(
            m["content"] if isinstance(m.get("content"), str) else str(m["content"])
            for m in messages
        )
        return model

    def test_web_search_empty_shared_context_is_regression_safe(self):
        tool = WebSearchTool(api_key="k", model_provider=self._mock_model())
        before = tool.build_analysis_prompt("q", "RESULTS")
        after = tool.build_analysis_prompt("q", "RESULTS", shared_context="")
        assert before == after

    def test_web_search_injects_context_above_task_instruction(self):
        tool = WebSearchTool(api_key="k", model_provider=self._mock_model())
        prompt = tool.build_analysis_prompt("q", "RESULTS", shared_context="MY_CTX_TOKEN")
        assert "MY_CTX_TOKEN" in prompt
        assert prompt.index("MY_CTX_TOKEN") < prompt.index("**Task Instruction:**")
        assert "**Shared context:**" in prompt

    def test_code_generator_empty_shared_context_is_regression_safe(self):
        tool = CodeGeneratorTool(model_provider=self._mock_model())
        before = tool.build_task_prompt("sum two ints")
        after = tool.build_task_prompt("sum two ints", shared_context="")
        assert before == after

    def test_code_generator_injects_above_attachment_context(self):
        tool = CodeGeneratorTool(model_provider=self._mock_model())
        prompt = tool.build_task_prompt(
            "sum two ints",
            attachment_context="ATT_CTX_TOKEN",
            shared_context="SHARED_CTX_TOKEN",
        )
        assert "SHARED_CTX_TOKEN" in prompt
        assert "ATT_CTX_TOKEN" in prompt
        assert prompt.index("SHARED_CTX_TOKEN") < prompt.index("ATT_CTX_TOKEN")

    def test_text_inspector_prepends_context_in_user_prompt(self):
        # Patch out the internal file-reading by calling _analyze_with_llm directly.
        tool = TextInspectorTool(model_provider=self._mock_model())
        captured = {}
        tool.model_provider.generate = lambda prompts: (
            captured.setdefault("prompt", prompts[0])
            or [MagicMock(text="ANS", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})]
        )
        out, _ = tool._analyze_with_llm(
            "FILE_CONTENT", "What?", shared_context="SHARED_CTX_TOKEN"
        )
        assert "SHARED_CTX_TOKEN" in captured["prompt"]
        # Context must come before the actual question prompt.
        assert captured["prompt"].index("SHARED_CTX_TOKEN") < captured["prompt"].index("Question:")


# ---------------------------------------------------------------------------
# 3. web_search analysis-cache bypass
# ---------------------------------------------------------------------------

class TestAnalysisCacheBypass:
    def test_bypass_when_enabled(self):
        orch = _make_orchestrator(enabled=True)
        tool = MagicMock()
        tool.direct_mode = False
        tool.build_analysis_prompt = MagicMock()
        tool.search_and_format = MagicMock(return_value={"results": [], "urls_to_fetch": [], "url_snippets": {}, "cached": True, "query": "q"})
        # Pre-populate the cache the way a previous run would.
        setattr(tool, "_analysis_cache", {"q": "CACHED_ANSWER"})

        state = _make_state()
        state.current_output = ""
        web_jobs = []
        immediate = []
        orch._schedule_web_job(state, {"name": "web_search", "arguments": {"query": "q"}},
                               tool, {"query": "q"}, web_jobs, immediate)
        # Should schedule a fresh job rather than serving the cache.
        assert len(immediate) == 0
        assert len(web_jobs) == 1

    def test_cache_hit_served_when_disabled(self):
        orch = _make_orchestrator(enabled=False)
        tool = MagicMock()
        tool.direct_mode = False
        tool.build_analysis_prompt = MagicMock()
        tool.search_and_format = MagicMock()
        setattr(tool, "_analysis_cache", {"q": "CACHED_ANSWER"})

        state = _make_state()
        state.current_output = ""
        web_jobs = []
        immediate = []
        orch._schedule_web_job(state, {"name": "web_search", "arguments": {"query": "q"}},
                               tool, {"query": "q"}, web_jobs, immediate)
        assert len(web_jobs) == 0
        assert len(immediate) == 1
        assert immediate[0].result.output == "CACHED_ANSWER"


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def _orch_model(self):
        return ModelConfig(name="m", family="qwen3", path_or_id="x", role="orchestrator")

    def test_baseline_plus_shared_memory_rejected(self):
        with pytest.raises(ValueError, match="baseline"):
            ExperimentConfig(
                name="t",
                models={"orchestrator": self._orch_model()},
                tools=ToolsConfig(subagent_shared_memory=True, direct_tool_call=False),
                baseline=True,
            )

    def test_direct_tool_call_plus_shared_memory_rejected(self):
        with pytest.raises(ValueError, match="direct_tool_call"):
            ExperimentConfig(
                name="t",
                models={"orchestrator": self._orch_model()},
                tools=ToolsConfig(subagent_shared_memory=True, direct_tool_call=True),
            )

    def test_negative_last_k_rejected(self):
        with pytest.raises(ValueError, match="last_k"):
            ExperimentConfig(
                name="t",
                models={"orchestrator": self._orch_model()},
                tools=ToolsConfig(subagent_shared_memory_last_k=-1),
            )

    def test_valid_shared_memory_config_ok(self):
        cfg = ExperimentConfig(
            name="t",
            models={"orchestrator": self._orch_model()},
            tools=ToolsConfig(subagent_shared_memory=True, direct_tool_call=False),
        )
        assert cfg.tools.subagent_shared_memory is True


# ---------------------------------------------------------------------------
# 5. Token accounting
# ---------------------------------------------------------------------------

class TestTokenAccounting:
    def test_counter_stays_zero_when_disabled(self):
        orch = _make_orchestrator(enabled=False)
        state = _make_state(query_analysis="a" * 100)
        _ = orch._maybe_attach_shared_context(
            tool_name="web_search", state=state,
            current_output="<sub_goal>cur</sub_goal>", task_payload="q",
        )
        assert state.subagent_shared_memory_tokens == 0

    def test_counter_increments_per_call(self):
        orch = _make_orchestrator(enabled=True)
        state = _make_state(query_analysis="a" * 40)
        ctx = orch._maybe_attach_shared_context(
            tool_name="web_search", state=state,
            current_output="<sub_goal>cur</sub_goal>", task_payload="q1",
        )
        assert state.subagent_shared_memory_tokens > 0
        first = state.subagent_shared_memory_tokens
        _ = orch._maybe_attach_shared_context(
            tool_name="code_generator", state=state,
            current_output="<sub_goal>cur</sub_goal>", task_payload="q2",
        )
        assert state.subagent_shared_memory_tokens > first
        assert ctx  # non-empty rendered context

    def test_tool_outside_scope_produces_no_context(self):
        orch = _make_orchestrator(enabled=True)
        state = _make_state(query_analysis="something")
        ctx = orch._maybe_attach_shared_context(
            tool_name="mind_map", state=state,
            current_output="", task_payload="",
        )
        assert ctx == ""
        assert state.subagent_shared_memory_tokens == 0

    def test_shared_memory_tools_set_matches_spec(self):
        assert _SHARED_MEMORY_TOOLS == frozenset({
            "web_search", "code_generator", "text_inspector", "image_inspector",
        })
