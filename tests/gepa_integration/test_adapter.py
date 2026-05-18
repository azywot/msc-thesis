import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry
from agent_engine.datasets.base import DatasetExample
from gepa.core.adapter import EvaluationBatch as GEPAEvaluationBatch
from gepa_integration.adapter import AgentGEPAAdapter, _extract_thinking


# ── _extract_thinking ────────────────────────────────────────────────────────

def test_extract_thinking_returns_content():
    text = "<think>internal reasoning here</think>visible output"
    assert _extract_thinking(text) == "internal reasoning here"


def test_extract_thinking_returns_empty_when_no_tags():
    assert _extract_thinking("no think tags here") == ""


def test_extract_thinking_handles_multiline():
    text = "<think>\nline1\nline2\n</think>answer"
    result = _extract_thinking(text)
    assert "line1" in result
    assert "line2" in result


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_adapter():
    model = MagicMock()
    model.config = MagicMock()
    model.config.supports_thinking = True
    model.config.family = "qwen3"
    tools = ToolRegistry()
    return AgentGEPAAdapter(
        model_provider=model,
        tool_registry=tools,
        use_thinking=True,
        max_turns=3,
    )


def _make_example(qid, question, answer, choices=None):
    meta = {}
    if choices is not None:
        meta["choices"] = choices
    return DatasetExample(question_id=qid, question=question, answer=answer, metadata=meta)


def _make_state(qid, question, answer, correct, tool_calls=None, raw_plan=None, action_history=None):
    state = ExecutionState(question_id=qid, question=question, messages=[], answer=answer, finished=True)
    state.metadata["ground_truth"] = "correct_answer" if correct else "other"
    state.query_analysis = "plan summary"
    state.raw_query_analysis = raw_plan or "<think>think</think>plan summary"
    state.tool_calls = tool_calls or []
    state.action_history = action_history or []
    return state


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_returns_correct_length():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q?", "4")]
    state = ExecutionState(question_id=1, question="Q?", messages=[], answer="4", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate, capture_traces=False)
    assert len(result.outputs) == 1
    assert len(result.scores) == 1
    assert result.trajectories is None


def test_evaluate_score_correct():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "Paris")]
    state = ExecutionState(question_id=1, question="Q", messages=[], answer="Paris", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate)
    assert result.scores[0] == 1.0


def test_evaluate_score_wrong():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "Paris")]
    state = ExecutionState(question_id=1, question="Q", messages=[], answer="Berlin", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate)
    assert result.scores[0] == 0.0


def test_evaluate_captures_trajectories():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "A")]
    state = ExecutionState(question_id=1, question="Q", messages=[], answer="A", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate, capture_traces=True)
    assert result.trajectories is not None
    assert result.trajectories[0] is state


def test_evaluate_passes_planning_suffix():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "MY_CUSTOM_SUFFIX"}
    examples = [_make_example(1, "Q", "A")]
    state = ExecutionState(question_id=1, question="Q", messages=[], answer="A", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        adapter.evaluate(examples, candidate)
    assert MockOrch.call_args.kwargs["planning_suffix"] == "MY_CUSTOM_SUFFIX"


def test_evaluate_stores_ground_truth():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "correct answer")]
    state = ExecutionState(question_id=1, question="Q", messages=[], answer="wrong", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        adapter.evaluate(examples, candidate, capture_traces=True)
    assert state.metadata["ground_truth"] == "correct answer"


# ── make_reflective_dataset ──────────────────────────────────────────────────

def test_make_reflective_dataset_system_prompt_key():
    adapter = _make_adapter()
    states = [_make_state(1, "Q", "correct_answer", correct=True)]
    batch = GEPAEvaluationBatch(outputs=["correct_answer"], scores=[1.0], trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert "system_prompt" in result


def test_make_reflective_dataset_planning_suffix_key():
    adapter = _make_adapter()
    states = [_make_state(1, "Q", "correct_answer", correct=True)]
    batch = GEPAEvaluationBatch(outputs=["correct_answer"], scores=[1.0], trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    assert "planning_suffix" in result


def test_make_reflective_dataset_correct_feedback():
    adapter = _make_adapter()
    states = [_make_state(1, "Q", "correct_answer", correct=True)]
    batch = GEPAEvaluationBatch(outputs=["correct_answer"], scores=[1.0], trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    feedback = result["system_prompt"][0]["Feedback"]
    assert feedback.startswith("CORRECT")
    assert "correct_answer" in feedback  # the predicted answer is echoed back


def test_make_reflective_dataset_wrong_contains_gt():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "wrong_answer", correct=False)
    state.metadata["ground_truth"] = "real_answer"
    batch = GEPAEvaluationBatch(outputs=["wrong_answer"], scores=[0.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert "real_answer" in result["system_prompt"][0]["Feedback"]


def test_make_reflective_dataset_planning_uses_raw():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "A", correct=True, raw_plan="<think>reasoning</think>plan")
    batch = GEPAEvaluationBatch(outputs=["A"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    raw = result["planning_suffix"][0]["Generated Outputs"]["raw_planning_output"]
    assert "<think>reasoning</think>" in raw


def test_make_reflective_dataset_capped_at_12():
    adapter = _make_adapter()
    states = [_make_state(i, "Q", "A", correct=True) for i in range(20)]
    batch = GEPAEvaluationBatch(outputs=["A"] * 20, scores=[1.0] * 20, trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert len(result["system_prompt"]) <= 12


# ── _diagnose (enriched feedback) ────────────────────────────────────────────

def _wrong_state(qid, pred, gt, eval_result=None, tool_counts=None,
                 action_history=None, max_turns_reached=False, choices=None):
    """Build an ExecutionState as evaluate() would leave it for a wrong answer."""
    state = ExecutionState(question_id=qid, question="Q?", messages=[], answer=pred, finished=True)
    state.metadata["ground_truth"] = gt
    state.metadata["eval_result"] = eval_result or {"em": 0.0, "f1": 0.0, "accuracy": 0.0}
    state.metadata["choices"] = choices
    if max_turns_reached:
        state.metadata["max_turns_reached"] = True
    if tool_counts:
        state.tool_counts.update(tool_counts)
    state.action_history = action_history or []
    state.turn = sum((tool_counts or {}).values())
    return state


def test_diagnose_correct_includes_tools_and_turns():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "Paris", correct=True)
    state.tool_counts["web_search"] = 2
    state.turn = 3
    fb = adapter._diagnose(state, score=1.0)
    assert fb.startswith("CORRECT")
    assert "web_search" in fb
    assert "turns=3" in fb


def test_diagnose_correct_no_tools_says_none():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "Paris", correct=True)
    fb = adapter._diagnose(state, score=1.0)
    assert "tools=none" in fb


def test_diagnose_wrong_includes_score_breakdown():
    adapter = _make_adapter()
    state = _wrong_state(1, "Berlin", "Paris",
                         eval_result={"em": 0.0, "f1": 0.0, "accuracy": 0.0})
    fb = adapter._diagnose(state, score=0.0)
    assert fb.startswith("WRONG.")
    assert "em=0.00" in fb
    assert "f1=0.00" in fb


def test_diagnose_wrong_includes_normalised_forms():
    adapter = _make_adapter()
    state = _wrong_state(1, "Berlin.", "Paris!")
    fb = adapter._diagnose(state, score=0.0)
    assert "Normalised:" in fb
    assert "'berlin'" in fb
    assert "'paris'" in fb


def test_diagnose_wrong_empty_prediction():
    adapter = _make_adapter()
    state = _wrong_state(1, "", "Paris")
    fb = adapter._diagnose(state, score=0.0)
    assert "No final answer was produced." in fb


def test_diagnose_wrong_format_mismatch_numeric_vs_prose():
    adapter = _make_adapter()
    # Punctuation in the prediction guarantees is_math_answer returns False
    # (MATH_TOKEN_PATTERN allows letters/digits/operators but not commas).
    state = _wrong_state(
        1,
        "pi, the ratio of a circle's circumference to its diameter",
        "3.14",
    )
    fb = adapter._diagnose(state, score=0.0)
    assert "Format mismatch" in fb


def test_diagnose_wrong_verbose_for_short_gold():
    adapter = _make_adapter()
    state = _wrong_state(
        1,
        "The capital city of France is the famous city of Paris located in Europe",
        "Berlin",
    )
    fb = adapter._diagnose(state, score=0.0)
    assert "much longer than the gold answer" in fb


def test_diagnose_wrong_high_f1_hint():
    adapter = _make_adapter()
    state = _wrong_state(
        1, "42 hours", "42",
        eval_result={"em": 0.0, "f1": 0.66, "accuracy": 0.0},
    )
    fb = adapter._diagnose(state, score=0.0)
    assert "Token overlap is high" in fb


def test_diagnose_wrong_parametric_memory_flag():
    adapter = _make_adapter()
    state = _wrong_state(1, "Berlin", "Paris")  # no tools called
    fb = adapter._diagnose(state, score=0.0)
    assert "parametric memory" in fb


def test_diagnose_wrong_counts_tool_errors():
    adapter = _make_adapter()
    state = _wrong_state(
        1, "x", "Paris",
        tool_counts={"web_search": 3},
        action_history=[
            {"tool_name": "web_search", "result": "Error: timeout"},
            {"tool_name": "web_search", "result": "Some valid content"},
            {"tool_name": "web_search", "result": "Exception: 429 rate limited"},
        ],
    )
    fb = adapter._diagnose(state, score=0.0)
    assert "2/3 tool calls returned an error" in fb


def test_diagnose_wrong_max_turns_reached():
    adapter = _make_adapter()
    state = _wrong_state(1, "", "Paris", max_turns_reached=True)
    fb = adapter._diagnose(state, score=0.0)
    assert "Max turns" in fb
    assert "without a final answer" in fb


def test_diagnose_wrong_multiple_choice_skips_format_checks():
    adapter = _make_adapter()
    # GPQA-style: gold is a letter, pred is a letter — none of the numeric/
    # verbose heuristics should fire because choices is set.
    state = _wrong_state(1, "B", "A", choices=["w", "x", "y", "z"])
    fb = adapter._diagnose(state, score=0.0)
    assert "Format mismatch" not in fb
    assert "much longer than the gold" not in fb


def test_evaluate_stashes_eval_result_and_choices():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "A", choices=["a", "b", "c", "d"])]
    state = ExecutionState(question_id=1, question="Q", messages=[], answer="A", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        adapter.evaluate(examples, candidate, capture_traces=True)
    assert "eval_result" in state.metadata
    assert state.metadata["eval_result"]["accuracy"] in (0.0, 1.0)
    assert state.metadata["choices"] == ["a", "b", "c", "d"]


def test_make_reflective_dataset_wrong_includes_tool_signal():
    """End-to-end check: the new feedback reaches the reflective record."""
    adapter = _make_adapter()
    state = _wrong_state(
        1, "Berlin", "Paris",
        tool_counts={"web_search": 2},
        action_history=[
            {"tool_name": "web_search", "result": "Error: search failed"},
            {"tool_name": "web_search", "result": "ok"},
        ],
    )
    state.raw_query_analysis = state.query_analysis = "plan"
    batch = GEPAEvaluationBatch(outputs=["Berlin"], scores=[0.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt", "planning_suffix"])
    sys_fb = result["system_prompt"][0]["Feedback"]
    plan_fb = result["planning_suffix"][0]["Feedback"]
    assert "1/2 tool calls returned an error" in sys_fb
    # Both record types share _diagnose, so both surface the same signal.
    assert "1/2 tool calls returned an error" in plan_fb
