"""Unit tests for classify_failure cascade."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import pytest
from failure_modes.analyze_failure_modes import classify_failure


def _rec(action_history=None, prediction="wrong", turns=2, tool_counts=None):
    """Build a minimal failed record."""
    return {
        "correct": False,
        "prediction": prediction,
        "turns": turns,
        "action_history": action_history or [],
        "tool_counts": tool_counts or {},
    }


def _step(tool_name, sub_goal="", result="some result"):
    return {"tool_name": tool_name, "sub_goal": sub_goal, "command": "{}", "result": result}


# ── Priority 1: modality_tool_gap ───────────────────────────────────────────

class TestModalityToolGap:
    def test_video_analysis_tool(self):
        rec = _rec(
            action_history=[_step("video_analysis", result="")],
            tool_counts={"video_analysis": 1},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_image_inspector_tool(self):
        rec = _rec(
            action_history=[_step("image_inspector", sub_goal="inspect image", result="")],
            tool_counts={"image_inspector": 1},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_multiple_text_inspector_empty_with_image_keyword(self):
        rec = _rec(
            action_history=[
                _step("text_inspector", sub_goal="inspect the attached image file", result=""),
                _step("text_inspector", sub_goal="inspect the attached image file again", result=""),
            ],
            prediction="",
            tool_counts={"text_inspector": 2},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_multiple_text_inspector_empty_with_diagram_keyword(self):
        rec = _rec(
            action_history=[
                _step("text_inspector", sub_goal="read the diagram in the file", result=""),
                _step("text_inspector", sub_goal="read the diagram in the file", result=""),
            ],
            tool_counts={"text_inspector": 2},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_single_text_inspector_empty_no_modality_keyword_not_gap(self):
        # Only 1 text_inspector call — not a modality gap by signal B
        rec = _rec(
            action_history=[_step("text_inspector", sub_goal="read the file", result="")],
            tool_counts={"text_inspector": 1},
        )
        assert classify_failure(rec) != "modality_tool_gap"

    def test_multiple_text_inspector_with_results_not_gap(self):
        # All have non-empty results — not a modality gap
        rec = _rec(
            action_history=[
                _step("text_inspector", sub_goal="inspect image", result="some content"),
                _step("text_inspector", sub_goal="inspect image again", result="more content"),
            ],
            tool_counts={"text_inspector": 2},
        )
        assert classify_failure(rec) != "modality_tool_gap"


# ── Priority 2: tool_loop_or_empty_final ────────────────────────────────────

class TestToolLoop:
    def test_empty_prediction(self):
        rec = _rec(
            action_history=[_step("web_search", result="info")],
            prediction="",
            tool_counts={"web_search": 1},
        )
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_whitespace_only_prediction(self):
        rec = _rec(
            action_history=[_step("web_search", result="info")],
            prediction="   ",
            tool_counts={"web_search": 1},
        )
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_turns_at_max(self):
        steps = [_step("web_search", result="info") for _ in range(14)]
        rec = _rec(action_history=steps, prediction="wrong", turns=15, tool_counts={"web_search": 14})
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_turns_above_max(self):
        steps = [_step("web_search", result="info") for _ in range(15)]
        rec = _rec(action_history=steps, prediction="wrong", turns=16, tool_counts={"web_search": 15})
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_same_tool_repeated_3_times(self):
        steps = [_step("code_generator", result="") for _ in range(3)]
        rec = _rec(action_history=steps, prediction="5", turns=4, tool_counts={"code_generator": 3})
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_two_repeats_not_loop(self):
        steps = [_step("code_generator", result="5"), _step("code_generator", result="5")]
        rec = _rec(action_history=steps, prediction="5", turns=3, tool_counts={"code_generator": 2})
        assert classify_failure(rec) != "tool_loop_or_empty_final"


# ── Priority 3: direct_reasoning_no_action ──────────────────────────────────

class TestDirectReasoning:
    def test_empty_action_history(self):
        rec = _rec(action_history=[], prediction="42", tool_counts={})
        assert classify_failure(rec) == "direct_reasoning_no_action"

    def test_none_action_history(self):
        rec = {
            "correct": False, "prediction": "42", "turns": 1,
            "action_history": None, "tool_counts": {},
        }
        assert classify_failure(rec) == "direct_reasoning_no_action"


# ── Priority 4: computational_subgoal_error ─────────────────────────────────

class TestComputational:
    def test_two_code_generator_calls(self):
        steps = [
            _step("code_generator", sub_goal="compute step 1", result="10"),
            _step("code_generator", sub_goal="compute step 2", result="42"),
        ]
        rec = _rec(action_history=steps, prediction="42", tool_counts={"code_generator": 2})
        assert classify_failure(rec) == "computational_subgoal_error"

    def test_one_code_generator_one_web_search(self):
        steps = [
            _step("web_search", result="some info"),
            _step("code_generator", result="42"),
        ]
        rec = _rec(action_history=steps, prediction="42", tool_counts={"web_search": 1, "code_generator": 1})
        assert classify_failure(rec) == "computational_subgoal_error"

    def test_single_code_generator_not_computational(self):
        # Only 1 action → single-shot, not computational
        rec = _rec(
            action_history=[_step("code_generator", result="42")],
            prediction="42",
            tool_counts={"code_generator": 1},
        )
        assert classify_failure(rec) != "computational_subgoal_error"


# ── Priority 5: retrieval_evidence_failure ───────────────────────────────────

class TestRetrieval:
    def test_web_search_used(self):
        steps = [
            _step("web_search", result="some info"),
            _step("web_search", result="more info"),
        ]
        rec = _rec(action_history=steps, prediction="wrong", tool_counts={"web_search": 2})
        assert classify_failure(rec) == "retrieval_evidence_failure"

    def test_single_web_search_is_retrieval_not_single_shot(self):
        # Single web_search: priority 4 (code_gen) no, priority 5 (web_search >= 1) YES
        # so retrieval wins before single_shot
        rec = _rec(
            action_history=[_step("web_search", result="some useful info")],
            prediction="wrong",
            tool_counts={"web_search": 1},
        )
        assert classify_failure(rec) == "retrieval_evidence_failure"


# ── Priority 6: single_shot_tool_trust ──────────────────────────────────────

class TestSingleShot:
    def test_single_code_generator_call(self):
        rec = _rec(
            action_history=[_step("code_generator", result="wrong answer")],
            prediction="wrong answer",
            tool_counts={"code_generator": 1},
        )
        assert classify_failure(rec) == "single_shot_tool_trust"

    def test_single_mind_map_call(self):
        rec = _rec(
            action_history=[_step("mind_map", result="some map")],
            prediction="wrong",
            tool_counts={"mind_map": 1},
        )
        assert classify_failure(rec) == "single_shot_tool_trust"
