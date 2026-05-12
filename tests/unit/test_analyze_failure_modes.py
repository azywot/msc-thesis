"""Unit tests for classify_failure cascade."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import pytest
from failure_modes.analyze_failure_modes import classify_failure


def _rec(action_history=None, prediction="wrong", turns=2, tool_counts=None, question=""):
    """Build a minimal failed record."""
    return {
        "correct": False,
        "prediction": prediction,
        "turns": turns,
        "action_history": action_history or [],
        "tool_counts": tool_counts or {},
        "question": question,
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

    def test_empty_action_history_with_image_keyword_in_question(self):
        # Signal C: non-empty prediction + no tools + visual question → modality gap
        rec = {
            "correct": False,
            "prediction": "15",
            "turns": 1,
            "action_history": [],
            "tool_counts": {},
            "question": "Using the provided image, identify the fractions shown.",
        }
        assert classify_failure(rec) == "modality_tool_gap"

    def test_empty_prediction_visual_question_is_tool_loop_not_modality(self):
        # Bug 1 guard: empty prediction takes Priority 2 even with a visual question
        rec = {
            "correct": False,
            "prediction": "",
            "turns": 3,
            "action_history": [],
            "tool_counts": {},
            "question": "Using the provided image, identify the fractions shown.",
        }
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_empty_action_history_no_visual_keyword_not_gap(self):
        # Empty action_history but no image keyword → direct reasoning
        rec = {
            "correct": False,
            "prediction": "42",
            "turns": 1,
            "action_history": [],
            "tool_counts": {},
            "question": "What is the sum of the first 10 prime numbers?",
        }
        assert classify_failure(rec) == "direct_reasoning_no_action"

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
            "action_history": None, "tool_counts": {}, "question": "What is 2+2?",
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

    def test_one_code_generator_one_web_search_is_single_shot(self):
        # One web_search + one code_generator: code_generator count = 1 (< 2),
        # so computational does NOT fire. web_search count = 1 (< 2), so
        # retrieval does NOT fire either. Falls through to single-shot.
        steps = [
            _step("web_search", result="some info"),
            _step("code_generator", result="42"),
        ]
        rec = _rec(action_history=steps, prediction="42", tool_counts={"web_search": 1, "code_generator": 1})
        assert classify_failure(rec) == "single_shot_tool_trust"

    def test_single_code_generator_not_computational(self):
        # code_generator count = 1 (< 2) → not computational
        rec = _rec(
            action_history=[_step("code_generator", result="42")],
            prediction="42",
            tool_counts={"code_generator": 1},
        )
        assert classify_failure(rec) != "computational_subgoal_error"


# ── Priority 5: retrieval_evidence_failure ───────────────────────────────────

class TestRetrieval:
    def test_multiple_web_search_is_retrieval(self):
        # >= 2 web_search calls: failure to reconcile evidence
        steps = [
            _step("web_search", result="some info"),
            _step("web_search", result="more info"),
        ]
        rec = _rec(action_history=steps, prediction="wrong", tool_counts={"web_search": 2})
        assert classify_failure(rec) == "retrieval_evidence_failure"

    def test_single_web_search_is_single_shot_not_retrieval(self):
        # 1 web_search → single-shot trust, not retrieval failure
        rec = _rec(
            action_history=[_step("web_search", result="some useful info")],
            prediction="wrong",
            tool_counts={"web_search": 1},
        )
        assert classify_failure(rec) == "single_shot_tool_trust"

    def test_single_web_search_plus_other_tool_is_single_shot(self):
        # web_search=1 + text_inspector=1: still only one search, no reconciliation
        # attempted → single-shot trust, not retrieval failure
        steps = [
            _step("web_search", result="some info"),
            _step("text_inspector", sub_goal="read the document", result="some text"),
        ]
        rec = _rec(action_history=steps, prediction="wrong",
                   tool_counts={"web_search": 1, "text_inspector": 1})
        assert classify_failure(rec) == "single_shot_tool_trust"


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

    def test_turns_none_does_not_crash(self):
        # turns=None should fall back to 0 (no loop trigger)
        rec = {
            "correct": False, "prediction": "42", "turns": None,
            "action_history": [_step("code_generator", result="42")],
            "tool_counts": {"code_generator": 1}, "question": "",
        }
        assert classify_failure(rec) == "single_shot_tool_trust"
