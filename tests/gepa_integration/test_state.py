import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.state import ExecutionState


def test_raw_query_analysis_defaults_to_none():
    state = ExecutionState(question_id=1, question="test question")
    assert state.raw_query_analysis is None


def test_raw_query_analysis_can_be_set():
    state = ExecutionState(question_id=1, question="test question")
    state.raw_query_analysis = "<think>internal reasoning</think>visible analysis"
    assert state.raw_query_analysis == "<think>internal reasoning</think>visible analysis"


def test_raw_query_analysis_independent_of_query_analysis():
    state = ExecutionState(question_id=1, question="test question")
    state.query_analysis = "stripped"
    state.raw_query_analysis = "<think>full</think>stripped"
    assert state.query_analysis == "stripped"
    assert state.raw_query_analysis == "<think>full</think>stripped"
