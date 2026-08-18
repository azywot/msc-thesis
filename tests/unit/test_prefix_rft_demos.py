"""Tests for the Prefix-RFT demonstration store."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_prefix_demos import question_key, records_to_demo_rows


def _record(idx, correct=True):
    return {
        "question_id": idx,
        "question": f"q{idx}",
        "data_source": "deepmath",
        "correct": correct,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"q{idx}"},
            {"role": "assistant", "content": "<think>hidden</think>plan text"},
            {"role": "assistant", "content": "call the tool"},
            {"role": "tool", "tool_name": "code_generator", "content": "42"},
            {"role": "assistant", "content": "the answer is 42"},
        ],
    }


def test_rows_have_one_entry_per_decision_in_order():
    rows = records_to_demo_rows([_record(7)])
    assert len(rows) == 1
    row = rows[0]
    assert row["question_key"] == question_key("q7")
    assert row["n_steps"] == 3
    assert [s["response"] for s in row["steps"]] == [
        "plan text",
        "call the tool",
        "the answer is 42",
    ]


def test_thinking_is_stripped():
    rows = records_to_demo_rows([_record(7)])
    assert "<think>" not in rows[0]["steps"][0]["response"]


def test_tool_results_are_attached_to_the_calling_step():
    steps = records_to_demo_rows([_record(7)])[0]["steps"]
    assert steps[1]["tool_name"] == "code_generator"
    assert steps[1]["tool_result"] == "42"
    assert steps[2]["tool_name"] is None
    assert steps[2]["tool_result"] is None


def test_incorrect_trajectories_are_dropped():
    assert records_to_demo_rows([_record(7, correct=False)]) == []


def test_one_row_per_question_when_duplicates_exist():
    rows = records_to_demo_rows([_record(7), _record(7)])
    assert len(rows) == 1


def test_questions_colliding_on_idx_stay_distinct():
    """extra_info.idx is assigned per data source and collides across them.

    Two different questions sharing an idx must produce two rows, or a maths
    demonstration would be replayed into a search question.
    """
    a = _record(669)
    b = _record(669)
    b["question"] = "which is currently more valuable"
    b["data_source"] = "hotpotqa"
    rows = records_to_demo_rows([a, b])
    assert len(rows) == 2
    assert len({r["question_key"] for r in rows}) == 2


def test_question_key_is_whitespace_insensitive():
    assert question_key(" q7 ") == question_key("q7")


from check_prefix_demos import check_row


def _good_row():
    """The shape every real trajectory has: plan, tool calls, answer."""
    return {
        "question_key": "abc12345",
        "n_steps": 3,
        "steps": [
            {"response": "plan text", "tool_name": None, "tool_result": None},
            {
                "response": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>',
                "tool_name": "web_search",
                "tool_result": "result text",
            },
            {"response": "final answer", "tool_name": None, "tool_result": None},
        ],
    }


def test_gate_accepts_a_well_formed_row():
    assert check_row(_good_row()) == []


def test_gate_accepts_a_single_decision_row():
    row = {
        "question_key": "abc12345",
        "n_steps": 1,
        "steps": [{"response": "answered directly", "tool_name": None, "tool_result": None}],
    }
    assert check_row(row) == []


def test_gate_rejects_a_missing_tool_result():
    row = _good_row()
    row["steps"][1]["tool_result"] = None
    assert any("stored tool_result" in p for p in check_row(row))


def test_gate_rejects_an_unparseable_tool_call():
    row = _good_row()
    row["steps"][1]["response"] = "I will search for it"
    assert any("does not parse" in p for p in check_row(row))


def test_gate_rejects_a_non_tool_step_in_the_middle():
    row = _good_row()
    row["steps"][1] = {"response": "musing", "tool_name": None, "tool_result": None}
    assert any("middle step" in p for p in check_row(row))


def test_gate_rejects_a_trajectory_ending_on_a_tool_call():
    row = _good_row()
    row["steps"] = row["steps"][:2]
    row["n_steps"] = 2
    assert any("no answer" in p for p in check_row(row))


def test_gate_rejects_surviving_thinking():
    row = _good_row()
    row["steps"][2]["response"] = "<think>oops</think>final answer"
    assert any("<think>" in p for p in check_row(row))


import pandas as pd

from verl_ext.prefix_rft.demos import DemoStore


def _write_store(tmp_path):
    path = tmp_path / "demos.parquet"
    pd.DataFrame(
        [
            {
                "question_key": question_key("q3"),
                "question_id": 3,
                "data_source": "deepmath",
                "question": "q3",
                "n_steps": 2,
                "steps": [
                    {"response": "a", "tool_name": "web_search", "tool_result": "r"},
                    {"response": "b", "tool_name": None, "tool_result": None},
                ],
            }
        ]
    ).to_parquet(path, index=False)
    return path


def test_store_returns_steps_for_a_known_question(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.n_steps("q3") == 2
    assert store.steps("q3")[0]["tool_result"] == "r"


def test_store_lookup_is_whitespace_insensitive(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.n_steps("  q3 ") == 2


def test_store_misses_return_zero_rather_than_raising(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.n_steps("never seen") == 0
    assert store.steps("never seen") == []


def test_store_reports_coverage(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.coverage() == (1, 2)
