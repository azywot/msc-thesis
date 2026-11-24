import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry
from rsi.intervention.records import (
    domain_of,
    intervention_candidates,
    load_collected,
    state_from_record,
)

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_sft_parquet", REPO / "scripts/build_sft_parquet.py"
)
build_sft_parquet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft_parquet)


def _rec(qid, p, correct=False, source="hotpotqa", turns=2, **kw):
    r = {
        "question_id": qid,
        "pass": p,
        "data_source": source,
        "correct": correct,
        "turns": turns,
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
    }
    r.update(kw)
    return r


def _full(n_actions=3):
    history, msgs = [], [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": "<think>t</think>plan"},
    ]
    for i in range(n_actions):
        call = {"name": "web_search", "arguments": {"query": "q%d" % i}}
        history.append(
            {
                "tool_name": "web_search",
                "sub_goal": "g%d" % i,
                "command": json.dumps(call),
                "result": "r%d" % i,
            }
        )
        msgs.append({"role": "assistant", "content": "call%d" % i})
        msgs.append({"role": "tool", "tool_name": "web_search", "content": "r%d" % i})
    msgs.append({"role": "assistant", "content": "final"})
    rec = _rec(
        7,
        1,
        turns=n_actions + 1,
        question="the question",
        messages=msgs,
        query_analysis="plan",
        raw_query_analysis="<think>t</think>plan",
        action_history=history,
        subagent_calls=[{"action_index": i} for i in range(n_actions)],
    )
    return rec


def test_candidates_need_all_passes_wrong():
    recs = [_rec("q1", p) for p in (1, 2, 3)]
    recs += [_rec("q2", 1), _rec("q2", 2), _rec("q2", 3, correct=True)]
    recs += [_rec("q3", 1), _rec("q3", 2)]
    cands, stats = intervention_candidates(recs, 3)
    assert [c["question_id"] for c in cands] == ["q1"]
    assert stats["search"]["zero_of_n"] == 1 and stats["search"]["candidates"] == 1


def test_candidate_is_pass_one_record():
    recs = [_rec("q1", p) for p in (3, 1, 2)]
    cands, _ = intervention_candidates(recs, 3)
    assert cands[0]["pass"] == 1


def test_skips_counted_per_domain():
    recs = []
    for p in (1, 2):
        recs.append(_rec("a", p, turns=0))
        recs.append(_rec("b", p, source="deepmath", error="boom", messages=[]))
        recs.append(_rec("c", p, source="deepmath"))
    cands, stats = intervention_candidates(recs, 2)
    assert [c["question_id"] for c in cands] == ["c"]
    assert stats["search"] == {
        "zero_of_n": 1,
        "skipped_error": 0,
        "skipped_planning_answer": 1,
        "candidates": 0,
    }
    assert stats["math"] == {
        "zero_of_n": 2,
        "skipped_error": 1,
        "skipped_planning_answer": 0,
        "candidates": 1,
    }


def test_domain_of_and_unknown():
    assert domain_of("nq") == "search" and domain_of("deepmath") == "math"
    with pytest.raises(ValueError):
        domain_of("gaia")


def test_load_collected_dedups(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"question_id": 1, "pass": 1, "v": "a"}) + "\n\n")
    b.write_text(
        json.dumps({"question_id": 1, "pass": 1, "v": "b"})
        + "\n"
        + json.dumps({"question_id": 1, "pass": 2, "v": "b"})
        + "\n"
    )
    out = load_collected([a, b])
    assert [(r["pass"], r["v"]) for r in out] == [(1, "a"), (2, "b")]


def test_state_from_record_prefix():
    rec = _full(3)
    st = state_from_record(rec, 2)
    assert st.turn == 2 and len(st.action_history) == 2 and len(st.output_messages) == 4
    assert st.output_messages[0]["content"] == "call0"
    assert st.output_messages[-1]["content"] == "r1"
    assert st.tool_counts["web_search"] == 2
    assert st.tool_calls[1] == {"name": "web_search", "arguments": {"query": "q1"}}
    assert st.metadata == {
        "subagent_calls": [{"action_index": 0}, {"action_index": 1}],
        "intervention_step": 2,
    }
    orch = AgenticOrchestrator(model_provider=MagicMock(), tool_registry=ToolRegistry())
    prompt = orch._build_memory_prompt(st, "sys")
    assert prompt[1]["content"] == build_sft_parquet._memory_user_content(
        "the question", "plan", rec["action_history"][:2]
    )


def test_state_from_record_step_zero_and_out_of_range():
    rec = _full(2)
    st = state_from_record(rec, 0)
    assert st.turn == 0 and st.output_messages == [] and st.action_history == []
    assert st.tool_counts["web_search"] == 0
    with pytest.raises(ValueError):
        state_from_record(rec, 3)
    with pytest.raises(ValueError):
        state_from_record(rec, -1)
