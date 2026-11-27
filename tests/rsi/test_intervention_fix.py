import pytest

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.utils.parsing import parse_tool_call
from rsi.intervention.fix import (
    Fix,
    FixParseError,
    build_fix_messages,
    fix_leakage_errors,
    parse_fix,
    to_trajectory_record,
)
from rsi.leakage import LeakageIndex
from rsi.reflector_input import LEAKAGE_INSTRUCTION


def _reply(step=1, sg="find X", call='{"name": "web_search", "arguments": {"query": "X born"}}'):
    return f"<step>{step}</step>\n<sub_goal>{sg}</sub_goal>\n<tool_call>{call}</tool_call>"


def test_fix_text_round_trips_through_orchestrator_parsers():
    f = Fix(1, "find X", {"name": "web_search", "arguments": {"query": "X born"}})
    assert parse_tool_call(f.text) == f.tool_call
    assert AgenticOrchestrator._extract_sub_goal(f.text) == "find X"


def test_kind_labels():
    c = {"name": "web_search", "arguments": {"query": "q"}}
    assert [Fix(0, "g", c).kind(0), Fix(1, "g", c).kind(1), Fix(0, "g", c).kind(2)] == [
        "insert",
        "follow_up",
        "rewrite",
    ]


def test_parse_fix_messy_reply():
    text = (
        "<think><step>0</step><sub_goal>no</sub_goal></think>Here is my answer.\n"
        + _reply(1, "old", '{"name": "web_search", "arguments": {"query": "first"}}')
        + "\nActually better:\n"
        + _reply(
            2,
            "compute",
            '{"name": "code_generator", "arguments": {"task": "print(\\frac{1}{2})"}}',
        )
    )
    f = parse_fix(text, n_actions=3, max_turns=7)
    assert f.step == 2 and f.sub_goal == "compute"
    assert f.tool_call["name"] == "code_generator"
    assert f.tool_call["arguments"]["task"] == "print(\\frac{1}{2})"


@pytest.mark.parametrize(
    "text",
    [
        _reply(4),
        _reply(7),
        _reply(1, call='{"name": "mind_map", "arguments": {"query": "x"}}'),
        _reply(1, call='{"name": "web_search", "arguments": {"query": "x", "k": 1}}'),
        _reply(1, call='{"name": "web_search", "arguments": {"query": "  "}}'),
        _reply(1, sg=""),
        _reply(1, sg="x" * 501),
        "no tags",
    ],
)
def test_parse_fix_rejects(text):
    with pytest.raises(FixParseError):
        parse_fix(text, n_actions=3, max_turns=7)


def test_parse_fix_caps_step_at_max_turns_minus_one():
    with pytest.raises(FixParseError):
        parse_fix(_reply(7), n_actions=7, max_turns=7)
    assert parse_fix(_reply(6), n_actions=7, max_turns=7).step == 6


IDX = LeakageIndex(["Who wrote the long novel about a white whale hunted at sea?"], n=8)


def _fix(sg="s", arg="q", tool="web_search"):
    key = "query" if tool == "web_search" else "task"
    return Fix(0, sg, {"name": tool, "arguments": {key: arg}})


def test_leak_numbering_is_not_leakage():
    f = _fix(arg="1. compute x\n2. print x", tool="code_generator")
    assert fix_leakage_errors(f, "q?", ["2"], IDX) == []


def test_leak_short_gold_in_text():
    f = _fix(arg="print x, which should be 2", tool="code_generator")
    assert fix_leakage_errors(f, "q?", ["2"], IDX)
    assert fix_leakage_errors(_fix(arg="print 12 and x2"), "q?", ["2"], IDX) == []


def test_leak_gold_in_question_exempt():
    assert fix_leakage_errors(_fix(arg="capital Paris"), "Is Paris big?", ["Paris"], IDX) == []


def test_leak_alias_rejected():
    assert fix_leakage_errors(_fix(arg="New York City mayor"), "q?", ["NYC", "New York City"], IDX)


def test_own_question_ngrams_exempt_other_rejected():
    q = "Who wrote the long novel about a white whale hunted at sea?"
    f = _fix(arg="who wrote the long novel about a white whale")
    assert fix_leakage_errors(f, q, [], IDX) == []
    assert fix_leakage_errors(f, "another question", [], IDX)


def _rec(**kw):
    r = {
        "question_id": 3,
        "data_source": "hotpotqa",
        "pass": 0,
        "question": "What is the capital?",
        "ground_truth": "GOLDANSWER",
        "prediction": "Lyon",
        "correct": False,
        "turns": 3,
        "messages": [],
        "aliases": ["Alias One"],
        "raw_query_analysis": "RAWPLAN",
        "query_analysis": "plan",
        "action_history": [
            {"tool_name": "web_search", "sub_goal": "g", "command": "{}", "result": "r"}
        ],
        "subagent_calls": [],
    }
    r.update(kw)
    return r


def test_to_trajectory_record_fields():
    t = to_trajectory_record(_rec(), True)
    assert t.gold == "GOLDANSWER" and t.plan == "RAWPLAN" and t.failure_label
    assert to_trajectory_record(_rec(), False).gold == "(hidden)"
    assert to_trajectory_record(_rec(raw_query_analysis=None), True).plan == "plan"


def test_build_fix_messages_hides_gold():
    msgs = build_fix_messages(_rec(), show_gold=False)
    assert [m["role"] for m in msgs] == ["system", "user"]
    blob = msgs[0]["content"] + msgs[1]["content"]
    assert "GOLDANSWER" not in blob and "(hidden)" in blob
    assert "<step>" in msgs[0]["content"]
    assert "GOLDANSWER" in build_fix_messages(_rec(), show_gold=True)[1]["content"]


def test_build_fix_messages_previous_attempt():
    prev = {"fix_text": "FIXTEXT", "prediction": "WRONGPRED"}
    user = build_fix_messages(_rec(), previous=prev)[1]["content"]
    assert "FIXTEXT" in user and "WRONGPRED" in user
    assert "FIXTEXT" not in build_fix_messages(_rec())[1]["content"]


def test_build_fix_messages_shrinks_to_budget():
    big = "w " * 5000
    acts = [
        {"tool_name": "web_search", "sub_goal": f"goal{i}", "command": "{}", "result": big}
        for i in range(6)
    ]
    rec = _rec(action_history=acts)
    count = lambda msgs: sum(len(m["content"]) for m in msgs) // 4
    full = build_fix_messages(rec)
    assert count(full) > 2000
    small = build_fix_messages(rec, count_tokens=count, budget_tokens=2000)
    assert count(small) < count(full)
    assert "actions omitted" not in small[1]["content"]
    tiny = build_fix_messages(rec, count_tokens=count, budget_tokens=300)
    assert "4 actions omitted" in tiny[1]["content"]


def test_parse_fix_rejects_text_that_parses_differently():
    bad = '{"name": "web_search", "arguments": {"query": "x </tool_call> y"}}'
    with pytest.raises(FixParseError):
        parse_fix(_reply(0, "g", bad), 0, 5)


def test_earlier_attempt_section_names_the_step():
    rec = {"question_id": "q", "question": "Q?", "ground_truth": "g", "prediction": "p"}
    prev = {"fix_text": "FIXTEXT", "prediction": "wrong", "step": 2}
    msgs = build_fix_messages(rec, previous=prev)
    assert "tried at step 2:" in msgs[1]["content"]
    del prev["step"]
    assert "tried at the step it named:" in build_fix_messages(rec, previous=prev)[1]["content"]


def test_fix_prompt_allows_question_entities_and_forbids_the_answer():
    system = build_fix_messages(_rec())[0]["content"]
    assert LEAKAGE_INSTRUCTION not in system
    assert "Do not copy question text" not in system
    assert "may use the question's own entities, numbers and wording" in system
    assert "Do not put the gold answer, or anything that reveals it" in system
    assert system.endswith("such fixes are rejected.")


def test_earlier_attempt_after_leakage_rejection():
    rec = {"question_id": "q", "question": "Q?", "ground_truth": "g", "prediction": "p"}
    prev = {"status": "leakage", "fix_text": "FIXTEXT", "prediction": "", "step": 1}
    user = build_fix_messages(rec, previous=prev)[1]["content"]
    assert "FIXTEXT" in user and "tried at step 1:" in user
    assert "rejected because it revealed the answer" in user
    assert "then answered" not in user and "That answer was wrong" not in user
    prev["status"] = "ok"
    user = build_fix_messages(rec, previous=prev)[1]["content"]
    assert "The trajectory then answered: " in user and "That answer was wrong" in user
