import json

import pytest

from rsi.intervention.rescue import RescueSettings, run_rescue
from rsi.leakage import LeakageIndex

MAX_TURNS = 5
ORD = {"a": 1, "b": 2, "c": 3}


def _cand(qid, gold="Paris", n_actions=2):
    history, messages = [], [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "plan"},
    ]
    for i in range(n_actions):
        call = {"name": "web_search", "arguments": {"query": f"q{i}"}}
        history.append({"command": json.dumps(call), "result": "r", "sub_goal": "g"})
        messages += [
            {"role": "assistant", "content": f"call{i}"},
            {"role": "tool", "content": f"res{i}"},
        ]
    return {
        "question_id": qid,
        "data_source": "hotpotqa",
        "pass": 1,
        "question": "Capital of the country of cheese?",
        "ground_truth": gold,
        "aliases": [],
        "prediction": "Lyon",
        "correct": False,
        "turns": n_actions,
        "messages": messages,
        "query_analysis": "plan",
        "raw_query_analysis": "plan",
        "action_history": history,
        "subagent_calls": [],
    }


def _reply(step, query="capital of France"):
    call = json.dumps({"name": "web_search", "arguments": {"query": query}})
    return f"<step>{step}</step><sub_goal>look it up</sub_goal><tool_call>{call}</tool_call>"


class FakeReflector:
    def __init__(self, scripts):
        self.scripts = scripts  # attempt-ordered list of lists of reply texts
        self.calls = []

    def sample_batch(self, messages_list, seeds):
        self.calls.append((messages_list, seeds))
        texts = self.scripts[len(self.calls) - 1]
        return [{"text": t, "finish_reason": "stop", "seed": s} for t, s in zip(texts, seeds)]


class FakeOrchestrator:
    def __init__(self, provider, fix_answer="Paris", control_answer="Lyon"):
        self.provider = provider
        self.fix_answer, self.control_answer = fix_answer, control_answer
        self.calls = []

    def resume_batch(self, states, injected):
        self.calls.append(([s.turn for s in states], list(injected), self.provider.config.seed))
        for s, inj in zip(states, injected):
            s.answer = self.fix_answer if inj is not None else self.control_answer
            s.output_messages = s.output_messages + [{"role": "assistant", "content": "x"}]
        return states


class Cfg:
    seed = None


class Prov:
    config = Cfg()


def _run(cands, scripts, tmp_path, orch=None, settings=None):
    orch = orch or FakeOrchestrator(Prov)
    refl = FakeReflector(scripts)
    rows = run_rescue(
        cands,
        orchestrator=orch,
        providers=[Prov],
        reflector=refl,
        index=LeakageIndex([]),
        settings=settings or RescueSettings(),
        out_path=tmp_path / "rescue.jsonl",
        max_turns=MAX_TURNS,
    )
    return rows, orch, refl


def test_rescued_on_first_attempt_not_retried(tmp_path):
    rows, orch, refl = _run([_cand(ORD["a"])], [[_reply(1)]], tmp_path)
    assert len(refl.calls) == 1
    assert [(r["attempt"], r["correct"]) for r in rows] == [(1, True)]
    assert rows[0]["control_correct"] is False


def test_second_attempt_sees_previous_fix(tmp_path):
    orch = FakeOrchestrator(Prov, fix_answer="Lyon")
    rows, _, refl = _run(
        [_cand(ORD["a"])], [[_reply(1, "first try")], [_reply(2, "second try")]], tmp_path, orch
    )
    assert [r["attempt"] for r in rows] == [1, 2]
    prompt2 = refl.calls[1][0][0][1]["content"]
    assert "first try" in prompt2 and "tried at step 1:" in prompt2
    assert refl.calls[1][1] == [1002]


def test_parse_error_and_leakage_still_run_control(tmp_path):
    cands = [_cand(ORD["a"], n_actions=2), _cand(ORD["b"], n_actions=9), _cand(ORD["c"])]
    scripts = [["garbage", "", _reply(1, "the answer is Paris")]]
    rows, orch, _ = _run(cands, scripts, tmp_path, settings=RescueSettings(max_attempts=1))
    by = {r["question_id"]: r for r in rows}
    assert by[ORD["a"]]["status"] == "parse_error" and by[ORD["a"]]["control_step"] == 2
    assert by[ORD["b"]]["status"] == "parse_error" and by[ORD["b"]]["control_step"] == MAX_TURNS - 1
    assert by[ORD["b"]]["reasons"] == ["stop"] or by[ORD["b"]]["reasons"]
    assert (
        by[ORD["c"]]["status"] == "leakage"
        and by[ORD["c"]]["reasons"]
        and by[ORD["c"]]["control_step"] == 1
    )
    assert by[ORD["a"]]["messages"] == [] and by[ORD["c"]]["fix_text"]
    assert all("control_prediction" in r for r in rows)
    # only no-op resumes for failed fixes
    assert len(orch.calls) == 1 and orch.calls[0][1] == [None, None, None]


def test_fix_and_control_share_seed(tmp_path):
    _, orch, _ = _run(
        [_cand(ORD["a"])], [[_reply(1)]], tmp_path, settings=RescueSettings(max_attempts=1)
    )
    assert [c[2] for c in orch.calls] == [1001, 1001]
    assert orch.calls[0][1][0] is not None and orch.calls[1][1] == [None]


def test_resume_after_crash_skips_done_rows(tmp_path):
    path = tmp_path / "rescue.jsonl"
    done = {
        "question_id": 1,
        "attempt": 1,
        "correct": False,
        "fix_text": "FT",
        "prediction": "Lyon",
        "step": 1,
    }
    path.write_text(json.dumps(done) + "\n")
    cands = [_cand(ORD["a"]), _cand(ORD["b"])]
    rows, _, refl = _run(cands, [[_reply(1)]], tmp_path, settings=RescueSettings(max_attempts=1))
    assert len(refl.calls) == 1 and len(refl.calls[0][0]) == 1  # only b
    assert sorted((r["question_id"], r["attempt"]) for r in rows) == [(1, 1), (2, 1)]
    rows2, _, refl2 = _run(cands, [], tmp_path, settings=RescueSettings(max_attempts=1))
    assert refl2.calls == [] and len(rows2) == 2


def test_rows_have_expected_fields(tmp_path):
    rows, _, _ = _run(
        [_cand(ORD["a"], gold="Paris")],
        [[_reply(1)]],
        tmp_path,
        settings=RescueSettings(max_attempts=1),
    )
    r = rows[0]
    assert set(r) == {
        "question_id",
        "data_source",
        "domain",
        "question",
        "ground_truth",
        "aliases",
        "attempt",
        "seed",
        "status",
        "reasons",
        "reflector_text",
        "step",
        "kind",
        "fix_text",
        "correct",
        "prediction",
        "messages",
        "control_step",
        "control_correct",
        "control_prediction",
    }
    assert r["domain"] == "search" and r["kind"] == "rewrite" and r["seed"] == 1001
    assert r["messages"][-1] == {"role": "assistant", "content": "x"}
    assert r["ground_truth"] == "Paris" and r["aliases"] == []


def test_rerun_tolerates_truncated_last_row(tmp_path):
    path = tmp_path / "rescue.jsonl"
    done = {"question_id": 1, "attempt": 1, "correct": False, "fix_text": "FT"}
    path.write_text(json.dumps(done) + "\n" + '{"question_id": 2, "att')
    cands = [_cand(ORD["a"]), _cand(ORD["b"])]
    rows, _, refl = _run(cands, [[_reply(1)]], tmp_path, settings=RescueSettings(max_attempts=1))
    assert len(refl.calls[0][0]) == 1
    assert sorted((r["question_id"], r["attempt"]) for r in rows) == [(1, 1), (2, 1)]
    assert len(path.read_text().splitlines()) == 2


def test_malformed_middle_line_still_raises(tmp_path):
    path = tmp_path / "rescue.jsonl"
    path.write_text('{"question_id": 2, "att\n' + json.dumps({"question_id": 1}) + "\n")
    with pytest.raises(json.JSONDecodeError):
        _run([_cand(ORD["a"])], [[_reply(1)]], tmp_path)


def test_show_gold_false_hides_gold_from_reflector(tmp_path):
    orch = FakeOrchestrator(Prov)
    refl = FakeReflector([[_reply(1)]])
    run_rescue(
        [_cand(ORD["a"])],
        orchestrator=orch,
        providers=[Prov],
        reflector=refl,
        index=LeakageIndex([]),
        settings=RescueSettings(max_attempts=1),
        out_path=tmp_path / "r.jsonl",
        max_turns=MAX_TURNS,
        show_gold=False,
    )
    prompt = refl.calls[0][0][0][1]["content"]
    assert "(hidden)" in prompt and "Paris" not in prompt
