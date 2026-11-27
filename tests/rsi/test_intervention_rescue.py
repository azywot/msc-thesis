import json

import pytest

from rsi.intervention.fix import parse_fix
from rsi.intervention.report import summarize
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


def _fix_text(step, query="capital of France"):
    return parse_fix(_reply(step, query), 9, 9).text


COUNTS1 = {"zero_of_n": 1, "skipped_error": 0, "skipped_planning_answer": 0, "candidates": 1}


class FakeReflector:
    def __init__(self, scripts):
        self.scripts = scripts  # attempt-ordered list of lists of reply texts
        self.calls = []

    def sample_batch(self, messages_list, seeds):
        self.calls.append((messages_list, seeds))
        texts = self.scripts[len(self.calls) - 1]
        return [{"text": t, "finish_reason": "stop", "seed": s} for t, s in zip(texts, seeds)]


class FakeOrchestrator:
    def __init__(self, provider, fix_answer="Paris", control_answer="Lyon", control_by_seed=None):
        self.provider = provider
        self.fix_answer, self.control_answer = fix_answer, control_answer
        self.control_by_seed = control_by_seed or {}
        self.fix_by_qid = {}
        self.calls = []

    def resume_batch(self, states, injected):
        seed = self.provider.config.seed
        self.calls.append(([s.turn for s in states], list(injected), seed))
        for s, inj in zip(states, injected):
            if inj is not None:
                s.answer = self.fix_by_qid.get(s.question_id, self.fix_answer)
            else:
                s.answer = self.control_by_seed.get(seed, self.control_answer)
            s.output_messages = s.output_messages + [{"role": "assistant", "content": "x"}]
        # a real run moves on; the next resume must set the seed again
        self.provider.config.seed = None
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
    assert [(r["attempt"], r["correct"]) for r in rows] == [(1, True), (2, False)]
    assert rows[0]["control_correct"] is False


def test_rescued_question_gets_control_only_draw_on_later_attempt(tmp_path):
    orch = FakeOrchestrator(Prov, control_by_seed={1002: "Paris"})
    cands = [_cand(ORD["a"]), _cand(ORD["b"])]
    # a is rescued at attempt 1 (step 1); b fails both attempts
    orch.fix_by_qid = {ORD["b"]: "Lyon"}
    rows, _, refl = _run(cands, [[_reply(1), _reply(2)], [_reply(0, "again")]], tmp_path, orch)
    # attempt 2 asks the reflector about b only
    assert len(refl.calls) == 2 and len(refl.calls[1][0]) == 1
    assert "Capital" in refl.calls[1][0][0][1]["content"]
    by = {(r["question_id"], r["attempt"]): r for r in rows}
    later = by[(ORD["a"], 2)]
    assert later["status"] == "rescued_earlier"
    assert later["reflector_text"] == "" and later["fix_text"] is None
    assert later["correct"] is False and later["messages"] == []
    assert later["step"] is None and later["kind"] is None and later["reasons"] == []
    assert later["control_step"] == 1 and later["seed"] == 1002
    assert later["control_correct"] is True and later["control_prediction"] == "Paris"
    # attempt 2: no fix continuation for a; both controls share the attempt seed
    fix2, control2 = orch.calls[2], orch.calls[3]
    assert fix2[1] == [_fix_text(0, "again")] and fix2[2] == 1002
    assert control2[1] == [None, None] and control2[0] == [1, 0] and control2[2] == 1002
    s = summarize(cands, rows, {"search": dict(COUNTS1, zero_of_n=2, candidates=2)})["search"]
    assert s["rescued"] == 1 and s["attempts_used"] == {"1": 1}
    assert s["control_solved"] == 2  # a via its control-only draw, b via its own
    assert s["parse_errors"] == 0 and s["leakage_rejections"] == 0
    assert s["kinds"] == {"rewrite": 1}


def test_rerun_after_rescue_does_not_redo_control_only_rows(tmp_path):
    _run([_cand(ORD["a"])], [[_reply(1)]], tmp_path)
    rows, orch, refl = _run([_cand(ORD["a"])], [], tmp_path)
    assert refl.calls == [] and orch.calls == []
    assert [(r["attempt"], r["status"]) for r in rows] == [(1, "ok"), (2, "rescued_earlier")]


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


class EchoReflector:
    """Replies with one fixed text per prompt; records how many rows were on disk per call."""

    def __init__(self, path, text=None, fail_on_call=None, error_on_call=None):
        self.path, self.text = path, text or _reply(1)
        self.fail_on_call, self.error_on_call = fail_on_call, error_on_call
        self.calls, self.lines_on_disk = [], []

    def sample_batch(self, messages_list, seeds):
        self.calls.append((messages_list, seeds))
        n = len(self.path.read_text().splitlines()) if self.path.exists() else 0
        self.lines_on_disk.append(n)
        if len(self.calls) == self.fail_on_call:
            raise KeyboardInterrupt("job killed")
        if len(self.calls) == self.error_on_call:
            return [{"text": "", "finish_reason": "error: oom", "seed": s} for s in seeds]
        return [{"text": self.text, "finish_reason": "stop", "seed": s} for s in seeds]


def _run_with(cands, refl, tmp_path, orch=None, settings=None):
    orch = orch or FakeOrchestrator(Prov)
    return run_rescue(
        cands,
        orchestrator=orch,
        providers=[Prov],
        reflector=refl,
        index=LeakageIndex([]),
        settings=settings,
        out_path=tmp_path / "rescue.jsonl",
        max_turns=MAX_TURNS,
    )


def test_chunks_append_rows_after_each_chunk_and_resume_after_crash(tmp_path):
    path = tmp_path / "rescue.jsonl"
    cands = [_cand(i) for i in range(1, 6)]
    settings = RescueSettings(max_attempts=1, chunk_size=2)
    orch = FakeOrchestrator(Prov)
    refl = EchoReflector(path)
    _run_with(cands, refl, tmp_path, orch, settings)
    assert [len(m) for m, _ in refl.calls] == [2, 2, 1]
    assert refl.lines_on_disk == [0, 2, 4]
    assert len(path.read_text().splitlines()) == 5
    # every resume_batch, fix and control, in every chunk, runs at the attempt seed
    assert [c[2] for c in orch.calls] == [1001] * 6

    path.unlink()
    crashing = EchoReflector(path, fail_on_call=2)
    with pytest.raises(KeyboardInterrupt):
        _run_with(cands, crashing, tmp_path, settings=settings)
    assert sorted(json.loads(ln)["question_id"] for ln in path.read_text().splitlines()) == [1, 2]
    again = EchoReflector(path)
    rows = _run_with(cands, again, tmp_path, settings=settings)
    asked = [m[1]["content"] for msgs, _ in again.calls for m in msgs]
    assert [len(m) for m, _ in again.calls] == [2, 1]
    assert len(asked) == 3
    assert sorted(r["question_id"] for r in rows) == [1, 2, 3, 4, 5]


def test_all_error_chunk_raises_before_writing_rows(tmp_path):
    path = tmp_path / "rescue.jsonl"
    cands = [_cand(i) for i in range(1, 5)]
    refl = EchoReflector(path, error_on_call=2)
    with pytest.raises(RuntimeError, match="error: oom"):
        _run_with(cands, refl, tmp_path, settings=RescueSettings(max_attempts=1, chunk_size=2))
    assert sorted(json.loads(ln)["question_id"] for ln in path.read_text().splitlines()) == [1, 2]


def test_single_error_item_among_good_ones_is_a_parse_error(tmp_path):
    class OneBad(FakeReflector):
        def sample_batch(self, messages_list, seeds):
            items = super().sample_batch(messages_list, seeds)
            items[0] = {"text": "", "finish_reason": "error: boom", "seed": seeds[0]}
            return items

    orch = FakeOrchestrator(Prov)
    rows = run_rescue(
        [_cand(ORD["a"]), _cand(ORD["b"])],
        orchestrator=orch,
        providers=[Prov],
        reflector=OneBad([[_reply(1), _reply(1)]]),
        index=LeakageIndex([]),
        settings=RescueSettings(max_attempts=1),
        out_path=tmp_path / "rescue.jsonl",
        max_turns=MAX_TURNS,
    )
    by = {r["question_id"]: r for r in rows}
    assert by[ORD["a"]]["status"] == "parse_error" and by[ORD["a"]]["reasons"] == ["error: boom"]
    assert by[ORD["b"]]["status"] == "ok"


def test_reflector_returning_wrong_item_count_raises(tmp_path):
    class Short(FakeReflector):
        def sample_batch(self, messages_list, seeds):
            return super().sample_batch(messages_list, seeds)[:1]

    with pytest.raises(RuntimeError, match="1 items for 2 prompts"):
        run_rescue(
            [_cand(ORD["a"]), _cand(ORD["b"])],
            orchestrator=FakeOrchestrator(Prov),
            providers=[Prov],
            reflector=Short([[_reply(1), _reply(1)]]),
            index=LeakageIndex([]),
            settings=RescueSettings(max_attempts=1),
            out_path=tmp_path / "rescue.jsonl",
            max_turns=MAX_TURNS,
        )
    assert not (tmp_path / "rescue.jsonl").exists()


def test_second_attempt_after_leakage_says_the_fix_was_rejected(tmp_path):
    rows, _, refl = _run(
        [_cand(ORD["a"])],
        [[_reply(1, "the answer is Paris")], [_reply(2, "second try")]],
        tmp_path,
        FakeOrchestrator(Prov, fix_answer="Lyon"),
    )
    assert [r["status"] for r in rows] == ["leakage", "ok"]
    prompt2 = refl.calls[1][0][0][1]["content"]
    assert "the answer is Paris" in prompt2
    assert "rejected because it revealed the answer" in prompt2
    assert "The trajectory then answered" not in prompt2


def test_second_attempt_after_parse_error_has_no_earlier_section(tmp_path):
    rows, _, refl = _run([_cand(ORD["a"])], [["garbage"], [_reply(2, "second try")]], tmp_path)
    assert [r["status"] for r in rows] == ["parse_error", "ok"]
    assert "Earlier attempt" not in refl.calls[1][0][0][1]["content"]


def test_single_prompt_chunk_error_is_a_parse_error_not_a_raise(tmp_path):
    path = tmp_path / "rescue.jsonl"
    refl = EchoReflector(path, error_on_call=1)
    rows = _run_with(
        [_cand(1)], refl, tmp_path, settings=RescueSettings(max_attempts=1, chunk_size=1)
    )
    assert [r["status"] for r in rows] == ["parse_error"]
    assert "error: oom" in rows[0]["reasons"][0]
