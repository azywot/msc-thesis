import json
from types import SimpleNamespace

import pytest

from agent_engine.datasets.base import DatasetExample
from agent_engine.models.base import ToolCallFormat
from rsi.bundle import seed_bundle
from rsi.data import Pools, PoolSizes
from rsi.executor import TrajectoryRecord
from rsi.leakage import LeakageIndex
from rsi.reflector import ReflectorSettings
from rsi.search import SearchConfig, complete_tool_descriptions, run_search

N_SHOWN, N_SCREEN, N_HELD = 6, 6, 10


class FakeExecutor:
    """Accuracy is a deterministic function of the bundle: 0.2 + 0.1 * count of 'check'."""

    def __init__(self):
        self.rollouts = 0

    def run(self, bundle, examples):
        frac = min(1.0, 0.2 + 0.1 * bundle.orchestrator["turn1n_prompt"].count("check"))
        n_right = round(frac * len(examples))
        self.rollouts += len(examples)
        return [
            TrajectoryRecord(
                question_id=e.question_id,
                question=e.question,
                gold=e.answer,
                aliases=list(e.metadata.get("answer_aliases", [])),
                prediction="x",
                correct=i < n_right,
                failure_label=None if i < n_right else "direct_reasoning_no_action",
            )
            for i, e in enumerate(examples)
        ]


class FakeReflector:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0
        self.messages = []

    def sample(self, messages, seeds):
        self.messages.append(messages)
        out = []
        for s in seeds:
            text = self.texts[self.calls % len(self.texts)] if self.texts else ""
            self.calls += 1
            out.append({"text": text, "finish_reason": "stop", "seed": s})
        return out


def proposal_json(unit, replacement):
    return json.dumps(
        {"unit": unit, "rationale": "r", "expected_effect": "e", "replacement": replacement}
    )


@pytest.fixture
def env(SCHEMAS):
    total = N_SHOWN + N_SCREEN + N_HELD
    examples = {
        i: DatasetExample(
            question_id=i,
            question=f"unique question number {i} about topic{i} zebra{i}",
            answer=f"answerword{i} qq{i}",
            metadata={"answer_aliases": []},
        )
        for i in range(total)
    }
    pools = Pools(
        shown=list(range(N_SHOWN)),
        screen=list(range(N_SHOWN, N_SHOWN + N_SCREEN)),
        heldout=list(range(N_SHOWN + N_SCREEN, total)),
    )

    def cfg(**kw):
        base = dict(
            data_path="unused.parquet",
            pools=PoolSizes(N_SHOWN, N_SCREEN, N_HELD),
            shown_minibatch=N_SHOWN,
            screen_minibatch=4,
            reflector=ReflectorSettings(model="m"),
        )
        base.update(kw)
        return SearchConfig(**base)

    seed = seed_bundle("search", SCHEMAS)
    leakage = LeakageIndex([e.question for e in examples.values()], n=8)

    def args(tmp_path, count_tokens=lambda s: len(s) // 4):
        return (
            examples,
            pools,
            seed,
            leakage,
            count_tokens,
            SCHEMAS,
            ToolCallFormat.JSON,
            tmp_path,
        )

    return SimpleNamespace(
        cfg=cfg,
        args=args,
        executor=FakeExecutor(),
        fresh_executor=FakeExecutor,
        examples=examples,
        pools=pools,
        seed=seed,
        heldout=N_HELD,
        a_shown_gold_answer=examples[0].answer,
    )


def test_strictly_better_candidate_becomes_champion(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check check"})] * 3)
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    assert state["champion"] != "v0" and state["champion_heldout"] > state["seed_heldout"]
    assert state["beats_seed"] is True
    assert (tmp_path / "bundles" / f"{state['champion']}.json").exists()
    assert (tmp_path / "heldout" / f"{state['champion']}.json").exists()
    assert {v["label"] for v in state["versions"]} == {"v0", state["champion"]}
    assert all(v["prompt_chars"] > 0 for v in state["versions"])


def test_tie_keeps_champion(tmp_path, env):
    refl = FakeReflector(
        [proposal_json("orchestrator", {"turn1n_prompt": "no change in score"})] * 3
    )
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    assert state["champion"] == "v0" and state["beats_seed"] is False


def test_round_with_only_unparseable_proposals_keeps_champion(tmp_path, env):
    refl = FakeReflector(["not json"] * 20)
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert state["champion"] == "v0" and all(p["errors"] for p in rec["proposals"])
    assert len(rec["proposals"]) == 3 * (1 + env.cfg().max_parse_retries)


def test_length_error_and_unknown_unit_are_resampled(tmp_path, env):
    class Scripted(FakeReflector):
        def sample(self, messages, seeds):
            out = super().sample(messages, seeds)
            for o in out:
                o["finish_reason"] = ["length", "error: boom"][self.calls % 2]
            return out

    refl = Scripted([proposal_json("orchestrator", {"turn1n_prompt": "check"})])
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert len(rec["proposals"]) == 9 and all(p["errors"] for p in rec["proposals"])

    refl2 = FakeReflector([proposal_json("nonsense_unit", "x")])
    run_search(env.cfg(rounds=1), "search", env.executor, refl2, *env.args(tmp_path / "b"))
    rec2 = json.loads((tmp_path / "b" / "rounds" / "round_01.json").read_text())
    assert len(rec2["proposals"]) == 9 and "unknown unit" in rec2["proposals"][0]["errors"][0]


def test_leaking_candidate_is_rejected_without_rollouts(tmp_path, env):
    refl = FakeReflector(
        [proposal_json("orchestrator", {"turn1n_prompt": env.a_shown_gold_answer})] * 3
    )
    before = env.executor.rollouts
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert all(p["leak_errors"] for p in rec["proposals"] if not p["errors"])
    assert any(not p["errors"] for p in rec["proposals"])
    # only seed held-out + champion on the shown minibatch were run; no screen without survivors
    assert env.executor.rollouts - before == env.heldout + env.cfg().shown_minibatch


def test_resume_skips_completed_rounds(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check"})] * 30)
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    calls_after_first = refl.calls
    state = run_search(env.cfg(rounds=2), "search", env.fresh_executor(), refl, *env.args(tmp_path))
    assert state["rounds_done"] == 2
    assert refl.calls - calls_after_first == 3  # only round 2 sampled
    assert (tmp_path / "rounds" / "round_01.json").exists()
    assert state["rollouts"] > env.heldout  # counter continued, seed held-out not rerun


def test_reflector_never_sees_screen_or_heldout_questions(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check"})] * 9)
    run_search(env.cfg(rounds=3), "search", env.executor, refl, *env.args(tmp_path))
    unseen = {env.examples[i].question for i in env.pools.screen + env.pools.heldout}
    assert not any(q in m["content"] for msgs in refl.messages for m in msgs for q in unseen)


def test_partial_tool_descriptions_are_completed_from_champion(tmp_path, env):
    champ_td = env.seed.orchestrator["tool_descriptions"]
    name = sorted(champ_td)[0]
    payload = {"tool_descriptions": {name: {"description": "A new, better description."}}}
    done = complete_tool_descriptions(payload, env.seed)
    td = done["tool_descriptions"]
    assert set(td) == set(champ_td)
    assert td[name]["description"] == "A new, better description."
    assert td[name]["parameters"] == champ_td[name]["parameters"]
    other = [n for n in champ_td if n != name][0]
    assert td[other] == champ_td[other]
    # missing single parameter entries are filled too
    pname = sorted(champ_td[name]["parameters"])[0]
    p2 = {"tool_descriptions": {name: {"description": "d", "parameters": {pname: "new p"}}}}
    td2 = complete_tool_descriptions(p2, env.seed)["tool_descriptions"][name]["parameters"]
    assert td2 == {**champ_td[name]["parameters"], pname: "new p"}

    # end to end: the partial payload survives validation and creates a candidate
    refl = FakeReflector([proposal_json("orchestrator", payload)] * 3)
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert all(not p["errors"] for p in rec["proposals"])


def test_prompt_over_budget_halves_history_then_skips(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check"})] * 3)
    state = run_search(
        env.cfg(rounds=2),
        "search",
        env.executor,
        refl,
        *env.args(tmp_path, count_tokens=lambda s: 10**9),
    )
    rec = json.loads((tmp_path / "rounds" / "round_02.json").read_text())
    assert rec["skipped"] == "prompt_over_budget" and rec["proposals"] == []
    assert state["rounds_done"] == 2 and state["champion"] == "v0" and refl.calls == 0


def test_history_is_halved_until_it_fits(tmp_path, env, monkeypatch):
    import rsi.search as search

    sizes = []
    real = search.build_reflector_messages

    def fake(ctx, count_tokens, budget, pcc):
        sizes.append(len(ctx.history))
        if len(ctx.history) > 1:
            raise ValueError("too long")
        return real(ctx, count_tokens, budget, pcc)

    monkeypatch.setattr(search, "build_reflector_messages", fake)
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "no change"})] * 30)
    run_search(env.cfg(rounds=4, history_max=8), "search", env.executor, refl, *env.args(tmp_path))
    rec = json.loads((tmp_path / "rounds" / "round_04.json").read_text())
    assert rec["skipped"] is None
    assert sizes[-4:] == [8, 4, 2, 1]


def test_leakage_is_relative_to_the_champion(tmp_path, env):
    env.examples[0].answer = "london"
    env.seed.orchestrator["tool_use_note"] += " Prefer london sources."
    keep = proposal_json("orchestrator", {"turn1n_prompt": "check"})
    leak = proposal_json("orchestrator", {"turn1n_prompt": env.examples[1].answer})
    refl = FakeReflector([keep, leak, keep])
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    props = json.loads((tmp_path / "rounds" / "round_01.json").read_text())["proposals"]
    assert [bool(p["leak_errors"]) for p in props] == [False, True, False]
    assert all(not p["errors"] for p in props)


def test_resume_rejects_a_changed_config(tmp_path, env):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": "check"})] * 30)
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    with pytest.raises(ValueError, match="shown_minibatch"):
        run_search(
            env.cfg(rounds=2, shown_minibatch=3),
            "search",
            env.fresh_executor(),
            refl,
            *env.args(tmp_path),
        )
    state = run_search(env.cfg(rounds=2), "search", env.fresh_executor(), refl, *env.args(tmp_path))
    assert state["rounds_done"] == 2
    assert json.loads((tmp_path / "config.json").read_text())["rounds"] == 2
