import json

import pytest

from agent_engine.datasets.base import DatasetExample
from agent_engine.models.base import ToolCallFormat
from rsi.confirm import confirm_question_ids, run_confirm
from rsi.data import Pools
from rsi.report import write_report
from rsi.search import run_search

from .test_search import FakeExecutor, FakeReflector, env, proposal_json  # noqa: F401

N_EXTRA = 20


def _add_unused(env):
    start = max(env.examples) + 1
    for i in range(start, start + N_EXTRA):
        env.examples[i] = DatasetExample(
            question_id=i, question=f"fresh question {i}", answer=f"fresh{i}", metadata={}
        )
    return set(range(start, start + N_EXTRA))


def _search(env, tmp_path, text):
    refl = FakeReflector([proposal_json("orchestrator", {"turn1n_prompt": text})] * 3)
    return run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))


def test_confirm_ids_avoid_every_pool_and_are_seeded():
    pools = Pools(shown=[0, 1], screen=[2], heldout=[3])
    ids = confirm_question_ids(range(20), pools, 5, pool_seed=0)
    assert len(set(ids)) == 5 and not set(ids) & {0, 1, 2, 3}
    assert ids == confirm_question_ids(range(20), pools, 5, pool_seed=0)
    assert ids != confirm_question_ids(range(20), pools, 5, pool_seed=1)
    with pytest.raises(ValueError, match="16 unused"):
        confirm_question_ids(range(20), pools, 17, pool_seed=0)


def test_confirm_scores_seed_and_champion_on_fresh_questions(tmp_path, env, SCHEMAS):
    unused = _add_unused(env)
    state = _search(env, tmp_path, "check check")
    champion = state["champion"]
    assert champion != "v0"
    before = json.loads((tmp_path / "state.json").read_text())

    result = run_confirm(tmp_path, FakeExecutor(), env.examples, 10)
    saved = json.loads((tmp_path / "confirm.json").read_text())
    assert saved == result
    assert saved["n"] == 10 and set(saved["question_ids"]) <= unused
    assert saved["champion"] == champion
    # FakeExecutor: 0.2 + 0.1 per "check"
    assert saved["seed_acc"] == pytest.approx(0.2) and saved["champion_acc"] == pytest.approx(0.4)
    assert saved["paired"] == {"both": 2, "seed_only": 0, "champion_only": 2, "neither": 6}
    for label in ("v0", champion):
        recs = json.loads((tmp_path / "confirm" / f"{label}.json").read_text())
        assert [r["question_id"] for r in recs] == saved["question_ids"]
    assert json.loads((tmp_path / "state.json").read_text()) == before  # search state untouched

    # a rerun reuses the per-version records and runs nothing
    again = FakeExecutor()
    assert run_confirm(tmp_path, again, env.examples, 10) == result and again.rollouts == 0

    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    assert "optimistic" in text
    assert f"Confirmation on 10 unused questions: seed 20.0%, champion {champion} 40.0%" in text


def test_confirm_with_seed_champion_runs_v0_once(tmp_path, env, SCHEMAS):
    _add_unused(env)
    _search(env, tmp_path, "no change")
    ex = FakeExecutor()
    result = run_confirm(tmp_path, ex, env.examples, 8)
    assert result["champion"] == "v0" and ex.rollouts == 8
    assert result["seed_acc"] == result["champion_acc"]
    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    assert "Confirmation on 8 unused questions" in text


def test_report_without_confirm_says_how_to_run_it(tmp_path, env, SCHEMAS):
    _search(env, tmp_path, "check")
    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    assert "optimistic" in text and "--confirm N" in text
    assert "[reflector input](rounds/round_01_reflector_input.json)" in text


def test_confirm_needs_a_finished_search(tmp_path, env):
    with pytest.raises(ValueError, match="state.json"):
        run_confirm(tmp_path, FakeExecutor(), env.examples, 5)
