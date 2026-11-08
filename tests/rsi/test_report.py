import json

from agent_engine.models.base import ToolCallFormat
from rsi.report import write_report
from rsi.search import run_search

from .test_search import FakeReflector, env, proposal_json  # noqa: F401

HEADINGS = [
    "## Summary",
    "## Search curve",
    "## Rejections",
    "## Units edited",
    "## Final prompt versus seed",
    "## Rendered prompts",
    "## Paired trajectories",
]


def _run(env, tmp_path, texts, rounds=1):
    refl = FakeReflector(texts)
    state = run_search(env.cfg(rounds=rounds), "search", env.executor, refl, *env.args(tmp_path))
    return state


def _no_em_dash(text):
    assert "—" not in text


def test_report_has_all_sections(tmp_path, env, SCHEMAS):
    state = _run(
        env, tmp_path, [proposal_json("orchestrator", {"turn1n_prompt": "check check"})] * 3
    )
    assert state["champion"] != "v0"
    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    positions = [text.index(h) for h in HEADINGS]
    assert positions == sorted(positions)
    assert '+  "turn1n_prompt": "check' in text
    assert "Question " in text and "Champion (v1)" in text
    assert "<question>" in text and "<query analysis>" in text and "<formatted_results>" in text
    assert f"| 1 | {state['rollouts']} |" in text  # derived rollouts match the counter
    _no_em_dash(text)


def test_report_counts_rejections_and_units(tmp_path, env, SCHEMAS):
    texts = ["not json", proposal_json("orchestrator", {"turn1n_prompt": "check"})]
    _run(env, tmp_path, texts)
    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    assert "| parse error |" in text and "| orchestrator |" in text


def test_report_with_zero_rounds(tmp_path, env, SCHEMAS):
    _run(env, tmp_path, [], rounds=0)
    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    for h in HEADINGS:
        assert h in text
    assert "no differing questions" in text and text.count("unchanged") == 3
    assert json.loads((tmp_path / "state.json").read_text())["rounds_done"] == 0


def test_report_with_skipped_round_and_seed_champion(tmp_path, env, SCHEMAS):
    _run(env, tmp_path, [proposal_json("orchestrator", {"turn1n_prompt": "x"})] * 3)
    path = tmp_path / "rounds" / "round_01.json"
    rec = json.loads(path.read_text())
    rec.update(skipped="prompt_over_budget", proposals=[], heldout=None, chosen=None)
    path.write_text(json.dumps(rec))
    text = write_report(tmp_path, SCHEMAS, ToolCallFormat.JSON).read_text()
    assert "skipped (prompt_over_budget)" in text
