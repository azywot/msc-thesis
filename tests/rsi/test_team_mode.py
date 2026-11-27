"""Team edits: one plain-text reply rewrites the orchestrator and the sub-agents together."""

import json

import pytest

from agent_engine.models.base import ToolCallFormat
from agent_engine.prompts.templating import find_placeholders
from agent_engine.tools.code_generator import CODE_GENERATOR_TASK_TEMPLATE
from agent_engine.tools.web_search import WEB_SEARCH_ANALYSIS_TEMPLATE
from rsi.bundle import seed_bundle
from rsi.confirm import mcnemar_p, run_ablation, run_confirm
from rsi.reflector import ReflectorSettings, parse_proposal
from rsi.reflector_input import ReflectorContext, build_reflector_messages
from rsi.report import write_report
from rsi.search import SearchConfig, run_search
from rsi.team import (
    UNCHANGED,
    apply_team,
    changed_sections,
    compose_subagent,
    parse_team_reply,
    render_team_document,
    resolve_sections,
    team_sections,
    validate_team,
)

from .test_confirm import _add_unused
from .test_search import FakeExecutor, FakeReflector, env  # noqa: F401

FMT = ToolCallFormat.JSON
ENABLED = ["web_search", "code_generator"]


def _seed(domain, schemas):
    return seed_bundle(domain, schemas, orchestrator_mode="full", tool_call_format=FMT)


def team_reply(**sections):
    """A reflector reply in the team format; omitted sections are left out of the reply."""
    headers = {
        "system_prompt": "ORCHESTRATOR SYSTEM PROMPT",
        "turn0_prompt": "PLANNING TURN",
        "turn1n_prompt": "AFTER EACH TOOL RESULT",
        "web_search": "WEB_SEARCH SUB-AGENT",
        "code_generator": "CODE_GENERATOR SUB-AGENT",
    }
    parts = ["=== RATIONALE ===\nSearch twice when unsure."]
    parts += [f"=== {headers[k]} ===\n{v}" for k, v in sections.items()]
    return "\n\n".join(parts)


WS_NEW = "Read the pages. Reply as:\nANSWER: ...\nCONFIDENCE: HIGH or LOW\n\n{inputs}"


# --- sections: split, render, round trip --------------------------------------------------


@pytest.mark.parametrize("domain", ["search", "math"])
def test_seed_sections_round_trip_byte_identically(domain, SCHEMAS):
    seed = _seed(domain, SCHEMAS)
    secs = team_sections(seed)
    assert list(secs) == list(
        ("system_prompt", "turn0_prompt", "turn1n_prompt", "web_search", "code_generator")
    )
    assert secs["web_search"].endswith("No helpful information found.\n\n{inputs}")
    assert "{inputs}\n\nRequirements:" in secs["code_generator"]
    assert apply_team(seed, secs).to_dict() == seed.to_dict()
    # the document parses back to the same sections
    parsed, rationale, unknown = parse_team_reply(render_team_document(seed))
    assert unknown == [] and rationale == ""
    assert resolve_sections(parsed, seed) == secs


def test_subagent_inputs_are_appended_when_inputs_marker_is_missing():
    text = compose_subagent("web_search", "Just read the pages.")
    assert text.startswith("Just read the pages.\n\n**Inputs:**")
    assert find_placeholders(text).count("query") == 2
    assert "{formatted_results}" in text
    assert compose_subagent("code_generator", "Code only.\n{inputs}\nPrint the result.") == (
        "Code only.\n{context_block}Problem: {task}\nPrint the result."
    )
    # the seed templates are the composition of their own sections
    assert CODE_GENERATOR_TASK_TEMPLATE.count("{context_block}Problem: {task}") == 1
    assert WEB_SEARCH_ANALYSIS_TEMPLATE.endswith('current search query "{query}".\n')


# --- parsing ------------------------------------------------------------------------------


def test_parse_reads_unchanged_omitted_fenced_and_thinking():
    text = (
        "<think>long thoughts</think>\n```text\n"
        + team_reply(turn0_prompt="UNCHANGED", turn1n_prompt="(empty)", web_search=WS_NEW)
        + "\n```"
    )
    sections, rationale, unknown = parse_team_reply(text)
    assert rationale == "Search twice when unsure."
    assert sections == {"turn0_prompt": UNCHANGED, "turn1n_prompt": "", "web_search": WS_NEW}
    assert unknown == []


def test_parse_accepts_loose_headers_and_reports_unknown_ones():
    text = "==== web_search sub-agent ====\nNew.\n\n=== PLANNER ===\nx"
    sections, _, unknown = parse_team_reply(text)
    assert sections == {"web_search": "New."} and unknown == ["PLANNER"]


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("", "empty reflector output"),
        ('{"unit": "orchestrator"}', "no '=== SECTION ===' headers"),
        ("=== RATIONALE ===\nonly a rationale", "no team section"),
        (team_reply(web_search="a") + "\n\n=== WEB_SEARCH SUB-AGENT ===\nb", "appears twice"),
    ],
)
def test_parse_errors_name_the_problem(text, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_team_reply(text)


# --- validation ---------------------------------------------------------------------------


def _check(SCHEMAS, **sections):
    seed = _seed("search", SCHEMAS)
    base = team_sections(seed)
    parsed, _, unknown = parse_team_reply(team_reply(**sections))
    return validate_team(resolve_sections(parsed, seed), unknown, ENABLED, base)


def test_a_joint_edit_is_valid(SCHEMAS):
    assert _check(SCHEMAS, turn1n_prompt="Check CONFIDENCE.", web_search=WS_NEW) == []


@pytest.mark.parametrize(
    "sections, fragment",
    [
        ({"turn0_prompt": UNCHANGED}, "no section changed"),
        ({"system_prompt": "No tools here. {answer_format}"}, "{tools} exactly once (found 0)"),
        ({"turn0_prompt": "(empty)"}, "PLANNING TURN must be non-empty"),
        ({"turn1n_prompt": "Use {tools}."}, "must not contain ['tools']"),
        ({"web_search": "Read {query} in {formatted_results}"}, "write {inputs} instead"),
        ({"web_search": "{inputs} and {inputs}"}, "at most once (found 2)"),
        ({"code_generator": "{inputs}"}, "must contain instructions"),
    ],
)
def test_invalid_edits_are_rejected_with_a_reason(SCHEMAS, sections, fragment):
    errors = _check(SCHEMAS, **sections)
    assert any(fragment in e for e in errors), errors


def test_a_disabled_tool_section_is_rejected(SCHEMAS):
    seed = _seed("search", SCHEMAS)
    base = team_sections(seed, ["web_search"])
    parsed, _, unknown = parse_team_reply(team_reply(code_generator="Code only."))
    payload = resolve_sections(parsed, seed, ["web_search"])
    errors = validate_team(payload, unknown, ["web_search"], base)
    assert any("'code_generator' is not enabled" in e for e in errors), errors


def test_literal_braces_in_instructions_stay_literal(SCHEMAS):
    seed = _seed("search", SCHEMAS)
    text = 'Reply as {"answer": "..."} or {answer}.\n{inputs}'
    assert _check(SCHEMAS, web_search=text) == []
    cand = apply_team(seed, resolve_sections({"web_search": text}, seed))
    assert cand.subagents["web_search"].startswith('Reply as {"answer": "..."} or {answer}.')


# --- reflector input ----------------------------------------------------------------------


def test_reflector_messages_show_the_team_document_and_format(SCHEMAS):
    seed = _seed("search", SCHEMAS)
    ctx = ReflectorContext(
        domain="search",
        champion=seed,
        seed=seed,
        best=seed,
        rendered_system_prompt="RENDERED",
        units=["team"],
        team=True,
        subagents=ENABLED,
    )
    msgs, _ = build_reflector_messages(ctx, lambda s: len(s) // 4, 14000, 1500)
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "=== RATIONALE ===" in system and "Reply in plain text (no JSON)" in system
    assert "=== CODE_GENERATOR SUB-AGENT ===\n<new text, or UNCHANGED>" in system
    assert "{inputs}" in system and "hand-offs" in system
    assert "Team document:" in user and "=== WEB_SEARCH SUB-AGENT ===" in user
    assert "Editable units:" not in user


# --- search loop, confirm and ablation ----------------------------------------------------


def _team_cfg(env, **kw):
    return env.cfg(orchestrator_mode="full", edit_mode="team", **kw)


def _team_args(env, tmp_path, SCHEMAS):
    args = list(env.args(tmp_path))
    args[2] = _seed("search", SCHEMAS)
    return args


def test_config_rules_for_team_mode():
    base = dict(data_path="x.parquet", reflector=ReflectorSettings(model="m"))
    with pytest.raises(ValueError, match='needs orchestrator_mode "full"'):
        SearchConfig(edit_mode="team", **base)
    with pytest.raises(ValueError, match="slot_units"):
        SearchConfig(edit_mode="team", orchestrator_mode="full", slot_units=["web_search"], **base)
    cfg = SearchConfig(edit_mode="team", orchestrator_mode="full", **base)
    assert cfg.units_for_slot(0) == ["team"]


def test_team_search_promotes_a_joint_edit(tmp_path, env, SCHEMAS):
    refl = FakeReflector([team_reply(turn1n_prompt="check check", web_search=WS_NEW)])
    state = run_search(
        _team_cfg(env, rounds=1), "search", env.executor, refl, *_team_args(env, tmp_path, SCHEMAS)
    )
    assert state["champion"] != "v0"
    champ = json.loads((tmp_path / "bundles" / f"{state['champion']}.json").read_text())
    assert champ["orchestrator"]["turn1n_prompt"] == "check check"
    assert champ["subagents"]["web_search"].startswith("Read the pages.")
    assert "{formatted_results}" in champ["subagents"]["web_search"]
    record = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    p = record["proposals"][0]
    assert p["unit"] == "team" and p["errors"] == [] and p["rationale"].startswith("Search twice")
    assert p["sections_changed"] == ["turn1n_prompt", "web_search"]
    entry = state["history"][0]
    assert entry["unit"] == "team" and "candidate/WEB_SEARCH SUB-AGENT" in entry["diff"]
    report = write_report(tmp_path, SCHEMAS, FMT).read_text()
    assert "| team | 3 | 1 |" in report
    assert "| AFTER EACH TOOL RESULT + WEB_SEARCH SUB-AGENT | 3 |" in report


def test_invalid_team_reply_is_recorded_with_its_reason(tmp_path, env, SCHEMAS):
    refl = FakeReflector([team_reply(web_search="Read {query}.")])
    cfg = _team_cfg(env, rounds=1, candidates_per_round=1)
    state = run_search(cfg, "search", env.executor, refl, *_team_args(env, tmp_path, SCHEMAS))
    assert state["champion"] == "v0"
    record = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert "write {inputs} instead" in record["proposals"][0]["errors"][0]


def test_json_reply_in_team_mode_is_resampled(tmp_path, env, SCHEMAS):
    refl = FakeReflector(['{"unit": "web_search", "replacement": "x"}'])
    cfg = _team_cfg(env, rounds=1, candidates_per_round=1)
    run_search(cfg, "search", env.executor, refl, *_team_args(env, tmp_path, SCHEMAS))
    record = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert len(record["proposals"]) == cfg.max_parse_retries + 1
    assert all(p["errors"][0].startswith("parse error: no '===") for p in record["proposals"])


def test_ablation_reverts_each_changed_section(tmp_path, env, SCHEMAS):
    _add_unused(env)
    refl = FakeReflector([team_reply(turn1n_prompt="check check", web_search=WS_NEW)])
    args = _team_args(env, tmp_path, SCHEMAS)
    run_search(_team_cfg(env, rounds=1), "search", env.executor, refl, *args)
    with pytest.raises(ValueError, match="run --confirm before --ablate"):
        run_ablation(tmp_path, FakeExecutor(), env.examples)
    run_confirm(tmp_path, FakeExecutor(), env.examples, 10)
    result = run_ablation(tmp_path, FakeExecutor(), env.examples)
    reverted = result["reverted"]
    assert set(reverted) == {"turn1n_prompt", "web_search"}
    # FakeExecutor scores 0.2 + 0.1 per "check" in turn1n_prompt
    assert reverted["turn1n_prompt"]["acc"] == pytest.approx(0.2)
    assert reverted["turn1n_prompt"]["lost_vs_champion"] == 2
    assert reverted["web_search"]["acc"] == pytest.approx(0.4)
    assert reverted["web_search"]["lost_vs_champion"] == 0
    assert json.loads((tmp_path / "ablation.json").read_text()) == result
    report = write_report(tmp_path, SCHEMAS, FMT).read_text()
    assert "With AFTER EACH TOOL RESULT reverted to the seed: 20.0% (lost 2, gained 0" in report


def test_ablation_needs_full_mode(tmp_path, env):
    _add_unused(env)
    refl = FakeReflector(
        [json.dumps({"unit": "orchestrator", "replacement": {"turn1n_prompt": "check check"}})]
    )
    run_search(env.cfg(rounds=1), "search", env.executor, refl, *env.args(tmp_path))
    run_confirm(tmp_path, FakeExecutor(), env.examples, 10)
    with pytest.raises(ValueError, match="orchestrator_mode: full"):
        run_ablation(tmp_path, FakeExecutor(), env.examples)


# --- small fixes that come with this mode -------------------------------------------------


def test_mcnemar_p_matches_the_exact_binomial():
    assert mcnemar_p(14, 23) == pytest.approx(0.18774, abs=1e-5)
    assert mcnemar_p(23, 26) == pytest.approx(0.77545, abs=1e-5)
    assert mcnemar_p(0, 0) == 1.0


def test_json_backspace_escapes_become_latex_again():
    text = json.dumps({"unit": "orchestrator", "replacement": {"turn1n_prompt": "PLACEHOLDER"}})
    text = text.replace("PLACEHOLDER", "Answer as \\boxed{x} or \\frac{a}{b}")
    proposal = parse_proposal(text)
    assert proposal.replacement["turn1n_prompt"] == "Answer as \\boxed{x} or \\frac{a}{b}"
    assert "\b" not in json.dumps(proposal.replacement)
