"""Fixed unit per candidate slot, full-control orchestrator mode, explicit validity rules."""

import json

import pytest

from agent_engine.models.base import ToolCallFormat
from agent_engine.prompts.templating import find_placeholders
from rsi.bundle import (
    FULL_PROMPT_KEY,
    editable_texts,
    is_full,
    render_system_prompt,
    replace_unit,
    seed_bundle,
    unit_text,
)
from rsi.reflector import ReflectorSettings
from rsi.reflector_input import _system_message, output_schema_text
from rsi.search import SearchConfig, run_search, write_run_header
from rsi.validate import validate_unit
from tests.rsi.test_search import FakeReflector, env, proposal_json  # noqa: F401

FMT = ToolCallFormat.JSON


def _full(domain, schemas):
    return seed_bundle(domain, schemas, orchestrator_mode="full", tool_call_format=FMT)


# --- full mode: seed, rendering, validation ---------------------------------------------


@pytest.mark.parametrize("domain", ["search", "math"])
def test_full_seed_renders_byte_identically_to_parts_seed(domain, SCHEMAS):
    parts = seed_bundle(domain, SCHEMAS)
    full = _full(domain, SCHEMAS)
    assert is_full(full) and not is_full(parts)
    assert render_system_prompt(full, SCHEMAS, FMT) == render_system_prompt(parts, SCHEMAS, FMT)
    names = find_placeholders(full.orchestrator[FULL_PROMPT_KEY])
    assert names.count("tools") == 1 and names.count("answer_format") == 1
    assert full.orchestrator["turn0_prompt"] == parts.orchestrator["turn0_prompt"]
    assert full.subagents == parts.subagents


def test_unknown_orchestrator_mode_is_rejected(SCHEMAS):
    with pytest.raises(ValueError, match="orchestrator_mode"):
        seed_bundle("search", SCHEMAS, orchestrator_mode="whole")


def test_rewritten_full_prompt_renders_protected_sections(SCHEMAS):
    seed = _full("search", SCHEMAS)
    payload = {FULL_PROMPT_KEY: "Be brief.\n\n{tools}\n\nThink twice.\n\n{answer_format}"}
    assert validate_unit("orchestrator", payload, seed, SCHEMAS) == []
    cand = replace_unit(seed, "orchestrator", payload)
    text = render_system_prompt(cand, SCHEMAS, FMT)
    assert text.startswith("Be brief.\n\n")
    assert "<tools>" in text and '"name": "web_search"' in text
    assert "\\boxed{YOUR_ANSWER}" in text
    assert "{tools}" not in text and "{answer_format}" not in text


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({FULL_PROMPT_KEY: "no placeholders {answer_format}"}, "{tools} exactly once (found 0)"),
        ({FULL_PROMPT_KEY: "{tools} {answer_format} {answer_format}"}, "(found 2)"),
        ({"example": {}}, "unknown orchestrator part 'example'"),
        ({"tool_use_note": "x"}, "unknown orchestrator part 'tool_use_note'"),
        ({"turn0_prompt": "  "}, "turn0_prompt must be non-empty"),
        ({FULL_PROMPT_KEY: 3}, "system_prompt must be a string"),
    ],
)
def test_full_mode_validation_names_the_rule(payload, fragment, SCHEMAS):
    errors = validate_unit("orchestrator", payload, _full("search", SCHEMAS), SCHEMAS)
    assert any(fragment in e for e in errors), errors


def test_full_mode_allows_empty_turn1n_and_other_braces(SCHEMAS):
    payload = {
        FULL_PROMPT_KEY: '{tools}\nExample call: {"name": "x"} then \\boxed{answer}\n{answer_format}',
        "turn1n_prompt": "",
    }
    assert validate_unit("orchestrator", payload, _full("search", SCHEMAS), SCHEMAS) == []


def test_full_mode_text_and_leakage_cover_the_prompt(SCHEMAS):
    seed = _full("search", SCHEMAS)
    shown = unit_text(seed, "orchestrator")
    assert "--- system_prompt ---" in shown and "--- turn0_prompt ---" in shown
    assert "tool_descriptions" not in shown
    assert seed.orchestrator[FULL_PROMPT_KEY] in editable_texts(seed)


# --- explicit validity rules in the reflector prompt -----------------------------------


def test_schema_spells_out_required_placeholders():
    text = output_schema_text()
    assert "{query}" in text and "{formatted_results}" in text
    assert "{context_block}" in text and "{task}" in text
    assert "no other word in curly braces" in text


def test_single_unit_full_mode_message():
    msg = _system_message("math", ["orchestrator"], full=True)
    assert "This turn you edit the orchestrator unit" in msg
    assert '"unit" must be "orchestrator"' in msg
    assert "{tools} exactly once" in msg and "{answer_format} exactly once" in msg
    assert "guidance_opening" not in msg and "web_search:" not in msg


def test_default_message_lists_every_unit():
    msg = _system_message("search")
    assert "Each turn you edit exactly one unit" in msg
    assert '"orchestrator", "web_search", "code_generator"' in msg


# --- slot units in the search loop -----------------------------------------------------


def test_slot_units_are_validated_and_cycled():
    base = dict(data_path="x.parquet", reflector=ReflectorSettings(model="m"))
    with pytest.raises(ValueError, match="unknown slot_units"):
        SearchConfig(slot_units=["planner"], **base)
    cfg = SearchConfig(slot_units=["orchestrator", "web_search"], **base)
    assert [cfg.units_for_slot(k) for k in range(3)] == [
        ["orchestrator"],
        ["web_search"],
        ["orchestrator"],
    ]
    assert SearchConfig(**base).units_for_slot(1) == ["orchestrator", "web_search", "code_generator"]


class UnitAwareReflector(FakeReflector):
    """Answers with the unit its messages ask for, editing turn1n_prompt or a template."""

    def sample(self, messages, seeds):
        self.messages.append(messages)
        system = messages[0]["content"]
        out = []
        for s in seeds:
            if '"unit" must be "orchestrator"' in system:
                text = proposal_json("orchestrator", {"turn1n_prompt": "check check"})
            elif '"unit" must be "web_search"' in system:
                text = proposal_json("web_search", "Q {query} R {formatted_results}")
            else:
                text = proposal_json("code_generator", "C {context_block}{task}")
            out.append({"text": text, "finish_reason": "stop", "seed": s})
        return out


def test_each_slot_gets_its_own_unit_and_input(tmp_path, env):
    cfg = env.cfg(rounds=1, slot_units=["orchestrator", "web_search", "code_generator"])
    refl = UnitAwareReflector([])
    run_search(cfg, "search", env.executor, refl, *env.args(tmp_path))
    record = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert [p["unit"] for p in record["proposals"]] == [
        "orchestrator",
        "web_search",
        "code_generator",
    ]
    assert all(not p["errors"] for p in record["proposals"])
    assert len(refl.messages) == 3  # one input per unit
    saved = json.loads((tmp_path / "rounds" / "round_01_reflector_input.json").read_text())
    assert saved["slot_units"] == [["orchestrator"], ["web_search"], ["code_generator"]]
    assert set(saved["messages_by_units"]) == {"orchestrator", "web_search", "code_generator"}


def test_a_slot_proposing_another_unit_is_resampled(tmp_path, env):
    cfg = env.cfg(rounds=1, candidates_per_round=1, slot_units=["orchestrator"])
    refl = FakeReflector([proposal_json("web_search", "Q {query} R {formatted_results}")])
    run_search(cfg, "search", env.executor, refl, *env.args(tmp_path))
    record = json.loads((tmp_path / "rounds" / "round_01.json").read_text())
    assert len(record["proposals"]) == cfg.max_parse_retries + 1
    assert all("was not asked for" in p["errors"][0] for p in record["proposals"])


def test_full_mode_search_promotes_a_full_rewrite(tmp_path, env, SCHEMAS):
    seed = _full("search", SCHEMAS)
    args = list(env.args(tmp_path))
    args[2] = seed
    refl = FakeReflector(
        [
            proposal_json(
                "orchestrator",
                {FULL_PROMPT_KEY: "New.\n{tools}\n{answer_format}", "turn1n_prompt": "check check"},
            )
        ]
    )
    state = run_search(env.cfg(rounds=1), "search", env.executor, refl, *args)
    assert state["champion"] != "v0"
    champ = json.loads((tmp_path / "bundles" / f"{state['champion']}.json").read_text())
    assert champ["orchestrator"][FULL_PROMPT_KEY].startswith("New.")


def test_old_run_dirs_resume_with_the_new_default_fields(tmp_path, env):
    cfg = env.cfg()
    write_run_header(cfg, env.pools, tmp_path)
    old = json.loads((tmp_path / "config.json").read_text())
    for key in ("orchestrator_mode", "slot_units"):
        old.pop(key)
    (tmp_path / "config.json").write_text(json.dumps(old))
    write_run_header(cfg, env.pools, tmp_path)  # same settings: accepted
    with pytest.raises(ValueError, match="slot_units"):
        write_run_header(env.cfg(slot_units=["orchestrator"]), env.pools, tmp_path)
