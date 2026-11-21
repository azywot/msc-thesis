import pytest

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
from agent_engine.models.base import ToolCallFormat
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools.code_generator import CODE_GENERATOR_TASK_TEMPLATE
from agent_engine.tools.web_search import WEB_SEARCH_ANALYSIS_TEMPLATE
from rsi.bundle import (
    acting_suffix,
    apply_subagent_templates,
    apply_tool_descriptions,
    editable_texts,
    load_bundle,
    planning_suffix,
    render_system_prompt,
    replace_unit,
    save_bundle,
    seed_bundle,
    unit_text,
)


@pytest.mark.parametrize("domain,dataset", [("search", "gaia"), ("math", "deepmath")])
def test_seed_bundle_renders_todays_system_prompt(domain, dataset, SCHEMAS):
    b = seed_bundle(domain, SCHEMAS)
    assert render_system_prompt(
        b, SCHEMAS, ToolCallFormat.JSON
    ) == PromptBuilder().build_system_prompt(dataset, SCHEMAS, direct_tool_call=False)


def test_seed_suffixes_match_todays_defaults(SCHEMAS):
    b = seed_bundle("search", SCHEMAS)
    assert planning_suffix(b) == _DEFAULT_PLANNING_SUFFIX_TOOLS
    assert acting_suffix(b) == ""


def test_seed_subagent_templates_are_todays(SCHEMAS):
    b = seed_bundle("math", SCHEMAS)
    assert b.subagents == {
        "web_search": WEB_SEARCH_ANALYSIS_TEMPLATE,
        "code_generator": CODE_GENERATOR_TASK_TEMPLATE,
    }


def test_apply_tool_descriptions_changes_only_descriptions(SCHEMAS):
    out = apply_tool_descriptions(
        SCHEMAS, {"web_search": {"description": "D", "parameters": {"query": "P"}}}
    )
    ws = next(s for s in out if s["function"]["name"] == "web_search")
    assert ws["function"]["description"] == "D"
    assert ws["function"]["parameters"]["properties"]["query"] == {
        "type": "string",
        "description": "P",
    }
    assert SCHEMAS[0]["function"]["description"] != "D"  # input not mutated


def test_round_trip_and_replace_unit(tmp_path, SCHEMAS):
    b = seed_bundle("search", SCHEMAS)
    save_bundle(b, tmp_path / "b.json")
    assert load_bundle(tmp_path / "b.json") == b
    b2 = replace_unit(b, "orchestrator", {"turn1n_prompt": "Check links."})
    assert (
        acting_suffix(b2) == "\n\nCheck links."
        and b2.orchestrator["turn0_prompt"] == b.orchestrator["turn0_prompt"]
    )
    assert b.orchestrator["turn1n_prompt"] == ""


def test_seed_with_edited_descriptions_still_renders_seed(SCHEMAS):
    b = seed_bundle("search", SCHEMAS)
    b2 = replace_unit(b, "web_search", "new {x}")
    assert b2.subagents["web_search"] == "new {x}" and b.subagents["web_search"] != "new {x}"
    assert unit_text(b2, "web_search") == "new {x}"
    assert '"guidance_opening"' in unit_text(b, "orchestrator")


def test_apply_subagent_templates_and_editable_texts(SCHEMAS):
    calls = {}

    class T:
        def __init__(self, n):
            self.n = n

        def set_analysis_template(self, t):
            calls["ws"] = t

        def set_task_template(self, t):
            calls["cg"] = t

    b = seed_bundle("search", SCHEMAS)

    class R:
        def get(self, name):
            return T(name) if name != "code_generator" else None

    apply_subagent_templates(b, R())
    assert calls == {"ws": b.subagents["web_search"]}
    texts = editable_texts(b)
    assert b.orchestrator["turn0_prompt"] in texts
    assert b.orchestrator["tool_descriptions"]["web_search"]["description"] in texts
    assert b.orchestrator["example"]["question"] in texts
