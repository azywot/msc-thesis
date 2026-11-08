"""PromptBuilder overrides for the editable orchestrator parts."""

import pytest

from agent_engine.prompts.builder import DEFAULT_TOOL_USE_NOTE, PromptBuilder
from agent_engine.tools.code_generator import CodeGeneratorTool
from agent_engine.tools.web_search import WebSearchTool


class FakeProvider:
    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return ""

    def generate(self, prompts):
        return []


SCHEMAS = [
    WebSearchTool(
        api_key="k", search_cache={}, url_cache={}, model_provider=FakeProvider()
    ).get_schema(),
    CodeGeneratorTool(model_provider=FakeProvider()).get_schema(),
]


@pytest.mark.parametrize("dataset", ["gaia", "deepmath"])
def test_resolved_parts_as_overrides_render_identically(dataset):
    pb = PromptBuilder()
    plain = pb.build_system_prompt(dataset, SCHEMAS, direct_tool_call=False)
    parts = pb.resolve_orchestrator_parts(dataset, SCHEMAS, direct_tool_call=False)
    assert set(parts) == {"guidance_opening", "guidance_closing", "example", "tool_use_note"}
    assert parts["tool_use_note"] == DEFAULT_TOOL_USE_NOTE
    assert (
        pb.build_system_prompt(dataset, SCHEMAS, direct_tool_call=False, overrides=parts) == plain
    )


def test_guidance_opening_override_replaces_only_first_section():
    pb = PromptBuilder()
    plain = pb.build_system_prompt("gaia", SCHEMAS, direct_tool_call=False)
    out = pb.build_system_prompt(
        "gaia",
        SCHEMAS,
        direct_tool_call=False,
        overrides={"guidance_opening": "You are a research coordinator."},
    )
    assert out.startswith("You are a research coordinator.\n\n# Tools")
    assert out.split("# Tools", 1)[1] == plain.split("# Tools", 1)[1]


def test_tool_use_note_override():
    out = PromptBuilder().build_system_prompt(
        "gaia",
        SCHEMAS,
        direct_tool_call=False,
        overrides={"tool_use_note": "Delegate every look-up."},
    )
    assert "Delegate every look-up." in out
    assert DEFAULT_TOOL_USE_NOTE not in out
    assert "<sub_goal>A specific, actionable goal for this step</sub_goal>" in out


def test_example_override_renders_through_builder():
    ex = {
        "question": "Q?",
        "steps": [
            {
                "step": 1,
                "reasoning": "r",
                "sub_goal": "g",
                "tool_call": '{"name": "web_search", "arguments": {"query": "x"}}',
                "tool_response": "y",
            },
            {"step": "final", "answer": "\\boxed{y}"},
        ],
    }
    out = PromptBuilder().build_system_prompt(
        "gaia", SCHEMAS, direct_tool_call=False, overrides={"example": ex}
    )
    assert 'Question: "Q?"' in out and "<tool_call>\n" in out and "**Answer:** \\boxed{y}" in out
    assert "atomic number of Oxygen" not in out


def test_boxed_rule_is_not_overridable():
    out = PromptBuilder().build_system_prompt(
        "gaia", SCHEMAS, direct_tool_call=False, overrides={"guidance_closing": "- Be brief."}
    )
    assert "Provide your final answer in the format \\boxed{YOUR_ANSWER}" in out
    assert "- Be brief." in out
