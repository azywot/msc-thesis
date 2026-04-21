"""Unit tests for agent_engine.prompts.builder."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.models.base import ToolCallFormat


@pytest.fixture
def builder():
    return PromptBuilder()


@pytest.fixture
def search_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]


@pytest.fixture
def code_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "code_generator",
                "description": "Run Python code",
                "parameters": {
                    "type": "object",
                    "properties": {"task": {"type": "string"}},
                    "required": ["task"],
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# JSON format (Qwen3 / default)
# ---------------------------------------------------------------------------

class TestBuildSystemPromptJSON:
    def test_has_sub_goal(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON)
        assert "<sub_goal>" in prompt

    def test_has_tool_call_xml(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON)
        assert "<tool_call>" in prompt

    def test_no_tool_call_json_key(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON)
        # "tool_call" appears only as part of the XML tag or inside JSON examples
        # The key "tool_call" as a JSON field should not be in the instructions
        assert '"tool_call"' not in prompt.split("### EXAMPLE")[0]

    def test_has_tools_section(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON)
        assert "# Tools" in prompt
        assert "web_search" in prompt

    def test_no_tools_omits_tool_section(self, builder):
        prompt = builder.build_system_prompt("gaia", [], tool_call_format=ToolCallFormat.JSON)
        assert "# Tools" not in prompt
        assert "<tool_call>" not in prompt

    def test_baseline_omits_sub_goal(self, builder, search_schema):
        prompt = builder.build_system_prompt(
            "gaia", search_schema, baseline=True, tool_call_format=ToolCallFormat.JSON
        )
        assert "<sub_goal>" not in prompt
        assert "<tool_call>" in prompt


# ---------------------------------------------------------------------------
# JSON_SINGLE format (DeepSeek)
# ---------------------------------------------------------------------------

class TestBuildSystemPromptJsonSingle:
    def test_has_sub_goal(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        assert "<sub_goal>" in prompt

    def test_no_xml_tool_call_tag(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        # Must not contain XML-style tool_call open/close tags
        assert "<tool_call>" not in prompt
        assert "</tool_call>" not in prompt

    def test_has_tool_call_json_key(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        assert '"tool_call"' in prompt

    def test_example_wraps_call_in_single_object(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        assert "### EXAMPLE" in prompt
        # Example must use {"tool_call": {...}} not {"tool_calls": [...]}
        assert '"tool_call"' in prompt
        assert '"tool_calls"' not in prompt

    def test_example_has_sub_goal(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        assert "<sub_goal>" in prompt

    def test_instructions_mention_one_tool_per_turn(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        assert "one tool per turn" in prompt.lower() or "at most one" in prompt.lower()

    def test_baseline_no_sub_goal(self, builder, search_schema):
        prompt = builder.build_system_prompt(
            "gaia", search_schema, baseline=True, tool_call_format=ToolCallFormat.JSON_SINGLE
        )
        assert "<sub_goal>" not in prompt
        assert "<tool_call>" not in prompt
        assert '"tool_call"' in prompt

    def test_has_tools_section(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.JSON_SINGLE)
        assert "# Tools" in prompt
        assert "web_search" in prompt

    def test_direct_mode_adds_reasoning_rule(self, builder, search_schema):
        prompt = builder.build_system_prompt(
            "gaia", search_schema, direct_tool_call=True, tool_call_format=ToolCallFormat.JSON_SINGLE
        )
        assert "tool_response" in prompt.lower()


# ---------------------------------------------------------------------------
# PYTHONIC format (OLMo)
# ---------------------------------------------------------------------------

class TestBuildSystemPromptPythonic:
    def test_has_function_calls_tag(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.PYTHONIC)
        assert "<function_calls>" in prompt

    def test_no_tool_call_xml(self, builder, search_schema):
        prompt = builder.build_system_prompt("gaia", search_schema, tool_call_format=ToolCallFormat.PYTHONIC)
        assert "<tool_call>" not in prompt


# ---------------------------------------------------------------------------
# _json_tool_call_to_single
# ---------------------------------------------------------------------------

class TestJsonToolCallToSingle:
    def test_basic(self, builder):
        json_str = '{"name": "web_search", "arguments": {"query": "foo"}}'
        result = builder._json_tool_call_to_single(json_str)
        parsed = json.loads(result)
        assert parsed == {"tool_call": {"name": "web_search", "arguments": {"query": "foo"}}}

    def test_roundtrip(self, builder):
        json_str = '{"name": "code_generator", "arguments": {"task": "write a function"}}'
        result = builder._json_tool_call_to_single(json_str)
        parsed = json.loads(result)
        assert parsed["tool_call"]["name"] == "code_generator"

    def test_invalid_json_fallback(self, builder):
        result = builder._json_tool_call_to_single("NOT JSON")
        assert "tool_call" in result
        assert "NOT JSON" in result

    def test_not_an_array(self, builder):
        """Result must be a single object, not a list."""
        json_str = '{"name": "web_search", "arguments": {"query": "foo"}}'
        result = builder._json_tool_call_to_single(json_str)
        parsed = json.loads(result)
        assert not isinstance(parsed.get("tool_call"), list)


# ---------------------------------------------------------------------------
# _json_tool_call_to_pythonic
# ---------------------------------------------------------------------------

class TestJsonToolCallToPythonic:
    def test_basic(self, builder):
        json_str = '{"name": "web_search", "arguments": {"query": "foo"}}'
        result = builder._json_tool_call_to_pythonic(json_str)
        assert result == 'web_search(query="foo")'

    def test_multiple_args(self, builder):
        json_str = '{"name": "some_tool", "arguments": {"a": "x", "b": 1}}'
        result = builder._json_tool_call_to_pythonic(json_str)
        assert result.startswith("some_tool(")
        assert "a=" in result
        assert "b=" in result

    def test_invalid_json_fallback(self, builder):
        result = builder._json_tool_call_to_pythonic("NOT JSON")
        assert result == "NOT JSON"


# ---------------------------------------------------------------------------
# Dataset template routing
# ---------------------------------------------------------------------------

class TestTemplateRouting:
    def test_hle_uses_gaia_template(self, builder, search_schema):
        prompt_gaia = builder.build_system_prompt("gaia", search_schema)
        prompt_hle = builder.build_system_prompt("hle", search_schema)
        assert prompt_gaia == prompt_hle

    def test_musique_uses_gaia_template(self, builder, search_schema):
        prompt_gaia = builder.build_system_prompt("gaia", search_schema)
        prompt_musique = builder.build_system_prompt("musique", search_schema)
        assert prompt_gaia == prompt_musique

    def test_aime_uses_math_template(self, builder):
        prompt_aime = builder.build_system_prompt("aime", [])
        prompt_math500 = builder.build_system_prompt("math500", [])
        assert prompt_aime == prompt_math500

    def test_case_insensitive(self, builder, search_schema):
        prompt_upper = builder.build_system_prompt("GAIA", search_schema)
        prompt_lower = builder.build_system_prompt("gaia", search_schema)
        assert prompt_upper == prompt_lower
