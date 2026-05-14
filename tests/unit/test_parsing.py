"""Unit tests for agent_engine.utils.parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from agent_engine.utils.parsing import extract_answer, parse_tool_call, strip_thinking_tags


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------

class TestParseToolCallQwen3:
    """<tool_call>JSON</tool_call> format."""

    def test_basic(self):
        text = '<tool_call>{"name": "web_search", "arguments": {"query": "foo"}}</tool_call>'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {"query": "foo"}}

    def test_last_tag_wins(self):
        text = (
            '<tool_call>{"name": "web_search", "arguments": {"query": "first"}}</tool_call>'
            "\nsome reasoning\n"
            '<tool_call>{"name": "web_search", "arguments": {"query": "second"}}</tool_call>'
        )
        result = parse_tool_call(text)
        assert result["arguments"]["query"] == "second"

    def test_missing_arguments_defaults_to_empty_dict(self):
        text = '<tool_call>{"name": "web_search"}</tool_call>'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {}}

    def test_invalid_json_falls_through(self):
        text = "<tool_call>NOT JSON</tool_call>"
        result = parse_tool_call(text)
        assert result is None

    def test_with_surrounding_text(self):
        text = "I'll search for this.\n<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"foo\"}}</tool_call>"
        result = parse_tool_call(text)
        assert result["name"] == "web_search"

    def test_with_think_block_before_tag(self):
        text = (
            "<think>Let me search for this.</think>\n"
            '<tool_call>{"name": "web_search", "arguments": {"query": "real"}}</tool_call>'
        )
        result = parse_tool_call(text)
        assert result["arguments"]["query"] == "real"


class TestParseToolCallOLMo:
    """<function_calls>pythonic</function_calls> format."""

    def test_basic(self):
        text = '<function_calls>\nweb_search(query="foo")\n</function_calls>'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {"query": "foo"}}

    def test_json_boolean_literals(self):
        text = '<function_calls>\nsome_tool(flag=true, other=false, val=null)\n</function_calls>'
        result = parse_tool_call(text)
        assert result == {"name": "some_tool", "arguments": {"flag": True, "other": False, "val": None}}

    def test_first_valid_call_returned(self):
        text = '<function_calls>\n\nweb_search(query="first")\ncode_generator(task="second")\n</function_calls>'
        result = parse_tool_call(text)
        assert result["name"] == "web_search"

    def test_invalid_line_skipped(self):
        text = '<function_calls>\nnot_a_call\nweb_search(query="real")\n</function_calls>'
        result = parse_tool_call(text)
        assert result["name"] == "web_search"


class TestParseToolCallDeepSeekJsonSingle:
    """{"tool_call": {...}} format (DeepSeek JSON_SINGLE)."""

    def test_basic(self):
        text = '{"tool_call": {"name": "web_search", "arguments": {"query": "hello"}}}'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {"query": "hello"}}

    def test_missing_arguments_defaults_to_empty_dict(self):
        text = '{"tool_call": {"name": "web_search"}}'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {}}

    def test_think_block_hallucination_ignored(self):
        """JSON_SINGLE inside <think> must NOT be parsed; only the real one outside is."""
        text = (
            "<think>I could call "
            '{"tool_call": {"name": "web_search", "arguments": {"query": "fake"}}}'
            "</think>\n"
            '{"tool_call": {"name": "web_search", "arguments": {"query": "real"}}}'
        )
        result = parse_tool_call(text)
        assert result["arguments"]["query"] == "real"

    def test_only_in_think_block_returns_none(self):
        """Tool call only inside <think> — should not be returned."""
        text = (
            "<think>"
            '{"tool_call": {"name": "web_search", "arguments": {"query": "fake"}}}'
            "</think>\nThe answer is 42."
        )
        result = parse_tool_call(text)
        assert result is None

    def test_with_sub_goal_prefix(self):
        """Typical DeepSeek AF output: <sub_goal> then JSON."""
        text = (
            "<sub_goal>Search for information about X.</sub_goal>\n"
            '{"tool_call": {"name": "web_search", "arguments": {"query": "X info"}}}'
        )
        result = parse_tool_call(text)
        assert result["name"] == "web_search"
        assert result["arguments"]["query"] == "X info"

    def test_with_think_and_sub_goal(self):
        text = (
            "<think>I need to search.</think>\n"
            "<sub_goal>Find info.</sub_goal>\n"
            '{"tool_call": {"name": "web_search", "arguments": {"query": "info"}}}'
        )
        result = parse_tool_call(text)
        assert result["name"] == "web_search"

    def test_first_occurrence_wins(self):
        """When multiple JSON_SINGLE objects appear, the FIRST one is kept (single-call contract)."""
        text = (
            '{"tool_call": {"name": "web_search", "arguments": {"query": "first"}}}\n'
            'Some reasoning.\n'
            '{"tool_call": {"name": "web_search", "arguments": {"query": "second"}}}'
        )
        result = parse_tool_call(text)
        assert result["arguments"]["query"] == "first"

    def test_invalid_json_falls_through(self):
        text = '{"tool_call": NOT JSON}'
        result = parse_tool_call(text)
        assert result is None

    def test_nested_arguments_handled(self):
        """Arguments with nested braces are parsed correctly via raw_decode."""
        text = '{"tool_call": {"name": "web_search", "arguments": {"query": "hello {world}"}}}'
        result = parse_tool_call(text)
        assert result["arguments"]["query"] == "hello {world}"


class TestParseToolCallCodeFence:
    """Code-fenced JSON fallback."""

    def test_json_fence(self):
        text = '```json\n{"name": "web_search", "arguments": {"query": "foo"}}\n```'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {"query": "foo"}}

    def test_plain_fence(self):
        text = '```\n{"name": "web_search", "arguments": {"query": "foo"}}\n```'
        result = parse_tool_call(text)
        assert result["name"] == "web_search"

    def test_json_inside_think_not_matched(self):
        """Code-fenced JSON inside <think> should not be returned."""
        text = (
            "<think>```json\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"fake\"}}\n```</think>\n"
            "The answer is 42."
        )
        result = parse_tool_call(text)
        assert result is None


class TestParseToolCallRawJson:
    """Raw JSON fallback."""

    def test_basic(self):
        text = 'Here is the call: {"name": "web_search", "arguments": {"query": "foo"}}'
        result = parse_tool_call(text)
        assert result == {"name": "web_search", "arguments": {"query": "foo"}}

    def test_inside_think_not_matched(self):
        text = (
            '<think>{"name": "web_search", "arguments": {"query": "fake"}}</think>\n'
            "The answer is 42."
        )
        result = parse_tool_call(text)
        assert result is None


class TestParseToolCallNone:
    def test_empty_string(self):
        assert parse_tool_call("") is None

    def test_plain_text(self):
        assert parse_tool_call("The answer is 42.") is None

    def test_only_think_block(self):
        assert parse_tool_call("<think>reasoning only</think>") is None


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    def test_boxed(self):
        assert extract_answer(r"\boxed{42}") == "42"

    def test_boxed_with_text(self):
        assert extract_answer(r"Therefore the answer is \boxed{Paris}.") == "Paris"

    def test_double_backslash_boxed(self):
        assert extract_answer(r"\\boxed{42}") == "42"

    def test_final_answer(self):
        assert extract_answer("Final Answer: 42") == "42"

    def test_answer_colon(self):
        assert extract_answer("Answer: Paris") == "Paris"

    def test_the_answer_is(self):
        assert extract_answer("The answer is 42.") == "42."

    def test_strips_think_before_matching(self):
        text = "<think>Answer: fake</think>\nAnswer: real"
        assert extract_answer(text) == "real"

    def test_boxed_not_in_think(self):
        text = r"<think>\boxed{fake}</think>" + "\n" + r"\boxed{real}"
        assert extract_answer(text) == "real"

    def test_no_match_returns_none(self):
        assert extract_answer("Nothing here.") is None

    def test_empty_returns_none(self):
        assert extract_answer("") is None


# ---------------------------------------------------------------------------
# strip_thinking_tags
# ---------------------------------------------------------------------------

class TestStripThinkingTags:
    def test_basic(self):
        assert strip_thinking_tags("<think>reasoning</think>result") == "result"

    def test_multiline_think(self):
        text = "<think>\nline1\nline2\n</think>\nThe answer."
        assert strip_thinking_tags(text) == "The answer."

    def test_orphaned_close_tag(self):
        text = "reasoning without open tag\n</think>\nresult"
        assert strip_thinking_tags(text) == "result"

    def test_no_tags(self):
        assert strip_thinking_tags("plain text") == "plain text"

    def test_empty_string(self):
        assert strip_thinking_tags("") == ""

    def test_none_passthrough(self):
        assert strip_thinking_tags(None) is None

    def test_multiple_think_blocks(self):
        text = "<think>first</think>middle<think>second</think>end"
        assert strip_thinking_tags(text) == "middleend"
