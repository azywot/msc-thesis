"""Parsing utilities for Qwen3 tool calls and answers.

This module provides utilities to parse tool calls from model outputs and
extract final answers from responses.
"""

import json
import re
from typing import Any, Dict, Optional


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the last tool call from model output.

    Extracts JSON from ``<tool_call>…</tool_call>`` tags.  When multiple tags
    are present (shouldn't happen in practice) the last one is used.

    Args:
        text: Raw model output text.

    Returns:
        Dict with ``"name"`` and ``"arguments"`` keys, or ``None`` if no valid
        tool call was found.
    """
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    tool_call_json = matches[-1].strip()

    try:
        tool_call = json.loads(tool_call_json)
        if isinstance(tool_call, dict) and "name" in tool_call:
            if "arguments" not in tool_call:
                tool_call["arguments"] = {}
            return tool_call
    except json.JSONDecodeError:
        return None

    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from model output.

    Strips ``<think>…</think>`` blocks first so that reasoning summaries inside
    thinking do not accidentally match answer patterns.

    Tries the following patterns in order of priority:

    1. ``\\boxed{answer}`` — LaTeX format used by math reasoning models.
    2. ``Final Answer: <answer>``
    3. ``Answer: <answer>``
    4. ``The answer is <answer>``
    5. Fenced code block (``` python … ```) — fallback for code-generation tasks.

    Args:
        text: Model output text.

    Returns:
        Extracted answer string, or ``None`` if no pattern matched.
    """
    # Strip thinking tags before any pattern matching so that reasoning
    # summaries inside <think> don't shadow the real answer.
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    boxed_pattern = r'\\+boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern1 = r'Final Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern1, stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern2 = r'(?<!Final )Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern2, stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern3 = r'The answer is[:\s]+(.+?)(?:\n|$)'
    match = re.search(pattern3, stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: fenced code block (for code-generation tasks like BigCodeBench).
    # Use the LAST match so that a final implementation block is preferred over
    # any example/intermediate snippets earlier in the response.
    code_blocks = re.findall(r'```(?:python)?\n?(.*?)\n?```', stripped, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    return None


def strip_thinking_tags(text: str) -> str:
    """Remove thinking tags from text.

    Models with thinking mode output <think>...</think> tags.
    Use this for any LLM output returned to the orchestrator so the orchestrator
    never sees thinking content. Safe to call with None or empty string.

    Args:
        text: Text potentially containing thinking tags (or None/empty)

    Returns:
        Text with thinking tags removed
    """
    if not text:
        return text
    # Normal case: both tags present (Qwen, DeepSeek, QwQ).
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # OLMo (and potentially others) emit <think> as a special token that vLLM
    # strips during decoding, leaving the thinking content as bare text followed
    # by an orphaned </think>.  Strip everything from the start of the text up
    # to (and including) the first </think>.
    if '</think>' in text:
        text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


