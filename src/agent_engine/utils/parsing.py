"""Parsing utilities for Qwen3 tool calls and answers.

This module provides utilities to parse tool calls from model outputs and
extract final answers from responses.
"""

import json
import re
from typing import Any, Dict, Optional


def parse_qwen3_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse Qwen3 tool call from model output.

    Qwen3 tool calls are in the format:
    <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

    Args:
        text: Model output text

    Returns:
        Dictionary with 'name' and 'arguments' keys, or None if no tool call found
    """
    # Look for <tool_call>...</tool_call> pattern
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    # Parse the JSON inside the tool_call tags
    tool_call_json = matches[-1].strip()  # Take the last tool call if multiple

    try:
        tool_call = json.loads(tool_call_json)
        if isinstance(tool_call, dict) and "name" in tool_call:
            # Ensure arguments exist
            if "arguments" not in tool_call:
                tool_call["arguments"] = {}
            return tool_call
    except json.JSONDecodeError:
        # Invalid JSON in tool call
        return None

    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from model output.

    Looks for patterns like:
    - "\\boxed{answer}" (LaTeX format from multi-agent-tools)
    - "Final Answer: <answer>"
    - "Answer: <answer>"
    - "The answer is <answer>"

    Args:
        text: Model output text

    Returns:
        Extracted answer string or None if not found
    """
    # Pattern 0: LaTeX boxed format (priority - matches multi-agent-tools)
    # Look for \boxed{answer} or \\boxed{answer}
    boxed_pattern = r'\\+boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 1: "Final Answer: <answer>"
    pattern1 = r'Final Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: "Answer: <answer>"
    pattern2 = r'(?<!Final )Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 3: "The answer is <answer>"
    pattern3 = r'The answer is[:\s]+(.+?)(?:\n|$)'
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return None


def strip_thinking_tags(text: str) -> str:
    """Remove thinking tags from text.

    Models with thinking mode output <think>...</think> tags.
    This function removes those tags from sub-agent LLM outputs.

    Args:
        text: Text potentially containing thinking tags

    Returns:
        Text with thinking tags removed
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def subagent_output_for_orchestrator(text: str) -> str:
    """Prepare sub-agent LLM output for the orchestrator: strip thinking tags.

    Any sub-agent (web_search, code_generator, text_inspector, image_inspector)
    that returns LLM-generated text to the orchestrator must use this so the
    planner never sees <think>...</think> content. Always strips; safe to call even
    when the model did not use thinking mode.

    Args:
        text: Raw sub-agent LLM output

    Returns:
        Text with thinking tags removed, suitable for tool_response to orchestrator
    """
    if not text:
        return text
    return strip_thinking_tags(text)


