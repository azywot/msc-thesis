"""Parsing utilities for tool calls and answers.

This module provides utilities to parse tool calls from model outputs and
extract final answers from responses.  It supports multiple tool-call formats
used by different model families:

1. Native DeepSeek special tokens — DeepSeek-R1-Distill / DeepSeek-R1-0528
2. ``<tool_call>…</tool_call>`` — Qwen3 / Qwen2.5 / QwQ
3. ``<|tool_call|>…<|/tool_call|>`` — Phi-4-mini
4. Markdown code blocks (`` ```json … ``` ``) — fallback
5. Bare JSON ``{"name": …, "arguments": …}`` — last-resort fallback
"""

import json
import re
from typing import Any, Dict, Optional


# ── Compiled regexes for tool-call detection ──────────────────────────────

# Format 1: Native DeepSeek tokens
# Matches: <｜tool▁call▁begin｜>{type}<｜tool▁sep｜>{name}\n```json\n{body}\n```
# Name is in group(1), JSON body (args only) is in group(2).
_RE_DEEPSEEK_NATIVE = re.compile(
    r'<｜tool▁call▁begin｜>\w+<｜tool▁sep｜>(\S+)\s*\n```json\s*\n(.*?)\n?```',
    re.DOTALL,
)

# Format 2: Qwen3-style XML tags
_RE_QWEN_TAG = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

# Format 3: Phi-4 pipe-style tags
_RE_PHI4_TAG = re.compile(r'<\|tool_call\|>(.*?)<\|/tool_call\|>', re.DOTALL)

# Format 4: Markdown code blocks
_RE_CODE_BLOCK = re.compile(r'```(?:json)?\s*\n?(.*?)```', re.DOTALL)

# Format 5: Bare JSON — match {"name": "...", ...} not inside tags/blocks
_RE_BARE_JSON = re.compile(
    r'\{["\']name["\']\s*:\s*["\'][^"\']+["\']\s*,\s*["\']arguments["\']\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)

# Matches the *opening* of any tool-call format (used by orchestrator to split).
TOOL_CALL_START = re.compile(
    r'<｜tool▁call▁begin｜>'
    r'|<tool_call>'
    r'|<\|tool_call\|>'
    r'|```(?:json)?\s*\n?\s*\{["\']name["\']',
    re.DOTALL,
)

# ── Stripping regexes (for strip_tool_calls) ─────────────────────────────

_STRIP_PATTERNS = [
    re.compile(r'<｜tool▁calls▁begin｜>.*?<｜tool▁calls▁end｜>', re.DOTALL),
    re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL),
    re.compile(r'<\|tool_call\|>.*?<\|/tool_call\|>', re.DOTALL),
    re.compile(r'```(?:json)?\s*\n?\s*\{["\']name["\'].*?```', re.DOTALL),
]


def _try_parse_tool_json(raw: str) -> Optional[Dict[str, Any]]:
    """Try to parse a JSON string as a tool call dict with 'name' key."""
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "name" in obj:
            if "arguments" not in obj:
                obj["arguments"] = {}
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _find_bare_json_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Scan for a bare JSON tool call not wrapped in any tags or code blocks."""
    for m in reversed(list(_RE_BARE_JSON.finditer(text))):
        result = _try_parse_tool_json(m.group(0))
        if result is not None:
            return result
    return None


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the last tool call from model output.

    Tries the following formats in priority order:

    1. Native DeepSeek tokens ``<｜tool▁call▁begin｜>…<｜tool▁call▁end｜>``.
       Name is extracted from the ``<｜tool▁sep｜>`` header; arguments from the
       JSON body (which may be args-only or the full ``{"name": …}`` wrapper).
    2. ``<tool_call>…</tool_call>`` XML tags (Qwen3 / Qwen2.5 / QwQ).
    3. ``<|tool_call|>…<|/tool_call|>`` pipe-style tags (Phi-4-mini).
    4. Markdown JSON code blocks (`` ```json … ``` ``) — fallback.
    5. Bare JSON ``{"name": …, "arguments": …}`` — last-resort fallback.

    When multiple matches exist within a format, the last one is used.

    Args:
        text: Raw model output text.

    Returns:
        Dict with ``"name"`` and ``"arguments"`` keys, or ``None`` if no valid
        tool call was found.
    """
    # 1. Native DeepSeek tokens (highest priority for DeepSeek models)
    ds_matches = _RE_DEEPSEEK_NATIVE.findall(text)
    if ds_matches:
        name_header, body = ds_matches[-1]
        try:
            parsed_body = json.loads(body.strip())
            if isinstance(parsed_body, dict):
                if "name" in parsed_body:
                    # Model produced the full {"name": …, "arguments": …} wrapper
                    result = _try_parse_tool_json(body)
                    if result is not None:
                        return result
                else:
                    # Native args-only JSON body — take name from header
                    return {"name": name_header.strip(), "arguments": parsed_body}
        except json.JSONDecodeError:
            pass

    # 2. Canonical <tool_call> tags (Qwen3 family)
    tag_matches = _RE_QWEN_TAG.findall(text)
    if tag_matches:
        result = _try_parse_tool_json(tag_matches[-1])
        if result is not None:
            return result

    # 3. Phi-4 pipe-style tags
    phi_matches = _RE_PHI4_TAG.findall(text)
    if phi_matches:
        result = _try_parse_tool_json(phi_matches[-1])
        if result is not None:
            return result

    # 4. Markdown code blocks (```json ... ``` or ``` ... ```)
    code_matches = _RE_CODE_BLOCK.findall(text)
    for block in reversed(code_matches):
        result = _try_parse_tool_json(block)
        if result is not None:
            return result

    # 5. Bare JSON — last resort
    result = _find_bare_json_tool_call(text)
    if result is not None:
        return result

    return None


def strip_tool_calls(text: str) -> str:
    """Remove tool call blocks in ALL recognised formats from *text*.

    Handles Qwen3 XML tags, Phi-4 pipe tags, and markdown code blocks that
    contain a tool-call JSON (i.e. a dict with a ``"name"`` key).

    Args:
        text: Text potentially containing tool call blocks.

    Returns:
        Text with tool call blocks removed and extra whitespace cleaned up.
    """
    if not text:
        return text
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from model output.

    Tries the following patterns in order of priority:

    1. ``\\boxed{answer}`` — LaTeX format used by math reasoning models.
    2. ``Final Answer: <answer>``
    3. ``Answer: <answer>``
    4. ``The answer is <answer>``

    Args:
        text: Model output text.

    Returns:
        Extracted answer string, or ``None`` if no pattern matched.
    """
    boxed_pattern = r'\\+boxed\{([^}]+)\}'
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern1 = r'Final Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern2 = r'(?<!Final )Answer:\s*(.+?)(?:\n|$)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    pattern3 = r'The answer is[:\s]+(.+?)(?:\n|$)'
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

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
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


