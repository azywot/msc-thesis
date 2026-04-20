"""Parsing utilities for tool calls and answers.

This module provides utilities to parse tool calls from model outputs and
extract final answers from responses.

Supported tool-call formats:
  - Qwen3/baseline: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
  - OLMo 3: ``<function_calls>\\ntool_name(arg=value)\\n</function_calls>``
    (pythonic; JSON boolean/null literals allowed alongside Python ones)
"""

import ast
import json
import re
from typing import Any, Dict, Optional


def _parse_pythonic_call(line: str) -> Optional[Dict[str, Any]]:
    """Parse one pythonic function-call line emitted by OLMo 3.

    Handles JSON boolean/null literals (``true``, ``false``, ``null``) in
    addition to the Python equivalents (``True``, ``False``, ``None``).

    Args:
        line: A single line such as ``web_search(query="foo")``.

    Returns:
        Dict with ``"name"`` and ``"arguments"`` keys, or ``None`` on failure.
    """
    # Normalise JSON boolean/null → Python literals before parsing.
    line = re.sub(r'\btrue\b', 'True', line)
    line = re.sub(r'\bfalse\b', 'False', line)
    line = re.sub(r'\bnull\b', 'None', line)
    try:
        tree = ast.parse(line.strip(), mode='eval')
    except SyntaxError:
        return None
    if not isinstance(tree.body, ast.Call):
        return None
    call = tree.body
    if not isinstance(call.func, ast.Name):
        return None
    try:
        arguments = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    except (ValueError, TypeError):
        return None
    return {"name": call.func.id, "arguments": arguments}


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first/last tool call from model output.

    Tries two formats in order:

    1. **Qwen3 format** — ``<tool_call>…</tool_call>`` with a JSON payload.
       When multiple tags are present the last one wins.
    2. **OLMo 3 format** — ``<function_calls>…</function_calls>`` containing
       newline-delimited pythonic calls.  The first parseable call is returned.

    Args:
        text: Raw model output text.

    Returns:
        Dict with ``"name"`` and ``"arguments"`` keys, or ``None`` if no valid
        tool call was found.
    """
    # ── Qwen3 / default: <tool_call>JSON</tool_call> ──────────────────────────
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if matches:
        try:
            tool_call = json.loads(matches[-1].strip())
            if isinstance(tool_call, dict) and "name" in tool_call:
                tool_call.setdefault("arguments", {})
                return tool_call
        except json.JSONDecodeError:
            pass

    # ── OLMo 3: <function_calls>pythonic calls</function_calls> ──────────────
    fc_match = re.search(r'<function_calls>(.*?)</function_calls>', text, re.DOTALL)
    if fc_match:
        for line in fc_match.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            result = _parse_pythonic_call(line)
            if result:
                return result

    # ── Fallback: JSON tool call in a fenced code block (```json / ```xml / ```) ──
    # DeepSeek and some other models wrap the JSON payload in a code fence instead
    # of <tool_call> tags.  Accept any fence language tag (or none).
    for fence_match in re.finditer(r'```(?:\w+)?\s*(\{.*?\})\s*```', text, re.DOTALL):
        try:
            tool_call = json.loads(fence_match.group(1).strip())
            if isinstance(tool_call, dict) and "name" in tool_call:
                tool_call.setdefault("arguments", {})
                return tool_call
        except json.JSONDecodeError:
            continue

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
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n?```', stripped, re.DOTALL)
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


