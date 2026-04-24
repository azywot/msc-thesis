"""Parsing utilities for tool calls and answers.

This module provides utilities to parse tool calls from model outputs and
extract final answers from responses.

Supported tool-call formats:
  - Qwen3 / default: ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
  - OLMo 3 (Instruct + Think): ``<function_calls>\\ntool(arg=value)\\n</function_calls>``
    pythonic calls, newline-delimited for parallel invocations.  The parser
    accepts both Python literals (``True``/``False``/``None``) and JSON
    literals (``true``/``false``/``null``).  When multiple calls are emitted
    we return only the first; the orchestrator dispatches one tool per turn.
    Matches the format documented by ``--tool-call-parser olmo3`` in vLLM
    and the official allenai/Olmo-3-{7B,32B}-{Instruct,Think} model cards.
  - DeepSeek R1 (JSON_SINGLE): ``{"tool_call": {"name": ..., "arguments": {...}}}``
    single JSON object per turn, no XML wrapper; the parser also accepts
    code-fenced JSON and bare ``{"name": ..., "arguments": {...}}`` as fallbacks.
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

    Tries formats in order:

    1. **Qwen3 format** — ``<tool_call>…</tool_call>`` with a JSON payload.
       When multiple tags are present the last one wins.
    2. **OLMo 3 format** — ``<function_calls>…</function_calls>`` containing
       newline-delimited pythonic calls.  The first parseable call is returned.
    3. **DeepSeek JSON_SINGLE** — ``{"tool_call": {"name": ..., "arguments": {...}}}``.
       Thinking tags are stripped first to avoid matching hallucinated calls inside
       ``<think>`` blocks.  The first occurrence wins (single-call-per-turn contract).
    4. **Code-fenced JSON** — JSON payload inside any `` ``` `` fence.
    5. **Raw JSON** — bare ``{"name": ..., "arguments": {...}}`` object.

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

    # For all remaining fallbacks strip thinking tags first so that tool calls
    # hallucinated inside <think> blocks (DeepSeek pattern) are not matched.
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '</think>' in stripped:
        stripped = re.sub(r'^.*?</think>', '', stripped, flags=re.DOTALL)
    stripped = stripped.strip()

    # ── DeepSeek JSON_SINGLE: {"tool_call": {"name": ..., "arguments": {...}}} ──
    # Use raw_decode so nested braces in arguments are handled correctly.
    # Take the FIRST occurrence (per-turn single-call contract).
    _decoder = json.JSONDecoder()
    for tc_match in re.finditer(r'\{"tool_call"\s*:', stripped):
        try:
            obj, _ = _decoder.raw_decode(stripped, tc_match.start())
            tc = obj.get("tool_call")
            if isinstance(tc, dict) and "name" in tc:
                tc.setdefault("arguments", {})
                return tc
        except json.JSONDecodeError:
            continue

    # ── Fallback: JSON tool call in a fenced code block (```json / ```xml / ```) ──
    # DeepSeek and some other models wrap the JSON payload in a code fence instead
    # of <tool_call> tags.  Accept any fence language tag (or none).
    for fence_match in re.finditer(r'```(?:\w+)?\s*(\{.*?\})\s*```', stripped, re.DOTALL):
        try:
            tool_call = json.loads(fence_match.group(1).strip())
            if isinstance(tool_call, dict) and "name" in tool_call:
                tool_call.setdefault("arguments", {})
                return tool_call
        except json.JSONDecodeError:
            continue

    # ── Fallback: raw JSON tool call with no wrapping tags or fences ──────────
    # DeepSeek sometimes emits {"name": ..., "arguments": ...} as bare text.
    # Search for the last occurrence to prefer the most recent tool call intent.
    for raw_match in reversed(list(re.finditer(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{.*?\}[^{}]*\}', stripped, re.DOTALL))):
        try:
            tool_call = json.loads(raw_match.group(0).strip())
            if isinstance(tool_call, dict) and "name" in tool_call:
                tool_call.setdefault("arguments", {})
                return tool_call
        except json.JSONDecodeError:
            continue

    return None


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
    # REVERT (A/B isolation §2.1): match on the raw text, do NOT strip
    # <think>…</think> first, and drop the fenced-code-block fallback.
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
    # Normal case: both tags present (Qwen, DeepSeek, QwQ).
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # OLMo (and potentially others) emit <think> as a special token that vLLM
    # strips during decoding, leaving the thinking content as bare text followed
    # by an orphaned </think>.  Strip everything from the start of the text up
    # to (and including) the first </think>.
    if '</think>' in text:
        text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


