"""Reasoning context extraction for tool sub-agents.

Provides previous reasoning to web_search and code_generator sub-agents,
mirroring the multi-agent-tools (MAT) behavior. Context is truncated to
keep first step, last 4 steps, and tool-related steps.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, List, Optional, Sequence

from .parsing import strip_thinking_tags

if TYPE_CHECKING:
    from ..core.state import ExecutionState


# Markers for tool-related content (keep these steps when truncating).
# MAT uses BEGIN_SEARCH_QUERY/BEGIN_SEARCH_RESULT; msc-thesis uses tool_call/tool_response.
_DEFAULT_TOOL_MARKERS = ("<tool_call>", "<tool_response>")


def get_accumulated_output_from_state(state: "ExecutionState") -> str:
    """Build the accumulated output string from conversation messages.
    Concatenates all assistant and tool messages in order.
    Strips <think> tags from assistant content so sub-agent context does not include them.

    Args:
        state: Execution state with messages

    Returns:
        Concatenated assistant + tool content (no <think> blocks)
    """
    parts: List[str] = []
    for msg in state.messages or []:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("assistant", "tool") and content:
            if role == "assistant":
                content = strip_thinking_tags(content)
            if content.strip():
                parts.append(content)
    return "\n\n".join(parts)


def extract_reasoning_context(
    all_reasoning_steps: Sequence[str],
    mind_map=None,
    tool_markers: Optional[Sequence[str]] = None,
) -> str:
    """Extract a truncated reasoning context for tool sub-agents.

    Mirrors MAT's extract_reasoning_context. With mind_map, queries GraphRAG
    for a summary. Without mind_map, truncates to: first step, last 4 steps,
    and steps containing tool markers (e.g. <tool_call>, <tool_response>).

    Args:
        all_reasoning_steps: List of reasoning lines/steps (e.g. from
            get_accumulated_output_from_state split by newlines)
        mind_map: Optional GraphRAG/mind_map for summarization (not used if None)
        tool_markers: Markers that indicate tool-related steps to keep.
            Defaults to <tool_call> and <tool_response>.

    Returns:
        Truncated context string for sub-agent prompts
    """
    markers = tool_markers or _DEFAULT_TOOL_MARKERS

    if not all_reasoning_steps:
        return ""

    meaningful = [s for s in all_reasoning_steps if s and s.strip()]
    if not meaningful:
        return ""

    total_len = sum(len(s) for s in meaningful)
    if mind_map and total_len >= 100:
        try:
            summary = _query_mind_map(mind_map)
            if summary:
                return summary
        except Exception:
            pass

    # Non-mind_map: truncation
    truncated = ""
    for i, step in enumerate(meaningful):
        truncated += f"Step {i + 1}: {step}\n\n"

    prev_steps = truncated.split("\n\n")
    prev_steps = [s for s in prev_steps if s.strip()]

    if len(prev_steps) <= 5:
        return "\n\n".join(prev_steps)

    result_parts = []
    for i, step in enumerate(prev_steps):
        keep = (
            i == 0
            or i >= len(prev_steps) - 4
            or any(m in step for m in markers)
        )
        if keep:
            result_parts.append(step)
        else:
            if result_parts and result_parts[-1] != "...":
                result_parts.append("...")

    return "\n\n".join(result_parts).strip("\n")


def get_reasoning_context_for_state(
    state: "ExecutionState",
    mind_map=None,
) -> str:
    """Convenience: extract reasoning context from an ExecutionState.

    Args:
        state: Current execution state
        mind_map: Optional GraphRAG for summarization

    Returns:
        Truncated reasoning context string
    """
    output = get_accumulated_output_from_state(state)
    if not output.strip():
        return ""
    steps = output.replace("\n\n", "\n").split("\n")
    return extract_reasoning_context(steps, mind_map=mind_map)


def _query_mind_map(mind_map) -> Optional[str]:
    """Query mind_map/GraphRAG for reasoning summary. Returns None on failure."""
    try:
        from nano_graphrag import QueryParam
        result = mind_map.query(
            "Summarize the reasoning process, be short and clear. Keep the summary under 500 words.",
            param=QueryParam(mode="local"),
        )
        if result:
            s = str(result).strip()
            return s[:2000] + "..." if len(s) > 2000 else s
    except Exception:
        pass
    return None


def get_attachment_context_for_code(state: "ExecutionState") -> str:
    """Augment code context with attachment path and prior text_inspector output.

    MAT-style behavior:
    - Always provide [ATTACHED_FILE_PATH] when a text attachment exists (so code
      generator knows where to read the file, even if text_inspector wasn't called).
    - When text_inspector was called, append <FILE_CONTENT> with its output.

    Format matches MAT: [ATTACHED_FILE_PATH] {path} (space after bracket).

    Args:
        state: Execution state with messages and attachments

    Returns:
        Additional context string (empty if no text attachment)
    """
    if not state.attachments:
        return ""

    text_exts = {
        ".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml",
        ".jsonl", ".xml", ".log",
    }
    att_path = None
    for p in (state.attachments or []):
        if not p or not isinstance(p, str):
            continue
        clean = p.strip().split("?", 1)[0].split("#", 1)[0]
        if "." in clean and clean.rsplit(".", 1)[-1].lower() in text_exts:
            att_path = os.path.abspath(p)
            break

    if not att_path:
        return ""

    # MAT format: [ATTACHED_FILE_PATH] {path} (space, no newline)
    path_note = f"[ATTACHED_FILE_PATH] {att_path}"

    # MAT: only add FILE_CONTENT when text_inspector was called and we have content
    tool_calls = state.tool_calls or []
    ti_indices = [i for i, tc in enumerate(tool_calls) if tc.get("name") == "text_inspector"]
    if not ti_indices or not state.messages:
        return path_note

    tool_messages = [m for m in state.messages if m.get("role") == "tool"]
    last_ti_idx = ti_indices[-1]
    if last_ti_idx >= len(tool_messages):
        return path_note

    content = tool_messages[last_ti_idx].get("content", "")
    if not content:
        return path_note

    # MAT: avoid duplicating if already in context
    truncated = content[:4000] + "\n...[truncated]" if len(content) > 4000 else content
    file_block = f"<FILE_CONTENT>\n{truncated}\n</FILE_CONTENT>"
    return f"{path_note}\n\n{file_block}"
