"""Reasoning context utilities for tool sub-agents.

After the structured-memory refactor, only the attachment context helper
remains. Previous functions (get_accumulated_output_from_state,
extract_reasoning_context, get_reasoning_context_for_state) have been
removed — the orchestrator now builds compact memory prompts instead.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.state import ExecutionState


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
