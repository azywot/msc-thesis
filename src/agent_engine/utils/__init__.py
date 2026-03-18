"""Utility functions for agent_engine.

This module provides utilities for parsing, logging, reproducibility,
and reasoning context extraction for tool sub-agents.
"""

from .parsing import parse_tool_call, extract_answer, strip_thinking_tags, strip_tool_calls, TOOL_CALL_START
from .logging import setup_logging, get_logger
from .seed import set_seed, get_seed_from_env
from .reasoning_context import get_attachment_context_for_code

__all__ = [
    "parse_tool_call",
    "extract_answer",
    "strip_thinking_tags",
    "strip_tool_calls",
    "TOOL_CALL_START",
    "setup_logging",
    "get_logger",
    "set_seed",
    "get_seed_from_env",
    "get_attachment_context_for_code",
]
