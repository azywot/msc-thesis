"""Parsing utilities for Qwen3 tool calls and answers.

Matches multi-agent-tools/scripts/evaluate.py extract_answer() exactly.
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
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    tool_call_json = matches[-1].strip()  # Take the last tool call if multiple

    try:
        tool_call = json.loads(tool_call_json)
        if isinstance(tool_call, dict) and "name" in tool_call:
            if "arguments" not in tool_call:
                tool_call["arguments"] = {}
            return tool_call
    except json.JSONDecodeError:
        return None

    return None


def extract_answer(output: str, mode: str = 'gen') -> str:
    """Extract final answer from model output.

    Exactly matches multi-agent-tools/scripts/evaluate.py extract_answer().

    Args:
        output: Full accumulated model output (all turns + tool responses)
        mode: Extraction mode: 'gen' (default), 'choose' (MC), 'qa' (open QA)

    Returns:
        Extracted answer string (empty string if not found)
    """
    extracted_text = ''

    # Pre-processing: strip thinking blocks.
    # Matches old: output.split('</think>')[-1]
    think_end_pattern = r'</think>'
    if think_end_pattern in output:
        output = output.split(think_end_pattern)[-1]

    # Try \boxed{...} first (greedy, last match — mirrors old)
    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, output)
    if matches:
        extracted_text = matches[-1]  # Take the last match
        if mode in ['choose', 'qa']:
            # Strip \text{...} wrapper inside boxed content
            inner_pattern = r'\\text\{(.*)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]
            extracted_text = extracted_text.strip("()")
    else:
        # Fallback: first non-empty line not starting with a known prefix
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Assistant:', 'Answer:', 'Final Answer:', '**')):
                extracted_text = line.rstrip('.,;:!?')
                break
        # If still empty, try splitting on "Assistant:" or take last chunk
        if not extracted_text:
            remainder = output.strip()
            if remainder:
                for prefix in ['Assistant:', 'Answer:', 'Final Answer:']:
                    if remainder.startswith(prefix):
                        remainder = remainder[len(prefix):].strip()
                        break
                first_line = remainder.split('\n')[0].strip()
                if first_line:
                    extracted_text = first_line.rstrip('.,;:!?')
                elif remainder:
                    extracted_text = remainder[-100:].strip().rstrip('.,;:!?')

    print(f">>> Extracted text: {extracted_text}")
    return extracted_text


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
