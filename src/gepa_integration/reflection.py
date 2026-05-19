"""Reflection prompt trimming for GEPA's reflective mutation loop.

The reflector LM has a hard context limit (e.g., 40K tokens for Qwen3-32B).
GEPA's reflection prompts include a minibatch of rollout traces that can
exceed this limit on long benchmarks. This module trims the prompt by
dropping interior examples from the minibatch while preserving the head
(task framing + current candidate) and tail (output directive).

Prompt structure (from gepa/strategies/instruction_proposal.py default template):
    HEAD: intro + ``` <curr_instructions> ```
    BODY: '# Example 1' ... '# Example N' (markdown h1 boundaries)
    TAIL: closing ``` + 'Your task is to write a new instruction...'
"""
from __future__ import annotations

import re
from typing import Callable, Optional


def count_tokens_char_heuristic(text: str) -> int:
    """~4 chars/token estimate when no tokenizer is available."""
    return len(text) // 4


def make_token_counter(tokenizer) -> Callable[[str], int]:
    """Build a token-counter closure from an HF tokenizer, or fall back to chars."""
    if tokenizer is None:
        return count_tokens_char_heuristic

    def _count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return _count


def drop_middle_examples(
    prompt: str,
    budget_tokens: int,
    count_tokens: Callable[[str], int],
) -> Optional[str]:
    """Drop interior examples from a GEPA reflection prompt until it fits.

    Keeps the first ceil(k/2) and last floor(k/2) examples, where k is the
    largest number that still fits within budget_tokens. Inserts a marker
    indicating how many were omitted.

    Returns None if:
      - Fewer than 2 examples are present (nothing to drop).
      - No closing wrapper ``` is found.
      - Even keeping a single example exceeds budget_tokens.
    """
    ex_matches = list(re.finditer(r"^# Example \d+", prompt, re.MULTILINE))
    if len(ex_matches) < 2:
        return None
    # The wrapper close is the LAST line-only ``` in the prompt. The default
    # template's tail uses ``` only inline (in "within ``` blocks."), so this
    # is robust even if an example body contains ``` code fences.
    all_closes = list(re.finditer(r"^```\s*$", prompt, re.MULTILINE))
    if not all_closes:
        return None
    body_end = all_closes[-1].start()
    if body_end < ex_matches[-1].start():
        return None
    head = prompt[: ex_matches[0].start()]
    tail = prompt[body_end:]
    examples = []
    for i, m in enumerate(ex_matches):
        s = m.start()
        e = ex_matches[i + 1].start() if i + 1 < len(ex_matches) else body_end
        examples.append(prompt[s:e])
    n = len(examples)
    for k in range(n - 1, 0, -1):
        h = (k + 1) // 2
        t = k - h
        head_ex = examples[:h]
        tail_ex = examples[-t:] if t > 0 else []
        marker = (
            f"\n\n[NOTE: {n - k} of {n} examples omitted to fit reflector context.]\n\n"
        )
        candidate = head + "".join(head_ex) + marker + "".join(tail_ex) + tail
        if count_tokens(candidate) <= budget_tokens:
            return candidate
    return None


def raw_middle_truncate(
    prompt: str,
    budget_tokens: int,
    tokenizer=None,
) -> str:
    """Last-resort middle truncation: drop the middle of the raw prompt."""
    if tokenizer is None:
        max_chars = budget_tokens * 4
        keep = max(max_chars - 200, 0)
        return (
            prompt[: keep // 2]
            + "\n\n[...TRIMMED FOR CONTEXT LIMIT...]\n\n"
            + prompt[-(keep // 2):]
        )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    keep = max(budget_tokens - 64, 0)
    head = tokenizer.decode(ids[: keep // 2], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-(keep // 2):], skip_special_tokens=True)
    return head + "\n\n[...TRIMMED FOR CONTEXT LIMIT...]\n\n" + tail


def trim_prompt(
    prompt: str,
    budget_tokens: int,
    tokenizer=None,
    on_trim: Optional[Callable[[str], None]] = None,
) -> str:
    """Trim a reflection prompt to fit within budget_tokens.

    Strategy:
      1. Under budget → return unchanged.
      2. Drop interior examples via `drop_middle_examples`.
      3. Fallback: raw middle-truncate (structurally lossy but never rejected).

    `on_trim` is invoked with a short status string when trimming occurs.
    """
    count = make_token_counter(tokenizer)
    if count(prompt) <= budget_tokens:
        return prompt
    trimmed = drop_middle_examples(prompt, budget_tokens, count)
    if trimmed is not None:
        if on_trim is not None:
            on_trim(f"reflector trim: dropped interior examples, ~{count(trimmed)} tokens")
        return trimmed
    result = raw_middle_truncate(prompt, budget_tokens, tokenizer)
    if on_trim is not None:
        on_trim(
            "reflector trim: structure unparseable or single example too big; "
            "raw middle-truncate"
        )
    return result
