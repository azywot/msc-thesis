import re

from gepa_integration.reflection import (
    count_tokens_char_heuristic,
    drop_middle_examples,
    make_token_counter,
    raw_middle_truncate,
    trim_prompt,
)


class CharTokenizer:
    """Test tokenizer where one character is one token. Lets tests use char-length budgets directly."""

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


def _make_prompt(n_examples: int, ex_size_chars: int) -> str:
    """Build a synthetic GEPA-formatted reflection prompt."""
    head = (
        "I provided an assistant with the following instructions to perform a task for me:\n"
        "```\n"
        "You are a helpful agent.\n"
        "```\n\n"
        "The following are examples of different task inputs:\n"
        "```\n"
    )
    body = ""
    for i in range(1, n_examples + 1):
        body += (
            f"# Example {i}\n"
            f"## input\n"
            f"### question\n"
            f"{'x' * ex_size_chars}\n\n"
            f"## output\n"
            f"### answer\n"
            f"{'y' * ex_size_chars}\n\n"
        )
    tail = (
        "```\n\n"
        "Your task is to write a new instruction for the assistant.\n\n"
        "Provide the new instructions within ``` blocks."
    )
    return head + body + tail


def _char_count(text: str) -> int:
    return len(text)


# ── count_tokens_char_heuristic / make_token_counter ─────────────────────────


def test_char_heuristic_quarter_of_length():
    assert count_tokens_char_heuristic("a" * 40) == 10


def test_make_token_counter_no_tokenizer_uses_heuristic():
    counter = make_token_counter(None)
    assert counter("a" * 40) == 10


def test_make_token_counter_uses_tokenizer():
    counter = make_token_counter(CharTokenizer())
    assert counter("hello") == 5


# ── drop_middle_examples ─────────────────────────────────────────────────────


def test_drop_returns_none_when_fewer_than_two_examples():
    p = "head\n# Example 1\nbody\n```\ntail"
    assert drop_middle_examples(p, 100, _char_count) is None


def test_drop_returns_none_when_no_wrapper_close():
    p = "head\n# Example 1\nx\n# Example 2\ny\nno close at line start"
    assert drop_middle_examples(p, 100, _char_count) is None


def test_drop_preserves_head_and_tail():
    prompt = _make_prompt(10, 200)
    result = drop_middle_examples(prompt, budget_tokens=1500, count_tokens=_char_count)
    assert result is not None
    assert "You are a helpful agent" in result
    assert "Your task is to write a new instruction" in result
    assert "Provide the new instructions within" in result


def test_drop_inserts_omission_marker():
    prompt = _make_prompt(10, 200)
    result = drop_middle_examples(prompt, budget_tokens=1500, count_tokens=_char_count)
    assert result is not None
    assert "examples omitted" in result.lower()


def test_drop_keeps_first_and_last_examples():
    prompt = _make_prompt(10, 200)
    result = drop_middle_examples(prompt, budget_tokens=1500, count_tokens=_char_count)
    assert result is not None
    assert re.search(r"^# Example 1\b", result, re.MULTILINE)
    assert re.search(r"^# Example 10\b", result, re.MULTILINE)


def test_drop_tighter_budget_keeps_fewer_examples():
    prompt = _make_prompt(10, 200)
    loose = drop_middle_examples(prompt, budget_tokens=4000, count_tokens=_char_count)
    tight = drop_middle_examples(prompt, budget_tokens=2000, count_tokens=_char_count)
    assert loose is not None and tight is not None
    n_loose = len(re.findall(r"^# Example \d+", loose, re.MULTILINE))
    n_tight = len(re.findall(r"^# Example \d+", tight, re.MULTILINE))
    assert n_tight < n_loose


def test_drop_returns_none_when_single_example_overflows():
    prompt = _make_prompt(3, 5000)
    assert drop_middle_examples(prompt, budget_tokens=1000, count_tokens=_char_count) is None


def test_drop_handles_embedded_code_fences_in_examples():
    """If an example body contains ```...``` fences, the wrapper close must
    still be located correctly (last line-only ``` in the prompt) — not
    confused by the inner fences."""
    head = (
        "I provided an assistant with the following instructions:\n"
        "```\nYou are a helpful agent.\n```\n\n"
        "The following are examples:\n```\n"
    )
    body = ""
    code_block = "code_line\n" * 100
    for i in range(1, 4):
        body += (
            f"# Example {i}\n## output\n```\n"
            + code_block
            + "```\n\n"
        )
    tail = "```\n\nYour task is to write a new instruction.\nProvide within ``` blocks."
    prompt = head + body + tail
    # Budget tight enough to force dropping at least one example, loose enough
    # that the marker fits.
    result = drop_middle_examples(
        prompt, budget_tokens=len(prompt) - 600, count_tokens=_char_count
    )
    assert result is not None
    assert "You are a helpful agent" in result  # head preserved
    assert "Your task is to write a new instruction" in result  # tail preserved
    # First and last examples should always be kept.
    assert re.search(r"^# Example 1\b", result, re.MULTILINE)
    assert re.search(r"^# Example 3\b", result, re.MULTILINE)
    # Wrapper close (tail) should be intact — verify by checking the tail string.
    assert result.rstrip().endswith("Provide within ``` blocks.")


# ── raw_middle_truncate ──────────────────────────────────────────────────────


def test_raw_truncate_no_tokenizer_marks_and_shrinks():
    p = "a" * 5000
    result = raw_middle_truncate(p, budget_tokens=100, tokenizer=None)
    assert "TRIMMED FOR CONTEXT LIMIT" in result
    assert len(result) < len(p)


def test_raw_truncate_preserves_outermost_content():
    p = "START_MARKER_HEAD" + "x" * 5000 + "END_MARKER_TAIL"
    result = raw_middle_truncate(p, budget_tokens=200, tokenizer=None)
    assert "START_MARKER_HEAD" in result
    assert "END_MARKER_TAIL" in result


def test_raw_truncate_with_tokenizer_path():
    tok = CharTokenizer()
    p = "HEAD_" + "x" * 5000 + "_TAIL"
    result = raw_middle_truncate(p, budget_tokens=200, tokenizer=tok)
    assert "TRIMMED FOR CONTEXT LIMIT" in result
    assert "HEAD_" in result
    assert "_TAIL" in result


# ── trim_prompt (top-level integration) ──────────────────────────────────────


def test_trim_under_budget_returns_unchanged():
    prompt = _make_prompt(2, 50)
    tok = CharTokenizer()
    assert trim_prompt(prompt, budget_tokens=len(prompt) + 100, tokenizer=tok) == prompt


def test_trim_uses_example_drop_when_parseable():
    prompt = _make_prompt(10, 200)
    tok = CharTokenizer()
    result = trim_prompt(prompt, budget_tokens=2000, tokenizer=tok)
    assert result != prompt
    assert "TRIMMED FOR CONTEXT LIMIT" not in result  # not the fallback path
    assert "examples omitted" in result.lower()


def test_trim_falls_back_to_raw_when_unparseable():
    prompt = "x" * 10000  # no # Example markers
    tok = CharTokenizer()
    result = trim_prompt(prompt, budget_tokens=200, tokenizer=tok)
    assert "TRIMMED FOR CONTEXT LIMIT" in result


def test_trim_callback_fires_on_trim():
    prompt = _make_prompt(10, 200)
    tok = CharTokenizer()
    log: list[str] = []
    trim_prompt(prompt, budget_tokens=2000, tokenizer=tok, on_trim=log.append)
    assert len(log) == 1
    assert "trim" in log[0].lower()


def test_trim_callback_not_called_when_under_budget():
    prompt = _make_prompt(2, 50)
    tok = CharTokenizer()
    log: list[str] = []
    trim_prompt(prompt, budget_tokens=len(prompt) + 100, tokenizer=tok, on_trim=log.append)
    assert log == []
