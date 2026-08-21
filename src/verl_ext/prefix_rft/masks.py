"""Build the prefix_mask that marks replayed teacher tokens in a training batch.

Kept free of verl imports so it stays unit-testable: verl is absent from the
agent_engine env and pytest is absent from cosmas-train, so any module importing
verl cannot be exercised by a test in either environment.
"""

from __future__ import annotations


def build_prefix_mask(trace_list, max_response_length):
    """One right-padded 0/1 row per kept turn, 1 on replayed teacher tokens.

    ``prefix_len`` is how many leading response tokens were the teacher's. Token mode
    splits a turn, so a row can be 1 on its head and 0 after. Step mode replays whole
    turns and writes ``is_prefix`` only; a prefixed turn with no ``prefix_len`` is read
    as wholly replayed, which reproduces the mask exactly as it was.

    ``prefix_len = 0`` is indistinguishable from the key being absent, so it cannot
    also mean "no tokens". A turn that replayed nothing carries ``is_prefix = False``.

    Mirrors the truncation and skip rules the vendored daemon applies to responses
    (``fine_tuning/agentflow/verl/daemon.py:740-772``) so the mask stays aligned with
    ``responses`` row for row. In particular a turn is dropped only when prompt *and*
    response are both empty, which is the base daemon's rule; dropping on a different
    condition would shift every later row's mask onto the wrong response.
    """
    rows = []
    for trace in trace_list:
        response_ids = trace.get("response_ids", [])
        prompt_ids = trace.get("prompt_ids", [])
        if len(prompt_ids) == 0 and len(response_ids) == 0:
            continue
        length = min(len(response_ids), max_response_length)
        n_prefix = int(trace.get("prefix_len", 0) or 0)
        if n_prefix == 0 and trace.get("is_prefix", False):
            n_prefix = length
        n_prefix = max(0, min(n_prefix, length))
        rows.append([1] * n_prefix + [0] * (max_response_length - n_prefix))
    return rows
