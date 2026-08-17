"""Build the prefix_mask that marks replayed teacher tokens in a training batch.

Kept free of verl imports so it stays unit-testable: verl is absent from the
agent_engine env and pytest is absent from cosmas-train, so any module importing
verl cannot be exercised by a test in either environment.
"""

from __future__ import annotations


def build_prefix_mask(trace_list, max_response_length):
    """One right-padded 0/1 row per kept turn, 1 on replayed tokens.

    Mirrors the truncation and skip rules the vendored daemon applies to
    responses (``fine_tuning/agentflow/verl/daemon.py:740-772``) so the mask stays
    aligned with ``responses`` row for row. In particular a turn is dropped only
    when prompt *and* response are both empty, which is the base daemon's rule;
    dropping on a different condition would shift every later row's mask onto the
    wrong response.
    """
    rows = []
    for trace in trace_list:
        response_ids = trace.get("response_ids", [])
        prompt_ids = trace.get("prompt_ids", [])
        if len(prompt_ids) == 0 and len(response_ids) == 0:
            continue
        length = min(len(response_ids), max_response_length)
        fill = 1 if trace.get("is_prefix", False) else 0
        rows.append([fill] * length + [0] * (max_response_length - length))
    return rows
