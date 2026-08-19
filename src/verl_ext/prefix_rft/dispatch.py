"""Decide how many teacher decisions each rollout replays.

The driver owns this because it owns ``global_step``, which the cosine schedule
needs. The result travels to the rollout worker in the task payload as
``prefix_k``.

Kept free of verl imports so it stays unit-testable: verl is absent from the
agent_engine env and pytest is absent from cosmas-train.
"""

from __future__ import annotations


def prefix_k_for(
    sample,
    rollout_index,
    is_train,
    schedule,
    demo_store,
    n_prefixed_rollouts,
    global_step,
):
    """Number of teacher decisions rollout ``rollout_index`` should replay.

    Paper A.2: "we sample 8 rollouts per prompt, and one of them starts with the
    sampled prefix", so only the first ``n_prefixed_rollouts`` of each group are
    hybrid. Validation never uses a prefix, so checkpoint selection measures the
    unaided policy.

    Returns 0, meaning ordinary GRPO, whenever there is no demonstration for the
    question or the demonstration has a single decision (a prefix must leave at
    least one on-policy decision to score).
    """
    if not is_train:
        return 0
    if schedule is None or demo_store is None:
        return 0
    if rollout_index >= n_prefixed_rollouts:
        return 0

    question = str((sample or {}).get("question", ""))
    if not question:
        return 0

    # Keyed on the question text, not extra_info.idx: prepare.py assigns idx per
    # data source, so it collides across them and would fetch the wrong
    # demonstration entirely.
    m = demo_store.n_steps(question)
    if m <= 1:
        return 0
    return schedule.sample_k(m, global_step=global_step)
