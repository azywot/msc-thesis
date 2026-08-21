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


def prefix_l_for(
    sample,
    rollout_index,
    is_train,
    schedule,
    demo_store,
    n_prefixed_rollouts,
    global_step,
):
    """Prefix fraction for rollout ``rollout_index``, or 0.0 for no prefix.

    Token mode's analogue of ``prefix_k_for``. The driver ships the raw ``l`` and the
    worker turns it into a token budget, because the worker is the process with a
    tokenizer and it would otherwise need the per-decision token counts precomputed
    into the demonstration store.

    There is deliberately no ``m > 1`` guard. Step mode needs one because ``k <= m - 1``
    leaves a single-decision demonstration unprefixable. The token budget's own
    ``B <= T - 1`` clamp already leaves a token to generate, so those 273 questions are
    prefixable here.
    """
    if not is_train:
        return 0.0
    if schedule is None or demo_store is None:
        return 0.0
    if rollout_index >= n_prefixed_rollouts:
        return 0.0

    question = str((sample or {}).get("question", ""))
    if not question:
        return 0.0
    if demo_store.n_steps(question) < 1:
        return 0.0

    l, _, _ = schedule.sample_l(global_step=global_step)
    return float(l)


def prefix_spec_for(
    sample,
    rollout_index,
    is_train,
    schedule,
    demo_store,
    n_prefixed_rollouts,
    global_step,
    mode="steps",
    min_demo_decisions=1,
):
    """The payload keys that tell the rollout worker what prefix to replay.

    One key per mode, and the worker reads the mode off whichever arrives. Keeping the
    choice in one process is why there is no PREFIX_MODE environment variable: two
    settings that must agree are two settings that can disagree, which is the failure
    PREFIX_DEMOS_PATH is already exposed to.

    The key is always present, with a zero value when this rollout is not prefixed.
    ``_make_controller`` warns when its key is missing entirely, because that means
    dispatch itself broke, and dropping the key would fire that warning constantly.
    """
    if mode not in ("steps", "tokens"):
        raise ValueError(f"unknown prefix mode {mode!r}; expected 'steps' or 'tokens'")

    # Eligibility gate, applied before either mode runs so the two cannot disagree
    # about which questions can carry a prefix. Step mode is structurally limited to
    # m >= 2 by its k <= m-1 clamp, while a token budget needs no second decision.
    # Left at 1 the modes prefix different question sets (1085 against 1358), which
    # confounds any comparison between them; set to 2 they prefix the same 1085.
    if min_demo_decisions > 1 and demo_store is not None:
        question = str((sample or {}).get("question", ""))
        if question and demo_store.n_steps(question) < min_demo_decisions:
            return {"prefix_l": 0.0} if mode == "tokens" else {"prefix_k": 0}

    kwargs = dict(
        sample=sample,
        rollout_index=rollout_index,
        is_train=is_train,
        schedule=schedule,
        demo_store=demo_store,
        n_prefixed_rollouts=n_prefixed_rollouts,
        global_step=global_step,
    )
    if mode == "tokens":
        return {"prefix_l": prefix_l_for(**kwargs)}
    return {"prefix_k": prefix_k_for(**kwargs)}


def read_prefix_spec(task):
    """Read the dispatched prefix off a task payload.

    Returns ``("tokens", l)``, ``("steps", k)``, or ``(None, 0)`` when neither key is
    present, which means dispatch itself broke rather than that this rollout is
    unprefixed (an unprefixed one carries its key with a zero).

    Kept here rather than in the rollout worker so it can be tested: the worker module
    imports agentflow, which needs agentops, absent from the agent_engine env.
    """
    task = task or {}
    has_k = "prefix_k" in task
    has_l = "prefix_l" in task
    if has_k and has_l:
        raise RuntimeError(
            "task payload carries both prefix_k and prefix_l; the driver and the "
            "rollout worker disagree about prefix_rft.mode."
        )
    if has_l:
        return "tokens", float(task["prefix_l"] or 0.0)
    if has_k:
        return "steps", int(task["prefix_k"] or 0)
    return None, 0
