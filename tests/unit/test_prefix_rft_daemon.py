"""Tests for the Prefix-RFT daemon's logic.

The logic lives in verl-free modules (``masks``, ``dispatch``) so it can be
tested on CPU: verl is not installed in the agent_engine env, and pytest is not
installed in cosmas-train, so anything importing verl cannot be exercised by a
test in either environment. ``PrefixRFTDaemon`` itself is a thin subclass and is
covered by the import check in 009_run_tests_for_prefix_rft.job.
"""

import pytest

from verl_ext.prefix_rft.dispatch import prefix_k_for
from verl_ext.prefix_rft.masks import build_prefix_mask


# --------------------------------------------------------------------------- #
# prefix_mask construction                                                     #
# --------------------------------------------------------------------------- #


def test_mask_is_one_over_prefix_turn_tokens_only():
    traces = [
        {"prompt_ids": [9], "response_ids": [1, 2, 3], "is_prefix": True},
        {"prompt_ids": [9], "response_ids": [4, 5], "is_prefix": False},
    ]
    assert build_prefix_mask(traces, max_response_length=5) == [
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_mask_truncates_with_the_response():
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3, 4, 5, 6], "is_prefix": True}]
    assert build_prefix_mask(traces, max_response_length=4) == [[1, 1, 1, 1]]


def test_mask_is_all_zero_when_nothing_was_replayed():
    traces = [{"prompt_ids": [9], "response_ids": [1, 2], "is_prefix": False}]
    assert build_prefix_mask(traces, max_response_length=3) == [[0, 0, 0]]


def test_turns_empty_on_both_sides_are_skipped_like_the_base_daemon_does():
    """The base daemon drops a turn only when prompt AND response are empty
    (daemon.py:746-750). The mask must drop exactly the same rows or it
    misaligns with `responses`."""
    traces = [
        {"prompt_ids": [], "response_ids": [], "is_prefix": True},
        {"prompt_ids": [9], "response_ids": [7], "is_prefix": True},
    ]
    assert build_prefix_mask(traces, max_response_length=2) == [[1, 0]]


def test_a_turn_with_a_prompt_but_no_response_is_kept():
    traces = [{"prompt_ids": [9], "response_ids": [], "is_prefix": True}]
    assert build_prefix_mask(traces, max_response_length=2) == [[0, 0]]


# --------------------------------------------------------------------------- #
# per-rollout k dispatch                                                       #
# --------------------------------------------------------------------------- #


class _Store:
    def __init__(self, n):
        self._n = n

    def n_steps(self, question):
        return self._n


class _Schedule:
    def __init__(self, k=2):
        self._k = k
        self.calls = []

    def sample_k(self, m, global_step):
        self.calls.append((m, global_step))
        return self._k


def _sample():
    return {"question": "what is 2+2", "data_source": "deepmath"}


def test_only_the_first_rollout_of_each_question_gets_a_prefix():
    schedule, store = _Schedule(2), _Store(4)
    common = dict(
        sample=_sample(),
        is_train=True,
        schedule=schedule,
        demo_store=store,
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_k_for(rollout_index=0, **common) == 2
    assert prefix_k_for(rollout_index=1, **common) == 0
    assert prefix_k_for(rollout_index=7, **common) == 0


def test_validation_rollouts_never_get_a_prefix():
    assert (
        prefix_k_for(
            sample=_sample(),
            rollout_index=0,
            is_train=False,
            schedule=_Schedule(2),
            demo_store=_Store(4),
            n_prefixed_rollouts=1,
            global_step=0,
        )
        == 0
    )


def test_single_decision_questions_get_no_prefix():
    schedule = _Schedule(2)
    assert (
        prefix_k_for(
            sample=_sample(),
            rollout_index=0,
            is_train=True,
            schedule=schedule,
            demo_store=_Store(1),
            n_prefixed_rollouts=1,
            global_step=0,
        )
        == 0
    )
    assert schedule.calls == []


def test_questions_without_a_demonstration_get_no_prefix():
    assert (
        prefix_k_for(
            sample=_sample(),
            rollout_index=0,
            is_train=True,
            schedule=_Schedule(2),
            demo_store=_Store(0),
            n_prefixed_rollouts=1,
            global_step=0,
        )
        == 0
    )


def test_missing_schedule_or_store_is_a_no_op_rather_than_a_crash():
    for schedule, store in ((None, _Store(4)), (_Schedule(2), None)):
        assert (
            prefix_k_for(
                sample=_sample(),
                rollout_index=0,
                is_train=True,
                schedule=schedule,
                demo_store=store,
                n_prefixed_rollouts=1,
                global_step=0,
            )
            == 0
        )


def test_the_schedule_is_asked_for_the_current_global_step():
    schedule = _Schedule(3)
    prefix_k_for(
        sample=_sample(),
        rollout_index=0,
        is_train=True,
        schedule=schedule,
        demo_store=_Store(5),
        n_prefixed_rollouts=1,
        global_step=42,
    )
    assert schedule.calls == [(5, 42)]


def test_lookup_uses_the_question_not_an_index():
    """extra_info.idx collides across data sources; the store is keyed on the
    question text, so that is what dispatch must pass through."""
    seen = []

    class _Recording:
        def n_steps(self, question):
            seen.append(question)
            return 4

    prefix_k_for(
        sample={"question": "what is 2+2", "extra_info": {"idx": 669}},
        rollout_index=0,
        is_train=True,
        schedule=_Schedule(1),
        demo_store=_Recording(),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert seen == ["what is 2+2"]
