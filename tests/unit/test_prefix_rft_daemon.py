"""Tests for the Prefix-RFT daemon's logic.

The logic lives in verl-free modules (``masks``, ``dispatch``) so it can be
tested on CPU: verl is not installed in the agent_engine env, and pytest is not
installed in cosmas-train, so anything importing verl cannot be exercised by a
test in either environment. ``PrefixRFTDaemon`` itself is a thin subclass and is
covered by the import check in 009_run_tests_for_prefix_rft.job.
"""

import pytest

from verl_ext.prefix_rft.dispatch import (
    prefix_k_for,
    prefix_l_for,
    prefix_spec_for,
    read_prefix_spec,
)
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


def test_mask_marks_only_the_replayed_head_of_a_split_turn():
    """Token mode hands the model the first r tokens of a turn and it writes the
    rest, so the row is 1 up to r and 0 after."""
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3, 4], "is_prefix": True, "prefix_len": 2}]
    assert build_prefix_mask(traces, max_response_length=6) == [[1, 1, 0, 0, 0, 0]]


def test_prefix_len_is_clamped_to_the_truncated_response():
    traces = [
        {"prompt_ids": [9], "response_ids": [1, 2, 3, 4, 5], "is_prefix": True, "prefix_len": 5}
    ]
    assert build_prefix_mask(traces, max_response_length=3) == [[1, 1, 1]]


def test_a_fully_replayed_turn_needs_no_prefix_len():
    """Step mode writes is_prefix only. It must keep meaning 'the whole turn'."""
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3], "is_prefix": True}]
    assert build_prefix_mask(traces, max_response_length=4) == [[1, 1, 1, 0]]


def test_prefix_len_without_is_prefix_still_marks_tokens():
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3], "prefix_len": 1}]
    assert build_prefix_mask(traces, max_response_length=3) == [[1, 0, 0]]


def test_a_zero_prefix_len_on_a_prefixed_turn_is_read_as_the_whole_turn():
    """0 is the absent-value default, so it cannot also mean 'no tokens'. A turn
    that genuinely replayed nothing carries is_prefix=False."""
    traces = [{"prompt_ids": [9], "response_ids": [1, 2], "is_prefix": True, "prefix_len": 0}]
    assert build_prefix_mask(traces, max_response_length=2) == [[1, 1]]


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


# --------------------------------------------------------------------------- #
# token-mode dispatch                                                          #
# --------------------------------------------------------------------------- #


class _LSchedule(_Schedule):
    def __init__(self, l=0.6):
        super().__init__()
        self._l = l
        self.l_calls = []

    def sample_l(self, global_step):
        self.l_calls.append(global_step)
        return self._l, 0.1, 0.95


def test_token_mode_dispatches_the_raw_fraction():
    sched = _LSchedule(l=0.6)
    l = prefix_l_for(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        schedule=sched,
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=7,
    )
    assert l == pytest.approx(0.6)
    assert sched.l_calls == [7]


def test_token_mode_prefixes_a_single_decision_demonstration():
    """Step mode cannot: k <= m - 1 = 0. The token guard is applied to the token
    total instead, so these 273 questions become prefixable."""
    l = prefix_l_for(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        schedule=_LSchedule(l=0.6),
        demo_store=_Store(1),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert l == pytest.approx(0.6)


def test_token_mode_skips_questions_with_no_demonstration():
    assert (
        prefix_l_for(
            sample={"question": "q"},
            rollout_index=0,
            is_train=True,
            schedule=_LSchedule(),
            demo_store=_Store(0),
            n_prefixed_rollouts=1,
            global_step=0,
        )
        == 0.0
    )


def test_token_mode_never_prefixes_validation_or_extra_rollouts():
    kwargs = dict(
        sample={"question": "q"},
        schedule=_LSchedule(),
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_l_for(rollout_index=0, is_train=False, **kwargs) == 0.0
    assert prefix_l_for(rollout_index=1, is_train=True, **kwargs) == 0.0


def test_the_spec_carries_exactly_one_key_per_mode():
    kwargs = dict(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_spec_for(schedule=_Schedule(k=2), mode="steps", **kwargs) == {"prefix_k": 2}
    assert prefix_spec_for(schedule=_LSchedule(l=0.6), mode="tokens", **kwargs) == pytest.approx(
        {"prefix_l": 0.6}
    )


def test_the_spec_still_carries_its_key_when_nothing_is_prefixed():
    """The worker warns when its key is missing entirely, because that means dispatch
    broke. A non-prefixed rollout must carry the key with a zero, not drop it."""
    kwargs = dict(
        sample={"question": "q"},
        rollout_index=5,
        is_train=True,
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_spec_for(schedule=_Schedule(), mode="steps", **kwargs) == {"prefix_k": 0}
    assert prefix_spec_for(schedule=_LSchedule(), mode="tokens", **kwargs) == {"prefix_l": 0.0}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown prefix mode"):
        prefix_spec_for(
            sample={"question": "q"},
            rollout_index=0,
            is_train=True,
            schedule=_Schedule(),
            demo_store=_Store(3),
            n_prefixed_rollouts=1,
            global_step=0,
            mode="fractions",
        )


# --------------------------------------------------------------------------- #
# reading the dispatched prefix back out in the worker                          #
# --------------------------------------------------------------------------- #


def test_a_prefix_l_payload_reads_as_token_mode():
    assert read_prefix_spec({"question": "q", "prefix_l": 0.8}) == ("tokens", 0.8)


def test_a_prefix_k_payload_reads_as_step_mode():
    assert read_prefix_spec({"question": "q", "prefix_k": 2}) == ("steps", 2)


def test_both_keys_at_once_is_refused():
    """One key per mode is the whole reason the worker cannot disagree with the
    driver. Both arriving means that invariant is already broken."""
    with pytest.raises(RuntimeError, match="disagree about prefix_rft.mode"):
        read_prefix_spec({"prefix_k": 1, "prefix_l": 0.5})


def test_neither_key_reads_as_no_dispatch():
    assert read_prefix_spec({"question": "q"}) == (None, 0)
    assert read_prefix_spec(None) == (None, 0)


def test_a_null_value_reads_as_a_zero_of_the_right_type():
    assert read_prefix_spec({"prefix_l": None}) == ("tokens", 0.0)
    assert read_prefix_spec({"prefix_k": None}) == ("steps", 0)
