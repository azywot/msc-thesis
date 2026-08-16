"""Tests for the Prefix-RFT prefix length schedule."""

import pytest

from verl_ext.prefix_rft.schedule import (
    ConstController,
    CosineDecayController,
    PrefixStepSchedule,
)


def test_cosine_decay_endpoints_match_the_paper():
    ctrl = CosineDecayController(init=0.95, target=0.05, n_steps=500)
    assert ctrl.value(global_step=0) == pytest.approx(0.95)
    assert ctrl.value(global_step=500) == pytest.approx(0.05)
    assert ctrl.value(global_step=250) == pytest.approx(0.5)


def test_cosine_decay_is_monotone_and_clamps_past_the_end():
    ctrl = CosineDecayController(init=0.95, target=0.05, n_steps=500)
    values = [ctrl.value(global_step=s) for s in range(0, 501, 25)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert ctrl.value(global_step=900) == pytest.approx(0.05)


def test_const_controller_ignores_the_step():
    ctrl = ConstController(init=0.8)
    assert ctrl.value(global_step=0) == 0.8
    assert ctrl.value(global_step=999) == 0.8


def test_sampled_l_stays_inside_the_scheduled_window():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    for step in (0, 100, 250, 499):
        for _ in range(50):
            l, low, high = sched.sample_l(global_step=step)
            assert low <= l <= high
            assert high == pytest.approx(0.95)


def test_k_is_clamped_to_leave_one_on_policy_decision():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    for step in (0, 250, 499):
        for m in (1, 2, 3, 4, 8, 13):
            for _ in range(20):
                k = sched.sample_k(m, global_step=step)
                assert 0 <= k <= m - 1


def test_single_decision_demonstrations_never_carry_a_prefix():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    assert all(sched.sample_k(1, global_step=s) == 0 for s in range(0, 500, 10))


def test_k_shrinks_as_training_advances():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    early = sum(sched.sample_k(8, global_step=0) for _ in range(200))
    late = sum(sched.sample_k(8, global_step=490) for _ in range(200))
    assert early > late


def test_zero_demonstration_steps_gives_zero():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    assert sched.sample_k(0, global_step=0) == 0
