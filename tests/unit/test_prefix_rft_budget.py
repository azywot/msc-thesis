"""Tests for the Prefix-RFT token budget.

Pure arithmetic over decision token lengths: no tokenizer, no verl. The clamp
under test is the paper's ``prefix_len <= demo_len - 1`` guard applied to the
concatenated demonstration, so there is always a token left to generate.
"""

import pytest

from verl_ext.prefix_rft.budget import split_for_budget


def test_budget_splits_the_decision_that_straddles_it():
    # total 30, l = 0.8 -> budget 24: two whole decisions (20) then 4 tokens.
    assert split_for_budget([10, 10, 10], 0.8) == (2, 4)


def test_exact_boundary_gives_no_split():
    # total 20, l = 0.5 -> budget 10, which is exactly one whole decision.
    assert split_for_budget([10, 10], 0.5) == (1, 0)


def test_zero_fraction_replays_nothing():
    assert split_for_budget([10, 10, 10], 0.0) == (0, 0)


def test_the_top_of_the_range_still_leaves_a_token_to_generate():
    n_full, r = split_for_budget([10, 10, 10], 0.95)
    assert (n_full, r) == (2, 8)
    assert n_full * 10 + r == 28  # 29 tokens is the cap; 28 is floor(0.95 * 30)


def test_a_fraction_of_one_never_exceeds_the_cap():
    n_full, r = split_for_budget([10, 10, 10], 1.0)
    assert n_full * 10 + r == 29


def test_a_single_decision_demonstration_is_prefixable():
    """Step mode forces k <= m - 1 = 0 here. The token guard does not, which is
    what takes prefixable coverage from 1085 questions to all 1358."""
    assert split_for_budget([5], 0.9) == (0, 4)


def test_a_single_token_demonstration_is_not_prefixable():
    assert split_for_budget([1], 0.9) == (0, 0)


def test_an_empty_demonstration_is_not_prefixable():
    assert split_for_budget([], 0.9) == (0, 0)


@pytest.mark.parametrize("l", [0.0, 0.05, 0.3, 0.5, 0.75, 0.95])
def test_at_least_one_token_is_always_generated(l):
    lengths = [3, 11, 2, 7]
    n_full, r = split_for_budget(lengths, l)
    assert sum(lengths[:n_full]) + r <= sum(lengths) - 1
    assert n_full < len(lengths) or r == 0


@pytest.mark.parametrize("l", [0.1, 0.4, 0.6, 0.9])
def test_the_split_decision_is_never_replayed_whole(l):
    lengths = [4, 4, 4, 4]
    n_full, r = split_for_budget(lengths, l)
    if r > 0:
        assert r < lengths[n_full]
