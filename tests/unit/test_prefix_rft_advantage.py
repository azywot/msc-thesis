"""Tests for the Prefix-RFT advantage correction."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verl_ext.prefix_rft.advantage import apply_prefix_advantage


def _batch(scores, is_prefix_rollout, rows_per_rollout=1, prefix_len=2, resp_len=4):
    """Build a flat batch of rows, rows_per_rollout rows per rollout.

    Mirrors Flow GRPO's layout: every turn of a rollout is its own row, all
    carrying the same uid and the same reward.
    """
    n_rollouts = len(scores)
    n_rows = n_rollouts * rows_per_rollout
    token_level_rewards = torch.zeros(n_rows, resp_len)
    response_mask = torch.ones(n_rows, resp_len)
    prefix_mask = torch.zeros(n_rows, resp_len, dtype=torch.long)
    uid, rollout_id, flags = [], [], []
    row = 0
    for r in range(n_rollouts):
        for _ in range(rows_per_rollout):
            token_level_rewards[row, -1] = scores[r]
            if is_prefix_rollout[r]:
                prefix_mask[row, :prefix_len] = 1
            uid.append("q0")
            rollout_id.append(f"r{r}")
            flags.append(is_prefix_rollout[r])
            row += 1
    return (
        torch.zeros(n_rows, resp_len),
        token_level_rewards,
        response_mask,
        prefix_mask,
        np.array(uid),
        np.array(rollout_id),
        np.array(flags),
    )


def test_prefix_tokens_get_score_minus_the_unprefixed_mean():
    # 4 unprefixed rollouts scoring 1, 0, 0, 0 -> mean 0.25; hybrid scores 1.
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    hybrid_row = 4
    assert out[hybrid_row, 0].item() == pytest.approx(0.75, rel=1e-5)
    assert out[hybrid_row, 1].item() == pytest.approx(0.75, rel=1e-5)


def test_non_prefix_tokens_of_the_hybrid_rollout_pass_through_uncentred():
    # Reference behaviour: a singleton prefix group takes mean 0 and std 1
    # (core_algos.py:189-191), so the raw score reaches the continuation tokens.
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    assert out[4, 2].item() == pytest.approx(1.0, rel=1e-5)
    assert out[4, 3].item() == pytest.approx(1.0, rel=1e-5)


def test_unprefixed_rollouts_are_centred_over_the_unprefixed_group_only():
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0])
    expected = (1.0 - scores.mean()) / (scores.std() + 1e-6)
    assert out[0, 0].item() == pytest.approx(expected.item(), rel=1e-4)


def test_the_hybrid_rollout_is_excluded_from_the_on_policy_baseline():
    """Including it would lift the mean and bias every on-policy advantage down."""
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    over_seven = torch.tensor([1.0, 0.0, 0.0, 0.0])
    over_eight = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    excl = ((1.0 - over_seven.mean()) / (over_seven.std() + 1e-6)).item()
    incl = ((1.0 - over_eight.mean()) / (over_eight.std() + 1e-6)).item()
    assert out[0, 0].item() == pytest.approx(excl, rel=1e-4)
    assert out[0, 0].item() != pytest.approx(incl, rel=1e-4)


def test_grouping_is_per_rollout_not_per_row():
    # The hybrid rollout has 3 turns. A row-level port would centre them against
    # themselves and produce exactly zero prefix advantage.
    args = _batch(
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [False, False, False, False, True],
        rows_per_rollout=3,
    )
    out = apply_prefix_advantage(*args)
    hybrid_first_row = 4 * 3
    assert out[hybrid_first_row, 0].item() == pytest.approx(0.75, rel=1e-5)
    assert out[hybrid_first_row, 0].item() != 0.0


def test_every_turn_of_the_hybrid_rollout_carries_the_prefix_advantage():
    args = _batch(
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [False, False, False, False, True],
        rows_per_rollout=3,
    )
    out = apply_prefix_advantage(*args)
    for row in (12, 13, 14):
        assert out[row, 0].item() == pytest.approx(0.75, rel=1e-5)


def test_questions_without_a_prefixed_rollout_are_left_untouched():
    args = list(_batch([1.0, 0.0], [False, False]))
    args[0][:] = 42.0
    before = args[0].clone()
    out = apply_prefix_advantage(*args)
    assert torch.equal(out, before)


def test_padding_positions_stay_zero():
    args = list(_batch([1.0, 0.0, 1.0], [False, False, True]))
    args[2][:, -1] = 0  # response_mask marks the last position as padding
    out = apply_prefix_advantage(*args)
    assert out[:, -1].abs().sum().item() == pytest.approx(0.0)


def test_group_baseline_recentres_only_the_continuation():
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args, singleton_baseline="group")
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0])
    expected = ((1.0 - scores.mean()) / (scores.std() + 1e-6)).item()
    assert out[4, 2].item() == pytest.approx(expected, rel=1e-4)
    assert out[4, 0].item() == pytest.approx(0.75, rel=1e-5)


def test_several_questions_are_handled_independently():
    args = list(_batch([1.0, 0.0, 1.0], [False, False, True]))
    uid = np.array(["q0", "q0", "q0"])
    args[4] = uid
    out_one = apply_prefix_advantage(*args)

    # Same rows, but the prefixed rollout belongs to a different question with no
    # unprefixed peers: its baseline is then 0, so the prefix keeps its raw score.
    args[4] = np.array(["q0", "q0", "q1"])
    out_two = apply_prefix_advantage(*args)
    assert out_one[2, 0].item() == pytest.approx(0.5, rel=1e-5)
    assert out_two[2, 0].item() == pytest.approx(1.0, rel=1e-5)


def test_a_failing_prefix_gets_a_negative_advantage():
    """The prefix is only reinforced when it actually helped."""
    args = _batch([1.0, 1.0, 1.0, 1.0, 0.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    assert out[4, 0].item() == pytest.approx(-1.0, rel=1e-5)
