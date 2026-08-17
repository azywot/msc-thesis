"""Tests for Prefix-RFT entropy clipping.

The clip itself lives in ``entropy.py``, which imports no verl and so is
testable on CPU. ``PrefixRFTActor`` and ``PrefixRFTWorker`` are thin wrappers
covered by the import check in 009_run_tests_for_prefix_rft.job.
"""

import pytest

torch = pytest.importorskip("torch")

from verl_ext.prefix_rft.entropy import clip_prefix_advantage_by_entropy


def test_keeps_exactly_the_top_20_percent_of_prefix_tokens():
    adv = torch.ones(1, 10)
    prefix_mask = torch.ones(1, 10, dtype=torch.long)
    entropy = torch.arange(10, dtype=torch.float).unsqueeze(0)
    out, n_zeroed = clip_prefix_advantage_by_entropy(
        adv, prefix_mask, entropy, keep_ratio=0.2
    )
    assert n_zeroed == 8
    assert out[0, :8].abs().sum().item() == 0.0
    assert out[0, 8:].tolist() == [1.0, 1.0]


def test_non_prefix_tokens_are_never_touched():
    adv = torch.ones(1, 6)
    prefix_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    entropy = torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0, 0.0]])
    out, _ = clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=0.5)
    assert out[0, 3:].tolist() == [1.0, 1.0, 1.0]


def test_selection_is_global_across_the_micro_batch():
    """The reference sorts the flattened prefix tokens, not per row
    (dp_actor.py:138-139)."""
    adv = torch.ones(2, 4)
    prefix_mask = torch.ones(2, 4, dtype=torch.long)
    entropy = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    out, n_zeroed = clip_prefix_advantage_by_entropy(
        adv, prefix_mask, entropy, keep_ratio=0.25
    )
    assert n_zeroed == 6
    assert out[0].abs().sum().item() == 0.0
    assert out[1].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_no_prefix_tokens_is_a_no_op():
    adv = torch.ones(1, 4)
    prefix_mask = torch.zeros(1, 4, dtype=torch.long)
    entropy = torch.zeros(1, 4)
    out, n_zeroed = clip_prefix_advantage_by_entropy(
        adv, prefix_mask, entropy, keep_ratio=0.2
    )
    assert n_zeroed == 0
    assert torch.equal(out, adv)


def test_keep_ratio_of_one_keeps_everything():
    adv = torch.ones(1, 4)
    prefix_mask = torch.ones(1, 4, dtype=torch.long)
    entropy = torch.arange(4, dtype=torch.float).unsqueeze(0)
    out, n_zeroed = clip_prefix_advantage_by_entropy(
        adv, prefix_mask, entropy, keep_ratio=1.0
    )
    assert n_zeroed == 0
    assert torch.equal(out, adv)


def test_the_highest_entropy_tokens_are_the_survivors():
    """Not merely 'some' 20%: the paper's ablation shows bottom-20% and
    random-20% both underperform, so which tokens survive is the point."""
    adv = torch.ones(1, 5)
    prefix_mask = torch.ones(1, 5, dtype=torch.long)
    entropy = torch.tensor([[3.0, 0.1, 9.0, 0.2, 0.3]])
    out, _ = clip_prefix_advantage_by_entropy(
        adv, prefix_mask, entropy, keep_ratio=0.4
    )
    kept = [i for i in range(5) if out[0, i].item() != 0.0]
    assert kept == [0, 2]


def test_the_input_tensor_is_not_mutated():
    adv = torch.ones(1, 4)
    prefix_mask = torch.ones(1, 4, dtype=torch.long)
    entropy = torch.arange(4, dtype=torch.float).unsqueeze(0)
    clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=0.25)
    assert adv.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_negative_advantages_are_zeroed_too():
    """Zeroing is by entropy rank alone, not by sign."""
    adv = torch.full((1, 4), -2.0)
    prefix_mask = torch.ones(1, 4, dtype=torch.long)
    entropy = torch.arange(4, dtype=torch.float).unsqueeze(0)
    out, n_zeroed = clip_prefix_advantage_by_entropy(
        adv, prefix_mask, entropy, keep_ratio=0.25
    )
    assert n_zeroed == 3
    assert out[0].tolist() == [0.0, 0.0, 0.0, -2.0]


def test_the_copied_update_policy_is_in_sync_with_verl():
    """actor.py copies verl's update_policy. A verl upgrade that changes the
    original must show up here, not as training silently running the old body.

    Skipped where verl is absent; scripts/check_prefix_rft_actor_sync.py runs the
    same check under cosmas-train, which is where verl actually lives.
    """
    pytest.importorskip("verl")
    from verl_ext.prefix_rft.actor_edits import (
        actual_prefix_rft_update_policy,
        expected_prefix_rft_update_policy,
    )

    assert actual_prefix_rft_update_policy() == expected_prefix_rft_update_policy()


def test_the_edit_anchors_are_unique():
    """apply_edits must fail loudly rather than patch the wrong line twice."""
    from verl_ext.prefix_rft.actor_edits import EDITS, apply_edits

    with pytest.raises(ValueError, match="anchor appears 0 times"):
        apply_edits("nothing to anchor onto\n", EDITS)

    doubled = (EDITS[0][0] * 2) + EDITS[1][0] + EDITS[2][0]
    with pytest.raises(ValueError, match="anchor appears 2 times"):
        apply_edits(doubled, EDITS)


def test_the_copied_train_step_is_in_sync_with_the_vendored_agentflow():
    """trainer.py copies the vendored AgentFlow _train_step. A re-vendor that
    changes it must show up here rather than as silent divergence."""
    pytest.importorskip("verl")
    from verl_ext.prefix_rft.trainer_edits import (
        actual_prefix_rft_train_step,
        expected_prefix_rft_train_step,
    )

    assert actual_prefix_rft_train_step() == expected_prefix_rft_train_step()


def test_trainer_edit_anchors_are_unique():
    from verl_ext.prefix_rft.trainer_edits import EDITS, apply_edits

    with pytest.raises(ValueError, match="anchor appears 0 times"):
        apply_edits("nothing to anchor onto\n", EDITS)
