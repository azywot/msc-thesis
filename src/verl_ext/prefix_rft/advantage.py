"""Prefix-aware GRPO advantage.

Port of ``compute_grpo_prefix_outcome_advantage``
(``repos/prefix_rft/recipe/prefix_rft/core_algos.py:162-217``), lifted from the
reference's one-row-per-rollout layout to Flow GRPO's one-row-per-turn layout.

Three properties of the reference are preserved:

1. Groups are ``(question, prefix identity)``. The unprefixed rollouts share one
   group; the hybrid rollout is alone in its own. Excluding the hybrid from the
   on-policy baseline matters: at one of eight with a hybrid reward near 1.0,
   including it would lift the group mean and bias every on-policy advantage down.
2. A singleton group takes mean 0 and std 1 (core_algos.py:189-191), so the hybrid
   rollout's continuation tokens pass through uncentred. The reference authors flag
   this at core_algos.py:294-296; it is kept for fidelity and watched via metrics.
   ``singleton_baseline="group"`` is the opt-in mitigation.
3. Prefix tokens are overwritten with ``score - mean(unprefixed)``, divided by the
   rollouts-per-prefix count. This is the quantity the paper's Figure 4 plots as
   the gap between reward-with-prefix and overall training reward, and it is what
   makes the prefix's influence fade as the policy improves.

Grouping is per rollout: scores are deduplicated by ``rollout_id`` before any
statistic is taken. A row-level port would put the hybrid rollout's several turns
in a group of their own, centre them against themselves, and yield a prefix
advantage of exactly zero on every step.

Kept free of verl imports so it stays unit-testable on CPU.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def apply_prefix_advantage(
    advantages: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_mask: torch.Tensor,
    uid: np.ndarray,
    rollout_id: np.ndarray,
    is_prefix_rollout: np.ndarray,
    num_rollouts_per_prefix: int = 1,
    epsilon: float = 1e-6,
    singleton_baseline: str = "none",
) -> torch.Tensor:
    """Rewrite advantages for questions that have a prefixed rollout.

    Questions without one, which covers every undemonstrated question and every
    ``k = 0`` draw, are left exactly as verl computed them.

    ``singleton_baseline`` is "none" for the reference behaviour, or "group" to
    recentre the hybrid rollout's continuation tokens against the unprefixed
    rollouts. See the spec's risk note; flip it only if training destabilises.
    """
    out = advantages.clone()
    row_scores = token_level_rewards.sum(dim=-1)
    device = row_scores.device

    # Rows belonging to each rollout, in first-seen order.
    rollout_rows: dict[str, list[int]] = defaultdict(list)
    for i, rid in enumerate(rollout_id):
        rollout_rows[str(rid)].append(i)

    # Question -> its rollouts.
    question_rollouts: dict[str, list[str]] = defaultdict(list)
    for rid, rows in rollout_rows.items():
        question_rollouts[str(uid[rows[0]])].append(rid)

    with torch.no_grad():
        for _question, rids in question_rollouts.items():
            prefixed = [r for r in rids if bool(is_prefix_rollout[rollout_rows[r][0]])]
            if not prefixed:
                continue  # plain GRPO; verl's advantage stands

            prefixed_set = set(prefixed)
            unprefixed = [r for r in rids if r not in prefixed_set]

            if unprefixed:
                base = torch.stack([row_scores[rollout_rows[r][0]] for r in unprefixed])
                mean_np, std_np = base.mean(), base.std()
                if torch.isnan(std_np):  # a single unprefixed rollout
                    std_np = torch.tensor(1.0, device=device)
            else:
                mean_np = torch.tensor(0.0, device=device)
                std_np = torch.tensor(1.0, device=device)

            # Unprefixed rollouts: standard GRPO over their own group.
            for rid in unprefixed:
                centred = (row_scores[rollout_rows[rid][0]] - mean_np) / (std_np + epsilon)
                for row in rollout_rows[rid]:
                    out[row] = centred * response_mask[row]

            # Prefixed rollouts.
            group = torch.stack([row_scores[rollout_rows[r][0]] for r in prefixed])
            if len(prefixed) == 1:
                own_mean = torch.tensor(0.0, device=device)
                own_std = torch.tensor(1.0, device=device)
            else:
                own_mean, own_std = group.mean(), group.std()
                if torch.isnan(own_std):
                    own_std = torch.tensor(1.0, device=device)

            for rid in prefixed:
                score = row_scores[rollout_rows[rid][0]]
                passthrough = (score - own_mean) / (own_std + epsilon)
                prefix_value = (passthrough - mean_np) / num_rollouts_per_prefix
                if singleton_baseline == "group":
                    # Opt-in mitigation for the uncentred continuation the
                    # reference authors flag at core_algos.py:294-296. Prefix
                    # tokens keep the reference value; only the continuation moves.
                    continuation = (score - mean_np) / (std_np + epsilon)
                else:
                    continuation = passthrough
                for row in rollout_rows[rid]:
                    filled = torch.where(
                        prefix_mask[row].bool(),
                        prefix_value.expand_as(out[row]),
                        continuation.expand_as(out[row]),
                    )
                    out[row] = filled * response_mask[row]

    return out
