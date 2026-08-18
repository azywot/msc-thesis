"""Entropy-based clipping of demonstration-token advantages.

Paper section 3: "we propose an entropy-based clipping approach, i.e. that
involves only the top-k% high-entropy demonstration tokens. Regarding
implementation, we directly set the corresponding advantages of all other tokens
to zero." Appendix A.2 fixes k at 20.

Why it is needed (paper Table 4, A.3): gradients from off-policy demonstration
tokens are large enough that prefix tokens roughly double the batch gradient norm
while making up 5-10% of it. Unclipped, the model fits superficial features of
the demonstration, notably response length, instead of learning from both signals.

Why high entropy specifically (A.4): low-entropy prefix tokens are in two useless
regimes. If the model already agrees with the demonstration the target-logit
gradient ``1 - p`` is near zero; if it confidently disagrees, updating toward the
demonstration is a sharp overwrite. High-entropy selection avoids both extremes
and leaves the reinforcement strength to the trajectory-level advantage.

Port of ``reshape_func``'s ``entropy`` branch
(``repos/prefix_rft/recipe/prefix_rft/dp_actor.py:132-158``). Kept free of verl
imports so it stays unit-testable on CPU.
"""

from __future__ import annotations

import torch


def clip_prefix_advantage_by_entropy(advantages, prefix_mask, entropy, keep_ratio=0.2):
    """Zero the advantage of all but the highest-entropy ``keep_ratio`` prefix tokens.

    Selection is global across the micro-batch, not per row: the reference sorts
    the flattened prefix tokens (dp_actor.py:138-139), so a row of uniformly
    low-entropy prefix tokens can be dropped entirely while another row keeps
    several.

    Returns ``(advantages, n_zeroed)``. The input is not mutated.
    """
    mask = prefix_mask.bool()
    n_prefix = int(mask.sum().item())
    if n_prefix == 0 or keep_ratio >= 1.0:
        return advantages, 0

    prefix_entropy = entropy[mask]
    order = torch.argsort(prefix_entropy)
    n_drop = int(len(order) * (1.0 - keep_ratio))
    if n_drop == 0:
        return advantages, 0

    keep_flat = torch.ones(n_prefix, dtype=torch.bool, device=advantages.device)
    keep_flat[order[:n_drop]] = False

    out = advantages.clone()
    prefix_values = out[mask]
    out[mask] = torch.where(keep_flat, prefix_values, torch.zeros_like(prefix_values))
    return out, n_drop
