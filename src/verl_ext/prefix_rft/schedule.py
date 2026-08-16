"""Prefix length schedule.

Ports the controllers from the reference implementation at
``repos/prefix_rft/recipe/prefix_rft/scheduler/global_step.py`` (CosineDecayController
lines 148-186, BetaSampler lines 256-275) and adds the step-level discretisation this
project uses in place of the paper's token ratio.

Paper Appendix A.2: "At each time step t, we sample l uniformly from [low_t, 0.95] to
decide the prefix length as l times the total demonstration length. And low_t follows a
cosine decay scheduler, starting from 0.95 and decaying to 0.05 at the 500th step."
"""

from __future__ import annotations

import math

import numpy as np

# Paper A.2. Named so the config and the tests read against one source of truth.
PAPER_HIGH = 0.95
PAPER_LOW_INIT = 0.95
PAPER_LOW_TARGET = 0.05


class ConstController:
    """Constant value. Reference: global_step.py:5-17."""

    def __init__(self, init: float = 0.0, **kwargs):
        self.c = init

    def value(self, global_step: int = 0, **kwargs) -> float:
        return self.c

    def __str__(self) -> str:
        return f"Constant({self.c})"


class CosineDecayController:
    """Cosine interpolation from ``init`` to ``target``. Reference: global_step.py:148-186."""

    def __init__(
        self,
        init=PAPER_LOW_INIT,
        target=PAPER_LOW_TARGET,
        n_steps=500,
        warmup_ratio=0.0,
        **kwargs,
    ):
        if init == target:
            raise ValueError("init and target must differ")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.init = init
        self.target = target
        self.n_steps = n_steps
        self.warmup_steps = int(warmup_ratio * n_steps)
        self.mode = "decay" if init > target else "rise"

    def value(self, global_step: int = 0, **kwargs) -> float:
        if self.warmup_steps and global_step < self.warmup_steps:
            return (global_step / self.warmup_steps) * self.init
        step = global_step - self.warmup_steps
        if step > self.n_steps:
            return self.target
        decay_ratio = 0.5 * (1 + math.cos(math.pi * step / self.n_steps))
        if self.mode == "decay":
            return self.target + decay_ratio * (self.init - self.target)
        return self.init + (1 - decay_ratio) * (self.target - self.init)

    def __str__(self) -> str:
        return (
            f"CosineDecay(init={self.init}, target={self.target}, "
            f"n_steps={self.n_steps}, warmup={self.warmup_steps})"
        )


CTRL_MAPPING = {"cosine_decay": CosineDecayController, "const": ConstController}


class PrefixStepSchedule:
    """Draw the number of teacher decisions to replay.

    ``l`` follows the paper exactly: a Beta(alpha, beta) draw rescaled onto
    ``[low_t, high]``, which at alpha = beta = 1 is the uniform draw A.2 specifies.
    Discretisation to whole decisions is ours (see the spec's "Adaptation" section):
    ``k = clamp(floor(l * m), 0, m - 1)``. The upper clamp is the step-level analogue of
    the reference's ``prefix_len >= demo_len -> demo_len - 1`` guard
    (rl_dataset.py:300-301) and guarantees at least one on-policy decision.
    """

    def __init__(
        self,
        low_init=PAPER_LOW_INIT,
        low_target=PAPER_LOW_TARGET,
        high=PAPER_HIGH,
        n_steps=500,
        alpha=1.0,
        beta=1.0,
        seed=None,
    ):
        self.low_ctrl = CosineDecayController(
            init=low_init, target=low_target, n_steps=n_steps
        )
        self.high_ctrl = ConstController(init=high)
        self.alpha = alpha
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def sample_l(self, global_step: int) -> tuple[float, float, float]:
        """Return (l, low_t, high). Reference: BetaSampler.value, global_step.py:268-275."""
        low = self.low_ctrl.value(global_step=global_step)
        high = self.high_ctrl.value(global_step=global_step)
        lower, higher = min(low, high), max(low, high)
        u = float(self._rng.beta(self.alpha, self.beta))
        return lower + (higher - lower) * u, lower, higher

    def sample_k(self, n_demo_steps: int, global_step: int) -> int:
        """Number of teacher decisions to replay for one rollout."""
        if n_demo_steps <= 1:
            return 0
        l, _, _ = self.sample_l(global_step)
        return max(0, min(int(math.floor(l * n_demo_steps)), n_demo_steps - 1))
