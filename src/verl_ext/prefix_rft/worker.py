"""Worker that installs the Prefix-RFT actor.

verl's ``fsdp_workers`` hardcodes ``DataParallelPPOActor`` (fsdp_workers.py:923)
with no configuration hook. Rather than fork verl or copy its several-hundred-line
``init_model``, the actor's class is reassigned after construction.
``PrefixRFTActor`` adds no instance state, so this is equivalent to having
constructed it directly.

Overriding a verl worker method requires re-applying ``@register``. The decorator
stores a dispatch descriptor on the function object, and ``RayWorkerGroup`` binds
only the methods carrying it, so a plain override is invisible to the driver: the
call fails with ``'RayWorkerGroup' object has no attribute 'init_model'`` at
``init_workers()``, before any training starts (job 25751449).
"""

from __future__ import annotations

import logging

from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .actor import PAPER_KEEP_RATIO, PrefixRFTActor

logger = logging.getLogger(__name__)


class PrefixRFTWorker(AsyncActorRolloutRefWorker):
    """AsyncActorRolloutRefWorker whose actor entropy-clips prefix advantages."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        out = super().init_model()
        actor = getattr(self, "actor", None)
        if actor is not None and getattr(self, "_is_actor", False):
            actor.__class__ = PrefixRFTActor
            # Read from actor_rollout_ref, not actor_rollout_ref.actor: verl turns the
            # latter into an FSDPActorConfig dataclass that rejects undeclared keys.
            # See the PrefixRFTActor docstring.
            actor.prefix_keep_ratio = float(
                self.config.get("prefix_entropy_keep_ratio", PAPER_KEEP_RATIO)
            )
            # print, not logger.info: see the note in trainer._ensure_prefix_daemon.
            print(
                "Installed PrefixRFTActor (entropy clipping on prefix tokens, "
                f"keep_ratio={actor.prefix_keep_ratio})"
            )
        return out


def _assert_registration_matches_verl() -> None:
    """Fail at import if our override no longer dispatches the way verl's does.

    The dispatch mode above is copied from verl, so it can drift. Being wrong here is
    not a crash: a mismatched mode would dispatch init_model to the wrong ranks and
    the run would proceed with the actor installed on some of them. Checked at import
    because there is no later point where it would surface.
    """
    ours = getattr(PrefixRFTWorker.init_model, MAGIC_ATTR, None)
    theirs = getattr(AsyncActorRolloutRefWorker.init_model, MAGIC_ATTR, None)
    if ours is None:
        raise RuntimeError(
            "PrefixRFTWorker.init_model lost its @register marker; RayWorkerGroup "
            "would not bind it and init_workers() would fail."
        )
    if theirs is not None and ours != theirs:
        raise RuntimeError(
            "PrefixRFTWorker.init_model is registered differently from verl's:\n"
            f"  ours:   {ours}\n"
            f"  verl's: {theirs}\n"
            "Update the @register decorator in verl_ext/prefix_rft/worker.py to match."
        )


_assert_registration_matches_verl()
