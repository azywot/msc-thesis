"""Worker that installs the Prefix-RFT actor.

verl's ``fsdp_workers`` hardcodes ``DataParallelPPOActor`` (fsdp_workers.py:923)
with no configuration hook. Rather than fork verl or copy its several-hundred-line
``init_model``, the actor's class is reassigned after construction.
``PrefixRFTActor`` adds no instance state, so this is equivalent to having
constructed it directly.
"""

from __future__ import annotations

import logging

from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .actor import PrefixRFTActor

logger = logging.getLogger(__name__)


class PrefixRFTWorker(AsyncActorRolloutRefWorker):
    """AsyncActorRolloutRefWorker whose actor entropy-clips prefix advantages."""

    def init_model(self):
        out = super().init_model()
        actor = getattr(self, "actor", None)
        if actor is not None and getattr(self, "_is_actor", False):
            actor.__class__ = PrefixRFTActor
            logger.info(
                "Installed PrefixRFTActor (entropy clipping on prefix tokens, "
                "keep_ratio=%s)",
                actor.prefix_keep_ratio,
            )
        return out
