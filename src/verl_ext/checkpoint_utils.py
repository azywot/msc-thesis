"""Reconstruct full tensors from a VERL FSDP SHARDED_STATE_DICT checkpoint.

Shared by `scripts/finalize_sft_run.py` (LoRA extraction) and `scripts/merge_lora.py`
(full-parameter consolidation). Both need the same DTensor-gathering logic; this module is
the single place it is implemented so the two never drift apart.

Two VERL checkpoint layouts exist in this repo and must not be confused:

- **RL path** (`verl/workers/fsdp_workers.py`): saves with FSDP `FULL_STATE_DICT`,
  `rank0_only=True`. Exactly one file is written, `model_world_size_<W>_rank_0.pt`, and it
  already holds the complete, correctly-shaped state dict regardless of world size. Nothing
  in this module is needed for that case — `torch.load` the single file directly.
- **SFT path** (`verl.trainer.sft_trainer`): saves with FSDP `SHARDED_STATE_DICT`. One file
  per rank is written, `model_world_size_<W>_rank_0.pt` ... `rank_<W-1>.pt`, and each holds
  only that rank's local slice (a DTensor) of every tensor. Loading `rank_0` alone silently
  yields a model with `1/W` of every weight actually populated — the shapes present in the
  file are the *local* shard shapes, not the full ones, so a naive `load_state_dict` fails
  loudly on shape mismatch rather than corrupting silently. This module exists to gather all
  ranks into the true full tensor before anything is loaded into a real model.

The presence of more than one `model_world_size_*_rank_*.pt` file in a checkpoint directory
is the reliable signal for which layout you have (see `find_model_shards` below).
"""

from pathlib import Path
from typing import Callable

import logging

logger = logging.getLogger(__name__)


def find_model_shards(actor_dir: Path) -> list[Path]:
    """All `model_world_size_*_rank_*.pt` files in an actor checkpoint dir, rank-ordered."""
    return sorted(
        actor_dir.glob("model_world_size_*_rank_*.pt"),
        key=lambda p: int(p.stem.rsplit("rank_", 1)[1]),
    )


def reconstruct_full_tensors(ckpt: Path, keep_key: Callable[[str], bool]) -> dict:
    """Rebuild full tensors from all per-rank FSDP SHARDED_STATE_DICT shards.

    `keep_key` filters which state-dict keys to reconstruct (e.g. only LoRA keys, or every
    key for a full-parameter checkpoint) — reconstructing the whole model when only the
    adapter is needed wastes memory for no benefit.

    Each value is a DTensor whose placement says how it was split. `Shard(dim=d)` means the
    ranks hold consecutive slices along `d` and must be concatenated in rank order;
    `Replicate()` means every rank holds the same tensor, so rank 0 alone is enough.
    """
    import torch
    import torch.distributed.tensor  # noqa: F401 — registers DTensor for torch.load

    shards = find_model_shards(ckpt)
    if not shards:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt in {ckpt}")
    logger.info("  reading %d rank shard(s)", len(shards))

    per_rank = [torch.load(s, map_location="cpu", weights_only=True, mmap=True) for s in shards]
    keys = [k for k in per_rank[0] if keep_key(k)]

    out = {}
    for k in keys:
        vals = [sd[k] for sd in per_rank]
        placements = getattr(vals[0], "placements", None)
        if placements is None:                       # plain tensor, nothing to gather
            out[k] = vals[0].clone()
            continue
        placement = placements[0]
        if placement.is_shard():
            dim = placement.dim
            full = torch.cat([v.to_local() for v in vals], dim=dim)
            expected = tuple(vals[0].shape)
            if tuple(full.shape) != expected:
                raise ValueError(
                    f"Reconstruction of {k} gave {tuple(full.shape)}, expected {expected}. "
                    "The shard layout is not what this module assumes; do not trust the output."
                )
            out[k] = full
        else:                                        # Replicate / Partial
            out[k] = vals[0].to_local().clone()
    return out
