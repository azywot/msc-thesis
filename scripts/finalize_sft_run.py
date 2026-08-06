"""Turn a finished VERL SFT run into two small PEFT adapters, then delete the shards.

Why this exists
---------------
`verl.trainer.sft_trainer` does NOT write a `lora_adapter/` directory (unlike the RL path in
`fsdp_workers.py`, which the GRPO run relies on). It writes the full FSDP state dict instead:
~33 GB per checkpoint for Qwen3-8B, of which only ~250 MB is actually trained. Keeping every
checkpoint from a 187-step run would cost ~264 GB of scratch to store 2 GB of useful weights.

Two traps this script exists to avoid:

1. **The shards are sharded DTensors, not replicas.** With `world_size=2`, `rank_0` holds only
   half of each LoRA weight — `(32, 4096)` of a `(64, 4096)` `lora_A`. `scripts/merge_lora.py`
   reads `model_world_size_*_rank_0.pt` alone ("the consolidated shard"), which is not true for
   these checkpoints. Reconstruction must concatenate every rank along each tensor's own
   `Shard(dim=...)` placement.
2. **`target_modules="all-linear"` is resolved at wrap time.** Saving the literal string would
   make load-time re-resolution the thing that decides which modules exist. The concrete module
   names are instead derived from the checkpoint keys, so the adapter records what was actually
   trained.

The output is a PEFT adapter directory, the same shape the GRPO adapter has, so an inference
config can point `lora_adapter_path` straight at it.

Usage:
    python scripts/finalize_sft_run.py \\
        --ckpt-dir /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-sft-folded-v1/<run-tag> \\
        --log out/fine_tuning/sft_train/sft_folded_<jobid>_verl.log
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# verl's sft_trainer console line, e.g. "step:25 - val/loss:0.6025107502937317"
_VAL_LOSS_RE = re.compile(r"step:(\d+)\s*-\s*val/loss:([0-9.eE+-]+)")
_LORA_KEY_RE = re.compile(r"\.lora_[AB]\.")
# base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight -> q_proj
_TARGET_MODULE_RE = re.compile(r"\.([A-Za-z0-9_]+)\.lora_[AB]\.")


def parse_val_losses(log_path: Path) -> dict:
    """Map {step: val_loss} from a verl SFT console log. Later lines win (resumed runs)."""
    losses = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = _VAL_LOSS_RE.search(line)
        if m:
            losses[int(m.group(1))] = float(m.group(2))
    return losses


def _reconstruct_full_tensors(ckpt: Path, keep_key) -> dict:
    """Rebuild full tensors from all per-rank FSDP shards.

    Each value is a DTensor whose placement says how it was split. `Shard(dim=d)` means the
    ranks hold consecutive slices along `d` and must be concatenated in rank order;
    `Replicate()` means every rank holds the same tensor, so rank 0 alone is enough.
    """
    import torch
    import torch.distributed.tensor  # noqa: F401 — registers DTensor for torch.load

    shards = sorted(ckpt.glob("model_world_size_*_rank_*.pt"),
                    key=lambda p: int(p.stem.rsplit("rank_", 1)[1]))
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
                    "The shard layout is not what this script assumes; do not trust the output."
                )
            out[k] = full
        else:                                        # Replicate / Partial
            out[k] = vals[0].to_local().clone()
    return out


def extract_adapter(ckpt: Path, out_dir: Path, lora_rank: int, lora_alpha: int) -> dict:
    """Write a PEFT adapter directory from a VERL SFT checkpoint. Returns a small summary."""
    import torch
    from safetensors.torch import save_file

    logger.info("Extracting adapter from %s", ckpt)

    # verl writes the LoRA hyperparameters next to the shards. Prefer them over the CLI:
    # an alpha that disagrees with training silently rescales every adapter weight, and
    # nothing downstream would flag it.
    meta_path = ckpt / "lora_train_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        found_rank, found_alpha = meta.get("r"), meta.get("lora_alpha")
        if found_rank is not None and found_rank != lora_rank:
            logger.warning("rank: using %s from lora_train_meta.json (CLI said %s)",
                           found_rank, lora_rank)
            lora_rank = found_rank
        if found_alpha is not None and found_alpha != lora_alpha:
            logger.warning("alpha: using %s from lora_train_meta.json (CLI said %s)",
                           found_alpha, lora_alpha)
            lora_alpha = found_alpha
        logger.info("  hyperparameters from checkpoint: r=%s alpha=%s", lora_rank, lora_alpha)
    else:
        logger.warning("  no lora_train_meta.json; trusting CLI r=%s alpha=%s",
                       lora_rank, lora_alpha)

    tensors = _reconstruct_full_tensors(ckpt, lambda k: bool(_LORA_KEY_RE.search(k)))
    if not tensors:
        raise ValueError(f"No LoRA keys found in {ckpt} — was this a full-parameter run?")

    # Derive the trained module set from the checkpoint itself rather than re-resolving
    # "all-linear" against the base model at load time.
    targets = sorted({m.group(1) for k in tensors for m in [_TARGET_MODULE_RE.search(k)] if m})
    logger.info("  %d LoRA tensors over %d target modules: %s",
                len(tensors), len(targets), ",".join(targets))

    # PEFT stores single-adapter weights without the adapter name, in bf16.
    renamed = {k.replace(".default.weight", ".weight"): v.to(torch.bfloat16).contiguous()
               for k, v in tensors.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out_dir / "adapter_model.safetensors"))

    from peft import LoraConfig, TaskType
    LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
        target_modules=targets, lora_dropout=0.0, bias="none",
    ).save_pretrained(str(out_dir))

    # Tokenizer/config, so the adapter dir is self-describing.
    hf_dir = ckpt / "huggingface"
    if hf_dir.is_dir():
        for f in hf_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)

    size_mb = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / 1e6
    logger.info("  wrote %s (%.0f MB)", out_dir, size_mb)
    return {"source_checkpoint": str(ckpt), "n_lora_tensors": len(tensors),
            "target_modules": targets, "size_mb": round(size_mb, 1)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="Dir containing global_step_* subdirs.")
    p.add_argument("--log", required=True, help="verl console log, for val/loss per step.")
    p.add_argument("--lora-rank", type=int, default=64, help="MUST match training.")
    p.add_argument("--lora-alpha", type=int, default=64,
                   help="MUST match training — a wrong alpha silently rescales the adapter.")
    p.add_argument("--keep-shards", action="store_true",
                   help="Extract adapters but do not delete the FSDP checkpoints.")
    p.add_argument("--dry-run", action="store_true", help="Report the plan and stop.")
    args = p.parse_args()

    ckpt_dir, log_path = Path(args.ckpt_dir), Path(args.log)
    if not ckpt_dir.is_dir():
        logger.error("No such checkpoint dir: %s", ckpt_dir)
        return 1

    found = sorted((int(d.name.rsplit("_", 1)[1]), d)
                   for d in ckpt_dir.glob("global_step_*") if d.is_dir())
    if not found:
        logger.error("No global_step_* dirs under %s", ckpt_dir)
        return 1

    # A cancelled or crashed run leaves a created-but-unwritten step dir behind. Such a dir
    # must not be selected as "last", or the run's real final adapter is silently skipped.
    steps = [(s, d) for s, d in found if any(d.glob("model_world_size_*_rank_*.pt"))]
    incomplete = [(s, d) for s, d in found if (s, d) not in steps]
    for s, _ in incomplete:
        logger.warning("Ignoring incomplete checkpoint: global_step_%d (no model shards)", s)
    if not steps:
        logger.error("No COMPLETE checkpoints under %s (%d incomplete).", ckpt_dir, len(found))
        return 1

    losses = parse_val_losses(log_path) if log_path.is_file() else {}
    if not losses:
        logger.warning("No 'step:N - val/loss:X' lines in %s — cannot pick a best "
                       "checkpoint; keeping the last one only.", log_path)

    last_step, last_dir = steps[-1]
    scored = [(losses[s], s, d) for s, d in steps if s in losses]
    best_loss, best_step, best_dir = min(scored) if scored else (None, last_step, last_dir)

    logger.info("=" * 70)
    logger.info("Checkpoints found: %s", ", ".join(str(s) for s, _ in steps))
    for s, _ in steps:
        mark = ""
        if s == best_step:
            mark += "  <-- best"
        if s == last_step:
            mark += "  <-- last"
        logger.info("  step %-5d val/loss %s%s", s,
                    f"{losses[s]:.6f}" if s in losses else "(not evaluated)", mark)
    logger.info("Keeping: best=step %s%s, last=step %s",
                best_step, f" (val/loss {best_loss:.6f})" if best_loss is not None else "",
                last_step)
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("--dry-run: nothing written or deleted.")
        return 0

    summary = {"best_step": best_step, "best_val_loss": best_loss, "last_step": last_step,
               "val_losses": {str(k): v for k, v in sorted(losses.items())}, "adapters": {}}

    summary["adapters"]["best"] = extract_adapter(
        best_dir, ckpt_dir / "best_adapter", args.lora_rank, args.lora_alpha)
    if last_step == best_step:
        logger.info("Last checkpoint IS the best; writing one adapter only.")
        summary["adapters"]["last"] = "same as best"
    else:
        summary["adapters"]["last"] = extract_adapter(
            last_dir, ckpt_dir / "last_adapter", args.lora_rank, args.lora_alpha)

    (ckpt_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", ckpt_dir / "selection.json")

    if args.keep_shards:
        logger.info("--keep-shards: leaving %d checkpoint dir(s) in place.", len(steps))
        return 0

    # Only ever delete after both adapters exist on disk — a failed extraction must not
    # take the shards with it.
    for name in ("best_adapter", "last_adapter"):
        d = ckpt_dir / name
        if name == "last_adapter" and last_step == best_step:
            continue
        if not (d / "adapter_model.safetensors").is_file():
            logger.error("Refusing to delete shards: %s is missing.", d)
            return 1

    freed = 0
    for _, d in steps + incomplete:
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d)
    logger.info("Deleted %d checkpoint dir(s), freed %.1f GB.",
                len(steps) + len(incomplete), freed / 1e9)
    logger.info("Kept: %s", ", ".join(
        str(p) for p in sorted(ckpt_dir.iterdir()) if p.is_dir() or p.suffix == ".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
