"""Turn a finished VERL SFT run into two small checkpoints, then delete the shards.

Why this exists
---------------
`verl.trainer.sft_trainer` does NOT write a `lora_adapter/` directory (unlike the RL path in
`fsdp_workers.py`, which the GRPO run relies on). It writes the full FSDP state dict instead:
~33 GB per checkpoint for Qwen3-8B. For a LoRA run only ~250 MB of that is actually trained,
so keeping every checkpoint from a 187-step run would cost ~264 GB of scratch to store 2 GB of
useful weights (`--mode lora`, the default). For a full-parameter run every byte of it is
trained (`--mode full`) — there is no adapter to extract, but the same per-checkpoint size and
the same "keep only best + last" logic still apply, so this script handles both.

Two traps `--mode lora` exists to avoid:

1. **The shards are sharded DTensors, not replicas.** With `world_size=2`, `rank_0` holds only
   half of each LoRA weight — `(32, 4096)` of a `(64, 4096)` `lora_A`. `scripts/merge_lora.py`'s
   single-rank-file fallback assumes a *consolidated* shard, which is not true for these
   checkpoints. Reconstruction (`verl_ext.checkpoint_utils.reconstruct_full_tensors`) must
   concatenate every rank along each tensor's own `Shard(dim=...)` placement.
2. **`target_modules="all-linear"` is resolved at wrap time.** Saving the literal string would
   make load-time re-resolution the thing that decides which modules exist. The concrete module
   names are instead derived from the checkpoint keys, so the adapter records what was actually
   trained.

`--mode lora` output is a PEFT adapter directory, the same shape the GRPO adapter has, so an
inference config can point `lora_adapter_path` straight at it. `--mode full` output is a
complete HuggingFace model directory (`config.json` + sharded `.safetensors` + tokenizer), so
an inference config can point `path_or_id` straight at it with no `lora_adapter_path` at all.

Usage:
    python scripts/finalize_sft_run.py \\
        --ckpt-dir /scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-sft-folded-v1/<run-tag> \\
        --log out/fine_tuning/sft_train/sft_folded_<jobid>_verl.log

    python scripts/finalize_sft_run.py --mode full \\
        --ckpt-dir /scratch-shared/$USER/fine_tuning/full_ft/qwen3-8b-sft-full-v1/<run-tag> \\
        --log out/fine_tuning/sft_train/sft_full_<jobid>_verl.log --base-model Qwen/Qwen3-8B
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

from verl_ext.checkpoint_utils import reconstruct_full_tensors  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# verl's sft_trainer console line, e.g. "step:25 - val/loss:0.6025107502937317"
_VAL_LOSS_RE = re.compile(r"step:(\d+)\s*-\s*val/loss:([0-9.eE+-]+)")
_LORA_KEY_RE = re.compile(r"\.lora_[AB]\.")
# base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight -> q_proj
_TARGET_MODULE_RE = re.compile(r"\.([A-Za-z0-9_]+)\.lora_[AB]\.")
# Where sft_checkpoint_janitor.py puts adapters/checkpoints it extracts while training runs.
STEP_ADAPTERS_DIRNAME = "step_adapters"
# --mode full analog of STEP_ADAPTERS_DIRNAME.
STEP_CHECKPOINTS_DIRNAME = "step_checkpoints"


def _step_num(path: Path):
    """Step number from `global_step_25` or `step_25`, or None if the name has no integer."""
    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None


def dir_size(path: Path) -> int:
    """Total bytes of files under `path`, for reporting what a deletion reclaimed."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def parse_val_losses(log_path: Path) -> dict:
    """Map {step: val_loss} from a verl SFT console log. Later lines win (resumed runs)."""
    losses = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = _VAL_LOSS_RE.search(line)
        if m:
            losses[int(m.group(1))] = float(m.group(2))
    return losses


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

    tensors = reconstruct_full_tensors(ckpt, lambda k: bool(_LORA_KEY_RE.search(k)))
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


def extract_full_checkpoint(ckpt: Path, out_dir: Path, base_model: str) -> dict:
    """Write a complete HuggingFace model directory from a full-parameter VERL SFT checkpoint.

    Unlike `extract_adapter`, every key in the checkpoint is trained weight — there is no
    filtering, `reconstruct_full_tensors` is called with `keep_key=lambda k: True`. The base
    architecture is instantiated on the meta device (no real memory for the random init) so the
    only real allocation is the reconstructed state dict itself (~16 GB bf16 for Qwen3-8B), then
    `load_state_dict(..., assign=True)` binds the real tensors in place rather than copying into
    already-allocated ones. `save_pretrained` writes sharded safetensors + the index file, the
    exact layout `AutoModelForCausalLM.from_pretrained` and vLLM's `path_or_id` expect.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info("Extracting full checkpoint from %s", ckpt)
    tensors = reconstruct_full_tensors(ckpt, lambda k: True)
    if not tensors:
        raise ValueError(f"No model keys found in {ckpt}")
    tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in tensors.items()}

    # verl writes the tokenizer/config into every checkpoint's huggingface/ subdir. Prefer that
    # over --base-model so the config actually used at train time (not a caller's guess) wins.
    hf_dir = ckpt / "huggingface"
    config_source = str(hf_dir) if hf_dir.is_dir() else base_model
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, trust_remote_code=True)

    result = model.load_state_dict(tensors, strict=False, assign=True)
    if result.missing_keys:
        logger.warning("  %d missing key(s) after load: %s", len(result.missing_keys),
                       result.missing_keys[:5])
    if result.unexpected_keys:
        logger.warning("  %d unexpected key(s) after load: %s", len(result.unexpected_keys),
                       result.unexpected_keys[:5])
    # Tied weights (e.g. lm_head <-> embed_tokens when config.tie_word_embeddings) are not
    # written to the checkpoint at all, so they land in missing_keys and stay on the meta
    # device unless re-tied here — save_pretrained would otherwise fail on a meta tensor.
    model.tie_weights()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    if hf_dir.is_dir():
        for f in hf_dir.iterdir():
            if f.is_file() and not (out_dir / f.name).exists():
                shutil.copy2(f, out_dir / f.name)
    else:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True).save_pretrained(str(out_dir))

    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    logger.info("  wrote %s (%.0f MB)", out_dir, size_mb)
    return {"source_checkpoint": str(ckpt), "n_tensors": len(tensors), "size_mb": round(size_mb, 1)}


# --mode-dependent naming, so main() below stays a single code path for both.
_DONE_MARKER = {"lora": "adapter_model.safetensors", "full": "config.json"}
_OUT_NAMES = {"lora": ("best_adapter", "last_adapter"), "full": ("best_checkpoint", "last_checkpoint")}
_STEP_DIRNAME = {"lora": STEP_ADAPTERS_DIRNAME, "full": STEP_CHECKPOINTS_DIRNAME}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="Dir containing global_step_* subdirs.")
    p.add_argument("--log", required=True, help="verl console log, for val/loss per step.")
    p.add_argument("--mode", choices=["lora", "full"], default="lora",
                   help="'lora' extracts a PEFT adapter (default); 'full' writes a complete "
                        "HuggingFace model directory from a full-parameter run.")
    p.add_argument("--lora-rank", type=int, default=64, help="--mode lora only. MUST match training.")
    p.add_argument("--lora-alpha", type=int, default=64,
                   help="--mode lora only. MUST match training — a wrong alpha silently "
                        "rescales the adapter.")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B",
                   help="--mode full only. Fallback config/tokenizer source if the checkpoint's "
                        "own huggingface/ subdir is missing.")
    p.add_argument("--keep-shards", action="store_true",
                   help="Extract checkpoints but do not delete the FSDP shards.")
    p.add_argument("--dry-run", action="store_true", help="Report the plan and stop.")
    args = p.parse_args()

    ckpt_dir, log_path = Path(args.ckpt_dir), Path(args.log)
    if not ckpt_dir.is_dir():
        logger.error("No such checkpoint dir: %s", ckpt_dir)
        return 1

    done_marker = _DONE_MARKER[args.mode]
    best_name, last_name = _OUT_NAMES[args.mode]
    step_dirname = _STEP_DIRNAME[args.mode]

    # Candidates come from two places, because sft_checkpoint_janitor.py collapses
    # checkpoints *during* the run: steps it already converted live in step_dirname/step_N,
    # and anything it did not reach is still an FSDP shard dir.
    pre_extracted = {}
    step_adapters_dir = ckpt_dir / step_dirname
    if step_adapters_dir.is_dir():
        for d in step_adapters_dir.glob("step_*"):
            s = _step_num(d)
            if s is not None and (d / done_marker).is_file():
                pre_extracted[s] = d

    found = sorted((s, d) for d in ckpt_dir.glob("global_step_*") if d.is_dir()
                   for s in [_step_num(d)] if s is not None)
    # A cancelled or crashed run leaves a created-but-unwritten step dir behind. Such a dir
    # must not be selected as "last", or the run's real final adapter is silently skipped.
    with_shards = [(s, d) for s, d in found if any(d.glob("model_world_size_*_rank_*.pt"))]
    incomplete = [(s, d) for s, d in found if (s, d) not in with_shards]
    for s, _ in incomplete:
        logger.warning("Ignoring incomplete checkpoint: global_step_%d (no model shards)", s)

    # step -> (kind, path). An already-extracted checkpoint wins over leftover shards for the
    # same step: it is the cheaper source and was produced by this same code.
    candidates = {s: ("shards", d) for s, d in with_shards}
    candidates.update({s: ("extracted", d) for s, d in pre_extracted.items()})
    if not candidates:
        logger.error("Nothing to finalize under %s (%d incomplete checkpoint dir(s)).",
                     ckpt_dir, len(incomplete))
        return 1

    losses = parse_val_losses(log_path) if log_path.is_file() else {}
    if not losses:
        logger.warning("No 'step:N - val/loss:X' lines in %s — cannot pick a best "
                       "checkpoint; keeping the last one only.", log_path)

    ordered = sorted(candidates)
    last_step = ordered[-1]
    scored = [(losses[s], s) for s in ordered if s in losses]
    best_loss, best_step = min(scored) if scored else (None, last_step)

    logger.info("=" * 70)
    logger.info("Steps available: %s (%d already collapsed during training)",
                ", ".join(str(s) for s in ordered), len(pre_extracted))
    for s in ordered:
        mark = ""
        if s == best_step:
            mark += "  <-- best"
        if s == last_step:
            mark += "  <-- last"
        logger.info("  step %-5d [%-7s] val/loss %s%s", s, candidates[s][0],
                    f"{losses[s]:.6f}" if s in losses else "(not evaluated)", mark)
    logger.info("Keeping: best=step %s%s, last=step %s",
                best_step, f" (val/loss {best_loss:.6f})" if best_loss is not None else "",
                last_step)
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("--dry-run: nothing written or deleted.")
        return 0

    def _materialise(step: int, out_dir: Path) -> dict:
        """Put step's checkpoint at out_dir, extracting from shards only if needed."""
        kind, src = candidates[step]
        if kind == "extracted":
            logger.info("Reusing checkpoint already extracted during training: %s", src)
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(src, out_dir)
            size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
            return {"source_checkpoint": str(src), "extracted_during_training": True,
                    "size_mb": round(size_mb, 1)}
        if args.mode == "lora":
            return extract_adapter(src, out_dir, args.lora_rank, args.lora_alpha)
        return extract_full_checkpoint(src, out_dir, args.base_model)

    summary = {"best_step": best_step, "best_val_loss": best_loss, "last_step": last_step,
               "val_losses": {str(k): v for k, v in sorted(losses.items())}, "checkpoints": {}}

    summary["checkpoints"]["best"] = _materialise(best_step, ckpt_dir / best_name)
    if last_step == best_step:
        logger.info("Last checkpoint IS the best; writing one copy only.")
        summary["checkpoints"]["last"] = "same as best"
    else:
        summary["checkpoints"]["last"] = _materialise(last_step, ckpt_dir / last_name)

    (ckpt_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", ckpt_dir / "selection.json")

    if args.keep_shards:
        logger.info("--keep-shards: leaving %d checkpoint dir(s) in place.", len(with_shards))
        return 0

    # Only ever delete after both outputs exist on disk — a failed extraction must not
    # take the shards with it.
    for name in (best_name, last_name):
        d = ckpt_dir / name
        if name == last_name and last_step == best_step:
            continue
        if not (d / done_marker).is_file():
            logger.error("Refusing to delete shards: %s is missing.", d)
            return 1

    freed = 0
    for _, d in with_shards + incomplete:
        freed += dir_size(d)
        shutil.rmtree(d)
    if step_adapters_dir.is_dir():
        freed += dir_size(step_adapters_dir)
        shutil.rmtree(step_adapters_dir)
    logger.info("Deleted %d checkpoint dir(s) and %d per-step extraction(s), freed %.1f GB.",
                len(with_shards) + len(incomplete), len(pre_extracted), freed / 1e9)
    logger.info("Kept: %s", ", ".join(
        str(p) for p in sorted(ckpt_dir.iterdir()) if p.is_dir() or p.suffix == ".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
