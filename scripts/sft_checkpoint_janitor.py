"""Convert VERL SFT checkpoints to LoRA adapters *during* training, then delete the shards.

Why this exists
---------------
`verl.trainer.sft_trainer` writes the full FSDP state dict at every save: ~32 GB per
checkpoint for Qwen3-8B, of which ~350 MB is trained weight. Converting only at the end of
the run means the peak on disk is `save_freq` count x 32 GB (~256 GB for a 187-step run at
save_freq=25), and a cancelled job leaves all of it behind — which is exactly what happened
to job 25268454.

This janitor runs alongside training and collapses each checkpoint to its adapter as soon as
the checkpoint is complete, so the peak is one or two checkpoints (~32-64 GB) plus a few GB
of adapters, and a cancelled job leaves almost nothing.

Knowing when a checkpoint is complete
-------------------------------------
Not by guessing from mtimes. `verl/utils/checkpoint/fsdp_checkpoint_manager.py` ends
`save_checkpoint` with a `torch.distributed.barrier()` after every rank has written and
closed its shard; `CheckpointHandler.save_checkpoint` then has rank 0 write
`latest_checkpointed_iteration.txt` via write-to-temp + `os.rename`, which is atomic. So
"the tracker names step N" implies "global_step_N is fully written by all ranks". Only steps
at or below the tracker value are touched.

Deleting is safe because the job sets `trainer.resume_mode=disable`: nothing reads these
checkpoints back. A shard directory is removed only after its adapter exists on disk, and a
failed extraction leaves that checkpoint alone for `finalize_sft_run.py` to retry at the end.

Usage (see 007_train_sft_folded.job, which runs this in the background):
    python scripts/sft_checkpoint_janitor.py \\
        --ckpt-dir <trainer.default_local_dir> --stop-file <path touched when training ends>
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Same extraction code the end-of-run script uses, and the same layout constant, so the two
# cannot disagree about where per-step adapters live.
from finalize_sft_run import (  # noqa: E402
    STEP_ADAPTERS_DIRNAME,
    _step_num,
    dir_size,
    extract_adapter,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] JANITOR %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

TRACKER = "latest_checkpointed_iteration.txt"


def read_tracker(ckpt_dir: Path):
    """Highest step verl has finished writing, or None if it has not saved yet."""
    f = ckpt_dir / TRACKER
    if not f.is_file():
        return None
    try:
        return int(f.read_text().strip())
    except (ValueError, OSError):
        return None  # mid-rename or malformed; the next poll will see it


def sweep(ckpt_dir: Path, lora_rank: int, lora_alpha: int) -> int:
    """Convert every complete, unconverted checkpoint and delete its shards.

    Returns the number of checkpoints collapsed this pass. Never raises for a single bad
    checkpoint: it is left in place and reported, so training is not disturbed and
    finalize_sft_run.py can retry it at the end.
    """
    latest = read_tracker(ckpt_dir)
    if latest is None:
        return 0

    step_adapters = ckpt_dir / STEP_ADAPTERS_DIRNAME
    collapsed = 0

    for d in sorted(ckpt_dir.glob("global_step_*"), key=lambda p: _step_num(p) or -1):
        step = _step_num(d)
        if step is None or not d.is_dir():
            continue
        if step > latest:
            continue  # still being written — the tracker has not reached it yet
        if not any(d.glob("model_world_size_*_rank_*.pt")):
            continue  # created but unwritten (a cancelled save); leave it for finalize

        out = step_adapters / f"step_{step}"
        try:
            if not (out / "adapter_model.safetensors").is_file():
                logger.info("Converting global_step_%d ...", step)
                # Extract into a hidden staging dir and rename into place. A janitor killed
                # mid-write (scancel, node failure) would otherwise leave a truncated
                # adapter_model.safetensors that looks finished to the next pass, and the
                # shards would be deleted against it. The staging name does not match
                # `step_*`, so nothing downstream can pick it up; the rename is atomic
                # within the filesystem.
                staging = step_adapters / f".tmp_step_{step}"
                if staging.exists():
                    shutil.rmtree(staging)
                extract_adapter(d, staging, lora_rank, lora_alpha)
                if out.exists():
                    shutil.rmtree(out)
                staging.rename(out)
        except Exception as e:  # noqa: BLE001 — one bad checkpoint must not stop the run
            logger.error("Extraction failed for global_step_%d (%s). Leaving shards in "
                         "place for finalize_sft_run.py.", step, e)
            continue

        if not (out / "adapter_model.safetensors").is_file():
            logger.error("No adapter written for global_step_%d; not deleting its shards.", step)
            continue

        freed = dir_size(d)
        shutil.rmtree(d)
        collapsed += 1
        logger.info("Collapsed global_step_%d -> %s (freed %.1f GB)", step, out, freed / 1e9)

    return collapsed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="trainer.default_local_dir")
    p.add_argument("--stop-file", required=True,
                   help="Poll until this file exists, then do one final sweep and exit.")
    p.add_argument("--poll-seconds", type=float, default=60.0)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64,
                   help="Overridden by lora_train_meta.json when the checkpoint has it.")
    p.add_argument("--once", action="store_true", help="Single sweep, then exit (for testing).")
    args = p.parse_args()

    ckpt_dir, stop_file = Path(args.ckpt_dir), Path(args.stop_file)
    logger.info("Watching %s (poll %.0fs, stop file %s)", ckpt_dir, args.poll_seconds, stop_file)

    if args.once:
        n = sweep(ckpt_dir, args.lora_rank, args.lora_alpha) if ckpt_dir.is_dir() else 0
        logger.info("Single sweep collapsed %d checkpoint(s).", n)
        return 0

    total = 0
    while not stop_file.exists():
        if ckpt_dir.is_dir():
            total += sweep(ckpt_dir, args.lora_rank, args.lora_alpha)
        time.sleep(args.poll_seconds)

    # Training has stopped, but the last checkpoint may have landed during the final sleep.
    if ckpt_dir.is_dir():
        total += sweep(ckpt_dir, args.lora_rank, args.lora_alpha)
    logger.info("Stop file seen. Collapsed %d checkpoint(s) in total. Exiting.", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
