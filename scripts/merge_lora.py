"""Merge LoRA adapter weights from a VERL FSDP checkpoint into the base model.

VERL 0.7.1 saves actor checkpoints to:
    <ckpt_dir>/global_step_<N>/actor/
        ├── model_world_size_<W>_rank_*.pt            # FSDP full-state-dict (base + LoRA layers)
        └── lora_adapter/                             # written only when USE_LORA=true
            ├── adapter_model.safetensors             # PEFT-format LoRA tensors
            └── adapter_config.json                   # rank/alpha/target_modules baked in

This script prefers the PEFT-format ``lora_adapter/`` dir when present — no CLI
hyperparams needed, since they're already in ``adapter_config.json``.
If the adapter dir is missing (e.g. an older checkpoint), it falls back to the
FSDP shard path and reconstructs the PEFT model from CLI args.

For full-parameter training (USE_LORA=false) there are no LoRA keys; the script
saves the base model with the FSDP-shard state dict applied.

Default LoRA adapter root on Snellius (set by ``scripts/launch_verl.py`` when
USE_LORA=true): ``/scratch-shared/$USER/fine_tuning/lora_adapters/<experiment>/<run-tag>/``.

Usage:
    python scripts/merge_lora.py \\
        --checkpoint /scratch-shared/$USER/fine_tuning/lora_adapters/<exp>/<run>/global_step_<N> \\
        --base-model Qwen/Qwen3-8B \\
        --output-dir <output_path>

    # Fallback path (no lora_adapter/ subdir) — must pass LoRA hyperparams that match training:
    python scripts/merge_lora.py \\
        --checkpoint /path/to/global_step_5 \\
        --base-model Qwen/Qwen3-8B \\
        --output-dir merged_model/ \\
        --lora-rank 64 --lora-alpha 64 --lora-target-modules all-linear
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_model_shard(actor_dir: Path) -> Path:
    """Return the rank-0 consolidated shard from the actor checkpoint directory.

    VERL saves with FSDP FULL_STATE_DICT (rank0_only=True), producing a single file
    named model_world_size_<W>_rank_0.pt regardless of training world size.
    """
    candidates = sorted(actor_dir.glob("model_world_size_*_rank_0.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model_world_size_*_rank_0.pt in {actor_dir}\n"
            "Check that the checkpoint completed successfully (job 009/010 step 5)."
        )
    if len(candidates) > 1:
        print(f"  Warning: multiple rank-0 shards found; using {candidates[0]}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters from VERL checkpoint into HuggingFace model."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to global_step_N directory (contains actor/ subdir)",
    )
    parser.add_argument(
        "--base-model", required=True,
        help="HuggingFace model ID or local path for the base architecture",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write the merged (or full-param) HuggingFace model",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=64,
        help="Fallback only — used if actor/lora_adapter/ is missing.",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64,
        help="Fallback only — used if actor/lora_adapter/ is missing. "
             "MUST match training (see lora.alpha in experiments/configs/fine_tuning/config.yaml).",
    )
    parser.add_argument(
        "--lora-target-modules", default="all-linear",
        help="Fallback only — comma-separated list or 'all-linear'.",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="dtype for model loading (default: bfloat16)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    actor_dir = ckpt_path / "actor"
    output_dir = Path(args.output_dir)

    if not actor_dir.exists():
        print(f"ERROR: actor directory not found: {actor_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    adapter_dir = actor_dir / "lora_adapter"
    has_peft_adapter = (adapter_dir / "adapter_model.safetensors").exists() and (
        adapter_dir / "adapter_config.json"
    ).exists()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True,
    )

    if has_peft_adapter:
        print(f"Found verl-written PEFT adapter at: {adapter_dir}")
        _merge_from_peft_dir(base_model, adapter_dir, output_dir, dtype_str=args.dtype)
    else:
        # Older checkpoint or full-FT — read the FSDP shard.
        shard_path = find_model_shard(actor_dir)
        print(f"No lora_adapter/ found; falling back to FSDP shard: {shard_path}")
        state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
        print(f"  {len(state_dict)} keys")

        lora_keys = [k for k in state_dict if ".lora_A." in k or ".lora_B." in k]
        is_lora = bool(lora_keys)
        print(f"  Type: {'LoRA' if is_lora else 'full-parameter'} "
              f"({len(lora_keys)} adapter keys)")

        if is_lora:
            _merge_from_shard(base_model, state_dict, lora_keys, args, output_dir)
        else:
            _save_full_param(base_model, state_dict, output_dir)

    # Copy tokenizer from checkpoint's huggingface/ subdir if present.
    hf_dir = actor_dir / "huggingface"
    if hf_dir.exists():
        print(f"Copying tokenizer from checkpoint ({hf_dir})")
        for f in hf_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)
    else:
        print(f"Saving tokenizer from {args.base_model}")
        AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        ).save_pretrained(output_dir)

    print(f"\nSaved to: {output_dir}")
    print(f"Load with: AutoModelForCausalLM.from_pretrained('{output_dir}')")


def _merge_from_peft_dir(
    base_model: "AutoModelForCausalLM",
    adapter_dir: Path,
    output_dir: Path,
    dtype_str: str,
) -> None:
    """Preferred path: verl 0.7.1 writes a PEFT-format adapter dir.

    Hyperparams (rank, alpha, target_modules) live in adapter_config.json so the
    caller doesn't have to pass them — eliminates a class of silent-merge bugs.
    """
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print("Merging LoRA adapter into base weights ...")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    print(f"  Saved merged model ({dtype_str})")


def _merge_from_shard(
    base_model: "AutoModelForCausalLM",
    state_dict: dict,
    lora_keys: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Fallback path: reconstruct PEFT model from CLI args and load the FSDP shard."""
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = args.lora_target_modules
    if "," in target_modules:
        target_modules = [m.strip() for m in target_modules.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    print(f"LoRA config (from CLI args): rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"target_modules={target_modules}")
    print("  WARNING: these must match training — mismatched alpha silently rescales the merge.")

    peft_model = get_peft_model(base_model, lora_config)

    # VERL saves FSDP state dict with HF-style keys (no base_model.model. prefix).
    # PeftModel.load_state_dict expects keys prefixed with "base_model.model.".
    sample = lora_keys[0]
    if not sample.startswith("base_model.model."):
        print("  Normalising checkpoint keys: adding 'base_model.model.' prefix")
        state_dict = {"base_model.model." + k: v for k, v in state_dict.items()}

    result = peft_model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"  Unexpected keys after load ({len(result.unexpected_keys)}): "
              f"{result.unexpected_keys[:5]}")

    print("Merging LoRA adapter into base weights ...")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    print(f"  Saved merged model ({args.dtype})")


def _save_full_param(
    model: "AutoModelForCausalLM",
    state_dict: dict,
    output_dir: Path,
) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing keys")
        for k in result.missing_keys[:5]:
            print(f"    {k}")
    if result.unexpected_keys:
        print(f"  WARNING: {len(result.unexpected_keys)} unexpected keys")
    print("Saving full-parameter model ...")
    model.save_pretrained(output_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
