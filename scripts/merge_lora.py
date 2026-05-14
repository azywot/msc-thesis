"""Merge LoRA adapter weights from a VERL FSDP checkpoint into the base model.

VERL saves actor checkpoints as model_world_size_W_rank_0.pt in:
    <ckpt_dir>/global_step_<N>/actor/

For LoRA training (USE_LORA=true), the checkpoint state dict contains both base
weights and LoRA adapter weights (keys with .lora_A. / .lora_B.).

For full-parameter training (USE_LORA=false), only standard HuggingFace keys are
present and the script saves the checkpoint directly without merging.

Usage:
    python scripts/merge_lora.py \\
        --checkpoint experiments/results/training/<exp>/<run>/global_step_<N> \\
        --base-model Qwen/Qwen3-8B \\
        --output-dir <output_path>

    # LoRA hyperparams (must match training config):
    python scripts/merge_lora.py \\
        --checkpoint /path/to/global_step_5 \\
        --base-model Qwen/Qwen3-8B \\
        --output-dir merged_model/ \\
        --lora-rank 64 --lora-alpha 16 --lora-target-modules all-linear
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
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-target-modules", default="all-linear",
                        help="Comma-separated list or 'all-linear'")
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

    shard_path = find_model_shard(actor_dir)
    print(f"Loading checkpoint: {shard_path}")
    state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
    print(f"  {len(state_dict)} keys")

    lora_keys = [k for k in state_dict if ".lora_A." in k or ".lora_B." in k]
    is_lora = bool(lora_keys)
    print(f"  Type: {'LoRA' if is_lora else 'full-parameter'} "
          f"({len(lora_keys)} adapter keys)")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True,
    )

    if is_lora:
        _merge_lora(model, state_dict, lora_keys, args, output_dir)
    else:
        _save_full_param(model, state_dict, output_dir)

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


def _merge_lora(
    base_model: "AutoModelForCausalLM",
    state_dict: dict,
    lora_keys: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = args.lora_target_modules
    # Accept comma-separated list as well as the bare "all-linear" string.
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
    print(f"LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, "
          f"target_modules={target_modules}")

    peft_model = get_peft_model(base_model, lora_config)

    # VERL saves FSDP state dict with HF-style keys (no base_model.model. prefix).
    # PeftModel.load_state_dict expects keys prefixed with "base_model.model.".
    # Add the prefix so load_state_dict can match PeftModel's parameter paths.
    sample = lora_keys[0]
    if not sample.startswith("base_model.model."):
        print("  Normalising checkpoint keys: adding 'base_model.model.' prefix")
        state_dict = {"base_model.model." + k: v for k, v in state_dict.items()}

    result = peft_model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        print(f"  Unexpected keys after load ({len(result.unexpected_keys)}): "
              f"{result.unexpected_keys[:5]}")
    # missing_keys are non-LoRA layers initialised from the base model — expected.

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
