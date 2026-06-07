"""Print trainable / total parameter counts for a LoRA-wrapped model.

Reads the same fields ``005_train.job`` reads from
``experiments/configs/fine_tuning/config.yaml`` (env.BASE_MODEL,
lora.rank, lora.alpha, lora.target_modules) and reproduces the wrap that
``verl/workers/fsdp_workers.py`` does at training start.

Usage:
    python scripts/count_lora_params.py
    python scripts/count_lora_params.py --config experiments/configs/fine_tuning/config.yaml
    python scripts/count_lora_params.py --base-model Qwen/Qwen3-8B \\
        --lora-rank 64 --lora-alpha 64 --lora-target-modules all-linear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM


def _from_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    env = cfg.get("env", {}) or {}
    lora = cfg.get("lora", {}) or {}
    return {
        "base_model": env.get("BASE_MODEL"),
        "lora_rank": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "lora_target_modules": lora.get("target_modules"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/fine_tuning/config.yaml")
    p.add_argument("--base-model")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--lora-target-modules")
    args = p.parse_args()

    defaults = _from_config(Path(args.config)) if Path(args.config).is_file() else {}
    base_model = args.base_model or defaults.get("base_model")
    lora_rank = args.lora_rank or defaults.get("lora_rank")
    lora_alpha = args.lora_alpha or defaults.get("lora_alpha")
    target_modules = args.lora_target_modules or defaults.get("lora_target_modules")

    if not all([base_model, lora_rank, lora_alpha, target_modules]):
        print("Missing one of: base_model, lora_rank, lora_alpha, target_modules", file=sys.stderr)
        return 1

    print(f"Base model:      {base_model}")
    print(f"LoRA rank:       {lora_rank}")
    print(f"LoRA alpha:      {lora_alpha}")
    print(f"Target modules:  {target_modules}")
    print()

    # Architecture-only init on the meta device: no weight download, ~zero RAM.
    # numel() on meta tensors still returns shape-based counts, so PEFT's
    # print_trainable_parameters() is exact.
    cfg_obj = AutoConfig.from_pretrained(base_model)
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(cfg_obj, torch_dtype=torch.bfloat16)

    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base, cfg)
    peft_model.print_trainable_parameters()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
