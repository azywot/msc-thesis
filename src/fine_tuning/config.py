"""Configuration dataclass for the orchestrator fine-tuning pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class FinetuningConfig:
    """All hyperparameters and paths for one fine-tuning run.

    Required fields must be supplied; optional fields carry sensible defaults
    that mirror AgentFlow's train/config.yaml.
    """

    # ── required ────────────────────────────────────────────────────────────
    base_model: str                # HF model ID, e.g. "Qwen/Qwen3-8B"
    train_data: str                # path to combined_train.parquet
    val_data: str                  # path to validation parquet
    output_dir: str                # checkpoint + config output directory

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_target_modules: str = "all-linear"

    # ── generation temperatures ──────────────────────────────────────────────
    train_temperature: float = 0.7
    test_temperature: float = 0.0

    # ── hardware ────────────────────────────────────────────────────────────
    n_gpus: int = 4
    rollout_tp_size: int = 2        # tensor parallel size for vLLM rollout

    # ── reproducibility ─────────────────────────────────────────────────────
    seed: int = 42

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_project: str = "cosmas-rl-finetuning"
    wandb_run_name: str = "qwen3-8b-grpo-search-math"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuningConfig":
        """Load config from a YAML file. Unknown keys are silently ignored."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
