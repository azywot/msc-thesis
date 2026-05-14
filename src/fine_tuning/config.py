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
    that mirror AgentFlow's train/config.yaml (full-parameter training, no LoRA).
    Set use_lora=True to enable PEFT LoRA adapters.
    """

    # ── required ────────────────────────────────────────────────────────────
    base_model: str                # HF model ID, e.g. "Qwen/Qwen3-8B"
    train_data: str                # path to combined_train.parquet
    val_data: str                  # path to validation parquet
    output_dir: str                # checkpoint + config output directory

    # ── LoRA (only used when use_lora=True) ─────────────────────────────────
    use_lora: bool = False         # default False = full-parameter training (matches AgentFlow)
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
        """Load config from a YAML file.

        Reads flat top-level keys, flattens the nested ``lora:`` section into
        ``lora_rank``/``lora_alpha``/``lora_target_modules``, and reads
        ``USE_LORA`` from the ``env:`` section into ``use_lora``.
        Unknown keys are silently ignored.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}

        # Flatten nested lora: section → lora_rank, lora_alpha, lora_target_modules
        lora_section = data.get("lora", {}) or {}
        if "rank" in lora_section:
            filtered["lora_rank"] = int(lora_section["rank"])
        if "alpha" in lora_section:
            filtered["lora_alpha"] = int(lora_section["alpha"])
        if "target_modules" in lora_section:
            filtered["lora_target_modules"] = str(lora_section["target_modules"])

        # Read USE_LORA from env: section
        env_section = data.get("env", {}) or {}
        if "USE_LORA" in env_section:
            val = str(env_section["USE_LORA"]).strip().lower()
            filtered["use_lora"] = val in ("1", "true", "yes", "on")

        return cls(**filtered)
