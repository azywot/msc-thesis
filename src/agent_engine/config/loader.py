"""Configuration loader for YAML-based experiment configs.

This module provides utilities to load and validate experiment configurations
from YAML files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from ..models.base import ModelConfig, ModelFamily
from .schema import DatasetConfig, ExperimentConfig, SlurmConfig, ThinkingMode, ToolsConfig


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        ExperimentConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty config file: {path}")

    # Pre-process models: construct ModelConfig instances from dicts
    if "models" in data:
        data["models"] = _load_models(data["models"])

    # Convert thinking_mode string to enum
    if "thinking_mode" in data:
        thinking_str = data["thinking_mode"]
        if not isinstance(thinking_str, str):
            raise ValueError(f"thinking_mode must be a string (NO, ORCHESTRATOR_ONLY, SUBAGENTS_ONLY, ALL), got {type(thinking_str).__name__}")
        try:
            data["thinking_mode"] = ThinkingMode(thinking_str)
        except ValueError:
            raise ValueError(f"Invalid thinking mode: {thinking_str}. Must be one of: NO, ORCHESTRATOR_ONLY, SUBAGENTS_ONLY, ALL")

    config = ExperimentConfig.model_validate(data)

    # Propagate top-level seed to any model that doesn't declare its own.
    for model_cfg in config.models.values():
        if model_cfg.seed is None:
            model_cfg.seed = config.seed

    return config


def _load_models(models_data: Dict[str, Any]) -> Dict[str, ModelConfig]:
    """Load model configurations from dict.

    Args:
        models_data: Dictionary of role -> model config

    Returns:
        Dictionary of role -> ModelConfig
    """
    return {role: ModelConfig.model_validate(cfg) for role, cfg in models_data.items()}


def save_experiment_config(config: ExperimentConfig, path: Path):
    """Save experiment configuration to YAML file.

    Args:
        config: ExperimentConfig instance
        path: Path to save YAML file
    """
    # Convert to dict
    data = _config_to_dict(config)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Convert ExperimentConfig to dictionary for YAML serialization.

    Args:
        config: ExperimentConfig instance

    Returns:
        Dictionary representation
    """
    data = config.model_dump()

    # Convert non-JSON-serializable types for YAML
    data["thinking_mode"] = config.thinking_mode.value
    data["output_dir"] = str(config.output_dir)
    data["cache_dir"] = str(config.cache_dir)

    for role in list(data.get("models", {})):
        mcfg = data["models"][role]
        mcfg["family"] = config.models[role].family.value
        # Only persist fields that differ from auto-derived defaults.
        for key in ["gpu_ids", "tensor_parallel_size", "gpu_memory_utilization"]:
            if mcfg.get(key) is None:
                mcfg.pop(key, None)
        if mcfg.get("backend") == "vllm":
            mcfg.pop("backend", None)

    if data.get("dataset"):
        data["dataset"]["data_dir"] = str(config.dataset.data_dir)

    if config.wandb_project:
        data["wandb_project"] = config.wandb_project

    return data
