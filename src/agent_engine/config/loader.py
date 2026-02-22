"""Configuration loader for YAML-based experiment configs.

This module provides utilities to load and validate experiment configurations
from YAML files.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from ..models.base import ModelConfig, ModelFamily
from .schema import DatasetConfig, ExperimentConfig, ThinkingMode, ToolsConfig


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

    # Convert nested configs
    if "models" in data:
        data["models"] = _load_models(data["models"])

    if "tools" in data:
        data["tools"] = ToolsConfig(**data["tools"])

    if "dataset" in data:
        data["dataset"] = DatasetConfig(**data["dataset"])

    # Convert thinking_mode string to enum
    if "thinking_mode" in data:
        thinking_str = data["thinking_mode"]
        if not isinstance(thinking_str, str):
            raise ValueError(f"thinking_mode must be a string (NO, ORCHESTRATOR_ONLY, SUBAGENTS_ONLY, ALL), got {type(thinking_str).__name__}")
        try:
            data["thinking_mode"] = ThinkingMode(thinking_str)
        except ValueError:
            raise ValueError(f"Invalid thinking mode: {thinking_str}. Must be one of: NO, ORCHESTRATOR_ONLY, SUBAGENTS_ONLY, ALL")

    # Create ExperimentConfig
    return ExperimentConfig(**data)


def _load_models(models_data: Dict[str, Any]) -> Dict[str, ModelConfig]:
    """Load model configurations from dict.

    Args:
        models_data: Dictionary of role -> model config

    Returns:
        Dictionary of role -> ModelConfig
    """
    models = {}
    for role, config_data in models_data.items():
        # Convert family string to enum
        if "family" in config_data:
            family_str = config_data["family"]
            try:
                config_data["family"] = ModelFamily(family_str)
            except ValueError:
                raise ValueError(f"Invalid model family: {family_str}")

        models[role] = ModelConfig(**config_data)

    return models


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
    data = {
        "name": config.name,
        "description": config.description,
        "models": {},
        "tools": {
            "enabled_tools": config.tools.enabled_tools,
            "direct_tool_call": config.tools.direct_tool_call,
            "max_search_limit": config.tools.max_search_limit,
            "top_k_results": config.tools.top_k_results,
            "max_doc_len": config.tools.max_doc_len,
        },
        "max_turns": config.max_turns,
        "seed": config.seed,
        "thinking_mode": config.thinking_mode.value,  # Convert enum to string
        "output_dir": str(config.output_dir),
        "use_wandb": config.use_wandb,
        "cache_dir": str(config.cache_dir),
    }

    # Add models
    for role, model_config in config.models.items():
        model_dict = {
            "name": model_config.name,
            "family": model_config.family.value,
            "path_or_id": model_config.path_or_id,
            "role": model_config.role,
            "max_model_len": model_config.max_model_len,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "repetition_penalty": model_config.repetition_penalty,
            "supports_thinking": model_config.supports_thinking,
            "seed": model_config.seed,
        }
        # Only persist resource fields when they carry non-default information.
        if model_config.gpu_ids is not None:
            model_dict["gpu_ids"] = model_config.gpu_ids
        if model_config.tensor_parallel_size is not None:
            model_dict["tensor_parallel_size"] = model_config.tensor_parallel_size
        if model_config.gpu_memory_utilization != 0.95:
            model_dict["gpu_memory_utilization"] = model_config.gpu_memory_utilization
        data["models"][role] = model_dict

    # Add dataset if present
    if config.dataset:
        data["dataset"] = {
            "name": config.dataset.name,
            "split": config.dataset.split,
            "data_dir": str(config.dataset.data_dir),
            "subset_num": config.dataset.subset_num,
        }

    # Add W&B project if present
    if config.wandb_project:
        data["wandb_project"] = config.wandb_project

    return data
