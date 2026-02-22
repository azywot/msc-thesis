"""Configuration schema for agent_engine experiments.

This module defines dataclasses for complete experiment configuration,
replacing scattered argparse arguments with declarative YAML configs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ..models.base import ModelConfig


class ThinkingMode(Enum):
    """Thinking mode configuration.

    Controls which components use thinking mode (for models that support it,
    like Qwen3).
    """
    NO = "NO"                      # No thinking enabled
    ORCHESTRATOR_ONLY = "ORCHESTRATOR_ONLY"  # Only orchestrator uses thinking
    SUBAGENTS_ONLY = "SUBAGENTS_ONLY"  # Only tool sub-agents use thinking
    ALL = "ALL"                    # Both orchestrator and sub-agents use thinking


@dataclass
class ToolsConfig:
    """Tool configuration.

    Attributes:
        enabled_tools: List of tool names to enable
        direct_tool_call: If True, tools execute directly (no sub-agent LLMs).
                         If False, tools use sub-agent LLMs for processing.
        max_search_limit: Maximum number of web searches allowed
        top_k_results: Number of search results to return
        max_doc_len: int = Maximum document length in characters
    """
    enabled_tools: List[str] = field(default_factory=lambda: ["web_search", "code_generator"])
    direct_tool_call: bool = True  # True = direct execution, False = sub-agent mode
    max_search_limit: int = 10
    top_k_results: int = 5
    max_doc_len: int = 3000


@dataclass
class DatasetConfig:
    """Dataset configuration.

    Attributes:
        name: Dataset name (gaia, gpqa, math500, etc.)
        split: Dataset split (all_validation, test, etc.)
        data_dir: Root directory for datasets
        subset_num: Number of examples to use (-1 for all)
    """
    name: str
    split: str
    data_dir: Path = Path("./data")
    subset_num: int = -1  # -1 uses the full split

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    This replaces all argparse arguments with a single declarative config
    that can be loaded from YAML files.

    Attributes:
        name: Experiment name
        description: Human-readable description
        models: Dictionary mapping role names to ModelConfig instances
        tools: Tool configuration
        dataset: Dataset configuration
        max_turns: Maximum reasoning turns per question
        seed: Random seed for reproducibility
        thinking_mode: Enable thinking mode for orchestrator
        output_dir: Directory for results
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        cache_dir: Directory for caching
    """

    # Metadata
    name: str
    description: str = ""

    # Models (dict of role -> ModelConfig)
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Tools
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    # Dataset
    dataset: Optional[DatasetConfig] = None

    # Execution
    max_turns: int = 15
    seed: int = 0
    thinking_mode: ThinkingMode = ThinkingMode.NO  # Thinking mode: NO, ORCHESTRATOR_ONLY, SUBAGENTS_ONLY, ALL

    # Output
    output_dir: Path = Path("./experiments/results")
    use_wandb: bool = False
    wandb_project: Optional[str] = None

    # Caching
    cache_dir: Path = Path("./cache")

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

    def get_model(self, role: str) -> Optional[ModelConfig]:
        """Get model config by role.

        Args:
            role: Model role (e.g., "orchestrator")

        Returns:
            ModelConfig or None if not found
        """
        return self.models.get(role)

    def has_model(self, role: str) -> bool:
        """Check if a model role exists.

        Args:
            role: Model role

        Returns:
            True if model exists
        """
        return role in self.models

    def use_orchestrator_thinking(self) -> bool:
        """Check if orchestrator should use thinking mode.

        Returns:
            True if orchestrator uses thinking
        """
        return self.thinking_mode in (ThinkingMode.ORCHESTRATOR_ONLY, ThinkingMode.ALL)

    def use_subagent_thinking(self) -> bool:
        """Check if sub-agents should use thinking mode.

        Returns:
            True if sub-agents use thinking
        """
        return self.thinking_mode in (ThinkingMode.SUBAGENTS_ONLY, ThinkingMode.ALL)
