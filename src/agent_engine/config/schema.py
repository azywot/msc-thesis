"""Configuration schema for agent_engine experiments."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from ..models.base import ModelConfig


class ThinkingMode(Enum):
    """Controls which components use extended thinking (for models that support it)."""
    NO = "NO"
    ORCHESTRATOR_ONLY = "ORCHESTRATOR_ONLY"
    SUBAGENTS_ONLY = "SUBAGENTS_ONLY"
    ALL = "ALL"


@dataclass
class SlurmConfig:
    """SLURM job resource configuration."""
    partition: str = "gpu_h100"
    num_gpus: Optional[int] = 1
    ntasks: int = 1
    cpus_per_task: int = 8
    time: str = "04:00:00"
    conda_env: str = "agent_engine"


@dataclass
class ToolsConfig:
    """Tool configuration."""
    enabled_tools: List[str] = field(default_factory=lambda: ["web_search", "code_generator"])
    direct_tool_call: bool = True
    web_tool_provider: str = "serper"  # "serper" or "tavily"
    max_search_limit: int = 10
    top_k_results: int = 10
    max_doc_len: int = 3000


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    split: str
    data_dir: Path = Path("./data")
    subset_num: int = -1  # -1 uses the full split

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration loaded from YAML."""

    name: str
    description: str = ""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    dataset: Optional[DatasetConfig] = None
    max_turns: int = 15
    batch_size: int = -1  # -1 = all in one batch; 1 = no batching
    seed: int = 0
    thinking_mode: ThinkingMode = ThinkingMode.NO
    output_dir: Path = Path("./experiments/results")
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    cache_dir: Path = Path("./cache")
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

    def get_model(self, role: str) -> Optional[ModelConfig]:
        return self.models.get(role)

    def has_model(self, role: str) -> bool:
        return role in self.models

    def use_orchestrator_thinking(self) -> bool:
        return self.thinking_mode in (ThinkingMode.ORCHESTRATOR_ONLY, ThinkingMode.ALL)

    def use_subagent_thinking(self) -> bool:
        return self.thinking_mode in (ThinkingMode.SUBAGENTS_ONLY, ThinkingMode.ALL)
