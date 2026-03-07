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
    """Tool enablement and behaviour configuration.

    Attributes:
        enabled_tools: Names of tools to register. Valid values are
            ``"web_search"``, ``"code_generator"``, ``"context_manager"``,
            ``"text_inspector"``, and ``"image_inspector"``.
        direct_tool_call: When ``True`` the orchestrator calls tools directly
            (no sub-agent LLM). When ``False`` each tool spins up its own
            sub-agent LLM for analysis (requires corresponding model entries).
        web_tool_provider: Search API backend — ``"serper"`` or ``"tavily"``.
        max_search_limit: Maximum number of ``web_search`` calls per question.
        top_k_results: Number of search results returned per query.
        max_doc_len: Maximum characters per fetched document snippet.
    """
    enabled_tools: List[str] = field(default_factory=lambda: ["web_search", "code_generator"])
    direct_tool_call: bool = True
    web_tool_provider: str = "serper"
    max_search_limit: int = 10
    top_k_results: int = 5
    max_doc_len: int = 3000


@dataclass
class DatasetConfig:
    """Dataset loading configuration.

    Attributes:
        name: Registered dataset name (e.g. ``"gaia"``, ``"gpqa"``, ``"math500"``).
        split: Dataset split to load (e.g. ``"validation"``, ``"test"``).
        data_dir: Root directory that contains downloaded dataset files.
        subset_num: Number of examples to sample. ``-1`` uses the full split.
    """
    name: str
    split: str
    data_dir: Path = Path("./data")
    subset_num: int = -1

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration loaded from a YAML file.

    This is the top-level config object populated by
    :func:`~agent_engine.config.loader.load_experiment_config`.  Every field
    maps one-to-one to a YAML key; nested objects (``models``, ``tools``,
    ``dataset``, ``slurm``) are automatically parsed into their dataclass
    counterparts.

    Attributes:
        name: Short experiment identifier (used in output directory names).
        description: Free-text description of the experiment.
        models: Role → :class:`~agent_engine.models.base.ModelConfig` mapping.
            Required roles: ``"orchestrator"``.  Optional roles correspond to
            enabled tool sub-agents (e.g. ``"web_search"``, ``"code_generator"``).
        tools: Tool enablement and behaviour settings.
        dataset: Dataset to evaluate on.
        max_turns: Maximum reasoning turns per question before forcing an answer.
        batch_size: Number of questions to process in one batched generation call.
            ``-1`` groups the entire dataset into a single batch; ``1`` disables
            batching (equivalent to sequential processing).
        seed: Global random seed propagated to all model configs.
        thinking_mode: Which components (if any) use extended ``<think>`` output.
        output_dir: Root directory for experiment result subdirectories.
        use_wandb: Whether to log results to Weights & Biases.
        wandb_project: W&B project name (required when ``use_wandb=True``).
        cache_dir: Root directory for search and URL caches.
        slurm: SLURM resource configuration for HPC job submission.
    """

    name: str
    description: str = ""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    dataset: Optional[DatasetConfig] = None
    max_turns: int = 15
    batch_size: int = -1
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
        """Return the :class:`ModelConfig` for *role*, or ``None`` if not configured."""
        return self.models.get(role)

    def has_model(self, role: str) -> bool:
        """Return ``True`` if a model is configured for *role*."""
        return role in self.models

    def use_orchestrator_thinking(self) -> bool:
        """Return ``True`` when the orchestrator should generate ``<think>`` output."""
        return self.thinking_mode in (ThinkingMode.ORCHESTRATOR_ONLY, ThinkingMode.ALL)

    def use_subagent_thinking(self) -> bool:
        """Return ``True`` when tool sub-agents should generate ``<think>`` output."""
        return self.thinking_mode in (ThinkingMode.SUBAGENTS_ONLY, ThinkingMode.ALL)
