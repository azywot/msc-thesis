"""Base classes for model providers.

This module defines the core abstractions for working with different model providers
(vLLM, OpenAI, Anthropic, etc.) in a unified way.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelFamily(Enum):
    """Supported model families.

    Explicit enumeration used throughout the codebase. 
    Add a new entry here when onboarding a new family.
    """
    QWEN3 = "qwen3"
    QWEN2_5 = "qwen2.5"
    QWQ = "qwq"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    GPT4 = "gpt4"
    CLAUDE = "claude"


# Families whose models natively support extended <think> output.
_THINKING_FAMILIES = frozenset({ModelFamily.QWEN3, ModelFamily.QWQ, ModelFamily.DEEPSEEK})


@dataclass
class ModelConfig:
    """Complete model configuration.

    All generation parameters, resource management settings, and capability
    flags live here.  Populated by the YAML loader and passed to model
    provider constructors.

    Attributes:
        name: Human-readable label (used in logs and W&B).
        family: Explicit :class:`ModelFamily` — determines capability flags such
                as ``supports_thinking``.
        path_or_id: HuggingFace model ID or absolute path to a local checkpoint.
        role: Functional role in the experiment, e.g. ``"orchestrator"``,
              ``"web_search"``, ``"code_generator"``.
        max_model_len: Maximum sequence length (prompt + generation) in tokens.
        max_tokens: Maximum *new* tokens to generate per call.
        temperature: Sampling temperature.  ``0.0`` = greedy (default, for reproducibility).
        top_p: Nucleus sampling threshold.
        top_k: Top-K sampling limit.
        repetition_penalty: Penalty applied to repeated tokens (``1.0`` = no penalty).
        supports_thinking: Whether the model can produce ``<think>`` output.
            Derived from *family* in ``__post_init__`` if not set explicitly.
        tensor_parallel_size: Number of GPUs to shard the model across.
            ``None`` → auto-detected from ``gpu_ids`` or visible device count.
        gpu_memory_utilization: Fraction of GPU VRAM to reserve for the model
            (passed to vLLM).  ``None`` → resolved at load time by
            :func:`~agent_engine.models.vllm_provider.resolve_gpu_assignments`.
        gpu_ids: List of specific GPU indices to use.  ``None`` = use all visible.
        seed: Random seed for reproducible generation.
    """
    name: str
    family: ModelFamily
    path_or_id: str
    role: str

    max_model_len: int = 32768
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1

    supports_thinking: Optional[bool] = None

    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    gpu_ids: Optional[List[int]] = None

    seed: int = 0

    def __post_init__(self):
        if isinstance(self.family, str):
            self.family = ModelFamily(self.family)
        if self.supports_thinking is None:
            self.supports_thinking = self.family in _THINKING_FAMILIES


@dataclass
class GenerationResult:
    """Standardized result from any model provider.

    Attributes:
        text: Raw generated text (may contain ``<think>`` blocks for thinking models).
        finish_reason: Why generation stopped (``"stop"``, ``"length"``, etc.).
        usage: Token counts — keys ``"prompt_tokens"``, ``"completion_tokens"``,
               ``"total_tokens"``.
        metadata: Provider-specific extras (e.g. logprobs).
        messages: The input messages that produced this result (optional,
                  used for debugging API provider calls).
    """
    text: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    messages: Optional[List[Dict[str, Any]]] = None  # input messages that produced this result


class BaseModelProvider(ABC):
    """Abstract base for all model providers.

    This provides a unified interface for interacting with different model
    backends (vLLM, OpenAI, Anthropic, etc.).
    """

    def __init__(self, config: ModelConfig):
        """Initialize provider with configuration.

        Args:
            config: ModelConfig instance with all model settings
        """
        self.config = config

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate completions for batch of prompts.

        Args:
            prompts: List of formatted prompt strings

        Returns:
            List of GenerationResult objects
        """
        pass

    @abstractmethod
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False
    ) -> str:
        """Apply model-specific chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            use_thinking: Whether to enable thinking mode (for Qwen3)

        Returns:
            Formatted prompt string ready for generation
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Release resources (GPU memory, connections)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
