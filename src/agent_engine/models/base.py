"""Base classes for model providers.

This module defines the core abstractions for working with different model providers
(vLLM, OpenAI, Anthropic, etc.) in a unified way.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ModelFamily(Enum):
    """Explicit model family enumeration - no string matching!"""
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
    """Complete model configuration - replaces argparse.

    This dataclass encapsulates all model-related configuration including
    generation parameters, resource management, and capability flags.
    """
    name: str                          # Human-readable name
    family: ModelFamily                # Explicit family (no string matching)
    path_or_id: str                    # Local path or API model ID
    role: str                          # "orchestrator", "web_search", "code_generator", etc.

    # Generation params (defaults: temperature=0 for reproducibility)
    max_model_len: int = 32768
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1

    # Derived from family in __post_init__; set explicitly in YAML to override.
    supports_thinking: Optional[bool] = None

    # Resource management (tensor_parallel_size: None → auto from gpu_ids or device count)
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None  # None → auto-resolved at load time
    gpu_ids: Optional[List[int]] = None

    # Reproducibility
    seed: int = 0

    def __post_init__(self):
        if isinstance(self.family, str):
            self.family = ModelFamily(self.family)
        if self.supports_thinking is None:
            self.supports_thinking = self.family in _THINKING_FAMILIES


@dataclass
class GenerationResult:
    """Standardized result from any provider."""
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
