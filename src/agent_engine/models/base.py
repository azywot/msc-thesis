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

# Base generation defaults — match multi-agent-tools CLI defaults.
_BASE_GEN_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 8192,
    "repetition_penalty": 1.05,
}

# Role-specific overrides applied on top of _BASE_GEN_DEFAULTS when the
# field is not explicitly set (i.e. still None).  Mirrors multi-agent-tools
# hardcoded per-tool SamplingParams values.
_ROLE_GEN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # run_code.py always uses greedy decoding (temp=0, top_p=1, top_k=-1),
    # max_tokens=2048, no repetition penalty.
    "code_generator": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "repetition_penalty": 1.0,
    },
    # search and text inspector sub-agents receive max_tokens=8192 from CLI.
    "web_search":     {"max_tokens": 8192},
    "text_inspector": {"max_tokens": 8192},
}


@dataclass
class ModelConfig:
    """Complete model configuration - replaces argparse.

    This dataclass encapsulates all model-related configuration including
    generation parameters, resource management, and capability flags.

    Generation params default to None so that __post_init__ can distinguish
    "explicitly set by user" from "not set" and apply the correct role-specific
    default (matching multi-agent-tools hardcoded per-tool values).
    """
    name: str                          # Human-readable name
    family: ModelFamily                # Explicit family (no string matching)
    path_or_id: str                    # Local path or API model ID
    role: str                          # "orchestrator", "web_search", "code_generator", etc.

    max_model_len: int = 32768

    # Generation params: None → resolved per-role in __post_init__.
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

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

        # Resolve generation params: role-specific defaults override base defaults,
        # but any value explicitly set in the YAML (non-None) is kept as-is.
        defaults = {**_BASE_GEN_DEFAULTS, **_ROLE_GEN_DEFAULTS.get(self.role, {})}
        if self.max_tokens is None:
            self.max_tokens = defaults["max_tokens"]
        if self.temperature is None:
            self.temperature = defaults["temperature"]
        if self.top_p is None:
            self.top_p = defaults["top_p"]
        if self.top_k is None:
            self.top_k = defaults["top_k"]
        if self.repetition_penalty is None:
            self.repetition_penalty = defaults["repetition_penalty"]


@dataclass
class GenerationResult:
    """Standardized result from any provider."""
    text: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
