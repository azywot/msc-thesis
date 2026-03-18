"""Base classes for model providers.

This module defines the core abstractions for working with different model providers
(vLLM, OpenAI, Anthropic, etc.) in a unified way.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator


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
    DEEPSEEK_R1 = "deepseek_r1"            # DeepSeek-R1-Distill-Qwen-{7,14,32}B (Jan 2025, Qwen2.5 backbone)
    DEEPSEEK_R1_0528 = "deepseek_r1_0528"  # DeepSeek-R1-0528-Qwen3-8B (May 2025, Qwen3 backbone)
    PHI4 = "phi4"
    GPT4 = "gpt4"
    CLAUDE = "claude"


# Convenience group for all DeepSeek subfamily entries.
_DEEPSEEK_FAMILIES = frozenset({ModelFamily.DEEPSEEK_R1, ModelFamily.DEEPSEEK_R1_0528})

# Convenience group for all Qwen-derived families (use Qwen3-style tool call tags).
_QWEN_FAMILIES = frozenset({ModelFamily.QWEN3, ModelFamily.QWEN2_5, ModelFamily.QWQ})

# Families whose models natively support extended <think> output.
_THINKING_FAMILIES = frozenset({ModelFamily.QWEN3, ModelFamily.QWQ}) | _DEEPSEEK_FAMILIES


# ----- Generation-parameter defaults -----
# Base defaults (Qwen3 family — the project's primary model family).
_BASE_GEN_DEFAULTS: Dict[str, Any] = {
    "max_model_len": 32768,
    "max_tokens": 8192,
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.1,
}

# Per-family overrides (only keys that differ from _BASE_GEN_DEFAULTS).
_FAMILY_GEN_DEFAULTS: Dict[ModelFamily, Dict[str, Any]] = {
    ModelFamily.DEEPSEEK_R1: {
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "max_model_len": 32768,
    },
    ModelFamily.DEEPSEEK_R1_0528: {
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "max_model_len": 32768,
    },
}


class ModelConfig(BaseModel):
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
        temperature: Sampling temperature.  ``None`` → resolved from family defaults.
        top_p: Nucleus sampling threshold.
        top_k: Top-K sampling limit.
        repetition_penalty: Penalty applied to repeated tokens (``1.0`` = no penalty).
        supports_thinking: Whether the model can produce ``<think>`` output.
            Derived from *family* by a model validator if not set explicitly.
        tensor_parallel_size: Number of GPUs to shard the model across.
            ``None`` → auto-detected from ``gpu_ids`` or visible device count.
        gpu_memory_utilization: Fraction of GPU VRAM to reserve for the model
            (passed to vLLM).  ``None`` → resolved at load time by
            :func:`~agent_engine.models.vllm_provider.resolve_gpu_assignments`.
        gpu_ids: List of specific GPU indices to use.  ``None`` = use all visible.
        seed: Random seed for reproducible generation.
    """
    model_config = {"arbitrary_types_allowed": True}

    name: str
    family: ModelFamily
    path_or_id: str
    role: str

    max_model_len: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

    supports_thinking: Optional[bool] = None

    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    gpu_ids: Optional[List[int]] = None

    seed: int = 0

    backend: str = "vllm"  # "vllm", "mlx", "openai", "anthropic"

    @field_validator("family", mode="before")
    @classmethod
    def _coerce_family(cls, v):
        if isinstance(v, str):
            return ModelFamily(v)
        return v

    @model_validator(mode="after")
    def _resolve_defaults(self):
        """Fill ``None`` generation fields from family → base defaults, and derive flags."""
        family_overrides = _FAMILY_GEN_DEFAULTS.get(self.family, {})
        for field_name, base_val in _BASE_GEN_DEFAULTS.items():
            if getattr(self, field_name) is None:
                setattr(self, field_name, family_overrides.get(field_name, base_val))
        if self.supports_thinking is None:
            self.supports_thinking = self.family in _THINKING_FAMILIES
        return self


class GenerationResult(BaseModel):
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
    usage: Dict[str, int] = {}
    metadata: Dict[str, Any] = {}
    messages: Optional[List[Dict[str, Any]]] = None


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
        """Mind map entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Mind map exit - cleanup resources."""
        self.cleanup()
