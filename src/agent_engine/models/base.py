"""Base classes for model providers.

This module defines the core abstractions for working with different model providers
(vLLM, OpenAI, Anthropic, etc.) in a unified way.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator


class ModelFamily(Enum):
    """Supported model families.

    Explicit enumeration used throughout the codebase.
    Add a new entry here when onboarding a new family, then update
    ``_THINKING_FAMILIES`` and ``_TOOL_CALL_FORMAT`` below accordingly.
    """
    QWEN3 = "qwen3"
    QWEN2_5 = "qwen2.5"
    QWQ = "qwq"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    OLMO_THINK = "olmo-think"
    OLMO_INSTRUCT = "olmo-instruct"
    GPT4 = "gpt4"
    CLAUDE = "claude"


class ToolCallFormat(Enum):
    """Format a model family uses to emit tool calls.

    JSON       — ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
                 (Qwen3, most families)
    PYTHONIC   — ``<function_calls>\\ntool(arg=val)\\n</function_calls>``
                 (OLMo 3; also accepts JSON boolean/null literals)
    JSON_SINGLE — ``{"tool_call": {"name": ..., "arguments": {...}}}``
                  (DeepSeek R1; pure JSON, single call per turn, no XML wrapping)
    """
    JSON = "json"
    PYTHONIC = "pythonic"
    JSON_SINGLE = "json_single"


# Families whose models natively support extended <think> output.
_THINKING_FAMILIES = frozenset({ModelFamily.QWEN3, ModelFamily.QWQ, ModelFamily.DEEPSEEK, ModelFamily.OLMO_THINK})

# Families whose HF chat template accepts the ``enable_thinking`` kwarg.
# Other thinking-capable families (e.g. OLMo, DeepSeek) always think and don't expose this knob.
_ENABLE_THINKING_KWARG_FAMILIES = frozenset({ModelFamily.QWEN3, ModelFamily.QWQ})

# Families whose chat template has no system-role slot.
# Providers merge the system-message content into the first user message before tokenising.
_NO_SYSTEM_PROMPT_FAMILIES = frozenset({ModelFamily.DEEPSEEK})

# Families that require explicit <think>\n prefix-forcing to engage reasoning
# (as opposed to Qwen3/QwQ which toggle it via the enable_thinking kwarg).
# use_thinking=True  → append "<think>\n"          (prime the model to reason)
# use_thinking=False → append "<think>\n\n</think>\n"  (suppress reasoning)
_THINK_PREFIX_FAMILIES = frozenset({ModelFamily.DEEPSEEK})

# Per-family tool-call output format.  Unlisted families default to JSON.
_TOOL_CALL_FORMAT: Dict[ModelFamily, ToolCallFormat] = {
    ModelFamily.OLMO_THINK: ToolCallFormat.PYTHONIC,
    ModelFamily.OLMO_INSTRUCT: ToolCallFormat.PYTHONIC,
    ModelFamily.DEEPSEEK: ToolCallFormat.JSON_SINGLE,
}


def get_tool_call_format(family: ModelFamily) -> ToolCallFormat:
    """Return the tool-call format for *family* (defaults to JSON)."""
    return _TOOL_CALL_FORMAT.get(family, ToolCallFormat.JSON)


def merge_system_into_user(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fold a leading system message into the first user message.

    For families whose chat template has no system-role slot (e.g. DeepSeek R1),
    prepend the system content to the first user turn separated by a blank line.
    Returns the original list unchanged when there is no leading system message.
    """
    if not msgs or msgs[0].get("role") != "system":
        return msgs
    system_content = msgs[0]["content"]
    result = list(msgs[1:])
    for i, msg in enumerate(result):
        if msg.get("role") == "user":
            result[i] = {**msg, "content": system_content + "\n\n" + msg["content"]}
            return result
    return result


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
        temperature: Sampling temperature.  ``0.0`` = greedy (default, for reproducibility).
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

    backend: str = "vllm"  # "vllm", "mlx", "openai", "anthropic"

    _FAMILY_DEFAULTS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "deepseek":      {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
        "olmo-think":    {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
        "olmo-instruct": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768},
    }

    @model_validator(mode="before")
    @classmethod
    def _apply_family_defaults(cls, values):
        family = values.get("family", "")
        family_str = family.value if isinstance(family, ModelFamily) else str(family)
        for param, default in cls._FAMILY_DEFAULTS.get(family_str, {}).items():
            values.setdefault(param, default)
        return values

    @field_validator("family", mode="before")
    @classmethod
    def _coerce_family(cls, v):
        if isinstance(v, str):
            return ModelFamily(v)
        return v

    @model_validator(mode="after")
    def _derive_supports_thinking(self):
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
        use_thinking: bool = False,
        force_tool_call: bool = False,
    ) -> str:
        """Apply model-specific chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            use_thinking: Whether to enable thinking mode (for Qwen3)
            force_tool_call: For families that need prefix forcing (DeepSeek),
                             inject a closed think block + ``<sub_goal>`` suffix
                             so the model is forced to complete a tool call rather
                             than answering in its reasoning block.  Has no effect
                             on other families.

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
