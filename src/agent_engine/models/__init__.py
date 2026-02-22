"""Model provider system for agent_engine.

This module provides unified interfaces for working with different model
providers (vLLM, OpenAI, Anthropic, etc.).

Notes:
- Some providers have optional third-party dependencies (e.g. `vllm`).
  We import those conditionally so that config utilities can be used in
  lightweight environments (e.g. generating SLURM jobs) without installing
  the full inference stack.
"""

from .base import BaseModelProvider, GenerationResult, ModelConfig, ModelFamily
from .registry import ModelRegistry, get_global_registry
from .llm_shared import get_llm_lock

try:
    from .vllm_provider import VLLMProvider, resolve_gpu_assignments
except Exception:  # pragma: no cover
    VLLMProvider = None  # type: ignore
    resolve_gpu_assignments = None  # type: ignore

try:
    from .api_provider import OpenAIProvider, AnthropicProvider
except Exception:  # pragma: no cover
    OpenAIProvider = None  # type: ignore
    AnthropicProvider = None  # type: ignore

__all__ = [
    "BaseModelProvider",
    "GenerationResult",
    "ModelConfig",
    "ModelFamily",
    "ModelRegistry",
    "get_global_registry",
    "VLLMProvider",
    "resolve_gpu_assignments",
    "OpenAIProvider",
    "AnthropicProvider",
    "get_llm_lock",
]
