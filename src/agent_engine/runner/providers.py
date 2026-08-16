"""Model-provider construction for experiment runs.

Moved verbatim from ``scripts/run_experiment.py`` so the runner is importable
as a library rather than only executable as a script.
"""

import logging
from typing import Any, Dict, Optional

from agent_engine.models.base import ModelFamily

logger = logging.getLogger(__name__)


def setup_model_provider(model_config, api_keys: Dict[str, str], model_cache: Optional[Dict[str, Any]] = None):
    """Initialize model provider with instance caching for memory efficiency.
    
    This function implements model instance reuse: if the same model path is
    requested multiple times (e.g., for different roles like orchestrator/web_search/code_generator),
    it returns the cached instance instead of loading a new one. This is critical
    for memory efficiency when using the same model for multiple roles.
    
    Thread safety is handled via locks in VLLMProvider, so multiple roles can
    safely share the same model instance.

    Args:
        model_config: ModelConfig instance
        api_keys: Dictionary of API keys
        model_cache: Optional cache dict for model instances (keyed by path_or_id)

    Returns:
        Model provider instance (new or cached)
    """
    # Check cache for local models (API models are lightweight, no need to cache).
    # Include lora_adapter_path in the key so a LoRA instance and the base model
    # don't collide when both appear in the same experiment.
    cache_key = f"{model_config.path_or_id}|lora:{model_config.lora_adapter_path or ''}"
    if model_cache is not None and cache_key in model_cache:
        if model_config.family not in (ModelFamily.GPT4, ModelFamily.CLAUDE):
            cached_provider = model_cache[cache_key]
            logger.info("♻️ Reusing cached model instance for: %s (role: %s)", model_config.name, model_config.role)
            return cached_provider
    
    # Lazy imports so this script can run with only API deps installed.
    backend = getattr(model_config, "backend", "vllm")

    if backend == "openai" or model_config.family in [ModelFamily.GPT4]:
        from agent_engine.models.api_provider import OpenAIProvider
        provider = OpenAIProvider(model_config, api_key=api_keys.get("openai"))
    elif backend == "anthropic" or model_config.family == ModelFamily.CLAUDE:
        from agent_engine.models.api_provider import AnthropicProvider
        provider = AnthropicProvider(model_config, api_key=api_keys.get("anthropic"))
    elif backend == "mlx":
        from agent_engine.models.mlx_provider import MLXProvider
        provider = MLXProvider(model_config)

        if model_cache is not None:
            model_cache[cache_key] = provider
            logger.info(f"💾 Cached MLX model instance: {cache_key}")
    else:
        # Local vLLM model - cache these to avoid duplicate loading
        from agent_engine.models.vllm_provider import VLLMProvider
        provider = VLLMProvider(model_config)

        # Cache local model instances for reuse
        if model_cache is not None:
            model_cache[cache_key] = provider
            logger.info(f"💾 Cached model instance: {cache_key}")

    return provider
