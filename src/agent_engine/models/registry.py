"""Model registry for managing and reusing model instances.

This module provides a registry pattern to avoid loading the same model
multiple times, which is crucial for GPU memory efficiency.
"""

from typing import Dict, Optional

from .base import BaseModelProvider, ModelConfig


class ModelRegistry:
    """Registry for managing model instances and enabling reuse.

    This singleton-style registry tracks loaded models and allows different
    components (orchestrator, tools) to share the same model instance.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, BaseModelProvider] = {}

    def register(self, key: str, provider: BaseModelProvider):
        """Register a model provider instance.

        Args:
            key: Unique key to identify this model (e.g., "orchestrator", "search")
            provider: Model provider instance
        """
        self._models[key] = provider

    def get(self, key: str) -> Optional[BaseModelProvider]:
        """Get a registered model provider.

        Args:
            key: Model key

        Returns:
            Model provider instance or None if not found
        """
        return self._models.get(key)

    def has(self, key: str) -> bool:
        """Check if a model is registered.

        Args:
            key: Model key

        Returns:
            True if model exists in registry
        """
        return key in self._models

    def cleanup_all(self):
        """Cleanup all registered models."""
        for provider in self._models.values():
            provider.cleanup()
        self._models.clear()

    def __len__(self) -> int:
        """Get number of registered models."""
        return len(self._models)

    def list_models(self) -> list:
        """List all registered model keys.

        Returns:
            List of model keys
        """
        return list(self._models.keys())


# Global singleton instance
_global_registry = ModelRegistry()


def get_global_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        Global ModelRegistry singleton
    """
    return _global_registry
