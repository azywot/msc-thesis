"""Shared LLM instance management with thread-safe locking.

This module provides infrastructure for safely sharing LLM instances across
multiple roles/components while preventing race conditions during generation.
"""

import threading
from typing import Dict, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Global registry of locks per model path
# Allows parallel generation on different models while serializing access to same instance
_LLM_LOCKS: Dict[str, threading.RLock] = {}
_lock_registry_lock = threading.Lock()


def get_llm_lock(model_path_or_id: Optional[str] = None) -> threading.RLock:
    """Get or create a thread-safe lock for a specific model instance.
    
    This ensures that multiple components can safely share a single model instance
    without race conditions during generation. Each unique model path gets its own
    lock, allowing different models to generate in parallel.
    
    Args:
        model_path_or_id: Model path or ID to get lock for. If None, returns
                         a shared lock for remote/unknown models.
    
    Returns:
        A re-entrant lock for the specified model
    
    Example:
        >>> lock = get_llm_lock("Qwen/Qwen3-32B")
        >>> with lock:
        ...     outputs = llm.generate(prompts)  # Thread-safe
    """
    # Use special key for remote/unknown models
    key = model_path_or_id if model_path_or_id else "__REMOTE__"
    
    with _lock_registry_lock:
        if key not in _LLM_LOCKS:
            _LLM_LOCKS[key] = threading.RLock()
            logger.debug(f"Created lock for model: {key}")
        return _LLM_LOCKS[key]


def clear_locks():
    """Clear all registered locks. Useful for testing."""
    global _LLM_LOCKS
    with _lock_registry_lock:
        _LLM_LOCKS.clear()
