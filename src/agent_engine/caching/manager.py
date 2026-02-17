"""Cache manager for agent_engine.

This module provides a unified interface for managing different caches
(search results, URL content, etc.).
"""

from pathlib import Path
from typing import Any, Dict, Optional

from .backends.file import FileCacheBackend
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages multiple caches with a unified interface."""

    def __init__(self, cache_dir: Path):
        """Initialize cache manager.

        Args:
            cache_dir: Root directory for all caches
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create cache backends
        self.search_backend = FileCacheBackend(cache_dir, "search_cache")
        self.url_backend = FileCacheBackend(cache_dir, "url_cache")

        # Load caches
        self.search_backend.load()
        self.url_backend.load()

        logger.info(f"Cache manager initialized at {cache_dir}")

    @property
    def search_cache(self) -> Dict[str, Any]:
        """Get search cache as dict-like interface.

        Returns:
            Dictionary interface to search cache
        """
        return self._CacheDict(self.search_backend)

    @property
    def url_cache(self) -> Dict[str, Any]:
        """Get URL cache as dict-like interface.

        Returns:
            Dictionary interface to URL cache
        """
        return self._CacheDict(self.url_backend)

    def save(self):
        """Save all caches to disk."""
        self.search_backend.save()
        self.url_backend.save()
        logger.info("All caches saved")

    def clear_all(self):
        """Clear all caches."""
        self.search_backend.clear()
        self.url_backend.clear()
        logger.info("All caches cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache sizes
        """
        return {
            "search_cache_size": len(self.search_backend),
            "url_cache_size": len(self.url_backend),
        }

    class _CacheDict:
        """Dictionary-like wrapper for cache backend."""

        def __init__(self, backend: FileCacheBackend):
            self.backend = backend

        def get(self, key: str, default: Any = None) -> Any:
            """Get value from cache."""
            value = self.backend.get(key)
            return value if value is not None else default

        def __getitem__(self, key: str) -> Any:
            """Get value from cache."""
            value = self.backend.get(key)
            if value is None:
                raise KeyError(key)
            return value

        def __setitem__(self, key: str, value: Any):
            """Set value in cache."""
            self.backend.set(key, value)

        def __contains__(self, key: str) -> bool:
            """Check if key exists."""
            return self.backend.has(key)

        def __len__(self) -> int:
            """Get cache size."""
            return len(self.backend)

        def update(self, other: Dict[str, Any]):
            """Update cache with key-value pairs from dict."""
            for key, value in other.items():
                self.backend.set(key, value)
