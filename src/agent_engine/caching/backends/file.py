"""File-based cache backend.

This module provides a simple file-based cache using JSON.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ...utils.logging import get_logger

logger = get_logger(__name__)


class FileCacheBackend:
    """File-based cache backend using JSON."""

    def __init__(self, cache_dir: Path, cache_name: str):
        """Initialize file cache backend.

        Args:
            cache_dir: Directory for cache files
            cache_name: Name of this cache (used for filename)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_name = cache_name
        self.cache_path = self.cache_dir / f"{cache_name}.json"
        self._data: Dict[str, Any] = {}
        self._loaded = False

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        """Load cache from disk."""
        if self._loaded:
            return

        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                logger.info(f"Loaded cache from {self.cache_path} ({len(self._data)} entries)")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self._data = {}
        else:
            logger.info(f"Cache file not found, starting with empty cache: {self.cache_path}")
            self._data = {}

        self._loaded = True

    def save(self):
        """Save cache to disk."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved cache to {self.cache_path} ({len(self._data)} entries)")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        self.load()
        return self._data.get(key)

    def set(self, key: str, value: Any):
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self.load()
        self._data[key] = value

    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        self.load()
        return key in self._data

    def delete(self, key: str):
        """Delete key from cache.

        Args:
            key: Cache key
        """
        self.load()
        if key in self._data:
            del self._data[key]

    def clear(self):
        """Clear all cache entries."""
        self._data = {}
        self.save()

    def __len__(self) -> int:
        """Get number of cache entries."""
        self.load()
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return self.has(key)
