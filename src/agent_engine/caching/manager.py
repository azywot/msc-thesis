"""Cache manager for agent_engine.

Thread-safe, process-safe cache for search results and fetched URL content.

Key properties:
- A single lock file (.cache.lock) serialises cross-process writes via fcntl.
- Every write goes through an atomic temp-file + os.replace() sequence so
  concurrent readers always see a complete, consistent JSON file.
- save_caches() merges the current on-disk state into memory (disk ∪ memory,
  memory wins) before persisting, so parallel workers never overwrite each
  other's entries.
"""

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any


class CacheManager:
    def __init__(self, cache_dir: str = './cache'):
        self.cache_dir = cache_dir
        self.search_cache_path = os.path.join(cache_dir, 'search_cache.json')
        self.url_cache_path = os.path.join(cache_dir, 'url_cache.json')
        # Single lock for both files to avoid deadlocks and cross-file races.
        self._lock_path = os.path.join(cache_dir, '.cache.lock')
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        # Ensure lock file exists.
        try:
            open(self._lock_path, 'a', encoding='utf-8').close()
        except Exception:
            # If we can't create the lock file, proceed without hard-failing;
            # writes may become unsafe, but the run can continue.
            pass
        self.search_cache = self._load_cache(self.search_cache_path)
        self.url_cache = self._load_cache(self.url_cache_path)

    @staticmethod
    def _normalize_search_results(value: Any) -> list:
        """Ensure search cache value is always a list of dicts (Serper result shape).

        Cache files may come from other tools or older versions; normalizing on
        load guarantees the rest of the code sees list[dict] per query.
        """
        if not isinstance(value, list):
            return []
        return [r for r in value if isinstance(r, dict)]

    def _load_cache(self, path: str) -> dict:
        # Reading is safe even without locks thanks to atomic replace on writes,
        # but we still prefer a shared lock if available for consistency.
        if not os.path.exists(path):
            return {}
        try:
            with self._locked(shared=True):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            if path == self.search_cache_path:
                data = {k: self._normalize_search_results(v) for k, v in data.items()}
            return data
        except Exception:
            # If the file is temporarily unreadable/corrupt (e.g. external edit),
            # fall back to empty cache rather than crashing the run.
            return {}

    @contextmanager
    def _locked(self, shared: bool):
        """Cross-process file lock on a single lock file.

        - shared=True  → multiple readers allowed simultaneously
        - shared=False → exclusive writer; blocks until all readers release
        """
        try:
            lock_f = open(self._lock_path, 'r+', encoding='utf-8')
        except Exception:
            # Best-effort fallback if lock file can't be opened.
            yield
            return

        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            finally:
                lock_f.close()

    def _atomic_write_json(self, path: str, data: dict) -> None:
        """Write JSON atomically: write to a temp file, then os.replace().

        Concurrent readers always see either the old or the new file, never a
        partially-written one.
        """
        dir_name = os.path.dirname(path) or '.'
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + '.', suffix='.tmp', dir=dir_name
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_f:
                json.dump(data, tmp_f, ensure_ascii=False, indent=2)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, path)
        finally:
            # Clean up temp file if replace() didn't happen.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    def save_caches(self) -> None:
        """Persist both caches to disk.

        Acquires an exclusive lock, re-reads the on-disk state, merges it with
        the in-memory state (memory wins on conflicts), and atomically writes
        both files. In-memory dicts are then updated to reflect the merged
        state so subsequent calls remain consistent.
        """
        with self._locked(shared=False):
            disk_search = self._load_cache_unlocked(self.search_cache_path)
            disk_url = self._load_cache_unlocked(self.url_cache_path)

            # disk ← disk ∪ memory (memory wins)
            disk_search.update(self.search_cache)
            disk_url.update(self.url_cache)

            self._atomic_write_json(self.search_cache_path, disk_search)
            self._atomic_write_json(self.url_cache_path, disk_url)

            # Refresh in-memory dicts to include any concurrent writers' additions.
            self.search_cache.clear()
            self.search_cache.update(disk_search)
            self.url_cache.clear()
            self.url_cache.update(disk_url)

    def _load_cache_unlocked(self, path: str) -> dict:
        """Load JSON without acquiring a lock (caller must already hold one)."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if path == self.search_cache_path:
                data = {k: self._normalize_search_results(v) for k, v in data.items()}
            return data
        except Exception:
            return {}
