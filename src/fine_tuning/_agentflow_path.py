"""Repair wrong `agentflow` resolution when only AgentFlow/agentflow is on PYTHONPATH.

If ``.../AgentFlow/agentflow`` is prepended alone, Python loads the inner package
directory ``agentflow/agentflow/`` as the top-level name ``agentflow``, whose
``__init__.py`` does not export ``LitAgent``.  Prepending ``.../AgentFlow`` fixes
resolution so ``agentflow`` maps to the real SDK package.  This lives in
msc-thesis because cluster job scripts commonly add the inner path only."""

from __future__ import annotations

import sys
from pathlib import Path


def _purge_agentflow_modules() -> None:
    stale = [
        key
        for key in sys.modules
        if key == "agentflow" or key.startswith("agentflow.")
    ]
    for key in stale:
        del sys.modules[key]


def _fix_sys_path_if_nested_agentflow_loaded() -> bool:
    """Return True if we inserted a sys.path prefix and cleared agentflow caches."""
    loaded = sys.modules.get("agentflow")
    loc = getattr(loaded, "__file__", None) if loaded is not None else None
    if not loc:
        return False

    path = Path(loc).resolve()
    if path.name != "__init__.py":
        return False

    inner_pkg_dir = path.parent
    outer_pkg_dir = inner_pkg_dir.parent
    # ``.../<repo>/agentflow/agentflow/__init__.py``  → put ``.../<repo>`` first on path.
    if inner_pkg_dir.name != "agentflow" or outer_pkg_dir.name != "agentflow":
        return False

    repo_like_root = outer_pkg_dir.parent
    root_s = str(repo_like_root)
    # ``.../agentflow`` on PYTHONPATH exposes ``types.py`` as top-level ``types`` (stdlib shadow).
    for bad in (str(outer_pkg_dir), str(inner_pkg_dir)):
        while bad in sys.path:
            sys.path.remove(bad)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    _purge_agentflow_modules()
    return True


def ensure_agentflow_litagent_importable() -> None:
    """Ensure ``from agentflow import LitAgent`` resolves to the AgentFlow SDK package."""
    for _ in range(2):
        try:
            from agentflow import LitAgent as _LitAgentProbe  # noqa: F401, PLC0415

            return
        except ImportError:
            if not _fix_sys_path_if_nested_agentflow_loaded():
                raise
