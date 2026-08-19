"""Tool construction registry.

``BaseTool``/``ToolRegistry`` in :mod:`agent_engine.core.tool` handle tool
*behaviour*.  This module handles tool *construction*: how a name in
``config.tools.enabled_tools`` becomes a configured instance.

Adding a sub-agent means writing a factory next to the tool it builds and
decorating it with :func:`register_tool` -- not editing a dispatch chain in a
runner module that has to know about every tool in the system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..core.tool import BaseTool

_FACTORIES: Dict[str, Callable[["ToolDeps"], Optional[BaseTool]]] = {}


@dataclass(frozen=True)
class ToolDeps:
    """Everything a tool factory is allowed to depend on.

    ``orchestrator_model`` is unused by every current factory.  It is carried
    because the function this replaced accepted it, and dropping a parameter
    would change the call signature callers rely on.
    """

    config: Any
    cache_manager: Any
    api_keys: Dict[str, Optional[str]]
    model_providers: Optional[Dict[str, Any]] = None
    orchestrator_model: Any = None
    mind_map_storage_path: Optional[Path] = None

    @property
    def direct_mode(self) -> bool:
        return self.config.tools.direct_tool_call

    @property
    def use_subagent_thinking(self) -> bool:
        return self.config.use_subagent_thinking()

    def provider_for(self, tool_name: str):
        """Sub-agent model for *tool_name*, or ``None`` in direct mode.

        Mirrors the ``... if not direct_mode and model_providers else None``
        idiom the four non-``mind_map`` tools used.  ``mind_map`` deliberately
        does *not* use this -- see :func:`~agent_engine.tools.mind_map.build_mind_map`.
        """
        if self.direct_mode or not self.model_providers:
            return None
        return self.model_providers.get(tool_name)


def register_tool(name: str):
    """Register a factory under the name used in ``enabled_tools``."""

    def wrapper(factory):
        if name in _FACTORIES:
            raise ValueError(f"Tool factory '{name}' is already registered")
        _FACTORIES[name] = factory
        return factory

    return wrapper


def build_tool(name: str, deps: ToolDeps) -> Optional[BaseTool]:
    """Construct one tool.

    Returns ``None`` when no factory is registered for *name* (unknown tool
    names were silently skipped by the if/elif chain this replaced) or when the
    factory declines to build one (``image_inspector`` in direct mode).
    """
    factory = _FACTORIES.get(name)
    if factory is None:
        return None
    return factory(deps)


def registered_tools():
    """Names with a registered factory, sorted."""
    return sorted(_FACTORIES)
