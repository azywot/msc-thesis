"""Build the tool registry for a run.

This replaces the ``setup_tools`` if/elif chain: the runner no longer needs to
know how any individual tool is constructed, only the order to try them in.
"""

from ..core.tool import ToolRegistry

# Importing the package runs each module's @register_tool decorators, which is
# what populates the factory table.  Without this import the registry is empty.
from .. import tools as _tools  # noqa: F401
from ..tools.registry import ToolDeps, build_tool


def build_tool_registry(deps: ToolDeps) -> ToolRegistry:
    """Construct and register every enabled tool, in configured order.

    Names with no factory, and factories that decline to build (``None``), are
    skipped -- matching the old chain, where an unmatched name fell through
    every branch.
    """
    tools = ToolRegistry()
    for tool_name in deps.config.tools.enabled_tools:
        tool = build_tool(tool_name, deps)
        if tool is not None:
            tools.register(tool)
    return tools
