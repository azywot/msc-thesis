"""Core tool abstractions for agent_engine.

This module defines the base classes and registry for tool management,
providing a clean interface for tool execution and schema generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Standardized tool execution result.

    Attributes:
        success: Whether the tool executed successfully
        output: Formatted output text to return to the model
        metadata: Additional information (cached, num_results, etc.)
        error: Error message if execution failed
        usage: Optional token usage from LLM calls (prompt_tokens, completion_tokens, total_tokens)
    """
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class BaseTool(ABC):
    """Abstract base class for all tools.

    Tools must implement:
    - name: Tool identifier
    - description: Human-readable description
    - get_schema(): Return Qwen3-compatible JSON schema
    - execute(): Execute the tool with provided arguments
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for invocation (e.g., 'web_search').

        Returns:
            Tool identifier string
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does.

        Returns:
            Description string
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return Qwen3 JSON Schema for this tool.

        The schema should follow the format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Returns:
            JSON schema dictionary
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with provided arguments.

        Args:
            **kwargs: Tool-specific arguments from the model

        Returns:
            ToolResult with execution outcome
        """
        pass

    def validate_args(self, **kwargs) -> bool:
        """Validate arguments before execution.

        Override this method to add custom validation logic.

        Args:
            **kwargs: Tool arguments to validate

        Returns:
            True if arguments are valid, False otherwise
        """
        return True

    def cleanup(self):
        """Optional cleanup (release resources).

        Override if your tool needs cleanup (e.g., closing connections).
        """
        pass


class ToolRegistry:
    """Central registry for managing tools.

    The registry maintains a collection of tool instances and provides
    methods for registration, retrieval, and batch operations.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool instance.

        Args:
            tool: BaseTool instance to register

        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            BaseTool instance or None if not found
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            True if tool exists in registry
        """
        return name in self._tools

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for prompt construction.

        Returns:
            List of JSON schema dictionaries
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tools(self) -> List[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def cleanup_all(self):
        """Cleanup all registered tools."""
        for tool in self._tools.values():
            tool.cleanup()

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool name is in registry."""
        return name in self._tools
