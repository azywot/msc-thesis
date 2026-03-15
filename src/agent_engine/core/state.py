"""State management for agentic reasoning execution.

This module defines the state tracking for a single question's execution,
including conversation history, tool usage, and metadata.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExecutionState(BaseModel):
    """Tracks state of a single question execution.

    This model maintains all information about the execution of a single
    question, including conversation history, tool usage tracking, and results.

    Attributes:
        question_id: Unique identifier for the question
        question: The question text
        messages: Chat history (list of message dicts with 'role' and 'content')
        current_output: Current turn's output text
        turn: Current turn number
        finished: Whether execution is complete
        answer: Final extracted answer
        tool_calls: List of all tool calls made
        tool_counts: Dictionary tracking usage count per tool
        metadata: Additional execution metadata
    """
    question_id: int
    question: str

    # Attachments (e.g. image paths for image_inspector; injected by orchestrator)
    attachments: Optional[List[str]] = None

    # Conversation state
    messages: List[Dict[str, str]] = []
    current_output: str = ""

    # Execution tracking
    turn: int = 0
    finished: bool = False
    answer: Optional[str] = None

    # Tool usage tracking
    tool_calls: List[Dict[str, Any]] = []
    tool_counts: Dict[str, int] = {
        'web_search': 0,
        'code_generator': 0,
        'mind_map': 0,
        'text_inspector': 0,
        'image_inspector': 0,
    }

    # Output message history (assistant + tool turns, excluding system/user)
    output_messages: List[Dict[str, str]] = []

    # Structured memory (AgentFlow-inspired)
    query_analysis: str = ""
    action_history: List[Dict[str, str]] = []

    # Metadata
    metadata: Dict[str, Any] = {}

    def increment_tool_count(self, tool_name: str):
        """Increment the usage count for a tool.

        Args:
            tool_name: Name of the tool
        """
        if tool_name not in self.tool_counts:
            self.tool_counts[tool_name] = 0
        self.tool_counts[tool_name] += 1

    def get_tool_count(self, tool_name: str) -> int:
        """Get the usage count for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of times the tool has been called
        """
        return self.tool_counts.get(tool_name, 0)
