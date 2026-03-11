"""State management for agentic reasoning execution.

This module defines the state tracking for a single question's execution,
including conversation history, tool usage, and metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionState:
    """Tracks state of a single question execution.

    This dataclass maintains all information about the execution of a single
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
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_output: str = ""

    # Execution tracking
    turn: int = 0
    finished: bool = False
    answer: Optional[str] = None

    # Tool usage tracking
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_counts: Dict[str, int] = field(default_factory=lambda: {
        'web_search': 0,
        'code_generator': 0,
        'mind_map': 0,
        'text_inspector': 0,
        'image_inspector': 0,
    })

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, tool)
            content: Message content
        """
        self.messages.append({"role": role, "content": content})

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of the state
        """
        return {
            "question_id": self.question_id,
            "question": self.question,
            "messages": self.messages,
            "current_output": self.current_output,
            "turn": self.turn,
            "finished": self.finished,
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "tool_counts": self.tool_counts,
            "metadata": self.metadata,
            "attachments": self.attachments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionState":
        """Create ExecutionState from dictionary.

        Args:
            data: Dictionary with state data

        Returns:
            ExecutionState instance
        """
        return cls(**data)
