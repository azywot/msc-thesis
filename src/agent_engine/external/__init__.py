"""External integrations and API clients.

This module provides standalone implementations of external services.
"""

from .serper import SerperRM
from .tavily import TavilyRM

__all__ = ["SerperRM", "TavilyRM"]
