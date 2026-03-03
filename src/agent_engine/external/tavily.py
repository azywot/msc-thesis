"""Standalone Tavily API implementation.

This module provides a clean implementation of Tavily.com search API
without external dependencies on dspy or other frameworks.
"""

import os
from typing import Dict, List, Union

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TavilyRM:
    """Retrieve information from web search using Tavily API.

    This is a standalone implementation that does not depend on dspy or
    other external frameworks.
    """

    def __init__(
        self,
        tavily_api_key: str = None,
        k: int = 3,
        search_depth: str = "advanced",
    ):
        """Initialize Tavily search.

        Args:
            tavily_api_key: API key from tavily.com
            k: Number of results to return
            search_depth: Search depth - "basic" or "advanced"
        """
        self.k = k
        self.usage = 0
        self.search_depth = search_depth

        # Get API key
        self.tavily_api_key = tavily_api_key
        if not self.tavily_api_key:
            self.tavily_api_key = os.environ.get("TAVILY_API_KEY")

        if not self.tavily_api_key:
            raise RuntimeError(
                "You must supply tavily_api_key or set environment variable TAVILY_API_KEY"
            )

        # Import TavilyClient here to avoid requiring tavily-python if not used
        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=self.tavily_api_key)
        except ImportError:
            raise RuntimeError(
                "tavily-python package not installed. Install it with: pip install tavily-python"
            )

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        exclude_urls: List[str] = None
    ) -> List[Dict]:
        """Search using Tavily API.

        Args:
            query_or_queries: Single query string or list of queries
            exclude_urls: URLs to exclude (currently unused, for interface compatibility)

        Returns:
            List of search results, each with keys:
                - snippets: List of text snippets
                - title: Page title
                - url: Page URL
                - description: Content/snippet from Tavily
        """
        # Convert to list if single query
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )

        self.usage += len(queries)
        collected_results = []

        for query in queries:
            if query == "Queries:":  # Skip placeholder
                continue

            try:
                # Execute search with advanced depth
                response = self.client.search(
                    query=query,
                    search_depth=self.search_depth,
                    max_results=self.k
                )

                # Extract results from Tavily response
                # Tavily returns: {"results": [{"url": ..., "title": ..., "content": ...}, ...]}
                results = response.get("results", [])

                for result in results:
                    collected_results.append({
                        "snippets": [result.get("content", "")],
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("content", ""),
                    })

            except Exception as e:
                logger.error(f"Error searching query '{query}' with Tavily: {e}")
                continue

        return collected_results[: self.k]

    def get_usage_and_reset(self) -> Dict[str, int]:
        """Get API usage count and reset counter.

        Returns:
            Dictionary with usage count
        """
        usage = self.usage
        self.usage = 0
        return {"TavilyRM": usage}
