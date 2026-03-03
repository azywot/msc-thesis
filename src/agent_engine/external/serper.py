"""Standalone Serper API implementation.

This module provides a clean implementation of Serper.dev search API
without external dependencies on dspy or other frameworks.
"""

import os
from typing import Dict, List, Union

import requests

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SerperRM:
    """Retrieve information from Google Search using Serper.dev API.

    This is a standalone implementation that does not depend on dspy or
    other external frameworks.
    """

    def __init__(
        self,
        serper_search_api_key: str = None,
        k: int = 3,
        query_params: Dict = None,
    ):
        """Initialize Serper search.

        Args:
            serper_search_api_key: API key from serper.dev
            k: Number of results to return
            query_params: Additional query parameters for Serper API
        """
        self.k = k
        self.usage = 0

        # Set up query parameters
        if query_params is None:
            self.query_params = {"num": k, "autocorrect": True, "page": 1}
        else:
            self.query_params = query_params
            self.query_params.update({"num": k})

        # Get API key
        self.serper_search_api_key = serper_search_api_key
        if not self.serper_search_api_key:
            self.serper_search_api_key = os.environ.get("SERPER_API_KEY")

        if not self.serper_search_api_key:
            raise RuntimeError(
                "You must supply serper_search_api_key or set environment variable SERPER_API_KEY"
            )

        self.base_url = "https://google.serper.dev"
        self.search_url = f"{self.base_url}/search"

    def _serper_runner(self, query_params: Dict) -> Dict:
        """Execute Serper API request.

        Args:
            query_params: Query parameters

        Returns:
            API response as dictionary
        """
        headers = {
            "X-API-KEY": self.serper_search_api_key,
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.search_url,
            headers=headers,
            json=query_params,
            timeout=30
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Serper API error: {response.reason}, "
                f"status code {response.status_code}"
            )

        return response.json()

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        exclude_urls: List[str] = None
    ) -> List[Dict]:
        """Search using Serper API.

        Args:
            query_or_queries: Single query string or list of queries
            exclude_urls: URLs to exclude (currently unused, for interface compatibility)

        Returns:
            List of search results, each with keys:
                - title: Page title
                - url: Page URL
                - content: Search result snippet
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

            # Prepare query parameters
            query_params = self.query_params.copy()
            query_params["q"] = query
            query_params["type"] = "search"

            try:
                # Execute search
                result = self._serper_runner(query_params)

                # Extract organic results
                organic_results = result.get("organic", [])

                for organic in organic_results:
                    collected_results.append({
                        "title": organic.get("title", ""),
                        "url": organic.get("link", ""),
                        "content": organic.get("snippet", ""),
                    })

            except Exception as e:
                logger.error(f"Error searching query '{query}': {e}")
                continue

        return collected_results[: self.k]

    def get_usage_and_reset(self) -> Dict[str, int]:
        """Get API usage count and reset counter.

        Returns:
            Dictionary with usage count
        """
        usage = self.usage
        self.usage = 0
        return {"SerperRM": usage}
