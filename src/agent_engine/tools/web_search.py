"""Web search tool using Serper API.

This tool provides web search functionality using the Serper API,
with caching support for efficiency.

Supports two modes:
1. Direct mode: Executes search directly and returns results
2. Sub-agent mode: Uses an LLM to analyze and summarize search results
"""

from typing import Any, Dict, List, Optional

from ..core.tool import BaseTool, ToolResult
from ..utils.logging import get_logger
from ..utils.parsing import strip_thinking_tags
from ..external.serper import SerperRM
from ..external.url_fetcher import fetch_page_content, extract_snippet_with_context

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Web search using Serper API.

    Supports direct mode (no LLM) or sub-agent mode (with LLM analysis).
    """

    def __init__(
        self,
        serper_api_key: str,
        search_cache: Optional[Dict[str, str]] = None,
        url_cache: Optional[Dict[str, str]] = None,
        top_k: int = 10,
        max_doc_len: int = 3000,
        model_provider = None,  # Optional: for sub-agent mode
        use_thinking: bool = False,  # Whether sub-agent uses thinking
        use_jina: bool = False,  # Whether to use Jina for URL fetching
        fetch_urls: bool = True  # Whether to fetch full page content
    ):
        """Initialize web search tool.

        Args:
            serper_api_key: Serper API key
            search_cache: Optional shared cache dictionary for search results
            url_cache: Optional shared cache dictionary for fetched URL content
            top_k: Number of search results to return
            max_doc_len: Maximum document length in characters
            model_provider: Optional model provider for sub-agent mode
            use_thinking: Whether sub-agent uses thinking mode
            use_jina: Whether to use Jina AI reader API for URL fetching
            fetch_urls: Whether to fetch full page content (vs just snippets)
        """
        self.serper_rm = SerperRM(serper_search_api_key=serper_api_key, k=top_k)
        self.search_cache = search_cache if search_cache is not None else {}
        self.url_cache = url_cache if url_cache is not None else {}
        self.top_k = top_k
        self.max_doc_len = max_doc_len
        self.model_provider = model_provider
        self.use_thinking = use_thinking
        self.use_jina = use_jina
        self.fetch_urls = fetch_urls
        self.direct_mode = model_provider is None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information using Serper API"

    def get_schema(self) -> Dict[str, Any]:
        """Return Qwen3 JSON Schema."""
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for current information, facts, news, or any "
                    "information not in your training data. Returns relevant search "
                    "results with titles, URLs, and snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to execute"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str) -> ToolResult:
        """Execute web search.

        Flow (mirrors multi-agent-tools):
        1. search_cache stores raw Serper result dicts (list[dict]) so cache files
           are interchangeable between the two repos.
        2. url_cache stores full page content keyed by URL (same as multi-agent-tools).
        3. URL fetching is always done here, before formatting, so url_cache is
           populated on every cache miss and any missing URLs are back-filled on hits.
        4. _format_results is a pure formatting step — it reads url_cache but never
           fetches.

        Args:
            query: Search query string

        Returns:
            ToolResult with formatted search results (direct mode) or
            LLM-analyzed results (sub-agent mode)
        """
        logger.info(f"Executing web search ({'sub-agent' if not self.direct_mode else 'direct'} mode): {query}")

        # Cache hit
        if query in self.search_cache:
            logger.info(f"Cache hit for: {query}")
            cached_results = self.search_cache[query]
            self._fetch_missing_urls(cached_results)
            formatted = self._format_results(cached_results, query)
            output = formatted if self.direct_mode else self._analyze_with_llm(query, formatted)
            return ToolResult(
                success=True,
                output=output,
                metadata={"cached": True, "query": query, "mode": "direct" if self.direct_mode else "sub-agent"},
            )

        # Cache miss: fetch from Serper, fetch page content, then format/analyse.
        try:
            results = self.serper_rm.forward(query)
            logger.info(f"Retrieved {len(results)} search results")

            # Persist raw results — same structure as multi-agent-tools search_cache.
            self.search_cache[query] = results

            # Fetch full page content and populate url_cache
            self._fetch_missing_urls(results)

            formatted_results = self._format_results(results, query)

            if self.direct_mode:
                return ToolResult(
                    success=True,
                    output=formatted_results,
                    metadata={"cached": False, "num_results": len(results), "query": query, "mode": "direct"},
                )

            output = self._analyze_with_llm(query, formatted_results)
            return ToolResult(
                success=True,
                output=output,
                metadata={"cached": False, "num_results": len(results), "query": query, "mode": "sub-agent"},
            )

        except Exception as e:
            logger.error(f"Search execution error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output="",
                metadata={"query": query},
                error=str(e),
            )

    def _fetch_missing_urls(self, results: list) -> None:
        """Fetch page content for any URLs not yet in url_cache.

        Mirrors fetch_urls() in multi-agent-tools/scripts/tools/run_search.py:
        collects uncached URLs, fetches them in one batch, and updates url_cache.
        No-op when fetch_urls=False.

        Args:
            results: List of raw Serper result dicts
        """
        if not self.fetch_urls:
            return

        urls_to_fetch = []
        snippets = {}
        for result in results:
            url = result.get('url', '')
            if url and url not in self.url_cache:
                urls_to_fetch.append(url)
                snippets[url] = result.get('snippets', [''])[0] if result.get('snippets') else ''

        if not urls_to_fetch:
            return

        logger.info(f"Fetching {len(urls_to_fetch)} URLs")
        try:
            fetched = fetch_page_content(urls_to_fetch, use_jina=self.use_jina, snippets=snippets)
            self.url_cache.update(fetched)
            logger.info(f"Cached {len(fetched)} URLs")
        except Exception as e:
            logger.error(f"URL fetch error: {e}", exc_info=True)

    def _analyze_with_llm(self, query: str, search_results: str) -> str:
        """Use LLM to analyze search results (sub-agent mode).

        Args:
            query: Original search query
            search_results: Formatted search results

        Returns:
            LLM-analyzed summary of search results
        """
        prompt = self.build_analysis_prompt(query, search_results)
        result = self.model_provider.generate([prompt])[0]

        # Strip thinking tags if present
        output = result.text
        if self.use_thinking:
            output = strip_thinking_tags(output)

        return output

    def build_analysis_prompt(self, query: str, formatted_results: str) -> str:
        """Build the sub-agent LLM prompt for search-result analysis.

        This is used by both single and batched sub-agent execution.
        """
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that analyzes web search results and provides "
                    "concise, accurate summaries."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\nSearch Results:\n{formatted_results}\n\n"
                    "Please analyze these search results and provide a clear, concise answer to the query. "
                    "Include relevant facts and cite sources when possible."
                ),
            },
        ]
        return self.model_provider.apply_chat_template(prompt_messages, use_thinking=self.use_thinking)

    def search_and_format(self, query: str) -> Dict[str, Any]:
        """Run Serper search, populate url_cache, and return raw + formatted results.

        Used for batch processing pipelines that need both the raw results and the
        formatted string. URL fetching is handled via _fetch_missing_urls so this
        method stays consistent with execute().
        """
        results = self.serper_rm.forward(query)
        self._fetch_missing_urls(results)
        formatted_results = self._format_results(results, query)
        urls_fetched = [r.get('url', '') for r in results if r.get('url') in self.url_cache]
        return {
            "results": results,
            "formatted_results": formatted_results,
            "urls_fetched": urls_fetched,
        }

    def _format_results(self, results: list, query: str) -> str:
        """Format raw Serper results into an LLM-readable string.

        Pure formatting — reads url_cache but never fetches. Call
        _fetch_missing_urls() first if fresh page content is needed.

        Args:
            results: List of raw Serper result dicts
            query: Original search query

        Returns:
            Formatted string with search results
        """
        if not results:
            return f"No results found for query: {query}"

        output_lines = [f"Search results for: {query}\n"]

        for idx, result in enumerate(results[:self.top_k], 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            snippet = result.get('snippets', [''])[0] if result.get('snippets') else ''

            output_lines.append(f"\n[{idx}] {title}")
            output_lines.append(f"URL: {url}")

            if self.fetch_urls and url in self.url_cache:
                full_content = self.url_cache[url]
                success, context = extract_snippet_with_context(full_content, snippet, self.max_doc_len)
                label = "Content (with context)" if success else "Content"
                output_lines.append(f"{label}: {context[:self.max_doc_len]}")
            elif snippet:
                if len(snippet) > self.max_doc_len:
                    snippet = snippet[:self.max_doc_len] + "..."
                output_lines.append(f"Snippet: {snippet}")

        return "\n".join(output_lines)

    def validate_args(self, **kwargs) -> bool:
        """Validate search arguments.

        Args:
            **kwargs: Tool arguments

        Returns:
            True if valid, False otherwise
        """
        if 'query' not in kwargs:
            return False
        query = kwargs['query']
        return isinstance(query, str) and len(query.strip()) > 0
