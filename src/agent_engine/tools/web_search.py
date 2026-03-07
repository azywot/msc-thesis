"""Web search tool using Serper or Tavily API.

This tool provides web search functionality using either Serper or Tavily API,
with caching support for efficiency.

Supports two modes:
1. Direct mode: Executes search directly and returns results
2. Sub-agent mode: Uses an LLM to analyze and summarize search results
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.tool import BaseTool, ToolResult
from ..utils.logging import get_logger
from ..utils.parsing import strip_thinking_tags
from ..external.serper import SerperRM
from ..external.tavily import TavilyRM
from ..external.url_fetcher import fetch_page_content, extract_snippet_with_context

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Web search using Serper or Tavily API.

    Supports direct mode (no LLM) or sub-agent mode (with LLM analysis).
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "serper",
        search_cache: Optional[Dict[str, str]] = None,
        url_cache: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        max_doc_len: int = 3000,
        model_provider = None,  # Optional: for sub-agent mode
        use_thinking: bool = False,  # Whether sub-agent uses thinking
        use_jina: bool = False,  # Whether to use Jina for URL fetching
        fetch_urls: bool = True,  # Whether to fetch full page content
        cache_manager=None,  # Optional: persist cache on each update (for parallel runs)
    ):
        """Initialize web search tool.

        Args:
            api_key: API key for the selected provider (Serper or Tavily)
            provider: Web search provider - "serper" or "tavily"
                     Note: Tavily provides pre-cleaned content, so URL fetching is skipped
            search_cache: Optional shared cache dictionary for search results
            url_cache: Optional shared cache dictionary for fetched URL content
                      (only used for Serper; Tavily doesn't require URL fetching)
            top_k: Number of search results to return
            max_doc_len: Maximum document length in characters
            model_provider: Optional model provider for sub-agent mode
            use_thinking: Whether sub-agent uses thinking mode
            use_jina: Whether to use Jina AI reader API for URL fetching (Serper only)
            fetch_urls: Whether to fetch full page content (Serper only; ignored for Tavily)
            cache_manager: Optional cache manager for atomic cache persistence
        """
        self.provider = provider.lower()

        # Initialize the appropriate search provider
        if self.provider == "serper":
            self.search_rm: Union[SerperRM, TavilyRM] = SerperRM(
                serper_search_api_key=api_key, k=top_k
            )
        elif self.provider == "tavily":
            self.search_rm = TavilyRM(
                tavily_api_key=api_key, k=top_k, search_depth="advanced"
            )
        else:
            raise ValueError(f"Unknown web search provider: {provider}. Must be 'serper' or 'tavily'")

        self.search_cache = search_cache if search_cache is not None else {}
        self.url_cache = url_cache if url_cache is not None else {}
        self.top_k = top_k
        self.max_doc_len = max_doc_len
        self.model_provider = model_provider
        self.use_thinking = use_thinking
        self.use_jina = use_jina
        self.fetch_urls = fetch_urls
        self.cache_manager = cache_manager
        self.direct_mode = model_provider is None

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return f"Search the web for information using {self.provider.capitalize()} API"

    def get_schema(self) -> Dict[str, Any]:
        """Return Qwen3 JSON Schema."""
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web for information to help answer questions. Use this when you need to find facts, verify information, or get up-to-date knowledge that you don't have."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web. Be specific and include relevant keywords for better results."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str, prev_reasoning: str = "") -> ToolResult:
        """Execute web search.

        Flow:
        1. search_cache stores raw search result dicts (list[dict]) for both Serper and Tavily.
        2. url_cache stores full page content keyed by URL (Serper only; Tavily doesn't use this).
        3. URL fetching:
           - Serper: Fetches full page content before formatting, populates url_cache
           - Tavily: Skipped (content is already provided in search results)
        4. _format_results is a pure formatting step — reads url_cache for Serper,
           uses content field directly for Tavily.

        Args:
            query: Search query string

        Returns:
            ToolResult with formatted search results (direct mode) or
            LLM-analyzed results (sub-agent mode)
        """
        logger.info(f"Executing web search ({'sub-agent' if not self.direct_mode else 'direct'} mode): {query}")

        # Cache hit
        if query in self.search_cache:
            cached_results = self.search_cache[query]
            logger.info(f"Cache hit for: {query} ({len(cached_results)} results)")
            # Only fetch URLs for Serper (Tavily already provides cleaned content)
            if self.provider == "serper":
                url_cache_updated = self._fetch_missing_urls(cached_results)
                if self.cache_manager and url_cache_updated:
                    self.cache_manager.save_url_cache()
            formatted = self._format_results(cached_results, query)

            if self.direct_mode:
                output, usage = formatted, None
            else:
                output, usage = self._analyze_with_llm(query, formatted, prev_reasoning)
            return ToolResult(
                success=True,
                output=output,
                metadata={"cached": True, "query": query, "mode": "direct" if self.direct_mode else "sub-agent"},
                usage=usage,
            )

        # Cache miss: fetch from search provider, fetch page content, then format/analyse.
        try:
            raw_results = self.search_rm.forward(query)
            logger.info(f"Retrieved {len(raw_results)} search results from {self.provider}")

            # Persist raw results — same structure as multi-agent-tools search_cache.
            # Normalize so we only ever store list-of-dicts (Serper shape).
            results = self._normalize_search_results(raw_results)
            self.search_cache[query] = results

            # Fetch full page content and populate url_cache (only for Serper)
            # Tavily already provides cleaned content, no URL fetching needed
            if self.provider == "serper":
                self._fetch_missing_urls(results)

            if self.cache_manager:
                if self.provider == "serper":
                    self.cache_manager.save_caches()
                else:
                    # For Tavily, only save search cache (no URL cache needed)
                    self.cache_manager.save_search_cache()

            formatted_results = self._format_results(results, query)

            if self.direct_mode:
                return ToolResult(
                    success=True,
                    output=formatted_results,
                    metadata={"cached": False, "num_results": len(results), "query": query, "mode": "direct"},
                )

            output, usage = self._analyze_with_llm(query, formatted_results, prev_reasoning)
            return ToolResult(
                success=True,
                output=output,
                metadata={"cached": False, "num_results": len(results), "query": query, "mode": "sub-agent"},
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Search execution error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output="",
                metadata={"query": query},
                error=str(e),
            )

    @staticmethod
    def _normalize_search_results(results: list) -> list:
        """Return a list containing only dict items.

        Ensures we never persist non-dict entries to search_cache.
        """
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)]

    def _fetch_missing_urls(self, results: list) -> bool:
        """Fetch page content for any URLs not yet in url_cache.

        Mirrors fetch_urls() in multi-agent-tools/scripts/tools/run_search.py:
        collects uncached URLs, fetches them in one batch, and updates url_cache.
        No-op when fetch_urls=False.

        Args:
            results: List of raw Serper result dicts (normalized at load/write).

        Returns:
            True if url_cache was updated (new URLs fetched).
        """
        if not self.fetch_urls:
            return False

        urls_to_fetch = []
        snippets = {}
        for result in results or []:
            url = result.get('url', '')
            if url and url not in self.url_cache:
                urls_to_fetch.append(url)
                snippets[url] = result.get('content', '')

        if not urls_to_fetch:
            return False

        logger.info(f"Fetching {len(urls_to_fetch)} URLs")
        try:
            fetched = fetch_page_content(urls_to_fetch, use_jina=self.use_jina, snippets=snippets)
            self.url_cache.update(fetched)
            logger.info(f"Cached {len(fetched)} URLs")
            return bool(fetched)
        except Exception as e:
            logger.error(f"URL fetch error: {e}", exc_info=True)
            return False

    def _analyze_with_llm(
        self,
        query: str,
        search_results: str,
        prev_reasoning: str = "",
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """Use LLM to analyze search results (sub-agent mode).

        Args:
            query: Original search query
            search_results: Formatted search results
            prev_reasoning: Previous reasoning steps (MAT-style context)

        Returns:
            Tuple of (LLM-analyzed summary, token usage dict or None)
        """
        prompt = self.build_analysis_prompt(query, search_results, prev_reasoning)
        result = self.model_provider.generate([prompt])[0]
        output = strip_thinking_tags(result.text)
        return output, result.usage

    def build_analysis_prompt(
        self,
        query: str,
        formatted_results: str,
        prev_reasoning: str = "",
    ) -> str:
        """Build the sub-agent prompt for web-page analysis.

        Mirrors the multi-agent-tools get_webpage_to_reasonchain_instruction prompt.
        prev_reasoning is extracted from state by the orchestrator (MAT-style).
        """
        instruction = f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{query}

- **Searched Web Pages:**  
{formatted_results}

Now you should analyze each web page and find helpful information based on the current search query "{query}" and previous reasoning steps.
"""

        prompt_messages = [{"role": "user", "content": instruction}]
        prompt = self.model_provider.apply_chat_template(prompt_messages, use_thinking=self.use_thinking)

        return prompt

    def search_and_format(self, query: str) -> Dict[str, Any]:
        """Run search and return a batch-friendly payload.

        This is used by the batched orchestrator path:
        - Returns raw search results (cache-compatible)
        - Returns a list of uncached URLs and their snippets for batch URL fetching (Serper only)
        - Does NOT fetch URLs (that is done in a single batch across jobs)
        - For Tavily, returns empty urls_to_fetch since content is already provided
        """
        cached = False
        if query in self.search_cache:
            results = self.search_cache[query]
            cached = True
            logger.info(f"Cache hit for: {query} ({len(results)} results)")
        else:
            raw = self.search_rm.forward(query, exclude_urls=[])
            results = self._normalize_search_results(raw)
            self.search_cache[query] = results
            logger.info(f"Retrieved {len(results)} search results from {self.provider}")
            if self.cache_manager:
                self.cache_manager.save_search_cache()

        urls_to_fetch: List[str] = []
        url_snippets: Dict[str, str] = {}

        # Only collect URLs to fetch for Serper (Tavily already provides content)
        if self.provider == "serper":
            for r in (results or [])[: self.top_k]:
                url = (r.get("url", "") or "").strip()
                if not url:
                    continue
                snippet = r.get("content", "").replace("<b>", "").replace("</b>", "")

                # Collect uncached URLs only
                if url not in self.url_cache:
                    urls_to_fetch.append(url)
                    url_snippets[url] = snippet

        return {
            "results": results,
            "urls_to_fetch": urls_to_fetch,
            "url_snippets": url_snippets,
            "cached": cached,
            "query": query,
        }

    def _format_results(self, results: list, query: str) -> str:
        """Format raw search results into a document string.

        Pure formatting — reads url_cache but never fetches. Call
        _fetch_missing_urls() first if fresh page content is needed.

        Args:
            results: List of raw search result dicts (from Serper or Tavily)
            query: Original search query

        Returns:
            Repeated blocks of "**Web Page i:**" followed by JSON.
        """
        if not results:
            return f"No results found for query: {query}"

        formatted_documents = ""
        for i, doc_info in enumerate((results or [])[: self.top_k]):
            url = (doc_info.get("url", "") or "").strip()

            if self.provider == "tavily":
                base = doc_info.get("content") or ""
                content = base[: self.max_doc_len * 2] if base else ""
            else:
                hint = doc_info.get("content", "").replace("<b>", "").replace("</b>", "")
                raw_context = self.url_cache.get(url, "") if (self.fetch_urls and url) else ""
                if raw_context:
                    _, focused = extract_snippet_with_context(
                        raw_context,
                        hint,
                        context_chars=self.max_doc_len,
                    )
                    content = focused
                else:
                    content = ""

            minimal_doc = {
                "title": doc_info.get("title", ""),
                "url": url,
                "content": content,
            }

            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += json.dumps(minimal_doc, ensure_ascii=False, indent=2) + "\n"

        return formatted_documents.strip()

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
