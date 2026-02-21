"""Web search tool using Serper API.

This tool provides web search functionality using the Serper API,
with caching support for efficiency.

Supports two modes:
1. Direct mode: Executes search directly and returns results
2. Sub-agent mode: Uses an LLM to analyze and summarize search results
"""

import json
from typing import Any, Dict, List, Optional

from ..core.tool import BaseTool, ToolResult
from ..utils.logging import get_logger
from ..utils.parsing import subagent_output_for_orchestrator
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

        output = subagent_output_for_orchestrator(result.text)
        return output

    def build_analysis_prompt(self, query: str, formatted_results: str) -> str:
        """Build the sub-agent prompt for web-page analysis.

        Kept intentionally close to the legacy multi-agent-tools prompt shape so
        sub-agent mode behaves similarly across repositories.
        """
        prev_reasoning = ""
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
        return self.model_provider.apply_chat_template(prompt_messages, use_thinking=self.use_thinking)

    def search_and_format(self, query: str) -> Dict[str, Any]:
        """Run Serper search and return a batch-friendly payload.

        This is used by the batched orchestrator path:
        - Returns raw Serper results (cache-compatible)
        - Returns a list of uncached URLs and their snippets for batch URL fetching
        - Does NOT fetch URLs (that is done in a single batch across jobs)
        """
        cached = False
        if query in self.search_cache:
            results = self.search_cache[query]
            cached = True
        else:
            results = self.serper_rm.forward(query, exclude_urls=[])
            self.search_cache[query] = results

        urls_to_fetch: List[str] = []
        url_snippets: Dict[str, str] = {}
        for r in (results or [])[: self.top_k]:
            url = (r.get("url", "") or "").strip()
            if not url:
                continue
            snippet = ""
            if r.get("snippets"):
                snippet = (r.get("snippets") or [""])[0] or ""
            if not snippet:
                snippet = r.get("description", "") or ""
            snippet = snippet.replace("<b>", "").replace("</b>", "")

            # Collect uncached URLs only (legacy behavior).
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
        """Format raw Serper results into a legacy-compatible document string.

        Pure formatting — reads url_cache but never fetches. Call
        _fetch_missing_urls() first if fresh page content is needed.

        Args:
            results: List of raw Serper result dicts
            query: Original search query

        Returns:
            A document string similar to multi-agent-tools' `to_doc()` output:
            repeated blocks of "**Web Page i:**" followed by JSON.
        """
        if not results:
            return f"No results found for query: {query}"

        formatted_documents = ""
        for i, doc_info in enumerate(results[: self.top_k]):
            url = (doc_info.get("url", "") or "").strip()
            raw_context = self.url_cache.get(url, "") if (self.fetch_urls and url) else ""

            snippet = ""
            if doc_info.get("snippets"):
                snippet = (doc_info.get("snippets") or [""])[0] or ""
            if not snippet:
                snippet = doc_info.get("description", "") or ""
            snippet = snippet.replace("<b>", "").replace("</b>", "")

            success, filtered_context = extract_snippet_with_context(raw_context, snippet, context_chars=self.max_doc_len)
            context = filtered_context if success else (raw_context[: self.max_doc_len * 2] if raw_context else "")

            # Mutate in-place so search_cache persists the enriched structure
            # (matches legacy cache shape which includes `snippet` and `context`).
            doc_info["snippet"] = snippet
            doc_info["context"] = context

            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

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
