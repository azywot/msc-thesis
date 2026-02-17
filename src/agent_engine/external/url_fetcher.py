"""URL content fetching utilities with parallel execution support.

This module provides functions to fetch and extract text content from URLs,
with support for concurrent fetching to optimize performance.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from ..utils.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_url(url: str, use_jina: bool = False, snippet: Optional[str] = None) -> str:
    """Extract text content from a single URL.
    
    Args:
        url: URL to fetch
        use_jina: Whether to use Jina AI reader API (if True, requires JINA_API_KEY)
        snippet: Optional snippet to help with context extraction
        
    Returns:
        Extracted text content
    """
    if use_jina:
        return _fetch_with_jina(url)
    else:
        return _fetch_with_requests(url)


def _fetch_with_requests(url: str, timeout: int = 10) -> str:
    """Fetch URL content using requests + BeautifulSoup.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except requests.Timeout:
        return f"Error: Request timeout for URL {url}"
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return f"Error: {e}"


def _fetch_with_jina(url: str) -> str:
    """Fetch URL content using Jina AI reader API.
    
    Args:
        url: URL to fetch
        
    Returns:
        Extracted text content
    """
    try:
        import os
        jina_api_key = os.getenv('JINA_API_KEY')
        if not jina_api_key:
            logger.warning("JINA_API_KEY not found, falling back to requests")
            return _fetch_with_requests(url)
        
        # Jina reader API
        reader_url = f"https://r.jina.ai/{url}"
        headers = {
            'Authorization': f'Bearer {jina_api_key}',
            'X-Return-Format': 'text'
        }
        
        response = requests.get(reader_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
        
    except Exception as e:
        logger.error(f"Jina API error for {url}: {e}")
        return _fetch_with_requests(url)


def fetch_page_content(
    urls: List[str],
    max_workers: int = 4,
    use_jina: bool = False,
    snippets: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Concurrently fetch content from multiple URLs.
    
    Args:
        urls: List of URLs to scrape
        max_workers: Maximum number of concurrent threads
        use_jina: Whether to use Jina for extraction
        snippets: Optional dictionary mapping URLs to their respective snippets
        
    Returns:
        Dictionary mapping URLs to extracted content
    """
    if not urls:
        return {}
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_text_from_url,
                url,
                use_jina,
                snippets.get(url) if snippets else None
            ): url
            for url in urls
        }
        
        for future in as_completed(futures):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
                logger.info(f"Successfully fetched {url}")
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
                logger.error(f"Error fetching {url}: {exc}")
            time.sleep(0.1)  # Simple rate limiting
    
    return results


def extract_snippet_with_context(
    full_text: str,
    snippet: str,
    context_chars: int = 3000
) -> tuple[bool, str]:
    """Extract snippet location from full text and return surrounding context.
    
    Args:
        full_text: Full document text
        snippet: Snippet to search for
        context_chars: Number of characters of context to include
        
    Returns:
        Tuple of (success: bool, extracted_context: str)
    """
    if not snippet or not full_text:
        return False, full_text[:context_chars * 2] if full_text else ""
    
    # Clean snippet for matching
    snippet_clean = snippet.lower().replace('<b>', '').replace('</b>', '').strip()
    full_text_lower = full_text.lower()
    
    # Try to find snippet
    pos = full_text_lower.find(snippet_clean[:100])  # Use first 100 chars for matching
    
    if pos != -1:
        # Found snippet, extract context around it
        start = max(0, pos - context_chars)
        end = min(len(full_text), pos + len(snippet_clean) + context_chars)
        context = full_text[start:end]
        return True, context
    else:
        # Snippet not found, return beginning of document
        return False, full_text[:context_chars * 2]
