"""URL content fetching utilities with parallel execution support.

This module provides functions to fetch and extract text content from URLs,
with support for concurrent fetching to optimize performance.
"""

import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

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


def _remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def _f1_score(true_set: set, pred_set: set) -> float:
    """Calculate F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(
    full_text: str,
    snippet: str,
    context_chars: int = 2500
) -> Tuple[bool, str]:
    """Extract the sentence that best matches the snippet and its context.

    Mirrors multi-agent-tools/scripts/tools/bing_search.py:
    sentence-level F1 matching via sent_tokenize, context window around best sentence.

    Args:
        full_text: Full document text
        snippet: Snippet to search for
        context_chars: Characters of context before and after the best sentence

    Returns:
        Tuple of (success: bool, extracted_context: str)
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = _remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = _remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = _f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            return True, full_text[start_index:end_index]
        else:
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"
