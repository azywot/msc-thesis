"""``external/url_fetcher`` — HTML extraction, error text, and snippet windowing.

**No test here touches the network.**  ``requests.get`` is replaced at the
module boundary in every test that could reach out; a test that forgot would
hang on a real socket rather than fail fast, so the stub is applied through
fixtures rather than per-call.

The error paths matter more than the happy path: this module never raises.  It
returns the error *as the page text*, which then flows into the model's context
as if it were content.  Those strings are part of the prompt, so they are
pinned exactly.
"""

import pytest
import requests

from agent_engine.external import url_fetcher
from agent_engine.external.url_fetcher import (
    _f1_score,
    extract_snippet_with_context,
    extract_text_from_url,
    fetch_page_content,
)

HTML = b"""
<html>
  <head><style>body { color: red; }</style></head>
  <body>
    <nav>Home About Contact</nav>
    <header>Site Header</header>
    <p>The first paragraph.</p>
    <p>The second paragraph.</p>
    <script>console.log("tracking");</script>
    <footer>Copyright 2026</footer>
  </body>
</html>
"""


class FakeResponse:
    def __init__(self, content=b"", status_code=200, text=""):
        self.content = content
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if anything reaches for the network without opting in."""

    def forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real HTTP request")

    monkeypatch.setattr(url_fetcher.requests, "get", forbidden)
    return monkeypatch


# --- happy path -----------------------------------------------------------


def test_boilerplate_elements_are_removed_from_the_extracted_text(no_network):
    no_network.setattr(url_fetcher.requests, "get", lambda *a, **k: FakeResponse(content=HTML))

    text = extract_text_from_url("https://example.com")

    assert "The first paragraph." in text
    assert "The second paragraph." in text
    for boilerplate in ("color: red", "tracking", "Home About Contact", "Site Header", "Copyright"):
        assert boilerplate not in text


def test_the_request_carries_a_browser_user_agent_and_a_timeout(no_network):
    seen = {}

    def capture(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(content=b"<p>hi</p>")

    no_network.setattr(url_fetcher.requests, "get", capture)
    extract_text_from_url("https://example.com")

    assert seen["url"] == "https://example.com"
    assert "Mozilla/5.0" in seen["headers"]["User-Agent"]
    assert seen["timeout"] == 10


# --- error paths: the string *is* the content -----------------------------


def test_a_timeout_returns_an_error_string_not_an_exception(no_network):
    def timeout(*args, **kwargs):
        raise requests.Timeout("too slow")

    no_network.setattr(url_fetcher.requests, "get", timeout)

    assert extract_text_from_url("https://example.com") == (
        "Error: Request timeout for URL https://example.com"
    )


def test_an_http_error_status_returns_an_error_string(no_network):
    no_network.setattr(url_fetcher.requests, "get", lambda *a, **k: FakeResponse(status_code=500))

    result = extract_text_from_url("https://example.com")
    assert result.startswith("Error fetching https://example.com: ")
    assert "500" in result


def test_a_connection_error_returns_an_error_string(no_network):
    def refused(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    no_network.setattr(url_fetcher.requests, "get", refused)

    assert extract_text_from_url("https://example.com") == (
        "Error fetching https://example.com: connection refused"
    )


def test_a_non_requests_exception_uses_the_shorter_error_form(no_network):
    """A different prefix on purpose: ``Error: {e}`` without the URL.  Anything
    grepping run logs for fetch failures has to match both shapes."""

    def boom(*args, **kwargs):
        raise ValueError("something else went wrong")

    no_network.setattr(url_fetcher.requests, "get", boom)

    assert extract_text_from_url("https://example.com") == "Error: something else went wrong"


# --- jina branch ----------------------------------------------------------


def test_without_a_jina_key_the_jina_path_falls_back_to_requests(no_network):
    no_network.delenv("JINA_API_KEY", raising=False)
    no_network.setattr(
        url_fetcher.requests, "get", lambda *a, **k: FakeResponse(content=b"<p>x</p>")
    )

    assert extract_text_from_url("https://example.com", use_jina=True) == "x"


def test_the_jina_reader_url_is_used_when_a_key_is_present(no_network):
    seen = {}

    def capture(url, headers=None, timeout=None):
        seen["url"] = url
        return FakeResponse(text="jina text")

    no_network.setenv("JINA_API_KEY", "key-123")
    no_network.setattr(url_fetcher.requests, "get", capture)

    assert extract_text_from_url("https://example.com", use_jina=True) == "jina text"
    assert seen["url"] == "https://r.jina.ai/https://example.com"


def test_a_jina_failure_falls_back_to_requests(no_network):
    calls = []

    def maybe_fail(url, headers=None, timeout=None):
        calls.append(url)
        if url.startswith("https://r.jina.ai/"):
            raise requests.ConnectionError("jina down")
        return FakeResponse(content=b"<p>fallback</p>")

    no_network.setenv("JINA_API_KEY", "key-123")
    no_network.setattr(url_fetcher.requests, "get", maybe_fail)

    assert extract_text_from_url("https://example.com", use_jina=True) == "fallback"
    assert len(calls) == 2


# --- fetch_page_content ---------------------------------------------------


def test_no_urls_short_circuits_without_a_thread_pool(no_network):
    assert fetch_page_content([]) == {}


def test_every_url_appears_in_the_result_mapping(no_network):
    no_network.setattr(url_fetcher.time, "sleep", lambda _s: None)
    no_network.setattr(
        url_fetcher.requests, "get", lambda *a, **k: FakeResponse(content=b"<p>page</p>")
    )

    result = fetch_page_content(["https://a.test", "https://b.test"], max_workers=2)
    assert result == {"https://a.test": "page", "https://b.test": "page"}


def test_one_failing_url_does_not_lose_the_others(no_network):
    """Per-URL isolation: a failure becomes that URL's text, and the successful
    siblings are still returned."""
    no_network.setattr(url_fetcher.time, "sleep", lambda _s: None)

    def selective(url, headers=None, timeout=None):
        if "bad" in url:
            raise requests.Timeout("nope")
        return FakeResponse(content=b"<p>good page</p>")

    no_network.setattr(url_fetcher.requests, "get", selective)

    result = fetch_page_content(["https://good.test", "https://bad.test"], max_workers=2)
    assert result["https://good.test"] == "good page"
    assert result["https://bad.test"] == "Error: Request timeout for URL https://bad.test"


def test_the_snippet_argument_is_accepted_and_ignored(no_network):
    """``extract_text_from_url`` takes ``snippet`` for interface compatibility
    and never reads it; snippet windowing happens later, in the caller."""
    no_network.setattr(
        url_fetcher.requests, "get", lambda *a, **k: FakeResponse(content=b"<p>x</p>")
    )

    with_snippet = extract_text_from_url("https://example.com", snippet="anything")
    without = extract_text_from_url("https://example.com")
    assert with_snippet == without


# --- snippet windowing ----------------------------------------------------


def test_no_snippet_returns_the_leading_window_and_reports_failure():
    text = "x" * 100
    found, context = extract_snippet_with_context(text, "", context_chars=10)
    assert found is False
    assert context == "x" * 20


def test_no_text_returns_empty():
    assert extract_snippet_with_context("", "snippet") == (False, "")


def test_a_matching_sentence_is_returned_with_surrounding_context():
    full = (
        "Unrelated opening sentence. "
        "The capital of France is Paris and it is lovely. "
        "Another unrelated closing sentence."
    )
    found, context = extract_snippet_with_context(full, "capital of France is Paris", 20)

    assert found is True
    assert "The capital of France is Paris" in context


def test_an_unmatched_snippet_falls_back_to_the_leading_window():
    full = "Sentences about entirely different subjects. More of the same here."
    found, context = extract_snippet_with_context(full, "quantum chromodynamics lattice", 10)

    assert found is False
    assert context == full[:20]


def test_snippet_matching_ignores_case_punctuation_and_bold_tags():
    full = "The Eiffel Tower, built in 1889, stands in Paris."
    found, _ = extract_snippet_with_context(full, "<b>eiffel tower</b> built in 1889", 10)
    assert found is True


def test_only_the_first_50k_characters_are_searched():
    """A hard cap on tokenisation cost.  A sentence past it cannot be found."""
    filler = "Filler sentence here. " * 3000  # comfortably over 50k chars
    full = filler + "The unique marker phrase appears late."
    found, _ = extract_snippet_with_context(full, "unique marker phrase appears late", 10)
    assert found is False


def test_f1_is_zero_when_nothing_overlaps():
    """Guards the division: an empty intersection returns before computing
    precision, so an empty prediction set cannot raise ZeroDivisionError."""
    assert _f1_score({"a"}, set()) == 0.0
    assert _f1_score({"a"}, {"b"}) == 0.0


def test_f1_is_one_for_identical_sets():
    assert _f1_score({"a", "b"}, {"a", "b"}) == 1.0
