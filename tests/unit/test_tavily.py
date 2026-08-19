"""``external/tavily`` — the Tavily client.

**No test here touches the network.**  ``TavilyRM`` imports ``TavilyClient``
inside its constructor, so the boundary is stubbed by installing a fake
``tavily`` module in ``sys.modules``; that also keeps the suite green on a
checkout without ``tavily-python`` installed.

``TavilyRM`` and ``SerperRM`` are used interchangeably behind
``web_tool_provider``, so the shape they return has to agree.  The last test
asserts that directly.
"""

import sys
import types

import pytest

from agent_engine.external.tavily import TavilyRM


class FakeClient:
    """Stand-in for ``tavily.TavilyClient``."""

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.calls = []
        self.response = {"results": []}
        self.error = None

    def search(self, query=None, search_depth=None, max_results=None):
        self.calls.append(
            {"query": query, "search_depth": search_depth, "max_results": max_results}
        )
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def fake_tavily(monkeypatch):
    """Install a fake ``tavily`` module and hand back the client instances it
    hands out, so a test can inspect or program them."""
    created = []

    def make_client(api_key=None):
        client = FakeClient(api_key=api_key)
        created.append(client)
        return client

    module = types.ModuleType("tavily")
    module.TavilyClient = make_client
    monkeypatch.setitem(sys.modules, "tavily", module)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    return created


def _results(n):
    return {
        "results": [
            {
                "title": f"Title {i}",
                "url": f"https://example.com/{i}",
                "content": f"Content {i}",
            }
            for i in range(n)
        ]
    }


# --- construction ---------------------------------------------------------


def test_a_missing_api_key_fails_at_construction(fake_tavily):
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        TavilyRM()


def test_the_api_key_falls_back_to_the_environment(monkeypatch, fake_tavily):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    rm = TavilyRM()

    assert rm.tavily_api_key == "from-env"
    assert fake_tavily[0].api_key == "from-env"


def test_an_explicit_key_beats_the_environment(monkeypatch, fake_tavily):
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    assert TavilyRM(tavily_api_key="explicit").tavily_api_key == "explicit"


def test_a_missing_package_is_reported_as_an_install_instruction(monkeypatch):
    """The import lives in ``__init__`` precisely so an unused provider costs
    nothing; the price is that the failure surfaces at construction."""
    # A module with no ``TavilyClient`` makes the ``from ... import`` raise
    # ImportError, the same class the real missing-package case raises.
    monkeypatch.setitem(sys.modules, "tavily", types.ModuleType("tavily"))

    with pytest.raises(RuntimeError, match="pip install tavily-python"):
        TavilyRM(tavily_api_key="k")


def test_the_search_depth_defaults_to_advanced(fake_tavily):
    assert TavilyRM(tavily_api_key="k").search_depth == "advanced"


# --- request shape --------------------------------------------------------


def test_the_search_call_passes_depth_and_max_results(fake_tavily):
    rm = TavilyRM(tavily_api_key="k", k=7, search_depth="basic")
    fake_tavily[0].response = _results(1)

    rm.forward("what is a heron")

    assert fake_tavily[0].calls == [
        {"query": "what is a heron", "search_depth": "basic", "max_results": 7}
    ]


# --- forward --------------------------------------------------------------


def test_results_are_remapped_to_title_url_content(fake_tavily):
    rm = TavilyRM(tavily_api_key="k")
    fake_tavily[0].response = _results(1)

    assert rm.forward("q") == [
        {"title": "Title 0", "url": "https://example.com/0", "content": "Content 0"}
    ]


def test_a_null_content_becomes_an_empty_string(fake_tavily):
    """Tavily returns ``null`` content for some results; ``or ""`` keeps a
    string in the field so downstream formatting never sees None."""
    rm = TavilyRM(tavily_api_key="k")
    fake_tavily[0].response = {"results": [{"title": "t", "url": "u", "content": None}]}

    assert rm.forward("q") == [{"title": "t", "url": "u", "content": ""}]


def test_a_response_without_results_yields_nothing(fake_tavily):
    rm = TavilyRM(tavily_api_key="k")
    fake_tavily[0].response = {"answer": "42"}

    assert rm.forward("q") == []


def test_the_k_limit_applies_across_all_queries_not_per_query(fake_tavily):
    rm = TavilyRM(tavily_api_key="k", k=3)
    fake_tavily[0].response = _results(3)

    assert len(rm.forward(["first", "second"])) == 3


def test_the_placeholder_query_is_skipped(fake_tavily):
    rm = TavilyRM(tavily_api_key="k")
    fake_tavily[0].response = _results(1)

    rm.forward(["Queries:", "real one"])
    assert [c["query"] for c in fake_tavily[0].calls] == ["real one"]


def test_a_failing_query_is_swallowed_and_returns_no_results(fake_tavily):
    rm = TavilyRM(tavily_api_key="k")
    fake_tavily[0].error = RuntimeError("tavily is down")

    assert rm.forward("q") == []


# --- usage accounting -----------------------------------------------------


def test_usage_counts_queries_and_resets(fake_tavily):
    rm = TavilyRM(tavily_api_key="k", k=10)
    fake_tavily[0].response = _results(1)

    rm.forward("one")
    rm.forward(["two", "three"])

    assert rm.get_usage_and_reset() == {"TavilyRM": 3}
    assert rm.get_usage_and_reset() == {"TavilyRM": 0}


# --- interchangeability with Serper ---------------------------------------


def test_both_providers_return_the_same_result_shape(fake_tavily, monkeypatch):
    """``web_tool_provider`` swaps these two without the tool noticing, so the
    keys have to match.  Note the Tavily docstring claims ``snippets`` and
    ``description`` keys — the code returns title/url/content, same as Serper,
    and that is what the caller reads."""
    from agent_engine.external import serper
    from agent_engine.external.serper import SerperRM

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"organic": [{"title": "t", "link": "u", "snippet": "c"}]}

    monkeypatch.setattr(serper.requests, "post", lambda *a, **k: FakeResponse())
    serper_result = SerperRM(serper_search_api_key="k").forward("q")

    rm = TavilyRM(tavily_api_key="k")
    fake_tavily[0].response = {"results": [{"title": "t", "url": "u", "content": "c"}]}
    tavily_result = rm.forward("q")

    assert serper_result == tavily_result
