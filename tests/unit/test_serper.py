"""``external/serper`` — the Serper.dev client.

**No test here touches the network**; ``requests.post`` is replaced at the
module boundary and the default fixture makes an unstubbed call fail loudly.

The behaviour that matters downstream is the swallow-and-continue error
handling: a failed query does not raise, it silently contributes no results.
That is what makes a run finish with a partially-populated cache rather than
crashing, and it is also why a quota outage looks like "the web tool found
nothing" in the logs.
"""

import pytest

from agent_engine.external import serper
from agent_engine.external.serper import SerperRM


def _organic(n):
    return {
        "organic": [
            {"title": f"Title {i}", "link": f"https://example.com/{i}", "snippet": f"Snippet {i}"}
            for i in range(n)
        ]
    }


class FakeResponse:
    def __init__(self, payload=None, status_code=200, reason="OK"):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.reason = reason

    def json(self):
        return self._payload


@pytest.fixture
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test attempted a real HTTP request")

    monkeypatch.setattr(serper.requests, "post", forbidden)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    return monkeypatch


# --- construction ---------------------------------------------------------


def test_a_missing_api_key_fails_at_construction(no_network):
    """Better here than mid-run: the constructor is reached during setup, so a
    misconfigured job dies before consuming GPU time."""
    with pytest.raises(RuntimeError, match="SERPER_API_KEY"):
        SerperRM()


def test_the_api_key_falls_back_to_the_environment(no_network):
    no_network.setenv("SERPER_API_KEY", "from-env")
    assert SerperRM().serper_search_api_key == "from-env"


def test_an_explicit_key_beats_the_environment(no_network):
    no_network.setenv("SERPER_API_KEY", "from-env")
    assert SerperRM(serper_search_api_key="explicit").serper_search_api_key == "explicit"


def test_default_query_params(no_network):
    assert SerperRM(serper_search_api_key="k", k=5).query_params == {
        "num": 5,
        "autocorrect": True,
        "page": 1,
    }


def test_custom_query_params_keep_their_keys_but_lose_num(no_network):
    """``num`` is always overwritten with ``k`` — the two cannot disagree."""
    rm = SerperRM(serper_search_api_key="k", k=3, query_params={"num": 99, "gl": "nl"})
    assert rm.query_params == {"num": 3, "gl": "nl"}


# --- request shape --------------------------------------------------------


def test_the_request_carries_the_key_the_query_and_a_timeout(no_network):
    seen = {}

    def capture(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse(_organic(1))

    no_network.setattr(serper.requests, "post", capture)
    SerperRM(serper_search_api_key="key-123").forward("what is a heron")

    assert seen["url"] == "https://google.serper.dev/search"
    assert seen["headers"]["X-API-KEY"] == "key-123"
    assert seen["json"]["q"] == "what is a heron"
    assert seen["json"]["type"] == "search"
    assert seen["timeout"] == 30


def test_a_non_200_response_raises_inside_the_runner(no_network):
    no_network.setattr(
        serper.requests,
        "post",
        lambda *a, **k: FakeResponse(status_code=429, reason="Too Many Requests"),
    )
    rm = SerperRM(serper_search_api_key="k")

    with pytest.raises(RuntimeError, match="status code 429"):
        rm._serper_runner({"q": "x"})


# --- forward --------------------------------------------------------------


def test_organic_results_are_remapped_to_title_url_content(no_network):
    no_network.setattr(serper.requests, "post", lambda *a, **k: FakeResponse(_organic(1)))

    assert SerperRM(serper_search_api_key="k").forward("q") == [
        {"title": "Title 0", "url": "https://example.com/0", "content": "Snippet 0"}
    ]


def test_missing_fields_become_empty_strings(no_network):
    no_network.setattr(serper.requests, "post", lambda *a, **k: FakeResponse({"organic": [{}]}))

    assert SerperRM(serper_search_api_key="k").forward("q") == [
        {"title": "", "url": "", "content": ""}
    ]


def test_a_response_without_organic_results_yields_nothing(no_network):
    no_network.setattr(
        serper.requests, "post", lambda *a, **k: FakeResponse({"answerBox": {"answer": "42"}})
    )
    assert SerperRM(serper_search_api_key="k").forward("q") == []


def test_a_list_of_queries_is_searched_in_one_call(no_network):
    queries = []

    def capture(url, headers=None, json=None, timeout=None):
        queries.append(json["q"])
        return FakeResponse(_organic(1))

    no_network.setattr(serper.requests, "post", capture)
    SerperRM(serper_search_api_key="k", k=10).forward(["first", "second"])

    assert queries == ["first", "second"]


def test_the_k_limit_applies_across_all_queries_not_per_query(no_network):
    """A subtlety worth knowing when tuning ``k``: two queries at k=3 return 3
    results total, and they all come from the first query."""
    no_network.setattr(serper.requests, "post", lambda *a, **k: FakeResponse(_organic(3)))

    results = SerperRM(serper_search_api_key="k", k=3).forward(["first", "second"])
    assert len(results) == 3


def test_the_placeholder_query_is_skipped(no_network):
    queries = []

    def capture(url, headers=None, json=None, timeout=None):
        queries.append(json["q"])
        return FakeResponse(_organic(1))

    no_network.setattr(serper.requests, "post", capture)
    SerperRM(serper_search_api_key="k").forward(["Queries:", "real one"])

    assert queries == ["real one"]


def test_a_failing_query_is_swallowed_and_the_rest_still_run(no_network):
    """No exception escapes ``forward``: an API error costs its query's results
    and nothing else."""

    def selective(url, headers=None, json=None, timeout=None):
        if json["q"] == "bad":
            return FakeResponse(status_code=500, reason="Server Error")
        return FakeResponse(_organic(1))

    no_network.setattr(serper.requests, "post", selective)

    results = SerperRM(serper_search_api_key="k", k=10).forward(["bad", "good"])
    assert results == [{"title": "Title 0", "url": "https://example.com/0", "content": "Snippet 0"}]


def test_a_transport_exception_is_swallowed_too(no_network):
    def boom(*args, **kwargs):
        raise ConnectionError("network unreachable")

    no_network.setattr(serper.requests, "post", boom)
    assert SerperRM(serper_search_api_key="k").forward("q") == []


# --- usage accounting -----------------------------------------------------


def test_usage_counts_queries_and_resets(no_network):
    no_network.setattr(serper.requests, "post", lambda *a, **k: FakeResponse(_organic(1)))
    rm = SerperRM(serper_search_api_key="k", k=10)

    rm.forward("one")
    rm.forward(["two", "three"])

    assert rm.get_usage_and_reset() == {"SerperRM": 3}
    assert rm.get_usage_and_reset() == {"SerperRM": 0}


def test_usage_counts_the_skipped_placeholder_and_failed_queries(no_network):
    """Recorded as-is: usage is incremented by ``len(queries)`` up front, before
    the placeholder skip and before any request is made, so it measures queries
    *submitted*, not API calls billed."""

    def boom(*args, **kwargs):
        raise ConnectionError("down")

    no_network.setattr(serper.requests, "post", boom)
    rm = SerperRM(serper_search_api_key="k")
    rm.forward(["Queries:", "also-never-succeeds"])

    assert rm.get_usage_and_reset() == {"SerperRM": 2}
