"""``CacheManager`` — round-trip, persistence, merge, and normalisation.

Characterization, not specification: every assertion below records what the
current implementation *does*.  Where that differs from what the docstrings
promise, the test says so rather than asserting the promise.

The cache is the only thing standing between a re-run and a few thousand
paid search calls, so its merge semantics (disk ∪ memory, memory wins) are
load-bearing for reproducibility.
"""

import json
import os

import pytest

from agent_engine.caching.manager import CacheManager


@pytest.fixture
def cache(tmp_path):
    return CacheManager(cache_dir=str(tmp_path), web_tool_provider="serper", dataset_name="ds")


def _second_manager(tmp_path):
    """A fresh manager over the same directory — i.e. a later process."""
    return CacheManager(cache_dir=str(tmp_path), web_tool_provider="serper", dataset_name="ds")


# --- layout ---------------------------------------------------------------


def test_init_creates_the_namespaced_layout(tmp_path, cache):
    provider_dir = tmp_path / "serper" / "ds"
    assert provider_dir.is_dir()
    assert (provider_dir / ".cache.lock").exists()
    assert cache.search_cache_path == str(provider_dir / "search_cache.json")
    assert cache.url_cache_path == str(provider_dir / "url_cache.json")


def test_provider_and_dataset_namespace_separate_caches(tmp_path):
    a = CacheManager(cache_dir=str(tmp_path), web_tool_provider="serper", dataset_name="gaia")
    b = CacheManager(cache_dir=str(tmp_path), web_tool_provider="tavily", dataset_name="gaia")
    c = CacheManager(cache_dir=str(tmp_path), web_tool_provider="serper", dataset_name="gpqa")

    a.search_cache["q"] = [{"title": "from-serper-gaia"}]
    a.save_caches()

    assert _second_manager(tmp_path) is not None  # same namespace as `a` is checked below
    assert b.search_cache == {}
    assert c.search_cache == {}
    assert CacheManager(
        cache_dir=str(tmp_path), web_tool_provider="serper", dataset_name="gaia"
    ).search_cache == {"q": [{"title": "from-serper-gaia"}]}


def test_caches_start_empty_when_no_files_exist(cache):
    assert cache.search_cache == {}
    assert cache.url_cache == {}


# --- round trip and persistence -------------------------------------------


def test_save_caches_round_trips_both_caches(tmp_path, cache):
    cache.search_cache["query"] = [{"title": "t", "url": "u", "content": "c"}]
    cache.url_cache["https://example.com"] = "page text"
    cache.save_caches()

    reloaded = _second_manager(tmp_path)
    assert reloaded.search_cache == {"query": [{"title": "t", "url": "u", "content": "c"}]}
    assert reloaded.url_cache == {"https://example.com": "page text"}


def test_save_search_cache_leaves_the_url_cache_file_alone(tmp_path, cache):
    cache.search_cache["q"] = [{"title": "t"}]
    cache.url_cache["u"] = "text"
    cache.save_search_cache()

    assert os.path.exists(cache.search_cache_path)
    assert not os.path.exists(cache.url_cache_path)


def test_save_url_cache_leaves_the_search_cache_file_alone(tmp_path, cache):
    cache.search_cache["q"] = [{"title": "t"}]
    cache.url_cache["u"] = "text"
    cache.save_url_cache()

    assert os.path.exists(cache.url_cache_path)
    assert not os.path.exists(cache.search_cache_path)


# --- merge semantics ------------------------------------------------------


def test_a_concurrent_writers_entries_survive_our_save(tmp_path, cache):
    """The property parallel SLURM workers depend on: saving must not clobber
    entries another process wrote after we loaded."""
    other = _second_manager(tmp_path)
    other.search_cache["theirs"] = [{"title": "other worker"}]
    other.save_caches()

    cache.search_cache["ours"] = [{"title": "us"}]
    cache.save_caches()

    on_disk = json.loads(open(cache.search_cache_path, encoding="utf-8").read())
    assert set(on_disk) == {"theirs", "ours"}


def test_memory_wins_over_disk_on_a_conflicting_key(tmp_path, cache):
    other = _second_manager(tmp_path)
    other.search_cache["k"] = [{"title": "disk"}]
    other.save_caches()

    cache.search_cache["k"] = [{"title": "memory"}]
    cache.save_caches()

    assert json.loads(open(cache.search_cache_path, encoding="utf-8").read()) == {
        "k": [{"title": "memory"}]
    }


def test_saving_pulls_the_merged_state_back_into_memory(tmp_path, cache):
    """After a save the in-memory dict is *replaced* by the merged result, so a
    later `in` check sees the other worker's keys without a reload."""
    other = _second_manager(tmp_path)
    other.url_cache["theirs"] = "other page"
    other.save_caches()

    cache.url_cache["ours"] = "our page"
    cache.save_caches()

    assert cache.url_cache == {"theirs": "other page", "ours": "our page"}


# --- normalisation --------------------------------------------------------


def test_loading_normalises_search_values_to_lists_of_dicts(tmp_path, cache):
    raw = {"str": "not-a-list", "mixed": [{"ok": 1}, "junk", 7], "good": [{"a": 1}]}
    with open(cache.search_cache_path, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    reloaded = _second_manager(tmp_path)
    assert reloaded.search_cache == {"str": [], "mixed": [{"ok": 1}], "good": [{"a": 1}]}


def test_the_url_cache_is_not_normalised(tmp_path, cache):
    """Normalisation is keyed on the *search* cache path, so url values pass
    through whatever their type."""
    with open(cache.url_cache_path, "w", encoding="utf-8") as f:
        json.dump({"u": ["not", "a", "string"]}, f)

    assert _second_manager(tmp_path).url_cache == {"u": ["not", "a", "string"]}


def test_save_search_cache_normalises_but_save_caches_does_not(tmp_path, cache):
    """A real asymmetry, recorded rather than corrected.

    ``save_search_cache`` passes ``normalize=True``; ``save_caches`` writes the
    merged dict straight out.  So a malformed value written through
    ``save_caches`` reaches disk intact and is only cleaned on the next load.
    Anything reading the file directly — an analysis script, a cache
    inspector — sees the difference.
    """
    cache.search_cache["q"] = "not-a-list"
    cache.save_caches()
    assert json.loads(open(cache.search_cache_path, encoding="utf-8").read()) == {"q": "not-a-list"}

    other = _second_manager(tmp_path)  # load normalises the existing key
    other.search_cache["q2"] = "also-not-a-list"
    other.save_search_cache()
    assert json.loads(open(cache.search_cache_path, encoding="utf-8").read()) == {
        "q": [],
        "q2": [],
    }


# --- resilience -----------------------------------------------------------


def test_a_corrupt_cache_file_loads_as_empty_instead_of_crashing(tmp_path, cache):
    with open(cache.search_cache_path, "w", encoding="utf-8") as f:
        f.write("{ this is not json")

    assert _second_manager(tmp_path).search_cache == {}


def test_a_corrupt_file_is_overwritten_by_the_next_save(tmp_path, cache):
    """The flip side of loading corruption as empty: the bad file's contents
    are gone after the next save, not merged."""
    with open(cache.search_cache_path, "w", encoding="utf-8") as f:
        f.write("{ this is not json")

    fresh = _second_manager(tmp_path)
    fresh.search_cache["q"] = [{"title": "t"}]
    fresh.save_caches()

    assert json.loads(open(cache.search_cache_path, encoding="utf-8").read()) == {
        "q": [{"title": "t"}]
    }


def test_writes_leave_no_temp_files_behind(tmp_path, cache):
    cache.search_cache["q"] = [{"title": "t"}]
    cache.url_cache["u"] = "text"
    cache.save_caches()

    leftovers = [p.name for p in (tmp_path / "serper" / "ds").iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_saved_json_is_utf8_not_escaped(tmp_path, cache):
    """``ensure_ascii=False`` — caches hold non-English pages and the files are
    meant to stay greppable."""
    cache.url_cache["u"] = "café ünicode 漢字"
    cache.save_caches()

    assert (
        "café ünicode 漢字"
        in open(
            cache.search_cache_path.replace("search_cache", "url_cache"), encoding="utf-8"
        ).read()
    )
