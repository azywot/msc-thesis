"""Seam-completeness tests.

Each extension seam gets a test that fails loudly when someone wires a new
thing up only halfway.  These are the tests that make the seams trustworthy:
without them, a half-registered tool or model family degrades silently at
runtime instead of failing at import.
"""

import pytest

from agent_engine.models.base import (
    _NO_SYSTEM_PROMPT_FAMILIES,
    _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES,
    _THINK_PREFIX_FAMILIES,
    _THINKING_FAMILIES,
    _TOOL_CALL_FORMAT,
    _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES,
    ModelFamily,
    ToolCallFormat,
    get_tool_call_format,
)
from agent_engine.tools.registry import registered_tools


def test_every_default_tool_has_a_factory():
    assert set(registered_tools()) == {
        "web_search",
        "code_generator",
        "mind_map",
        "text_inspector",
        "image_inspector",
    }


@pytest.mark.parametrize("family", list(ModelFamily), ids=lambda f: f.name)
def test_every_model_family_resolves_to_a_tool_call_format(family):
    """Every family must resolve to a real format through the public accessor.

    Deliberately *not* asserting that every family appears in
    ``_TOOL_CALL_FORMAT``: that table is sparse on purpose (``base.py``:
    "Unlisted families default to JSON"), so requiring an entry per family
    would fail on seven families that are correct as they stand.  What matters
    is that resolution never breaks.
    """
    assert isinstance(get_tool_call_format(family), ToolCallFormat)


@pytest.mark.parametrize(
    "name, members",
    [
        ("_TOOL_CALL_FORMAT", set(_TOOL_CALL_FORMAT)),
        ("_THINKING_FAMILIES", _THINKING_FAMILIES),
        ("_NO_SYSTEM_PROMPT_FAMILIES", _NO_SYSTEM_PROMPT_FAMILIES),
        ("_THINK_PREFIX_FAMILIES", _THINK_PREFIX_FAMILIES),
        ("_TOOL_ROLE_AS_ENVIRONMENT_FAMILIES", _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES),
        ("_SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES", _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES),
    ],
)
def test_family_tables_contain_only_real_families(name, members):
    """Catches the failure the sparse-table test cannot: a stale or misspelled
    entry left behind when a family is renamed or removed.  Such an entry is
    silently inert -- the lookup simply never matches."""
    stale = [m for m in members if not isinstance(m, ModelFamily)]
    assert not stale, f"{name} contains non-ModelFamily entries: {stale}"
