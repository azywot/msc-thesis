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
from agent_engine.datasets.spec import DATASET_SPECS
from agent_engine.prompts import PromptBuilder
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


# --- dataset seam ---------------------------------------------------------
#
# Deliberately NOT asserting that every registered dataset has a DATASET_SPECS
# row: unknown names fall back to a default spec on purpose, which is how the
# hop/QA datasets (nq, triviaqa, ...) get the base template.  Requiring a row
# each would fail on datasets that are correct as they stand -- the same mistake
# the sparse model-family table invites.


@pytest.mark.parametrize("name", sorted(DATASET_SPECS), ids=str)
def test_every_dataset_spec_template_exists(name):
    """A spec naming a template that isn't on disk would silently degrade to the
    base prompt at runtime, via the FileNotFoundError fallback."""
    spec = DATASET_SPECS[name]
    if spec.template is None:
        return
    builder = PromptBuilder()
    for stem in (spec.template, f"{spec.template}_baseline"):
        template = builder.load_template(stem)
        assert template, f"template '{stem}' (for dataset '{name}') is empty"


@pytest.mark.parametrize("name", sorted(DATASET_SPECS), ids=str)
def test_stratified_datasets_declare_a_level_field(name):
    """`compute_metrics` emits per_level only for stratified datasets; one
    without a level_field would bucket every example under 'all'."""
    spec = DATASET_SPECS[name]
    if spec.stratified:
        assert spec.level_field, f"'{name}' is stratified but declares no level_field"
    if spec.level_fallback_field:
        assert spec.level_field, f"'{name}' has a fallback level field but no primary one"
