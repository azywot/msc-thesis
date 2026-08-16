"""Properties that must hold for the extension seams, whatever is plugged in.

These replace two snapshot gates (`configs.manifest` and `prompts.json`) that
were deleted once the refactor landed.  Those gates compared committed
baselines, so they failed on every *intended* change: adding a benchmark
changes the generated configs, editing a prompt changes the templates.  Since
the whole point of the handover is that a new researcher does exactly those
things, they would have gone red on their first honest commit -- and a test
that fails on intended changes trains people to regenerate reflexively, at
which point it protects nothing.

The tests here assert *properties* instead.  Adding a dataset, a template or a
tool leaves them green; wiring one up incorrectly does not.  Each one
corresponds to a mistake that is otherwise silent at runtime:

* a dataset with no usable prompt -> the base template is used and the run
  quietly measures the wrong thing;
* a malformed template -> the same, via the FileNotFoundError fallback;
* a tool whose schema name disagrees with its registry key -> the model is told
  about a tool it cannot successfully call.
"""

import pathlib

import pytest
import yaml

import agent_engine.tools  # noqa: F401  -- import side effect registers the factories
from agent_engine.datasets.spec import DATASET_SPECS
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools.registry import ToolDeps, build_tool, registered_tools

TEMPLATE_DIR = pathlib.Path(agent_engine.prompts.__file__).parent / "templates" / "system"


@pytest.fixture(scope="module")
def builder():
    return PromptBuilder()


@pytest.fixture(scope="module")
def tool_schemas():
    """A real schema list.

    Load-bearing: with ``tool_schemas=[]`` every template renders to the same
    ~130-character stub, so an empty list would make these assertions pass for
    a dataset whose template does not exist.
    """
    from agent_engine.tools.web_search import WebSearchTool

    return [WebSearchTool(api_key="k", search_cache={}, url_cache={}).get_schema()]


# --- datasets -------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", sorted(DATASET_SPECS))
@pytest.mark.parametrize("baseline", [False, True], ids=["agentflow", "baseline"])
def test_every_dataset_spec_resolves_to_a_template(dataset_name, baseline, builder, tool_schemas):
    """Every registered dataset must produce a real prompt in *both* modes.

    The baseline half is the one that catches things: `PromptBuilder` appends
    `_baseline` to the stem, so a benchmark shipped with only `<name>.yaml`
    silently falls back to the generic base template in baseline runs.  The
    run completes and the comparison is broken.
    """
    prompt = builder.build_system_prompt(dataset_name, tool_schemas, baseline=baseline)

    assert prompt.strip(), f"{dataset_name} ({'baseline' if baseline else 'agentflow'}) is empty"
    # Comfortably above the ~130-char stub an unresolved template renders to.
    assert len(prompt) > 500, f"{dataset_name} rendered only {len(prompt)} chars"
    assert "web_search" in prompt, "the tool schema did not reach the prompt"


@pytest.mark.parametrize("dataset_name", sorted(DATASET_SPECS))
def test_a_spec_template_names_a_file_that_exists(dataset_name):
    """`template=None` is legal (the raw name is tried, then base); a template
    that *is* named must exist in both variants."""
    spec = DATASET_SPECS[dataset_name]
    if spec.template is None:
        return

    for stem in (spec.template, f"{spec.template}_baseline"):
        assert (TEMPLATE_DIR / f"{stem}.yaml").exists(), f"{dataset_name} -> missing {stem}.yaml"


def test_a_stratified_dataset_declares_a_level_field():
    """`stratified=True` without `level_field` yields a `per_level` block whose
    only bucket is "all" -- technically valid, silently useless."""
    offenders = [
        name for name, spec in DATASET_SPECS.items() if spec.stratified and not spec.level_field
    ]
    assert offenders == []


def test_a_fallback_field_requires_a_primary_field():
    offenders = [
        name
        for name, spec in DATASET_SPECS.items()
        if spec.level_fallback_field and not spec.level_field
    ]
    assert offenders == []


# --- templates ------------------------------------------------------------


def _template_files():
    return sorted(TEMPLATE_DIR.glob("*.yaml"))


def test_the_template_directory_is_not_empty():
    """Guards the parametrised tests below: an empty glob would make them pass
    by vacuously generating no cases."""
    assert len(_template_files()) >= 5


# Present in all ten templates today.  Deliberately not the full key set: the
# `example_*` blocks vary by template (a math prompt has no search example), and
# requiring them would make this test fail on a legitimate new benchmark.
REQUIRED_TEMPLATE_KEYS = ("base_instruction", "base_instruction_tools", "final_instructions")


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_every_template_parses_and_carries_its_required_keys(path):
    """A malformed edit here does not raise at startup -- `PromptBuilder`
    catches the failure and falls back to the base template, so the run
    proceeds with the wrong prompt."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(loaded, dict), f"{path.name} is not a YAML mapping"
    assert loaded, f"{path.name} is empty"

    for key in REQUIRED_TEMPLATE_KEYS:
        assert key in loaded, f"{path.name} has no {key!r}"
        assert str(loaded[key]).strip(), f"{path.name} has an empty {key!r}"


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_every_agentflow_template_has_a_baseline_partner(path):
    """The pairing `PromptBuilder` assumes. Checked in one direction only --
    a stray `_baseline` file without a partner is harmless."""
    if path.stem.endswith("_baseline"):
        return
    assert (TEMPLATE_DIR / f"{path.stem}_baseline.yaml").exists()


# --- tools ----------------------------------------------------------------


def _deps(tmp_path):
    """A `ToolDeps` complete enough for every factory to build.

    Real objects rather than stand-ins, because the factories reach further
    than an obvious stub anticipates: `web_search` wants the provider name, a
    matching `api_keys` entry, *and* a cache manager to take `search_cache`
    from; `code_generator` reads `config.cache_dir`. Building this from a real
    `ExperimentConfig` and a real `CacheManager` under `tmp_path` means the
    test keeps working when a factory starts reading one more field.

    No network is touched: constructing the clients does not call out, and
    nothing here executes a tool.
    """
    from agent_engine.caching.manager import CacheManager
    from agent_engine.config.schema import ExperimentConfig, ToolsConfig

    config = ExperimentConfig(
        name="wiring-invariants",
        tools=ToolsConfig(direct_tool_call=True, web_tool_provider="serper"),
        cache_dir=tmp_path / "cache",
    )
    cache_manager = CacheManager(
        cache_dir=str(tmp_path / "cache"), web_tool_provider="serper", dataset_name="test"
    )
    return ToolDeps(
        config=config,
        cache_manager=cache_manager,
        api_keys={"serper": "test-key", "tavily": "test-key"},
        mind_map_storage_path=tmp_path / "mind_map",
    )


def test_the_registry_is_populated():
    """Registration is an import side effect; if the package import were
    dropped, every parametrised tool test below would vanish rather than fail."""
    assert len(registered_tools()) >= 5


@pytest.mark.parametrize("name", registered_tools())
def test_every_registered_tool_has_a_valid_schema(name, tmp_path):
    """Three-way agreement: registry key, `tool.name`, and the name inside the
    schema.  A copy-pasted factory that registers under a new key while the
    class still reports the old name produces a tool the model is told about
    but cannot call -- and nothing errors."""
    tool = build_tool(name, _deps(tmp_path))
    if tool is None:
        # A factory may legitimately decline (image_inspector in direct mode).
        return

    assert tool.name == name, f"registered as {name!r} but reports {tool.name!r}"

    schema = tool.get_schema()
    assert isinstance(schema, dict)

    function = schema.get("function", schema)
    for key in ("name", "description", "parameters"):
        assert key in function, f"{name} schema is missing {key!r}"

    assert function["name"] == name, f"{name} schema names {function['name']!r}"
    assert str(function["description"]).strip(), f"{name} has an empty description"

    parameters = function["parameters"]
    assert parameters.get("type") == "object"
    assert isinstance(parameters.get("properties"), dict)
    for required in parameters.get("required", []):
        assert (
            required in parameters["properties"]
        ), f"{name} requires {required!r} but does not declare it"


def test_an_unregistered_name_builds_nothing_rather_than_raising(tmp_path):
    """Preserved from the if/elif chain this registry replaced: an unknown name
    in `enabled_tools` is skipped silently."""
    assert build_tool("no_such_tool", _deps(tmp_path)) is None
