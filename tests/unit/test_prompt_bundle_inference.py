"""Tests for running a normal experiment under an RSI prompt bundle (diagnostic use).

Mirrors tests/unit/test_gepa_inference.py: schema field, loader, mutual exclusion with
gepa_prompt_path, and the runner branch (system prompt, suffixes, sub-agent templates,
bundle copy in the run directory).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_engine.config.loader import load_experiment_config
from agent_engine.config.schema import ExperimentConfig
from agent_engine.runner import experiment as exp
from agent_engine.tools.code_generator import CodeGeneratorTool
from agent_engine.tools.web_search import WebSearchTool
from rsi.bundle import replace_unit, save_bundle, seed_bundle

_BASE_YAML = """
name: test
models:
  orchestrator:
    name: mock
    family: qwen3
    path_or_id: mock
    role: orchestrator
dataset:
  name: gaia
  split: validation
tools:
  direct_tool_call: false
"""


class _FakeProvider:
    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return ""

    def generate(self, prompts):
        return []


@pytest.fixture
def SCHEMAS():
    return [
        WebSearchTool(
            api_key="k", search_cache={}, url_cache={}, model_provider=_FakeProvider()
        ).get_schema(),
        CodeGeneratorTool(model_provider=_FakeProvider()).get_schema(),
    ]


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(_BASE_YAML + extra)
    return p


def test_defaults_to_none():
    assert ExperimentConfig.model_validate({"name": "x", "models": {}}).prompt_bundle_path is None


def test_prompt_bundle_path_parses(tmp_path):
    cfg = load_experiment_config(_write_config(tmp_path, "prompt_bundle_path: some/bundle.json\n"))
    assert cfg.prompt_bundle_path == Path("some/bundle.json")
    assert isinstance(cfg.prompt_bundle_path, Path)


def test_both_prompt_sources_rejected(tmp_path):
    cfg_path = _write_config(tmp_path, "prompt_bundle_path: a.json\ngepa_prompt_path: b.json\n")
    with pytest.raises(ValueError):
        load_experiment_config(cfg_path)


def _run(monkeypatch, tmp_path, SCHEMAS, bundle):
    bundle_path = tmp_path / "champion.json"
    save_bundle(bundle, bundle_path)
    cfg_path = _write_config(
        tmp_path,
        f"output_dir: {tmp_path / 'out'}\ncache_dir: {tmp_path / 'cache'}\n"
        f"prompt_bundle_path: {bundle_path}\n",
    )
    captured = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            captured["orchestrator_kwargs"] = kwargs

        def run_batch(self, **kwargs):
            return []

        def cleanup(self):
            pass

    model = MagicMock()
    registry = MagicMock()
    registry.get_all_schemas.return_value = SCHEMAS
    dataset = MagicMock()
    example = MagicMock()
    example.get_attachments.return_value = []
    dataset.get_subset.return_value = [example]
    monkeypatch.setattr(exp, "AgenticOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(exp, "setup_model_provider", lambda *a, **k: model)
    monkeypatch.setattr(exp, "build_tool_registry", lambda deps: registry)
    monkeypatch.setattr(exp.DatasetRegistry, "get", staticmethod(lambda cfg: dataset))
    exp.run_experiment(SimpleNamespace(config=cfg_path, output_dir=None, seed=None))
    run_dir = next((tmp_path / "out").iterdir())
    captured["run_dir"] = run_dir
    captured["registry"] = registry
    captured["system_prompt"] = json.loads((run_dir / "config.json").read_text())["system_prompt"]
    return captured


def test_runner_uses_bundle_prompts(monkeypatch, tmp_path, SCHEMAS):
    bundle = replace_unit(
        seed_bundle("search", SCHEMAS),
        "orchestrator",
        {"guidance_opening": "You are a coordinator.", "turn1n_prompt": "Check links."},
    )
    captured = _run(monkeypatch, tmp_path, SCHEMAS, bundle)
    assert captured["system_prompt"].startswith("You are a coordinator.")
    kwargs = captured["orchestrator_kwargs"]
    assert kwargs["acting_suffix"] == "\n\nCheck links."
    assert kwargs["planning_suffix"].startswith("\n\n")
    captured["registry"].get.assert_any_call("web_search")
    assert (captured["run_dir"] / "prompt_bundle.json").read_text() == (
        tmp_path / "champion.json"
    ).read_text()


def test_fingerprint_unchanged_when_bundle_unset():
    import hashlib

    from agent_engine.config.loader import config_fingerprint

    cfg = ExperimentConfig.model_validate({"name": "x", "models": {}})
    d = cfg.model_dump(mode="json")
    for key in ("seed", "output_dir", "slurm", "use_wandb", "wandb_project", "description", "name"):
        d.pop(key, None)
    d.pop("prompt_bundle_path")
    expected = hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()
    assert config_fingerprint(cfg) == expected


def test_fingerprint_differs_with_bundle_path():
    from agent_engine.config.loader import config_fingerprint

    base = {"name": "x", "models": {}, "tools": {"direct_tool_call": False}}
    a = ExperimentConfig.model_validate({**base, "prompt_bundle_path": "a.json"})
    b = ExperimentConfig.model_validate({**base, "prompt_bundle_path": "b.json"})
    c = ExperimentConfig.model_validate({"name": "x", "models": {}})
    assert len({config_fingerprint(a), config_fingerprint(b), config_fingerprint(c)}) == 3


@pytest.mark.parametrize(
    "extra",
    [
        {"baseline": True, "tools": {"direct_tool_call": False}},
        {"tools": {"direct_tool_call": True}},
    ],
)
def test_bundle_rejected_outside_subagent_mode(extra):
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(
            {"name": "x", "models": {}, "prompt_bundle_path": "a.json", **extra}
        )


@pytest.mark.parametrize(
    "dataset, domain, ok",
    [
        ("gaia", "search", True),
        ("hle", "search", True),
        ("musique", "search", True),
        ("aime", "math", True),
        ("gpqa", "search", False),
        ("bigcodebench", "search", False),
        ("aime", "search", False),
        ("gaia", "math", False),
    ],
)
def test_bundle_template_must_match_the_benchmark(tmp_path, SCHEMAS, dataset, domain, ok):
    bundle_path = tmp_path / "champion.json"
    save_bundle(seed_bundle(domain, SCHEMAS), bundle_path)
    config = SimpleNamespace(prompt_bundle_path=bundle_path, dataset=SimpleNamespace(name=dataset))
    if ok:
        exp._check_bundle_template(config)
    else:
        with pytest.raises(ValueError, match="template"):
            exp._check_bundle_template(config)


def test_runner_refuses_mismatched_bundle_before_loading_models(monkeypatch, tmp_path, SCHEMAS):
    bundle_path = tmp_path / "champion.json"
    save_bundle(seed_bundle("search", SCHEMAS), bundle_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        _BASE_YAML.replace("name: gaia", "name: gpqa")
        + f"output_dir: {tmp_path / 'out'}\ncache_dir: {tmp_path / 'cache'}\n"
        f"prompt_bundle_path: {bundle_path}\n"
    )
    monkeypatch.setattr(
        exp, "setup_model_provider", lambda *a, **k: pytest.fail("models loaded before check")
    )
    with pytest.raises(ValueError, match="gpqa"):
        exp.run_experiment(SimpleNamespace(config=cfg_path, output_dir=None, seed=None))
    assert not (tmp_path / "out").exists()
