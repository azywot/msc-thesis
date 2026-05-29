"""Tests for GEPA inference integration.

Covers:
  - gepa_prompt_path field in ExperimentConfig (schema + loader)
  - Runner loading logic: system_prompt / planning_suffix extracted from best_candidate.json
  - Orchestrator receives the planning_suffix loaded from JSON
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.config.loader import load_experiment_config
from agent_engine.config.schema import ExperimentConfig
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
"""

_BEST_CANDIDATE = {
    "system_prompt": "You are a GEPA-optimised reasoning assistant.",
    "planning_suffix": "**Plan carefully before calling tools.**",
}


def _write_best_candidate(tmp_path: Path) -> Path:
    p = tmp_path / "best_candidate.json"
    p.write_text(json.dumps(_BEST_CANDIDATE))
    return p


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(_BASE_YAML + extra)
    return p


# ---------------------------------------------------------------------------
# Schema: gepa_prompt_path field
# ---------------------------------------------------------------------------

class TestGEPAPromptPathSchema:
    def test_defaults_to_none(self):
        cfg = ExperimentConfig.model_validate({
            "name": "x",
            "models": {},
        })
        assert cfg.gepa_prompt_path is None

    def test_string_is_coerced_to_path(self):
        cfg = ExperimentConfig.model_validate({
            "name": "x",
            "models": {},
            "gepa_prompt_path": "/some/path/best_candidate.json",
        })
        assert isinstance(cfg.gepa_prompt_path, Path)
        assert cfg.gepa_prompt_path == Path("/some/path/best_candidate.json")

    def test_path_object_accepted(self):
        p = Path("/tmp/best_candidate.json")
        cfg = ExperimentConfig.model_validate({
            "name": "x",
            "models": {},
            "gepa_prompt_path": p,
        })
        assert cfg.gepa_prompt_path == p


# ---------------------------------------------------------------------------
# Loader: gepa_prompt_path round-trips through YAML
# ---------------------------------------------------------------------------

class TestGEPAPromptPathLoader:
    def test_not_present_gives_none(self, tmp_path):
        cfg_path = _write_config(tmp_path)
        cfg = load_experiment_config(cfg_path)
        assert cfg.gepa_prompt_path is None

    def test_path_loaded_from_yaml(self, tmp_path):
        json_path = _write_best_candidate(tmp_path)
        cfg_path = _write_config(
            tmp_path,
            f'\ngepa_prompt_path: "{json_path}"\n',
        )
        cfg = load_experiment_config(cfg_path)
        assert cfg.gepa_prompt_path == json_path

    def test_path_is_path_instance(self, tmp_path):
        json_path = _write_best_candidate(tmp_path)
        cfg_path = _write_config(
            tmp_path,
            f'\ngepa_prompt_path: "{json_path}"\n',
        )
        cfg = load_experiment_config(cfg_path)
        assert isinstance(cfg.gepa_prompt_path, Path)


# ---------------------------------------------------------------------------
# Runner loading logic (isolated from the full script)
# ---------------------------------------------------------------------------

def _simulate_runner_gepa_load(gepa_prompt_path):
    """Mirrors the gepa_prompt_path branch in run_experiment.py."""
    gepa_planning_suffix = None
    system_prompt_for_config = None

    if gepa_prompt_path is not None:
        with open(gepa_prompt_path, "r", encoding="utf-8") as f:
            gepa_data = json.load(f)
        system_prompt_for_config = gepa_data["system_prompt"]
        gepa_planning_suffix = gepa_data.get("planning_suffix")

    return system_prompt_for_config, gepa_planning_suffix


class TestRunnerGEPALoad:
    def test_system_prompt_extracted(self, tmp_path):
        json_path = _write_best_candidate(tmp_path)
        sp, _ = _simulate_runner_gepa_load(json_path)
        assert sp == _BEST_CANDIDATE["system_prompt"]

    def test_planning_suffix_extracted(self, tmp_path):
        json_path = _write_best_candidate(tmp_path)
        _, suffix = _simulate_runner_gepa_load(json_path)
        assert suffix == _BEST_CANDIDATE["planning_suffix"]

    def test_none_path_returns_none_both(self):
        sp, suffix = _simulate_runner_gepa_load(None)
        assert sp is None
        assert suffix is None

    def test_missing_planning_suffix_returns_none(self, tmp_path):
        p = tmp_path / "no_suffix.json"
        p.write_text(json.dumps({"system_prompt": "Hello"}))
        _, suffix = _simulate_runner_gepa_load(p)
        assert suffix is None

    def test_empty_planning_suffix_is_preserved(self, tmp_path):
        p = tmp_path / "empty_suffix.json"
        p.write_text(json.dumps({"system_prompt": "Hello", "planning_suffix": ""}))
        _, suffix = _simulate_runner_gepa_load(p)
        assert suffix == ""


# ---------------------------------------------------------------------------
# End-to-end: planning_suffix from JSON reaches the orchestrator
# ---------------------------------------------------------------------------

class TestPlanningsSuffixReachesOrchestrator:
    def test_suffix_stored_on_orchestrator(self, tmp_path):
        json_path = _write_best_candidate(tmp_path)
        _, suffix = _simulate_runner_gepa_load(json_path)

        model = MagicMock()
        model.config = MagicMock()
        model.config.supports_thinking = False
        model.config.family = "qwen3"

        orch = AgenticOrchestrator(
            model_provider=model,
            tool_registry=ToolRegistry(),
            planning_suffix=suffix,
        )
        assert orch.planning_suffix == _BEST_CANDIDATE["planning_suffix"]

    def test_none_suffix_leaves_orchestrator_default(self):
        model = MagicMock()
        model.config = MagicMock()
        model.config.supports_thinking = False
        model.config.family = "qwen3"

        orch = AgenticOrchestrator(
            model_provider=model,
            tool_registry=ToolRegistry(),
            planning_suffix=None,
        )
        assert orch.planning_suffix is None
