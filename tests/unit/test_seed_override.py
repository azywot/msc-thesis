"""Tests for the --seed override, seeded run dir name and config fingerprint."""

from pathlib import Path

import yaml

from agent_engine.config.loader import (
    apply_seed_override,
    config_fingerprint,
    load_experiment_config,
)
from agent_engine.runner.experiment import _make_run_dir


def write_yaml(tmp_path, seed=0, roles=("orchestrator",), model_seed=None):
    models = {}
    for role in roles:
        m = {
            "name": "Qwen3-8B",
            "family": "qwen3",
            "path_or_id": "Qwen/Qwen3-8B",
            "role": role,
        }
        if model_seed is not None:
            m["seed"] = model_seed
        models[role] = m
    data = {
        "name": "t",
        "description": "t",
        "models": models,
        "tools": {"enabled_tools": ["web_search"], "direct_tool_call": True},
        "dataset": {"name": "gaia", "split": "all_validation", "data_dir": "./data"},
        "seed": seed,
        "thinking_mode": "NO",
        "baseline": True,
        "output_dir": "./out",
        "cache_dir": "./cache",
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_seed_override_sets_all_seeds(tmp_path):
    cfg = load_experiment_config(write_yaml(tmp_path, seed=0, roles=["orchestrator", "web_search"]))
    apply_seed_override(cfg, 3)
    assert cfg.seed == 3 and all(m.seed == 3 for m in cfg.models.values())


def test_seed_override_beats_model_level_seed(tmp_path):
    cfg = load_experiment_config(write_yaml(tmp_path, seed=0, model_seed=11))
    apply_seed_override(cfg, 4)
    assert cfg.get_model("orchestrator").seed == 4


def test_fingerprint_ignores_seed_and_output_dir(tmp_path):
    a = load_experiment_config(write_yaml(tmp_path, seed=0))
    b = load_experiment_config(write_yaml(tmp_path, seed=0))
    apply_seed_override(b, 2)
    b.output_dir = Path("/elsewhere")
    assert config_fingerprint(a) == config_fingerprint(b)


def test_fingerprint_ignores_bookkeeping_fields(tmp_path):
    base = config_fingerprint(load_experiment_config(write_yaml(tmp_path)))
    for field, value in [
        ("use_wandb", True),
        ("wandb_project", "other"),
        ("description", "changed"),
        ("name", "renamed"),
    ]:
        c = load_experiment_config(write_yaml(tmp_path))
        setattr(c, field, value)
        assert config_fingerprint(c) == base, field
    c = load_experiment_config(write_yaml(tmp_path))
    c.slurm.time = "99:00:00"
    assert config_fingerprint(c) == base


def test_fingerprint_sees_model_and_tool_changes(tmp_path):
    base = config_fingerprint(load_experiment_config(write_yaml(tmp_path)))
    c = load_experiment_config(write_yaml(tmp_path))
    c.get_model("orchestrator").temperature = 0.3
    assert config_fingerprint(c) != base
    c = load_experiment_config(write_yaml(tmp_path))
    c.tools.enabled_tools = ["code_generator"]
    assert config_fingerprint(c) != base


def test_fingerprint_sees_hidden_fields(tmp_path):
    a = load_experiment_config(write_yaml(tmp_path))
    b = load_experiment_config(write_yaml(tmp_path))
    b.tools.web_tool_provider = "tavily"
    assert config_fingerprint(a) != config_fingerprint(b)


def test_run_dir_name_with_and_without_seed(tmp_path):
    assert _make_run_dir(tmp_path, "all_validation_text", seed=3)[0].name.startswith(
        "all_validation_text_seed3_"
    )
    name = _make_run_dir(tmp_path, "all_validation", seed=None)[0].name
    assert name.startswith("all_validation_") and "_seed" not in name
