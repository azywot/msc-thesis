"""Tests for the text-split Qwen3 config suites (baseline_text, agentflow_text)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from agent_engine.config import load_experiment_config

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("generate_configs", REPO / "scripts" / "generate_configs.py")
gc = importlib.util.module_from_spec(_spec)
sys.modules["generate_configs"] = gc
_spec.loader.exec_module(gc)

PAIRS = [("baseline_text", "baseline", 8), ("agentflow_text", "agentflow", 4)]
SPLITS = {"gaia": "all_validation_text", "hle": "test_text_subset_200"}


def _load(path):
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("text_name,parent,n", PAIRS)
def test_text_suite_counts_and_values(tmp_path, text_name, parent, n):
    gc.generate_suite(text_name, tmp_path)
    for ds, split in SPLITS.items():
        files = sorted((tmp_path / gc.SUITES[text_name]["config_subdir"] / ds).glob("*.yaml"))
        assert len(files) == n
        for f in files:
            cfg = load_experiment_config(f)
            orch = cfg.models["orchestrator"]
            assert orch.sampling_by_thinking.thinking.temperature == 0.6
            assert orch.sampling_by_thinking.non_thinking.temperature == 0.7
            assert cfg.dataset.split == split


@pytest.mark.parametrize("text_name,parent,n", PAIRS)
def test_text_suite_mirrors_parent(tmp_path, text_name, parent, n):
    gc.generate_suite(text_name, tmp_path)
    gc.generate_suite(parent, tmp_path)
    t_root = tmp_path / gc.SUITES[text_name]["config_subdir"]
    p_root = tmp_path / gc.SUITES[parent]["config_subdir"]
    for ds in SPLITS:
        t_files = {f.name: f for f in (t_root / ds).glob("*.yaml")}
        p_files = {f.name: f for f in (p_root / ds).glob("*.yaml")}
        assert t_files.keys() == p_files.keys()
        for name, tf in t_files.items():
            t, p = _load(tf), _load(p_files[name])
            for d in (t, p):
                for k in ("name", "description", "output_dir", "wandb_project"):
                    d.pop(k)
                d["dataset"].pop("split")
            assert "sampling_by_thinking" in t["models"]["orchestrator"]
            t["models"]["orchestrator"].pop("sampling_by_thinking")
            assert t == p


def test_parent_suites_have_no_sampling_block(tmp_path):
    gc.generate_suite("baseline", tmp_path)
    for f in (tmp_path / "qwen3/baseline").glob("*/*.yaml"):
        assert "sampling_by_thinking" not in f.read_text()
