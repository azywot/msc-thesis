"""generate_job.py: optional per-seed SLURM job rendering."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "experiments/configs/qwen3/baseline/gaia/qwen8B_no_tools_none.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "generate_job", REPO / "jobs/scripts/generate_job.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg():
    from agent_engine.config import load_experiment_config

    return load_experiment_config(CFG)


def _render(tmp_path, **kw):
    out = tmp_path / "x.job"
    _load().generate_job(CFG, output_path=out, project_dir=tmp_path, **kw)
    return out.read_text()


def test_job_without_seed_is_unchanged(tmp_path):
    cfg = _cfg()
    text = _render(tmp_path, seed=None)
    assert "--seed" not in text
    assert f"#SBATCH --job-name={cfg.name}\n" in text
    assert f"{cfg.name}_%A.log\n" in text
    assert f"export PYTHONHASHSEED={cfg.seed}\n" in text
    assert text.endswith(
        f'  --output-dir {cfg.output_dir}\n\nEXIT_CODE=$?\n\necho "Exit code: $EXIT_CODE"\necho "End time: $(date)"\n\nexit $EXIT_CODE'
    )
    assert text == _render(tmp_path)


def test_job_with_seed(tmp_path, monkeypatch):
    cfg = _cfg()
    text = _render(tmp_path, seed=3)
    assert f"--job-name={cfg.name}_seed3\n" in text
    assert f"{cfg.name}_seed3_%A.log" in text
    assert "--seed 3" in text
    assert "PYTHONHASHSEED=3" in text
    assert f"--output-dir {cfg.output_dir} \\\n  --seed 3\n" in text

    # default output path: run from tmp_path so nothing lands in the repo
    monkeypatch.chdir(tmp_path)
    _load().generate_job(CFG, project_dir=tmp_path, seed=3)
    assert (tmp_path / "jobs/generated" / f"{cfg.name}_seed3.job").exists()
