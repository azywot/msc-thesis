"""The SFT folded job takes its data files and experiment name from the environment."""

import re
import subprocess
from pathlib import Path

JOB = Path(__file__).resolve().parents[2] / "jobs" / "fine_tuning" / "007_train_sft_folded.job"
VARS = ("TRAIN_FILE", "VAL_FILE", "NATIVE_FILE", "NATIVE_VAL_FILE", "EXPERIMENT_NAME")

DEFAULTS = {
    "TRAIN_FILE": "data/training/sft/sft_folded_train.parquet",
    "VAL_FILE": "data/training/sft/sft_folded_val.parquet",
    "NATIVE_FILE": "data/training/sft/sft_train.parquet",
    "NATIVE_VAL_FILE": "data/training/sft/sft_val.parquet",
    "EXPERIMENT_NAME": "qwen3-8b-sft-folded-v1",
}


def _resolve(env):
    lines = [
        line
        for line in JOB.read_text().splitlines()
        if re.match(r"^(" + "|".join(VARS) + r")=", line)
    ]
    script = "\n".join(lines) + "\n" + "\n".join(f'echo "{v}=[${v}]"' for v in VARS)
    out = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(re.fullmatch(r"(\w+)=\[(.*)\]", line).groups() for line in out.splitlines())


def test_defaults_unchanged_when_unset():
    assert _resolve({}) == DEFAULTS


def test_overrides_win():
    env = {v: f"x/{v}.parquet" for v in VARS}
    assert _resolve(env) == env


def test_explicitly_empty_native_disables_flag_but_others_keep_default():
    got = _resolve({"NATIVE_FILE": "", "NATIVE_VAL_FILE": "", "TRAIN_FILE": ""})
    assert got["NATIVE_FILE"] == ""
    assert got["NATIVE_VAL_FILE"] == ""
    assert got["TRAIN_FILE"] == DEFAULTS["TRAIN_FILE"]


def test_preflight_passes_native_only_when_set_and_hyperparameters_unchanged():
    text = JOB.read_text()
    assert '--folded "$TRAIN_FILE"' in text and '--folded "$VAL_FILE"' in text
    assert '--native "$NATIVE_FILE"' in text and '--native "$NATIVE_VAL_FILE"' in text
    assert "data.max_length=16384" in text
    assert "total_epochs=2" in text
