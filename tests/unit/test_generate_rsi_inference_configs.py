import pathlib
import subprocess
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BENCHES = ["gaia", "hle", "gpqa", "aime", "musique"]


def _generate(tmp_path):
    for suite in ("rsi_arm_c_inference", "rsi_arm_i_inference"):
        r = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/generate_configs.py"),
                "--suite",
                suite,
                "--output-root",
                str(tmp_path),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr


def _load(root, arm, bench):
    p = root / "qwen3" / "rsi_inference" / arm / bench / "qwen8B_sub8B_none.yaml"
    return yaml.safe_load(p.read_text())


def test_rsi_suites_generate_ten_configs(tmp_path):
    _generate(tmp_path)
    n = 0
    for arm in ("arm_c", "arm_i"):
        for b in BENCHES:
            cfg = _load(tmp_path, arm, b)
            n += 1
            models = cfg["models"]
            assert (
                f"qwen3-8b-rsi-{arm.replace('_', '-')}"
                in models["orchestrator"]["lora_adapter_path"]
            )
            assert models["orchestrator"]["path_or_id"] == "Qwen/Qwen3-8B"
            subs = {k: v for k, v in models.items() if k != "orchestrator"}
            assert subs
            for v in subs.values():
                assert v["path_or_id"] == "Qwen/Qwen3-8B"
                assert "lora_adapter_path" not in v
            assert cfg["thinking_mode"] == "NO"
    assert n == 10


def test_rsi_configs_match_base_reference_splits(tmp_path):
    _generate(tmp_path)
    for arm in ("arm_c", "arm_i"):
        for b in BENCHES:
            ref = yaml.safe_load(
                (
                    REPO
                    / "experiments/configs/qwen3/agentflow"
                    / b
                    / "qwen8B_subagent_tools_none.yaml"
                ).read_text()
            )
            assert _load(tmp_path, arm, b)["dataset"]["split"] == ref["dataset"]["split"]
