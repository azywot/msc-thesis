import importlib.util
import json
import random
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rsi.bundle import load_bundle

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "experiments/configs/rsi/phase1"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_prompt_search", REPO / "scripts/run_prompt_search.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load_script()
main = SCRIPT.main
load_search_config = SCRIPT.load_search_config


@pytest.fixture
def tiny_parquet(tmp_path):
    rows = ["hotpotqa"] * 40 + ["nq"] * 8 + ["deepmath"] * 12
    random.Random(0).shuffle(rows)
    data = []
    for i, src in enumerate(rows):
        golden = None if src == "deepmath" else [f"ans{i}", f"alias{i}"]
        data.append(
            {
                "data_source": src,
                "question": f"question {i}",
                "result": f"ans{i}",
                "extra_info": {"idx": i, "golden_answers": golden},
            }
        )
    path = tmp_path / "combined_train.parquet"
    pd.DataFrame(data).to_parquet(path)
    return path


def write_config(tmp_path, data_path, output_root, pools):
    cfg = yaml.safe_load((SHIPPED / "search_reflector_qwen3_8b.yaml").read_text())
    cfg["search"]["data_path"] = str(data_path)
    cfg["search"]["output_root"] = str(output_root)
    cfg["search"]["pools"] = pools
    path = tmp_path / "tiny_search.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_dry_run_writes_pools_and_seed_bundle(tmp_path, tiny_parquet, monkeypatch, capsys):
    cfg = write_config(
        tmp_path,
        data_path=tiny_parquet,
        output_root=tmp_path / "out",
        pools={"shown": 10, "screen": 6, "heldout": 4},
    )
    monkeypatch.setenv("SERPER_API_KEY", "x")
    rc = main(["--config", str(cfg), "--domain", "search", "--dry-run"])
    out = tmp_path / "out" / cfg.stem / "search"
    assert rc == 0
    assert "shown=10 screen=6 heldout=4" in capsys.readouterr().out
    pools = json.loads((out / "pools.json").read_text())
    assert [len(pools[k]) for k in ("shown", "screen", "heldout")] == [10, 6, 4]
    assert load_bundle(out / "bundles" / "v0.json").domain == "search"
    assert (out / "config.json").exists()


def test_shipped_configs_parse():
    paths = sorted(SHIPPED.glob("*.yaml"))
    assert len(paths) == 2
    for p in paths:
        exp, search = load_search_config(p)
        assert exp.thinking_mode.value == "NO" and exp.max_turns == 7
        r = search.reflector
        assert r.input_budget_tokens + r.max_tokens <= r.max_model_len


class _FakeProvider:
    def generate(self, prompts):
        return []

    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return ""


def _patch_full_run(monkeypatch, tmp_path):
    seen = {"providers": []}
    provider = _FakeProvider()

    def fake_setup(model_cfg, api_keys, cache):
        seen["providers"].append(provider)
        return provider

    def fake_build_reflector(settings, executor_provider):
        seen["reflector_provider"] = executor_provider
        return object()

    def fake_run_search(*args):
        seen["args"] = args
        return {}

    monkeypatch.setattr(SCRIPT, "setup_model_provider", fake_setup)
    monkeypatch.setattr(SCRIPT, "build_reflector", fake_build_reflector)
    monkeypatch.setattr(SCRIPT, "run_search", fake_run_search)
    monkeypatch.setattr(SCRIPT, "write_report", lambda *a: tmp_path / "report.md")
    monkeypatch.setattr(SCRIPT.signal, "signal", lambda *a: None)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    return seen, provider


def test_normal_run_wires_one_provider_everywhere(tmp_path, tiny_parquet, monkeypatch):
    cfg = write_config(
        tmp_path, tiny_parquet, tmp_path / "out", {"shown": 10, "screen": 6, "heldout": 4}
    )
    monkeypatch.setenv("SERPER_API_KEY", "x")
    seen, provider = _patch_full_run(monkeypatch, tmp_path)
    assert main(["--config", str(cfg), "--domain", "search"]) == 0
    assert seen["reflector_provider"] is provider
    assert all(p is provider for p in seen["providers"])
    cfg_, domain, executor, _refl, examples, pools, *_rest = seen["args"]
    assert domain == "search"
    assert isinstance(examples, dict)
    assert set(examples) >= set(pools.shown + pools.screen + pools.heldout)
    assert all(k == v.question_id for k, v in examples.items())
    inner = executor._inner
    assert inner.model_provider is provider
    assert inner.tool_limits == {"web_search": 10}
    for tool in inner.tool_registry._tools.values():
        assert tool.model_provider is provider


def test_in_process_reflector_model_mismatch_raises(tmp_path, tiny_parquet, monkeypatch):
    cfg = write_config(
        tmp_path, tiny_parquet, tmp_path / "out", {"shown": 10, "screen": 6, "heldout": 4}
    )
    raw = yaml.safe_load(cfg.read_text())
    raw["search"]["reflector"]["model"] = "Qwen/Qwen3-32B"
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("SERPER_API_KEY", "x")
    _patch_full_run(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Qwen3-32B"):
        main(["--config", str(cfg), "--domain", "search"])


def test_dry_run_refuses_changed_config_in_existing_dir(tmp_path, tiny_parquet, monkeypatch):
    cfg = write_config(
        tmp_path, tiny_parquet, tmp_path / "out", {"shown": 10, "screen": 6, "heldout": 4}
    )
    monkeypatch.setenv("SERPER_API_KEY", "x")
    assert main(["--config", str(cfg), "--domain", "search", "--dry-run"]) == 0
    raw = yaml.safe_load(cfg.read_text())
    raw["search"]["ngram"] = 5
    cfg.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="ngram"):
        main(["--config", str(cfg), "--domain", "search", "--dry-run"])


@pytest.mark.parametrize(
    "patch, match",
    [({"tools": {"direct_tool_call": True}}, "direct_tool_call"), ({"baseline": True}, "baseline")],
)
def test_non_subagent_modes_are_rejected(tmp_path, tiny_parquet, monkeypatch, patch, match):
    cfg = write_config(
        tmp_path, tiny_parquet, tmp_path / "out", {"shown": 10, "screen": 6, "heldout": 4}
    )
    raw = yaml.safe_load(cfg.read_text())
    for key, value in patch.items():
        if isinstance(value, dict):
            raw[key].update(value)
        else:
            raw[key] = value
    cfg.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("SERPER_API_KEY", "x")
    with pytest.raises(ValueError, match=match):
        main(["--config", str(cfg), "--domain", "search", "--dry-run"])


def test_confirm_mode_scores_an_existing_run(tmp_path, tiny_parquet, monkeypatch, capsys):
    cfg = write_config(
        tmp_path, tiny_parquet, tmp_path / "out", {"shown": 10, "screen": 6, "heldout": 4}
    )
    monkeypatch.setenv("SERPER_API_KEY", "x")
    seen, provider = _patch_full_run(monkeypatch, tmp_path)
    calls = {}

    def fake_confirm(out_dir, executor, examples, n):
        calls.update(out_dir=out_dir, executor=executor, examples=examples, n=n)
        return {"n": n, "seed_acc": 0.25, "champion_acc": 0.5, "champion": "v1"}

    monkeypatch.setattr(SCRIPT, "run_confirm", fake_confirm)
    monkeypatch.setattr(
        SCRIPT, "build_reflector", lambda *a: pytest.fail("confirm must not build a reflector")
    )
    out = tmp_path / "out" / cfg.stem / "search"
    with pytest.raises(ValueError, match="state.json"):
        main(["--config", str(cfg), "--domain", "search", "--confirm", "7"])
    assert "out_dir" not in calls

    out.mkdir(parents=True)
    (out / "state.json").write_text("{}")
    assert main(["--config", str(cfg), "--domain", "search", "--confirm", "7"]) == 0
    assert calls["n"] == 7 and calls["out_dir"] == out
    assert calls["executor"]._inner.model_provider is provider
    assert all(k == v.question_id for k, v in calls["examples"].items())
    assert "args" not in seen  # run_search never called
    printed = capsys.readouterr().out
    assert "confirm: n=7 seed=25.0% champion v1=50.0%" in printed


def test_confirm_is_exclusive_with_dry_run(tmp_path, tiny_parquet, monkeypatch):
    cfg = write_config(
        tmp_path, tiny_parquet, tmp_path / "out", {"shown": 10, "screen": 6, "heldout": 4}
    )
    monkeypatch.setenv("SERPER_API_KEY", "x")
    with pytest.raises(SystemExit):
        main(["--config", str(cfg), "--domain", "search", "--dry-run", "--confirm", "5"])
