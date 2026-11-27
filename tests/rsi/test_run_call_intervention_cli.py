import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from rsi.intervention.rescue import RescueSettings

from .test_intervention_rescue import FakeOrchestrator, FakeReflector, Prov, _cand, _reply

REPO = Path(__file__).resolve().parents[2]
PHASE2 = REPO / "experiments/configs/rsi/phase2"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run_call_intervention", REPO / "scripts/run_call_intervention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load_script()


class FakeCache:
    def __init__(self):
        self.saved = 0

    def save_caches(self):
        self.saved += 1


@pytest.fixture
def setup(tmp_path):
    cfg = yaml.safe_load((PHASE2 / "rescue_qwen3_8b.yaml").read_text())
    parquet = tmp_path / "train.parquet"
    pd.DataFrame({"question": ["Capital of the country of cheese?", "other"]}).to_parquet(parquet)
    cfg["intervention"]["data_path"] = str(parquet)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    collected = tmp_path / "collected.jsonl"
    with open(collected, "w") as f:
        for p in (1, 2, 3):
            rec = _cand(1)
            rec["pass"] = p
            f.write(json.dumps(rec) + "\n")
    return cfg_path, collected, tmp_path / "out"


def test_shipped_configs_load():
    for name in ("rescue_qwen3_8b.yaml", "rescue_qwen3_32b.yaml"):
        exp, icfg = SCRIPT.load_intervention_config(PHASE2 / name)
        assert icfg.n_passes == 3 and icfg.threshold_points == 10.0 and icfg.blind_n == 100
        assert icfg.rescue == RescueSettings()
    assert icfg.reflector.backend == "openai" and icfg.reflector.base_url.endswith(":8001/v1")


def test_messages_token_counter():
    count = SCRIPT.messages_token_counter(lambda text: len(text.split()))
    assert count([{"role": "system", "content": "a b"}, {"role": "user", "content": "c"}]) == 3


def test_cli_dry_end_to_end(tmp_path, setup, monkeypatch):
    cfg_path, collected, out = setup
    cache = FakeCache()
    reflector = FakeReflector([[_reply(1)], [_reply(1)]])
    orch = FakeOrchestrator(Prov)

    monkeypatch.setattr(
        SCRIPT,
        "build_runtime",
        lambda exp, icfg: (orch, [Prov], reflector, lambda msgs: 10, cache),
    )
    argv = ["--config", str(cfg_path), "--collected", str(collected), "--out-dir", str(out)]
    assert SCRIPT.main(argv + ["--blind"]) == 0
    for name in ("candidates.json", "rescue.jsonl", "gate.json", "gate.md", "blind.jsonl"):
        assert (out / name).exists(), name
    gate = json.loads((out / "gate.json").read_text())
    assert {"search", "math", "gate", "blind"} <= set(gate)
    assert gate["gate"]["pass"] is True and gate["search"]["net_points"] == 100.0
    assert gate["blind"] == {
        "n": 1,
        "rescued": 1,
        "rate": 100.0,
        "control_rescued": 0,
        "control_rate": 0.0,
    }
    blind_row = json.loads((out / "blind.jsonl").read_text().splitlines()[0])
    assert blind_row["seed"] == 1099
    blind_prompt = reflector.calls[-1][0][0][1]["content"]
    assert "(hidden)" in blind_prompt and "Paris" not in blind_prompt
    assert cache.saved == 1


def test_cli_report_only_does_not_load_models(tmp_path, setup, monkeypatch):
    cfg_path, collected, out = setup

    def boom(exp, icfg):
        raise AssertionError("models must not load")

    monkeypatch.setattr(SCRIPT, "build_runtime", boom)
    argv = ["--config", str(cfg_path), "--collected", str(collected), "--out-dir", str(out)]
    assert SCRIPT.main(argv + ["--report-only"]) == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["search"]["candidates"] == 1 and gate["search"]["rescued"] == 0
    assert gate["gate"]["pass"] is False


def _write_collected(path, per_question):
    """per_question: list of (qid, source, correct_by_pass, extra)."""
    with open(path, "w") as f:
        for qid, source, correct, extra in per_question:
            for p, ok in enumerate(correct, start=1):
                rec = _cand(qid)
                rec.update({"pass": p, "correct": ok, "data_source": source, **extra})
                f.write(json.dumps(rec) + "\n")


def test_subset_restricts_the_pool_before_counting(tmp_path, setup, monkeypatch):
    cfg_path, _, out = setup
    collected = tmp_path / "many.jsonl"
    # collected order: 1 solved, 2 zero-of-3, 3 zero-of-3 with error (skipped), 4 zero-of-3,
    # 5 zero-of-3; subset 3 keeps the first three 0-of-3 questions: 2, 3, 4 (and 1, solved)
    _write_collected(
        collected,
        [
            (1, "hotpotqa", [True, True, True], {}),
            (2, "hotpotqa", [False] * 3, {}),
            (3, "hotpotqa", [False] * 3, {"error": "boom"}),
            (4, "hotpotqa", [False] * 3, {}),
            (5, "hotpotqa", [False] * 3, {}),
        ],
    )
    reflector = FakeReflector([[_reply(1), _reply(1)]])
    orch = FakeOrchestrator(Prov)
    monkeypatch.setattr(
        SCRIPT,
        "build_runtime",
        lambda exp, icfg: (orch, [Prov], reflector, lambda msgs: 10, FakeCache()),
    )
    argv = ["--config", str(cfg_path), "--collected", str(collected), "--out-dir", str(out)]
    assert SCRIPT.main(argv + ["--subset", "3"]) == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["search"]["zero_of_n"] == 3
    assert gate["search"]["skipped_error"] == 1 and gate["search"]["candidates"] == 2
    # both candidates rescued by the fake orchestrator: 2 of the 3 questions, not 2 of 4
    assert gate["search"]["rescued"] == 2
    assert gate["search"]["rescue_rate"] == pytest.approx(100 * 2 / 3)
    cands = json.loads((out / "candidates.json").read_text())
    assert [c["question_id"] for c in cands["candidates"]] == [2, 4]
    assert cands["pool_question_ids"] == [1, 2, 3, 4]


def test_no_subset_keeps_counts_over_every_question(tmp_path, setup, monkeypatch):
    cfg_path, _, out = setup
    collected = tmp_path / "many.jsonl"
    _write_collected(collected, [(i, "hotpotqa", [False] * 3, {}) for i in (1, 2, 3)])
    argv = ["--config", str(cfg_path), "--collected", str(collected), "--out-dir", str(out)]
    assert SCRIPT.main(argv + ["--report-only"]) == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["search"]["zero_of_n"] == 3
    assert "pool_question_ids" not in json.loads((out / "candidates.json").read_text())
