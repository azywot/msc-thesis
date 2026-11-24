import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from agent_engine.core.state import ExecutionState
from rsi.intervention.records import set_sampling_seed, trajectory_messages

REPO = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "collect_sft_data", REPO / "scripts/collect_sft_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCRIPT = _load_script()


def _state(answer="x"):
    return ExecutionState(
        question="q",
        question_id=0,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        query_analysis="plan",
        raw_query_analysis="<think>t</think>plan",
        action_history=[
            {"tool_name": "web_search", "sub_goal": "g", "command": "c", "result": "r"}
        ],
        output_messages=[
            {"role": "assistant", "content": "call"},
            {"role": "tool", "content": "res"},
            {"role": "assistant", "content": "answer"},
        ],
        answer=answer,
        turn=2,
        tool_counts={"web_search": 1},
        metadata={"subagent_calls": [{"action_index": 0, "tool": "web_search"}]},
    )


def test_trajectory_messages_matches_old_builder():
    st = _state()
    msgs = trajectory_messages(st)
    assert msgs[:2] == st.messages[:2]
    assert msgs[2] == {"role": "assistant", "content": st.raw_query_analysis}
    assert msgs[3:] == st.output_messages
    assert SCRIPT._build_sft_messages(st) == msgs


def test_set_sampling_seed_sets_each_provider_once():
    class Cfg:
        def __init__(self):
            self.writes = 0
            self._seed = None

        @property
        def seed(self):
            return self._seed

        @seed.setter
        def seed(self, v):
            self.writes += 1
            self._seed = v

    shared = SimpleNamespace(config=Cfg())
    other = SimpleNamespace(config=Cfg())
    set_sampling_seed([shared, shared, other, None], 7)
    assert shared.config.seed == 7 and shared.config.writes == 1
    assert other.config.seed == 7 and other.config.writes == 1


def test_make_record_has_resume_fields():
    st = _state()
    row = {"_qid": 4, "question": "q", "data_source": "nq", "result": "a"}
    rec = SCRIPT.make_record(row, st, 1, 3, ["al"], "x", False)
    assert rec["query_analysis"] == st.query_analysis
    assert rec["raw_query_analysis"] == st.raw_query_analysis
    assert rec["action_history"] == st.action_history
    assert rec["subagent_calls"] == st.metadata["subagent_calls"]
    assert rec["seed"] == 3 and rec["aliases"] == ["al"]
    assert rec["pass"] == 2 and rec["question_id"] == 4
    assert rec["turns"] == 2 and rec["tool_counts"] == {"web_search": 1}
    assert rec["messages"] == trajectory_messages(st)
    st.metadata = {}
    assert SCRIPT.make_record(row, st, 0, 0, [], "x", True)["subagent_calls"] == []


def _setup_main(tmp_path, monkeypatch, rows, predict):
    parquet = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(parquet)
    cfg = {
        "name": "t",
        "models": {
            "orchestrator": {
                "name": "Qwen3-8B",
                "family": "qwen3",
                "path_or_id": "Qwen/Qwen3-8B",
                "role": "orchestrator",
            }
        },
        "tools": {"enabled_tools": ["code_generator"]},
        "max_turns": 3,
        "batch_size": -1,
        "seed": 10,
        "thinking_mode": "NO",
        "cache_dir": str(tmp_path / "cache"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    provider = SimpleNamespace(config=SimpleNamespace(seed=None))
    seeds = []

    class FakeOrch:
        def __init__(self, **kw):
            pass

        def run_batch(self, questions, question_ids, system_prompts):
            seeds.append(provider.config.seed)
            return [predict(q) for q in questions]

    monkeypatch.setattr(SCRIPT, "_setup_model_provider", lambda *a, **k: provider)
    monkeypatch.setattr(
        SCRIPT, "_setup_tools", lambda *a, **k: SimpleNamespace(get_all_schemas=lambda: [])
    )
    monkeypatch.setattr(SCRIPT, "AgenticOrchestrator", FakeOrch)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "--config",
            str(cfg_path),
            "--parquet",
            str(parquet),
            "--output-dir",
            str(out),
            "--n-samples",
            "3",
        ],
    )
    return seeds, out


def _records(out):
    (path,) = list(out.glob("collected_*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_passes_use_distinct_seeds(monkeypatch, tmp_path):
    rows = [{"question": "q", "result": "gold", "data_source": "nq", "extra_info": None}]
    seeds, out = _setup_main(tmp_path, monkeypatch, rows, lambda q: _state(answer="wrong"))
    SCRIPT.main()
    # config.seed is 10, so pass k uses 10 + k
    assert seeds == [10, 11, 12]
    recs = _records(out)
    assert [r["seed"] for r in recs] == [10, 11, 12]
    assert [r["pass"] for r in recs] == [1, 2, 3]


def test_grading_accepts_alias(monkeypatch, tmp_path):
    rows = [
        {
            "question": "q",
            "result": "NYC",
            "data_source": "nq",
            "extra_info": {"golden_answers": ["NYC", "New York City"]},
        }
    ]
    seeds, out = _setup_main(tmp_path, monkeypatch, rows, lambda q: _state(answer="New York City"))
    SCRIPT.main()
    recs = _records(out)
    assert len(recs) == 1 and recs[0]["correct"] is True
    assert recs[0]["aliases"] == ["NYC", "New York City"]
