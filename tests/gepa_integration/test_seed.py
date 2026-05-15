import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
from gepa_integration.seed import build_seed_candidate, build_splits


# ── build_seed_candidate ─────────────────────────────────────────────────────

def test_build_seed_candidate_returns_two_keys():
    candidate = build_seed_candidate(benchmark="gaia", tool_schemas=[], direct_tool_call=True)
    assert set(candidate.keys()) == {"system_prompt", "planning_suffix"}


def test_build_seed_candidate_planning_suffix_matches_constant():
    candidate = build_seed_candidate(benchmark="gaia", tool_schemas=[], direct_tool_call=True)
    assert candidate["planning_suffix"] == _DEFAULT_PLANNING_SUFFIX_TOOLS


def test_build_seed_candidate_system_prompt_is_string():
    candidate = build_seed_candidate(benchmark="gaia", tool_schemas=[], direct_tool_call=True)
    assert isinstance(candidate["system_prompt"], str)
    assert len(candidate["system_prompt"]) > 0


def test_build_seed_candidate_system_prompt_no_tools_when_empty():
    candidate = build_seed_candidate(benchmark="gaia", tool_schemas=[], direct_tool_call=True)
    assert "<tools>" not in candidate["system_prompt"]


def test_build_seed_candidate_system_prompt_contains_tool_schema_when_provided():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    candidate = build_seed_candidate(benchmark="gaia", tool_schemas=[tool_schema], direct_tool_call=True)
    assert "<tools>" in candidate["system_prompt"]
    assert "web_search" in candidate["system_prompt"]


# ── build_splits ─────────────────────────────────────────────────────────────

def _make_raw_results(n_correct: int, failures_by_mode: dict) -> list:
    records = []
    qid = 0
    for _ in range(n_correct):
        records.append({
            "question_id": qid,
            "question": f"q{qid}",
            "correct": True,
            "prediction": "right",
            "action_history": [],
            "turns": 1,
        })
        qid += 1
    for _ in range(failures_by_mode.get("tool_loop_or_empty_final", 0)):
        records.append({
            "question_id": qid,
            "question": f"q{qid}",
            "correct": False,
            "prediction": "",
            "action_history": [],
            "turns": 3,
        })
        qid += 1
    for _ in range(failures_by_mode.get("retrieval_evidence_failure", 0)):
        records.append({
            "question_id": qid,
            "question": f"q{qid}",
            "correct": False,
            "prediction": "wrong",
            "action_history": [
                {"tool_name": "web_search", "sub_goal": "s1", "result": "r1"},
                {"tool_name": "web_search", "sub_goal": "s2", "result": "r2"},
            ],
            "turns": 2,
        })
        qid += 1
    return records


def test_build_splits_returns_three_keys():
    records = _make_raw_results(60, {"tool_loop_or_empty_final": 20, "retrieval_evidence_failure": 20})
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)
    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    assert set(splits.keys()) == {"train", "val", "test"}


def test_build_splits_sizes_are_correct():
    records = _make_raw_results(60, {"tool_loop_or_empty_final": 20, "retrieval_evidence_failure": 20})
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)
    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    assert len(splits["train"]) == 50
    assert len(splits["val"]) == 20
    assert len(splits["test"]) == 100 - 50 - 20


def test_build_splits_no_overlap():
    records = _make_raw_results(60, {"tool_loop_or_empty_final": 20, "retrieval_evidence_failure": 20})
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)
    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["val"]) & set(splits["test"])


def test_build_splits_train_contains_mostly_failures():
    records = _make_raw_results(60, {"tool_loop_or_empty_final": 20, "retrieval_evidence_failure": 20})
    failed_qids = {r["question_id"] for r in records if not r["correct"]}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)
    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    train_failures = sum(1 for qid in splits["train"] if qid in failed_qids)
    assert train_failures >= 25


def test_build_splits_saves_json(tmp_path):
    records = _make_raw_results(60, {"tool_loop_or_empty_final": 20, "retrieval_evidence_failure": 20})
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        raw_path = Path(f.name)
    out_path = tmp_path / "splits.json"
    build_splits(raw_results_path=raw_path, train_n=50, val_n=20, seed=42, output_path=out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert set(loaded.keys()) == {"train", "val", "test"}
