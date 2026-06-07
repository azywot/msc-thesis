"""Tests for golden_answers / answer_aliases support in QA dataset loaders.

Covers:
  - _load_qa_jsonl: golden_answers and answer_aliases field extraction
  - NQ/TriviaQA/HotpotQA/Bamboogle/2Wiki/MuSiQue evaluate() using aliases
  - evaluate_musique export from evaluators __init__
  - numeric_match removed from evaluators __init__
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

pytest.importorskip("math_verify", reason="math_verify required for metrics")

from agent_engine.config.schema import DatasetConfig
from agent_engine.datasets.loaders.qa import (
    _load_qa_jsonl,
    NaturalQuestionsDataset,
    TriviaQADataset,
    HotpotQADataset,
    BamboogleDataset,
    TwoWikiDataset,
    MusiqueDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_config(name: str, data_dir: Path) -> DatasetConfig:
    return DatasetConfig(name=name, split="test", data_dir=data_dir)


# ---------------------------------------------------------------------------
# _load_qa_jsonl: field extraction
# ---------------------------------------------------------------------------

class TestLoadQaJsonlAliases:
    def test_golden_answers_stored_as_answer_aliases(self, tmp_path):
        _write_jsonl(tmp_path / "nq.jsonl", [
            {"question": "Q1", "answer": "A1", "golden_answers": ["A1", "answer one"]},
        ])
        examples = _load_qa_jsonl(tmp_path / "nq.jsonl", "nq")
        assert examples[0].metadata["answer_aliases"] == ["A1", "answer one"]

    def test_answer_aliases_takes_priority_over_golden_answers(self, tmp_path):
        _write_jsonl(tmp_path / "musique.jsonl", [
            {"question": "Q1", "answer": "A1",
             "answer_aliases": ["alias1"], "golden_answers": ["ga1"]},
        ])
        examples = _load_qa_jsonl(tmp_path / "musique.jsonl", "musique")
        assert examples[0].metadata["answer_aliases"] == ["alias1"]

    def test_no_aliases_when_field_absent(self, tmp_path):
        _write_jsonl(tmp_path / "nq.jsonl", [
            {"question": "Q1", "answer": "A1"},
        ])
        examples = _load_qa_jsonl(tmp_path / "nq.jsonl", "nq")
        assert "answer_aliases" not in examples[0].metadata

    def test_golden_answers_not_a_list_is_ignored(self, tmp_path):
        _write_jsonl(tmp_path / "nq.jsonl", [
            {"question": "Q1", "answer": "A1", "golden_answers": "not a list"},
        ])
        examples = _load_qa_jsonl(tmp_path / "nq.jsonl", "nq")
        assert "answer_aliases" not in examples[0].metadata

    def test_empty_golden_answers_list_is_stored(self, tmp_path):
        _write_jsonl(tmp_path / "nq.jsonl", [
            {"question": "Q1", "answer": "A1", "golden_answers": []},
        ])
        examples = _load_qa_jsonl(tmp_path / "nq.jsonl", "nq")
        assert examples[0].metadata["answer_aliases"] == []

    def test_handles_Question_Answer_capitalised(self, tmp_path):
        _write_jsonl(tmp_path / "nq.jsonl", [
            {"Question": "Big Q", "Answer": "Big A", "golden_answers": ["Big A", "big a"]},
        ])
        examples = _load_qa_jsonl(tmp_path / "nq.jsonl", "nq")
        assert examples[0].question == "Big Q"
        assert examples[0].answer == "Big A"
        assert examples[0].metadata["answer_aliases"] == ["Big A", "big a"]


# ---------------------------------------------------------------------------
# Dataset evaluate() — alias acceptance
# ---------------------------------------------------------------------------

_METADATA_WITH_ALIASES = {"answer_aliases": ["franchise", "suffrage"]}
_METADATA_NO_ALIASES = {}

_DATASETS = [
    ("nq", NaturalQuestionsDataset),
    ("triviaqa", TriviaQADataset),
    ("hotpotqa", HotpotQADataset),
    ("bamboogle", BamboogleDataset),
    ("2wiki", TwoWikiDataset),
    ("musique", MusiqueDataset),
]


@pytest.mark.parametrize("name,cls", _DATASETS)
class TestQADatasetEvaluate:
    def _dataset(self, name, cls, tmp_path):
        return cls(_make_config(name, tmp_path))

    def test_correct_when_prediction_matches_ground_truth(self, name, cls, tmp_path):
        ds = self._dataset(name, cls, tmp_path)
        result = ds.evaluate("political franchise", "political franchise", _METADATA_NO_ALIASES)
        assert result["correct"] is True

    def test_correct_when_prediction_matches_alias(self, name, cls, tmp_path):
        ds = self._dataset(name, cls, tmp_path)
        # "franchise" is in _METADATA_WITH_ALIASES but not the ground truth
        result = ds.evaluate("franchise", "political franchise", _METADATA_WITH_ALIASES)
        assert result["correct"] is True

    def test_correct_when_prediction_matches_second_alias(self, name, cls, tmp_path):
        ds = self._dataset(name, cls, tmp_path)
        result = ds.evaluate("suffrage", "political franchise", _METADATA_WITH_ALIASES)
        assert result["correct"] is True

    def test_wrong_when_prediction_matches_nothing(self, name, cls, tmp_path):
        ds = self._dataset(name, cls, tmp_path)
        result = ds.evaluate("something unrelated", "political franchise", _METADATA_WITH_ALIASES)
        assert result["correct"] is False

    def test_wrong_when_no_aliases_and_prediction_is_wrong(self, name, cls, tmp_path):
        ds = self._dataset(name, cls, tmp_path)
        result = ds.evaluate("wrong answer", "correct answer", _METADATA_NO_ALIASES)
        assert result["correct"] is False

    def test_aliases_none_treated_as_empty(self, name, cls, tmp_path):
        ds = self._dataset(name, cls, tmp_path)
        result = ds.evaluate("the answer", "the answer", {"answer_aliases": None})
        assert result["correct"] is True


# ---------------------------------------------------------------------------
# Public API: evaluate_musique exported; numeric_match removed
# ---------------------------------------------------------------------------

class TestEvaluatorsPublicAPI:
    def test_evaluate_musique_importable_from_evaluators(self):
        from agent_engine.datasets.evaluators import evaluate_musique
        assert callable(evaluate_musique)

    def test_evaluate_musique_in_all(self):
        import agent_engine.datasets.evaluators as ev
        assert "evaluate_musique" in ev.__all__

    def test_numeric_match_not_in_evaluators_init(self):
        import agent_engine.datasets.evaluators as ev
        assert not hasattr(ev, "numeric_match")

    def test_numeric_match_not_in_all(self):
        import agent_engine.datasets.evaluators as ev
        assert "numeric_match" not in ev.__all__

    def test_numeric_match_not_in_metrics(self):
        import agent_engine.datasets.evaluators.metrics as m
        assert not hasattr(m, "numeric_match")


# ---------------------------------------------------------------------------
# evaluate_musique logic (canonical + alias)
# ---------------------------------------------------------------------------

class TestEvaluateMusique:
    @pytest.fixture(autouse=True)
    def _import(self):
        from agent_engine.datasets.evaluators.metrics import evaluate_musique
        self.fn = evaluate_musique

    def test_matches_canonical(self):
        assert self.fn("Paris", "Paris")["correct"] is True

    def test_matches_alias(self):
        assert self.fn("City of Light", "Paris", ["City of Light"])["correct"] is True

    def test_matches_alias_case_insensitive(self):
        assert self.fn("city of light", "Paris", ["City of Light"])["correct"] is True

    def test_rejects_when_nothing_matches(self):
        assert self.fn("Berlin", "Paris", ["City of Light"])["correct"] is False

    def test_empty_aliases_still_uses_canonical(self):
        assert self.fn("Paris", "Paris", [])["correct"] is True

    def test_none_aliases_treated_as_empty(self):
        assert self.fn("Paris", "Paris", None)["correct"] is True

    def test_result_has_required_keys(self):
        result = self.fn("Paris", "Paris")
        assert {"correct", "accuracy", "em", "f1"} <= result.keys()
