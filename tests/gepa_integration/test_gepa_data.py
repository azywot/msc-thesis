"""Tests for src/gepa_integration/data/prepare.py and loader.py.

All HuggingFace downloads are mocked — tests run fully offline.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ---------------------------------------------------------------------------
# Helpers — use the fine_tuning normalised row schema (not the GEPA output schema)
# prepare.py's _norm_to_example expects: "question", "result", "data_source",
# "extra_info": {"golden_answers": [...]}
# ---------------------------------------------------------------------------

def _make_search_rows(n, source="hotpotqa"):
    """n fake Search-R1 rows in fine_tuning normalised format."""
    return [
        {
            "data_source": source,
            "question": f"Search Q {i}",
            "result": f"Answer {i}",
            "extra_info": {
                "idx": i,
                "groundtruth": f"Answer {i}",
                "golden_answers": [f"Answer {i}", f"Alt {i}"],
            },
        }
        for i in range(n)
    ]


def _make_math_rows(n):
    """n fake DeepMath rows in fine_tuning normalised format."""
    return [
        {
            "data_source": "deepmath",
            "question": f"Math Q {i}",
            "result": f"{i}",
            "extra_info": {
                "idx": i,
                "groundtruth": f"{i}",
            },
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# prepare.py: make_gepa_splits
# ---------------------------------------------------------------------------

class TestMakeGEPASplits:
    def _fn(self):
        from gepa_integration.data.prepare import make_gepa_splits
        return make_gepa_splits

    def test_correct_split_sizes(self):
        make = self._fn()
        all_ids = list(range(300))
        splits = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        assert len(splits["train"]) == 150
        assert len(splits["val"]) == 50
        assert len(splits["test"]) == 100

    def test_no_overlap(self):
        make = self._fn()
        all_ids = list(range(300))
        splits = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_assigned) == 300
        assert set(splits["train"]) & set(splits["val"]) == set()
        assert set(splits["train"]) & set(splits["test"]) == set()
        assert set(splits["val"]) & set(splits["test"]) == set()

    def test_deterministic_given_seed(self):
        make = self._fn()
        all_ids = list(range(300))
        s1 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        s2 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        assert s1 == s2

    def test_different_seeds_give_different_splits(self):
        make = self._fn()
        all_ids = list(range(300))
        s1 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        s2 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=2)
        assert s1["train"] != s2["train"]

    def test_raises_if_ids_too_few(self):
        make = self._fn()
        with pytest.raises(ValueError, match="n_feedback.*n_pareto.*n_test"):
            make(list(range(10)), n_feedback=150, n_pareto=50, n_test=100, seed=1)


# ---------------------------------------------------------------------------
# prepare.py: build_gepa_examples
# ---------------------------------------------------------------------------

class TestBuildGEPAExamples:
    def _fn(self):
        from gepa_integration.data.prepare import build_gepa_examples
        return build_gepa_examples

    def test_assigns_sequential_ids(self):
        build = self._fn()
        examples = build(
            feedback_search=_make_search_rows(2), feedback_math=_make_math_rows(1),
            pareto_search=_make_search_rows(2), pareto_math=_make_math_rows(1),
            test_search=_make_search_rows(2), test_math=_make_math_rows(1),
            seed=1,
        )
        ids = [ex["question_id"] for ex in examples]
        assert ids == list(range(len(examples)))

    def test_total_example_count(self):
        build = self._fn()
        examples = build(
            feedback_search=_make_search_rows(3), feedback_math=_make_math_rows(1),
            pareto_search=_make_search_rows(2), pareto_math=_make_math_rows(1),
            test_search=_make_search_rows(2), test_math=_make_math_rows(1),
            seed=1,
        )
        assert len(examples) == 10  # 4 + 3 + 3

    def test_required_fields_present(self):
        build = self._fn()
        examples = build(
            feedback_search=_make_search_rows(1), feedback_math=_make_math_rows(1),
            pareto_search=_make_search_rows(1), pareto_math=[],
            test_search=_make_search_rows(1), test_math=[],
            seed=1,
        )
        for ex in examples:
            assert "question_id" in ex
            assert "question" in ex
            assert "answer" in ex
            assert "answer_aliases" in ex
            assert "data_source" in ex

    def test_question_id_type_is_int(self):
        build = self._fn()
        examples = build(
            feedback_search=_make_search_rows(2), feedback_math=[],
            pareto_search=[], pareto_math=[],
            test_search=[], test_math=[],
            seed=1,
        )
        for ex in examples:
            assert isinstance(ex["question_id"], int)

    def test_norm_maps_result_to_answer(self):
        """_norm_to_example maps fine_tuning 'result' → GEPA 'answer'."""
        build = self._fn()
        rows = _make_search_rows(1)  # result = "Answer 0"
        examples = build(
            feedback_search=rows, feedback_math=[],
            pareto_search=[], pareto_math=[],
            test_search=[], test_math=[],
            seed=1,
        )
        assert examples[0]["answer"] == "Answer 0"

    def test_norm_maps_golden_answers_to_aliases(self):
        """_norm_to_example maps extra_info.golden_answers → answer_aliases."""
        build = self._fn()
        rows = _make_search_rows(1)  # golden_answers = ["Answer 0", "Alt 0"]
        examples = build(
            feedback_search=rows, feedback_math=[],
            pareto_search=[], pareto_math=[],
            test_search=[], test_math=[],
            seed=1,
        )
        assert "Answer 0" in examples[0]["answer_aliases"]
        assert "Alt 0" in examples[0]["answer_aliases"]


# ---------------------------------------------------------------------------
# prepare.py: save_gepa_data / round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_save_and_reload(self, tmp_path):
        from gepa_integration.data.prepare import save_gepa_data

        examples = [
            {"question_id": 0, "question": "Q?", "answer": "A",
             "answer_aliases": ["A"], "data_source": "hotpotqa"},
        ]
        splits = {"train": [0], "val": [], "test": []}
        data_file = tmp_path / "all_examples.json"
        splits_file = tmp_path / "splits.json"

        save_gepa_data(examples, splits, data_file, splits_file)

        loaded_examples = json.loads(data_file.read_text())
        loaded_splits = json.loads(splits_file.read_text())

        assert loaded_examples == examples
        assert loaded_splits == splits

    def test_output_dir_created_if_missing(self, tmp_path):
        from gepa_integration.data.prepare import save_gepa_data

        data_file = tmp_path / "sub" / "nested" / "data.json"
        splits_file = tmp_path / "splits.json"
        save_gepa_data([], {"train": [], "val": [], "test": []}, data_file, splits_file)
        assert data_file.exists()


# ---------------------------------------------------------------------------
# loader.py: load_gepa_examples
# ---------------------------------------------------------------------------

class TestLoadGEPAExamples:
    def _write_examples(self, tmp_path, examples):
        p = tmp_path / "all_examples.json"
        p.write_text(json.dumps(examples))
        return p

    def test_loads_by_id(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [
            {"question_id": 0, "question": "Q0", "answer": "A0",
             "answer_aliases": ["A0"], "data_source": "hotpotqa"},
            {"question_id": 1, "question": "Q1", "answer": "A1",
             "answer_aliases": [], "data_source": "deepmath"},
            {"question_id": 2, "question": "Q2", "answer": "A2",
             "answer_aliases": ["A2", "alt"], "data_source": "nq"},
        ]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [0, 2])
        assert len(examples) == 2
        assert {ex.question_id for ex in examples} == {0, 2}

    def test_returns_dataset_examples(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        from agent_engine.datasets.base import DatasetExample
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": [], "data_source": "hotpotqa"}]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [0])
        assert isinstance(examples[0], DatasetExample)

    def test_question_and_answer_set_correctly(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 5, "question": "What is X?", "answer": "42",
                "answer_aliases": ["forty-two"], "data_source": "nq"}]
        p = self._write_examples(tmp_path, raw)
        ex = load_gepa_examples(p, [5])[0]
        assert ex.question == "What is X?"
        assert ex.answer == "42"
        assert ex.question_id == 5

    def test_answer_aliases_in_metadata(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": ["A", "alt"], "data_source": "hotpotqa"}]
        p = self._write_examples(tmp_path, raw)
        ex = load_gepa_examples(p, [0])[0]
        assert ex.metadata["answer_aliases"] == ["A", "alt"]

    def test_unknown_ids_ignored(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": [], "data_source": "nq"}]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [0, 99])  # 99 doesn't exist
        assert len(examples) == 1

    def test_empty_ids_returns_empty(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": [], "data_source": "nq"}]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [])
        assert examples == []

    def test_file_not_found_raises(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        with pytest.raises(FileNotFoundError):
            load_gepa_examples(tmp_path / "missing.json", [0])
