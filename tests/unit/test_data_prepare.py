import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import pandas as pd
import fine_tuning.data.prepare as data_prepare
from fine_tuning.data.prepare import (
    normalise_search_r1_row,
    normalise_deepmath_row,
    normalise_aime_row,
    normalise_aime_local_row,
    validate_parquet_schema,
    REQUIRED_COLS,
)


class TestNormaliseSearchR1:
    def test_hub_golden_answers_row(self):
        raw = {
            "question": "Who wrote Hamlet?",
            "golden_answers": ["William Shakespeare", "Shakespeare"],
            "data_source": "nq",
        }
        row = normalise_search_r1_row(raw, idx=0)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "nq"
        assert row["question"] == "Who wrote Hamlet?"
        assert row["result"] == "William Shakespeare"
        assert row["extra_info"]["idx"] == 0
        assert row["extra_info"]["groundtruth"] == "William Shakespeare"
        assert row["extra_info"]["golden_answers"] == [
            "William Shakespeare",
            "Shakespeare",
        ]

    def test_nq_row(self):
        raw = {"question": "Who wrote Hamlet?", "answer": "Shakespeare", "dataset": "nq"}
        row = normalise_search_r1_row(raw, idx=0)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "nq"
        assert row["question"] == "Who wrote Hamlet?"
        assert row["result"] == "Shakespeare"
        assert row["extra_info"]["idx"] == 0
        assert row["extra_info"]["groundtruth"] == "Shakespeare"

    def test_hotpotqa_row(self):
        raw = {"question": "Where was X born?", "answer": "London", "dataset": "hotpotqa"}
        row = normalise_search_r1_row(raw, idx=5)
        assert row["data_source"] == "hotpotqa"
        assert row["extra_info"]["idx"] == 5

    def test_answer_list_takes_first(self):
        raw = {"question": "Q?", "answer": ["First", "Second"], "dataset": "nq"}
        row = normalise_search_r1_row(raw, idx=0)
        assert row["result"] == "First"

    def test_answers_field_fallback(self):
        raw = {"question": "Q?", "answers": ["A"], "dataset": "nq"}
        row = normalise_search_r1_row(raw, idx=0)
        assert row["result"] == "A"



class TestNormaliseDeepMath:
    def test_hub_question_final_answer_row(self):
        """zwhe99/DeepMath-103K uses question + final_answer (not problem/answer)."""
        raw = {"question": "Compute 7+8", "final_answer": 15, "source": "aime"}
        row = normalise_deepmath_row(raw, idx=1)
        assert row["question"] == "Compute 7+8"
        assert row["result"] == "15"
        assert row["extra_info"]["groundtruth"] == "15"

    def test_basic_row_legacy_problem_answer(self):
        raw = {"problem": "What is 2+2?", "answer": "4", "source": "math"}
        row = normalise_deepmath_row(raw, idx=3)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "deepmath"
        assert row["question"] == "What is 2+2?"
        assert row["result"] == "4"
        assert row["extra_info"]["idx"] == 3

    def test_missing_source_defaults_to_deepmath(self):
        raw = {"problem": "Solve x=2", "answer": "2"}
        row = normalise_deepmath_row(raw, idx=0)
        assert row["data_source"] == "deepmath"

    def test_final_answer_wins_over_answer_when_both_present(self):
        raw = {"question": "Q", "final_answer": "correct", "answer": "wrong"}
        row = normalise_deepmath_row(raw, idx=0)
        assert row["result"] == "correct"

    def test_difficulty_stored_in_extra_info(self):
        raw = {"question": "Hard problem", "final_answer": "42", "difficulty": 7}
        row = normalise_deepmath_row(raw, idx=0)
        assert row["extra_info"]["difficulty"] == 7

    def test_difficulty_coerced_from_string(self):
        raw = {"question": "Hard problem", "final_answer": "42", "difficulty": "6"}
        row = normalise_deepmath_row(raw, idx=0)
        assert row["extra_info"]["difficulty"] == 6

    def test_missing_difficulty_not_in_extra_info(self):
        raw = {"question": "Problem", "final_answer": "1"}
        row = normalise_deepmath_row(raw, idx=0)
        assert "difficulty" not in row["extra_info"]


    def test_validity_detects_blank_question_or_answer(self):
        ok = normalise_deepmath_row(
            {"question": "Has text", "final_answer": "1"}, idx=0
        )
        assert data_prepare._is_valid_deepmath_norm(ok)
        no_q = normalise_deepmath_row({"question": "", "final_answer": "1"}, idx=0)
        assert not data_prepare._is_valid_deepmath_norm(no_q)
        no_a = normalise_deepmath_row({"question": "x", "final_answer": ""}, idx=0)
        assert not data_prepare._is_valid_deepmath_norm(no_a)


class TestSearchSourceQuotas:
    def test_both_splits_by_ratio(self):
        from fine_tuning.data.prepare import _search_source_quotas
        hotpot, nq = _search_source_quotas(n=1000, source="both", hotpot_ratio=0.85)
        assert hotpot == 850
        assert nq == 150

    def test_hotpotqa_only(self):
        from fine_tuning.data.prepare import _search_source_quotas
        hotpot, nq = _search_source_quotas(n=1000, source="hotpotqa", hotpot_ratio=0.85)
        assert hotpot == 1000
        assert nq == 0

    def test_nq_only(self):
        from fine_tuning.data.prepare import _search_source_quotas
        hotpot, nq = _search_source_quotas(n=1000, source="nq", hotpot_ratio=0.85)
        assert hotpot == 0
        assert nq == 1000

    def test_quotas_sum_to_n(self):
        from fine_tuning.data.prepare import _search_source_quotas
        for n in [7, 100, 1000, 3333]:
            hotpot, nq = _search_source_quotas(n=n, source="both", hotpot_ratio=0.85)
            assert hotpot + nq == n


class TestPassesDifficultyFilter:
    def test_row_above_threshold_passes(self):
        from fine_tuning.data.prepare import _passes_difficulty_filter
        assert _passes_difficulty_filter({"difficulty": 7}, min_difficulty=5) is True

    def test_row_at_threshold_passes(self):
        from fine_tuning.data.prepare import _passes_difficulty_filter
        assert _passes_difficulty_filter({"difficulty": 5}, min_difficulty=5) is True

    def test_row_below_threshold_fails(self):
        from fine_tuning.data.prepare import _passes_difficulty_filter
        assert _passes_difficulty_filter({"difficulty": 3}, min_difficulty=5) is False

    def test_row_without_difficulty_passes(self):
        # If the Hub row has no difficulty field, don't filter it out — fail open.
        from fine_tuning.data.prepare import _passes_difficulty_filter
        assert _passes_difficulty_filter({}, min_difficulty=5) is True

    def test_string_difficulty_coerced(self):
        # Hub may return difficulty as a string.
        from fine_tuning.data.prepare import _passes_difficulty_filter
        assert _passes_difficulty_filter({"difficulty": "6"}, min_difficulty=5) is True
        assert _passes_difficulty_filter({"difficulty": "4"}, min_difficulty=5) is False


class TestNormaliseAime:
    def test_basic_row_2024(self):
        raw = {"problem": "Find the value of x", "answer": "42"}
        row = normalise_aime_row(raw, idx=0, year=2024)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "aime_2024"
        assert row["question"] == "Find the value of x"
        assert row["result"] == "42"
        assert row["extra_info"]["idx"] == 0
        assert row["extra_info"]["groundtruth"] == "42"
        assert row["extra_info"]["year"] == 2024

    def test_basic_row_2025(self):
        raw = {"problem": "Compute the sum", "answer": "113"}
        row = normalise_aime_row(raw, idx=3, year=2025)
        assert row["data_source"] == "aime_2025"
        assert row["extra_info"]["year"] == 2025
        assert row["extra_info"]["idx"] == 3

    def test_answer_coerced_to_str(self):
        raw = {"problem": "Q", "answer": 7}
        row = normalise_aime_row(raw, idx=0, year=2024)
        assert row["result"] == "7"
        assert row["extra_info"]["groundtruth"] == "7"

    def test_missing_fields_produce_empty_strings(self):
        row = normalise_aime_row({}, idx=0, year=2024)
        assert row["question"] == ""
        assert row["result"] == ""


class TestNormaliseAimeLocal:
    def test_basic_row_capitalised_fields(self):
        raw = {"ID": 7, "Question": "Find n", "Answer": "42", "Year": 2024}
        row = normalise_aime_local_row(raw, idx=2)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "aime_2024"
        assert row["question"] == "Find n"
        assert row["result"] == "42"
        assert row["extra_info"]["idx"] == 2
        assert row["extra_info"]["groundtruth"] == "42"
        assert row["extra_info"]["year"] == 2024
        assert row["extra_info"]["aime_id"] == 7

    def test_year_2025(self):
        raw = {"Question": "Q", "Answer": "1", "Year": 2025}
        row = normalise_aime_local_row(raw, idx=0)
        assert row["data_source"] == "aime_2025"
        assert row["extra_info"]["year"] == 2025

    def test_missing_year_defaults_to_zero(self):
        raw = {"Question": "Q", "Answer": "1"}
        row = normalise_aime_local_row(raw, idx=0)
        assert row["data_source"] == "aime"
        assert row["extra_info"]["year"] == 0

    def test_answer_coerced_to_str(self):
        raw = {"Question": "Q", "Answer": 7, "Year": 2024}
        row = normalise_aime_local_row(raw, idx=0)
        assert row["result"] == "7"

    def test_lowercase_field_fallback(self):
        # If a caller passes the HF-style problem/answer keys, still work.
        raw = {"problem": "Q", "answer": "5", "Year": 2024}
        row = normalise_aime_local_row(raw, idx=0)
        assert row["question"] == "Q"
        assert row["result"] == "5"

    def test_missing_id_not_in_extra_info(self):
        raw = {"Question": "Q", "Answer": "1", "Year": 2024}
        row = normalise_aime_local_row(raw, idx=0)
        assert "aime_id" not in row["extra_info"]


class TestLoadAimeLocal:
    def test_n_zero_returns_empty_without_reading(self, tmp_path):
        from fine_tuning.data.prepare import _load_aime_local
        rows = _load_aime_local(tmp_path / "missing.jsonl", n=0, seed=1)
        assert rows == []

    def test_missing_file_raises(self, tmp_path):
        from fine_tuning.data.prepare import _load_aime_local
        with pytest.raises(FileNotFoundError):
            _load_aime_local(tmp_path / "missing.jsonl", n=5, seed=1)

    def test_returns_correct_count_and_schema(self, tmp_path):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _load_aime_local

        fake = [
            {"ID": i, "Question": f"Q{i}", "Answer": str(i), "Year": 2024}
            for i in range(30)
        ]
        path = tmp_path / "aime.jsonl"
        path.write_text("non-empty")  # _load_aime_local checks existence
        with patch("fine_tuning.data.prepare._read_jsonl", return_value=fake):
            rows = _load_aime_local(path, n=20, seed=42)

        assert len(rows) == 20
        assert all(set(r.keys()) == REQUIRED_COLS for r in rows)
        assert all(r["data_source"] == "aime_2024" for r in rows)
        assert [r["extra_info"]["idx"] for r in rows] == list(range(20))

    def test_deterministic_for_same_seed(self, tmp_path):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _load_aime_local

        fake = [
            {"ID": i, "Question": f"Q{i}", "Answer": str(i), "Year": 2024}
            for i in range(30)
        ]
        path = tmp_path / "aime.jsonl"
        path.write_text("non-empty")
        with patch("fine_tuning.data.prepare._read_jsonl", return_value=fake):
            a = _load_aime_local(path, n=5, seed=42)
            b = _load_aime_local(path, n=5, seed=42)
        assert [r["extra_info"]["aime_id"] for r in a] == [
            r["extra_info"]["aime_id"] for r in b
        ]

    def test_raises_if_not_enough_rows(self, tmp_path):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _load_aime_local

        fake = [{"ID": 1, "Question": "Q", "Answer": "1", "Year": 2024}]
        path = tmp_path / "aime.jsonl"
        path.write_text("non-empty")
        with patch("fine_tuning.data.prepare._read_jsonl", return_value=fake):
            with pytest.raises(RuntimeError, match="only 1 rows"):
                _load_aime_local(path, n=20, seed=1)


class TestValidateSchema:
    def test_valid_df_passes(self):
        df = pd.DataFrame([
            {
                "data_source": "nq",
                "question": "Q?",
                "result": "A",
                "extra_info": {"idx": 0, "groundtruth": "A"},
            }
        ])
        validate_parquet_schema(df)  # should not raise

    def test_missing_column_raises(self):
        df = pd.DataFrame([{"data_source": "nq", "question": "Q?", "result": "A"}])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_parquet_schema(df)


class TestDownloadAimeVal:
    """Tests for _download_aime_val — HF calls are mocked."""

    def _fake_rows(self, n):
        return [{"problem": f"Problem {i}", "answer": str(i)} for i in range(n)]

    def test_returns_correct_count_and_sources(self):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(30)
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake):
            rows = _download_aime_val(n_2024=10, n_2025=10, seed=42)

        assert len(rows) == 20
        assert sum(1 for r in rows if r["data_source"] == "aime_2024") == 10
        assert sum(1 for r in rows if r["data_source"] == "aime_2025") == 10

    def test_raises_if_not_enough_rows(self):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(5)  # only 5 available, need 10
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake):
            with pytest.raises(RuntimeError, match="only 5 rows available"):
                _download_aime_val(n_2024=10, n_2025=10, seed=42)

    def test_zero_n_skips_that_year(self):
        from unittest.mock import patch, call
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(30)
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake) as mock_load:
            rows = _download_aime_val(n_2024=0, n_2025=10, seed=42)

        assert len(rows) == 10
        assert all(r["data_source"] == "aime_2025" for r in rows)
        assert mock_load.call_count == 1  # only called for 2025

    def test_indices_are_per_year(self):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(30)
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake):
            rows = _download_aime_val(n_2024=3, n_2025=3, seed=42)

        rows_2024 = [r for r in rows if r["data_source"] == "aime_2024"]
        rows_2025 = [r for r in rows if r["data_source"] == "aime_2025"]
        assert [r["extra_info"]["idx"] for r in rows_2024] == [0, 1, 2]
        assert [r["extra_info"]["idx"] for r in rows_2025] == [0, 1, 2]
