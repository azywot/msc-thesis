import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import pandas as pd
from fine_tuning.data.prepare import (
    normalise_search_r1_row,
    normalise_deepmath_row,
    validate_parquet_schema,
    REQUIRED_COLS,
)


class TestNormaliseSearchR1:
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
    def test_basic_row(self):
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
