import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

pytest.importorskip("math_verify", reason="math_verify required like other metrics tests")


from agent_engine.datasets.evaluators.metrics import is_math_answer


def test_paris_counts_as_plain_qa_not_math():
    assert not is_math_answer("Paris")


def test_yes_counts_as_plain_qa_not_math():
    assert not is_math_answer("yes")


def test_numeric_ground_truth_is_math():
    assert is_math_answer("42")
    assert is_math_answer("3.14")


def test_single_letter_keeps_math_path():
    """Short symbolic vars like x, n."""
    assert is_math_answer("x")
    assert is_math_answer("n")


def test_expression_with_operators_or_latex():
    assert is_math_answer("x + 1")
    assert is_math_answer(r"x^2")
