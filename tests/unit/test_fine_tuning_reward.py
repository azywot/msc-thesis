import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
pytest.importorskip("math_verify", reason="math_verify not installed in local dev env; tests run on cluster")

from fine_tuning.reward import OrchestratorReward

reward_fn = OrchestratorReward()


class TestSearchDomain:
    def test_correct_nq(self):
        assert reward_fn("Paris", "Paris", "nq") == 1.0

    def test_containment_nq(self):
        # GAIA scorer: prediction containing the ground truth → correct
        assert reward_fn("The capital is \\boxed{Paris}", "Paris", "nq") == 1.0

    def test_wrong_nq(self):
        assert reward_fn("London", "Paris", "nq") == 0.0

    def test_correct_hotpotqa(self):
        assert reward_fn("yes", "yes", "hotpotqa") == 1.0

    def test_wrong_hotpotqa(self):
        assert reward_fn("no", "yes", "hotpotqa") == 0.0


class TestMathDomain:
    def test_correct_math(self):
        assert reward_fn("42", "42", "math") == 1.0

    def test_correct_deepmath(self):
        assert reward_fn("7", "7", "deepmath") == 1.0

    def test_wrong_math(self):
        assert reward_fn("43", "42", "math") == 0.0

    def test_correct_aime(self):
        assert reward_fn("120", "120", "aime") == 1.0


class TestEdgeCases:
    def test_empty_prediction_returns_zero(self):
        assert reward_fn("", "Paris", "nq") == 0.0

    def test_none_prediction_returns_zero(self):
        assert reward_fn(None, "Paris", "nq") == 0.0

    def test_unknown_data_source_works(self):
        # unknown source falls through to evaluate_answer uniformly
        assert reward_fn("Paris", "Paris", "unknown_dataset") == 1.0
