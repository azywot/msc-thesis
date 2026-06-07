"""Reward function for the orchestrator fine-tuning pipeline.

Routes correctness evaluation to the existing metrics.py evaluate_answer()
function. The data_source field in the parquet identifies the domain, but
evaluate_answer() handles all cases uniformly — so the routing is informational.
"""

from typing import Optional


class OrchestratorReward:
    """Computes binary reward (1.0 / 0.0) for a predicted answer.

    Calls evaluate_answer() from metrics.py. Returns 1.0 if the prediction
    is judged correct, 0.0 otherwise (including None/empty predictions).
    """

    def __call__(
        self,
        prediction: Optional[str],
        ground_truth: str,
        data_source: str,
    ) -> float:
        """Return 1.0 if prediction is correct, 0.0 otherwise.

        Args:
            prediction: Extracted answer string (may be None).
            ground_truth: Reference answer from the dataset.
            data_source: Dataset name string (e.g. "nq", "math", "deepmath").
                         Used for logging; evaluation logic is uniform.

        Returns:
            1.0 if correct, 0.0 if wrong or prediction is None/empty.
        """
        if not prediction:
            return 0.0

        from agent_engine.datasets.evaluators.metrics import evaluate_answer  # requires math_verify
        result = evaluate_answer(
            prediction=str(prediction),
            ground_truth=str(ground_truth),
        )
        return 1.0 if result["correct"] else 0.0
