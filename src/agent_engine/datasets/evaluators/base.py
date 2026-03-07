"""Base evaluator interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
    """Abstract base class for dataset evaluators.

    Concrete evaluators implement :meth:`evaluate` to score a single
    model prediction against the ground truth.  All dataset-specific
    evaluation logic (GAIA, GPQA, MATH, QA) can be encapsulated in a
    subclass instead of calling the module-level helpers in
    :mod:`~agent_engine.datasets.evaluators.metrics` directly.

    Example::

        class GAIAEvaluator(BaseEvaluator):
            def evaluate(self, prediction, ground_truth, metadata):
                from .metrics import evaluate_gaia
                return evaluate_gaia(prediction, ground_truth, metadata)
    """

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score *prediction* against *ground_truth*.

        Args:
            prediction: Model's predicted answer string.
            ground_truth: Reference answer string.
            metadata: Example-level metadata (e.g. difficulty level, choices).

        Returns:
            Dictionary with at minimum a boolean ``"correct"`` key, and any
            additional metrics (``"f1"``, ``"level"``, etc.) the evaluator
            produces.
        """
