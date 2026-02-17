"""Base evaluator classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
    """Abstract base for evaluators."""

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a prediction.

        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            metadata: Example metadata

        Returns:
            Dictionary with evaluation results
        """
        pass
