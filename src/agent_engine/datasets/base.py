"""Base classes for dataset loading and evaluation.

This module defines the core abstractions for working with different datasets
in a unified way, with automatic registration support.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..config.schema import DatasetConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _random_subset(examples: list, num: int) -> list:
    """Randomly sample num examples, preserving original order.

    Matches multi-agent-tools behavior: shuffle indices with the current
    random state (seeded before dataset loading), take first num, then
    sort to restore original dataset order.
    """
    if num <= 0 or num >= len(examples):
        return examples
    indices = list(range(len(examples)))
    random.shuffle(indices)
    selected = sorted(indices[:num])
    return [examples[i] for i in selected]


@dataclass
class DatasetExample:
    """Single dataset example.

    Attributes:
        question_id: Unique identifier for the question
        question: Question text
        answer: Ground truth answer
        metadata: Additional information (choices, level, attachments, etc.)
    """
    question_id: int
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_attachments(self) -> List[str]:
        """Get list of file attachments.

        Returns:
            List of attachment file paths
        """
        return self.metadata.get('attachments', [])

    def has_attachments(self) -> bool:
        """Check if example has attachments.

        Returns:
            True if attachments exist
        """
        return bool(self.get_attachments())


class BaseDataset(ABC):
    """Abstract base for datasets.

    Subclasses must implement:
    - load(): Load dataset from disk
    - evaluate(): Evaluate a prediction
    """

    def __init__(self, config: DatasetConfig):
        """Initialize dataset with configuration.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self._examples: List[DatasetExample] = []
        self._loaded = False

    @abstractmethod
    def load(self) -> List[DatasetExample]:
        """Load dataset from disk.

        Returns:
            List of DatasetExample objects
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a prediction against ground truth.

        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            metadata: Example metadata for context-specific evaluation

        Returns:
            Dictionary with evaluation metrics (e.g., {"correct": True, "score": 1.0})
        """
        pass

    def load_if_needed(self):
        """Load dataset if not already loaded."""
        if not self._loaded:
            logger.info(f"Loading dataset: {self.config.name}")
            self._examples = self.load()
            self._loaded = True
            logger.info(f"Loaded {len(self._examples)} examples")

    def __len__(self) -> int:
        """Get number of examples."""
        self.load_if_needed()
        return len(self._examples)

    def __getitem__(self, idx: int) -> DatasetExample:
        """Get example by index."""
        self.load_if_needed()
        return self._examples[idx]

    def __iter__(self):
        """Iterate over examples."""
        self.load_if_needed()
        return iter(self._examples)

    def get_subset(self, num: int) -> List[DatasetExample]:
        """Get a subset of examples.

        Args:
            num: Number of examples to get (-1 for all)

        Returns:
            List of examples
        """
        self.load_if_needed()
        return _random_subset(self._examples, num)


class DatasetRegistry:
    """Registry for dataset loaders with decorator-based registration.

    Usage:
        @DatasetRegistry.register("my_dataset")
        class MyDataset(BaseDataset):
            ...
    """

    _datasets: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class.

        Args:
            name: Dataset name (e.g., "gaia", "gpqa")

        Returns:
            Decorator function
        """
        def wrapper(dataset_class: Type[BaseDataset]):
            if name in cls._datasets:
                logger.warning(f"Dataset '{name}' already registered, overwriting")
            cls._datasets[name] = dataset_class
            logger.debug(f"Registered dataset: {name}")
            return dataset_class
        return wrapper

    @classmethod
    def get(cls, config: DatasetConfig) -> BaseDataset:
        """Get a dataset instance from configuration.

        Args:
            config: Dataset configuration

        Returns:
            Dataset instance

        Raises:
            ValueError: If dataset not found
        """
        dataset_class = cls._datasets.get(config.name)
        if not dataset_class:
            available = ", ".join(cls._datasets.keys())
            raise ValueError(
                f"Unknown dataset: {config.name}. "
                f"Available datasets: {available}"
            )
        return dataset_class(config)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names.

        Returns:
            List of dataset names
        """
        return list(cls._datasets.keys())

    @classmethod
    def has_dataset(cls, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: Dataset name

        Returns:
            True if registered
        """
        return name in cls._datasets
