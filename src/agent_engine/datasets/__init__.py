"""Dataset loading and evaluation system.

This module provides unified interfaces for loading and evaluating
different datasets.
"""

from .base import BaseDataset, DatasetExample, DatasetRegistry

# Import loaders to trigger registration
from .loaders import gaia, gpqa, math, qa

# Import evaluators
from .evaluators import metrics

__all__ = [
    "BaseDataset",
    "DatasetExample",
    "DatasetRegistry",
]
