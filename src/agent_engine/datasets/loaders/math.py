"""Math dataset loaders.

Loaders for math benchmarks: MATH500, AIME, AMC.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry, _random_subset
from ..evaluators.metrics import evaluate_math
from ...utils.logging import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register("math500")
class MATH500Dataset(BaseDataset):
    """MATH500 dataset loader.

    A subset of 500 challenging math problems from the MATH dataset.
    """

    def load(self) -> List[DatasetExample]:
        """Load MATH500 dataset."""
        data_path = self.config.data_dir / "MATH500" / f"{self.config.split}.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"MATH500 dataset not found at: {data_path}\n"
                f"Please download MATH500 dataset and place it in {self.config.data_dir}/MATH500/"
            )

        logger.info(f"Loading MATH500 from: {data_path}")

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)

                    example = DatasetExample(
                        question_id=idx,
                        question=data['Question'],
                        answer=data['Answer'],
                        metadata={
                            'problem_type': data.get('type', 'unknown'),
                            'difficulty': data.get('level', 'unknown'),
                        }
                    )
                    examples.append(example)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing line {idx}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} MATH500 examples")

        if self.config.subset_num > 0:
            examples = _random_subset(examples, self.config.subset_num)

        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate math prediction."""
        return evaluate_math(prediction, ground_truth)


@DatasetRegistry.register("aime")
class AIMEDataset(BaseDataset):
    """AIME (American Invitational Mathematics Examination) dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load AIME dataset."""
        data_path = self.config.data_dir / "AIME" / f"{self.config.split}.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"AIME dataset not found at: {data_path}"
            )

        logger.info(f"Loading AIME from: {data_path}")

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)

                    example = DatasetExample(
                        question_id=idx,
                        question=data['Question'],
                        answer=data['Answer'],
                        metadata={
                            'year': data.get('year', 'unknown'),
                            'problem_number': data.get('problem_number', idx),
                        }
                    )
                    examples.append(example)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing line {idx}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} AIME examples")

        if self.config.subset_num > 0:
            examples = _random_subset(examples, self.config.subset_num)

        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate AIME prediction."""
        return evaluate_math(prediction, ground_truth)


@DatasetRegistry.register("amc")
class AMCDataset(BaseDataset):
    """AMC (American Mathematics Competition) dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load AMC dataset."""
        data_path = self.config.data_dir / "AMC" / f"{self.config.split}.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"AMC dataset not found at: {data_path}"
            )

        logger.info(f"Loading AMC from: {data_path}")

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)

                    example = DatasetExample(
                        question_id=idx,
                        question=data['Question'],
                        answer=data['Answer'],
                        metadata={
                            'year': data.get('year', 'unknown'),
                            'competition': data.get('competition', 'AMC'),
                        }
                    )
                    examples.append(example)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing line {idx}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} AMC examples")

        if self.config.subset_num > 0:
            examples = _random_subset(examples, self.config.subset_num)

        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate AMC prediction."""
        return evaluate_math(prediction, ground_truth)
