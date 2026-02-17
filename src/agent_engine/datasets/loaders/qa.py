"""QA dataset loaders.

Loaders for question answering benchmarks: NQ, TriviaQA, HotpotQA, etc.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry
from ..evaluators.metrics import evaluate_qa
from ...utils.logging import get_logger

logger = get_logger(__name__)


def _load_qa_jsonl(data_path: Path, dataset_name: str, subset_num: int = -1) -> List[DatasetExample]:
    """Common loader for QA datasets in JSONL format.

    Args:
        data_path: Path to JSONL file
        dataset_name: Name of the dataset
        subset_num: Number of examples to load (-1 for all)

    Returns:
        List of DatasetExample objects
    """
    if not data_path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset not found at: {data_path}")

    logger.info(f"Loading {dataset_name} from: {data_path}")

    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)

                # Handle different field names
                question = data.get('Question') or data.get('question') or data.get('input', '')
                answer = data.get('Answer') or data.get('answer') or data.get('output', '')

                example = DatasetExample(
                    question_id=idx,
                    question=question,
                    answer=answer,
                    metadata={
                        'dataset': dataset_name,
                    }
                )
                examples.append(example)

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing line {idx}: {e}")
                continue

    logger.info(f"Loaded {len(examples)} {dataset_name} examples")

    if subset_num > 0:
        examples = examples[:subset_num]
        logger.info(f"Using subset of {len(examples)} examples")

    return examples


@DatasetRegistry.register("nq")
class NaturalQuestionsDataset(BaseDataset):
    """Natural Questions dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load Natural Questions dataset."""
        data_path = self.config.data_dir / "QA_Datasets" / "nq.jsonl"
        return _load_qa_jsonl(data_path, "Natural Questions", self.config.subset_num)

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate NQ prediction."""
        return evaluate_qa(prediction, ground_truth)


@DatasetRegistry.register("triviaqa")
class TriviaQADataset(BaseDataset):
    """TriviaQA dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load TriviaQA dataset."""
        data_path = self.config.data_dir / "QA_Datasets" / "triviaqa.jsonl"
        return _load_qa_jsonl(data_path, "TriviaQA", self.config.subset_num)

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate TriviaQA prediction."""
        return evaluate_qa(prediction, ground_truth)


@DatasetRegistry.register("hotpotqa")
class HotpotQADataset(BaseDataset):
    """HotpotQA dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load HotpotQA dataset."""
        data_path = self.config.data_dir / "QA_Datasets" / "hotpotqa.jsonl"
        return _load_qa_jsonl(data_path, "HotpotQA", self.config.subset_num)

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate HotpotQA prediction."""
        return evaluate_qa(prediction, ground_truth)


@DatasetRegistry.register("musique")
class MusiqueDataset(BaseDataset):
    """MuSiQue dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load MuSiQue dataset."""
        data_path = self.config.data_dir / "QA_Datasets" / "musique.jsonl"
        return _load_qa_jsonl(data_path, "MuSiQue", self.config.subset_num)

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate MuSiQue prediction."""
        return evaluate_qa(prediction, ground_truth)


@DatasetRegistry.register("bamboogle")
class BamboogleDataset(BaseDataset):
    """Bamboogle dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load Bamboogle dataset."""
        data_path = self.config.data_dir / "QA_Datasets" / "bamboogle.jsonl"
        return _load_qa_jsonl(data_path, "Bamboogle", self.config.subset_num)

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate Bamboogle prediction."""
        return evaluate_qa(prediction, ground_truth)


@DatasetRegistry.register("2wiki")
class TwoWikiDataset(BaseDataset):
    """2WikiMultihopQA dataset loader."""

    def load(self) -> List[DatasetExample]:
        """Load 2WikiMultihopQA dataset."""
        data_path = self.config.data_dir / "QA_Datasets" / "2wiki.jsonl"
        return _load_qa_jsonl(data_path, "2WikiMultihopQA", self.config.subset_num)

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate 2Wiki prediction."""
        return evaluate_qa(prediction, ground_truth)
