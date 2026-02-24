"""GAIA dataset loader.

GAIA (General AI Assistant) benchmark for evaluating AI assistants
on complex real-world tasks.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry
from ..evaluators.metrics import evaluate_gaia
from ...utils.logging import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register("gaia")
class GAIADataset(BaseDataset):
    """GAIA dataset loader.

    GAIA consists of questions that may require:
    - Web search
    - File analysis (attachments)
    - Multi-step reasoning
    - Tool use
    """

    def load(self) -> List[DatasetExample]:
        """Load GAIA dataset from JSONL file.

        Expected format:
        {
            "Question": "question text",
            "Answer": "answer",
            "Level": 1-3,
            "file_name": "optional_attachment.ext",
            "file_path": "relative/path/inside_HF_repo.ext",
            "full_file_path": "/absolute/path/to/downloaded/file.ext"  # preferred for tools
        }

        Returns:
            List of DatasetExample objects
        """
        # Construct path
        data_path = self.config.data_dir / "GAIA" / f"{self.config.split}.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"GAIA dataset not found at: {data_path}\n"
                f"Please download GAIA dataset and place it in {self.config.data_dir}/GAIA/"
            )

        logger.info(f"Loading GAIA from: {data_path}")

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)

                    # Extract attachments
                    attachments = []
                    if data.get("file_name"):
                        # Prefer the absolute full_file_path written by download_gaia(),
                        # since GAIA files live inside a HF snapshot directory.
                        full_fp = data.get("full_file_path")
                        if full_fp:
                            attachments.append(str(full_fp))
                        else:
                            file_path = data.get("file_path", "")
                            if file_path:
                                # Fallback: construct path relative to data_dir/GAIA
                                attachment_path = self.config.data_dir / "GAIA" / file_path
                                attachments.append(str(attachment_path))

                    example = DatasetExample(
                        question_id=idx,
                        question=data['Question'],
                        answer=data['Answer'],
                        metadata={
                            "level": data.get("Level", 1),
                            "file_name": data.get("file_name", ""),
                            "file_path": data.get("file_path", ""),
                            "full_file_path": data.get("full_file_path", ""),
                            "attachments": attachments,
                        }
                    )
                    examples.append(example)

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {idx}: {e}")
                    continue
                except KeyError as e:
                    logger.error(f"Missing required field in line {idx}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} GAIA examples")

        # Apply subset if specified
        if self.config.subset_num > 0:
            examples = examples[:self.config.subset_num]
            logger.info(f"Using subset of {len(examples)} examples")

        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        result = evaluate_gaia(prediction, ground_truth)
        result['level'] = metadata.get('level', 1)
        return result
