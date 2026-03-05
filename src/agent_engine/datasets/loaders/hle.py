"""HLE dataset loader.

HLE (Humanity's Last Exam) consists of single-question QA items,
optionally with associated images. The dataset is stored as JSONL, similar
to GAIA, and is evaluated with the same unified QA metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry
from ..evaluators.metrics import evaluate_hle
from ...utils.logging import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register("hle")
class HLEDataset(BaseDataset):
    """HLE dataset loader.

    HLE examples are single QA pairs, some with images. Attachments are taken
    from absolute paths written by `scripts/download_datasets.py` when
    `--dataset hle` and `--extract-images` are used.
    """

    def _resolve_data_path(self) -> Path:
        """Resolve the JSONL path for the requested split.

        Primary expectation:
            <data_dir>/HLE/<split>.jsonl  (e.g. test.jsonl, dev.jsonl)

        When subsets were saved via `--subset N`, files follow:
            <split>_subset_N.jsonl  (e.g. test_subset_200.jsonl)
        """
        base_dir = self.config.data_dir / "HLE"
        preferred = base_dir / f"{self.config.split}.jsonl"
        if preferred.exists():
            return preferred

        # Fallback: any file that starts with the split name.
        candidates = sorted(base_dir.glob(f"{self.config.split}*.jsonl"))
        if candidates:
            logger.warning(
                "HLE split '%s' not found at %s, using fallback file %s",
                self.config.split,
                preferred,
                candidates[0],
            )
            return candidates[0]

        raise FileNotFoundError(
            f"HLE dataset not found for split='{self.config.split}' at: {preferred}\n"
            f"Expected files like '{self.config.split}.jsonl' or "
            f"'{self.config.split}_subset_N.jsonl' in {base_dir}"
        )

    def load(self) -> List[DatasetExample]:
        """Load HLE dataset from JSONL file."""
        data_path = self._resolve_data_path()
        logger.info(f"Loading HLE from: {data_path}")

        examples: List[DatasetExample] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)

                    # Attachments: use absolute full_file_path when available.
                    attachments: List[str] = []
                    full_fp = data.get("full_file_path")
                    if full_fp:
                        attachments.append(str(full_fp))

                    category = data.get("category", "")

                    example = DatasetExample(
                        question_id=idx,
                        question=data.get("Question", ""),
                        answer=data.get("Answer", ""),
                        metadata={
                            "id": data.get("id", f"hle_{self.config.split}_{idx}"),
                            "answer_type": data.get("Answer_type", ""),
                            "raw_subject": data.get("raw_subject", ""),
                            "category": category,
                            "level": category,
                            # "has_image": data.get("has_image", bool(resolved_fp)),
                            "file_name": data.get("file_name", ""),
                            "file_path": data.get("file_path", ""),
                            "full_file_path": full_fp or "",
                            # "image_base64": data.get("image_base64", ""),
                            "input_output": data.get("input_output", ""),
                            "attachments": attachments,
                        },
                    )
                    examples.append(example)

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {idx}: {e}")
                    continue
                except KeyError as e:
                    logger.error(f"Missing required field in line {idx}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} HLE examples")

        # Apply subset if specified at runtime (independent of any on-disk subset)
        if self.config.subset_num > 0 and self.config.subset_num < len(examples):
            examples = examples[: self.config.subset_num]
            logger.info(f"Using subset of {len(examples)} examples")

        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate HLE prediction using 'gen' mode (same as old)."""
        result = evaluate_hle(prediction, ground_truth)
        # Treat HLE's category as the "level" field for stratified metrics.
        result["level"] = metadata.get("level", metadata.get("category", "unknown"))
        return result

