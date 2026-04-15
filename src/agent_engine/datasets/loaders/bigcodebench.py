"""BigCodeBench dataset loader.

BigCodeBench (bigcode/bigcodebench) is a code-generation benchmark where the
model must implement a function described in natural language.  Correctness is
determined by running the official test harness, not by string matching.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry
from ..evaluators.bigcodebench_scorer import evaluate_bigcodebench
from ...utils.logging import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register("bigcodebench")
class BigCodeBenchDataset(BaseDataset):
    """Loader for the BigCodeBench instruct split.

    Expected JSONL format (one object per line):
        task_id, instruct_prompt, code_prompt, test, entry_point, libs
    """

    def load(self) -> List[DatasetExample]:
        """Load BigCodeBench from a local JSONL file.

        File location: ``<data_dir>/BigCodeBench/<split>.jsonl``
        """
        data_path = self.config.data_dir / "BigCodeBench" / f"{self.config.split}.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"BigCodeBench dataset not found at: {data_path}\n"
                f"Run: python scripts/download_datasets.py --dataset bigcodebench\n"
                f"Or place the JSONL file at: {data_path}"
            )

        logger.info("Loading BigCodeBench from: %s", data_path)

        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    example = DatasetExample(
                        question_id=idx,
                        question=data["instruct_prompt"],
                        answer="",  # correctness is determined by test execution
                        metadata={
                            "task_id": data.get("task_id", f"bigcodebench/{idx}"),
                            "code_prompt": data.get("code_prompt", ""),
                            "test": data.get("test", ""),
                            "entry_point": data.get("entry_point", ""),
                            "libs": data.get("libs", []),
                        },
                    )
                    examples.append(example)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error("Error parsing line %d: %s", idx, e)
                    continue

        logger.info("Loaded %d BigCodeBench examples", len(examples))
        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the test harness and return pass/fail.

        Args:
            prediction: Model's generated code (may contain markdown fences).
            ground_truth: Unused (empty string for BigCodeBench).
            metadata: Must contain ``task_id``, ``code_prompt``, ``test``, ``entry_point``.

        Returns:
            Dict with ``correct``, ``score``, ``task_id``, ``error``.
        """
        result = evaluate_bigcodebench(prediction, metadata)
        # Mirror the ``accuracy`` key used by _compute_metrics in run_experiment.py
        result["accuracy"] = result["score"]
        return result
