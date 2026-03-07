"""GPQA dataset loader.

GPQA (Graduate-level Physics Questions and Answers) benchmark for
evaluating reasoning on graduate-level science questions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry
from ..evaluators.metrics import evaluate_gpqa
from ...utils.logging import get_logger

logger = get_logger(__name__)


@DatasetRegistry.register("gpqa")
class GPQADataset(BaseDataset):
    """GPQA dataset loader.

    GPQA consists of multiple-choice questions in physics, chemistry,
    and biology at the graduate level.
    """

    def load(self) -> List[DatasetExample]:
        """Load GPQA dataset from JSONL file.

        Expected format:
        {
            "Question": "question text",
            "Choices": ["option A text", "option B text", "option C text", "option D text"],
            "Answer": "B",              # correct option letter
            "AnswerText": "option B text"  # (optional) sanity-check copy of the correct option text
        }

        Returns:
            List of DatasetExample objects
        """
        # Construct path
        data_path = self.config.data_dir / "GPQA" / f"{self.config.split}.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(
                f"GPQA dataset not found at: {data_path}\n"
                f"Please download GPQA dataset and place it in {self.config.data_dir}/GPQA/"
            )

        logger.info(f"Loading GPQA from: {data_path}")

        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)

                    # Extract choices
                    choices = data.get("Choices", []) or []

                    # Ground-truth letter and answer text from JSONL.
                    raw_answer_letter = str(data.get("Answer", "") or "").strip().upper()
                    answer_letter = raw_answer_letter or None
                    raw_answer_text = data.get("AnswerText")
                    answer_text = str(raw_answer_text).strip() if isinstance(raw_answer_text, str) and raw_answer_text.strip() else None


                    # Format question with choices
                    question_text = data["Question"]
                    if choices:
                        question_text += "\n\nChoices:"
                        for i, choice in enumerate(choices):
                            letter = chr(ord("A") + i)
                            question_text += f"\n{letter}. {choice}"

                    example = DatasetExample(
                        question_id=idx,
                        question=question_text,
                        answer=answer_letter or "",
                        metadata={
                            "choices": choices,
                            "answer_letter": answer_letter,
                            "answer_text": answer_text,
                        }
                    )
                    examples.append(example)

                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {idx}: {e}")
                    continue
                except KeyError as e:
                    logger.error(f"Missing required field in line {idx}: {e}")
                    continue

        logger.info(f"Loaded {len(examples)} GPQA examples")

        return examples

    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate GPQA prediction.

        Args:
            prediction: Model's predicted answer
            ground_truth: Ground truth answer (letter)
            metadata: Example metadata with 'choices'

        Returns:
            Evaluation results with 'correct' and 'score' keys
        """
        choices = metadata.get('choices', [])
        return evaluate_gpqa(prediction, ground_truth, choices)
