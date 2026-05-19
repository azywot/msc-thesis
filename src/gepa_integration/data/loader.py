"""Load GEPA data JSON files as DatasetExample objects."""
from __future__ import annotations

import json
from pathlib import Path

from agent_engine.datasets.base import DatasetExample


def load_gepa_examples(data_file: Path, question_ids: list[int]) -> list[DatasetExample]:
    """Load examples from a GEPA data JSON file, filtered by question_id.

    Args:
        data_file: Path to ``all_examples.json`` written by ``prepare.py``.
        question_ids: IDs to load. Unknown IDs are silently ignored.

    Returns:
        List of ``DatasetExample`` objects in the order they appear in the file.
    """
    if not data_file.exists():
        raise FileNotFoundError(f"GEPA data file not found: {data_file}")

    with open(data_file, encoding="utf-8") as f:
        all_examples = json.load(f)

    id_set = set(question_ids)
    result = []
    for raw in all_examples:
        if raw["question_id"] not in id_set:
            continue
        result.append(
            DatasetExample(
                question_id=raw["question_id"],
                question=raw["question"],
                answer=raw["answer"],
                metadata={
                    "answer_aliases": raw.get("answer_aliases", []),
                    "data_source": raw["data_source"],
                },
            )
        )
    return result
