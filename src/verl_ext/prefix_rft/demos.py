"""Runtime access to the Prefix-RFT demonstration store.

Lookup is by question text, hashed the same way ``scripts/build_prefix_demos.py``
hashes it. Not by ``extra_info.idx``: prepare.py assigns idx per data source, so
it collides across them and would attach a maths demonstration to a search
question.

Coverage is partial by design: 1358 of the 1800 RL training questions carry a
teacher demonstration, and 273 of those have a single decision and so can never
carry a prefix. A miss is not an error, it means that question trains as ordinary
GRPO, which is what the reference implementation's ``demo_ratio`` mechanism does
(rl_dataset.py:194-203) and what the paper's Table 2 validates down to 1%
coverage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def question_key(question: str) -> str:
    """Must stay identical to ``scripts/build_prefix_demos.question_key``."""
    return hashlib.sha1(str(question).strip().encode("utf-8")).hexdigest()


class DemoStore:
    """Maps a training question to the teacher's ordered decisions."""

    def __init__(self, by_key: dict[str, list[dict]]):
        self._by_key = by_key

    @classmethod
    def from_parquet(cls, path) -> "DemoStore":
        frame = pd.read_parquet(Path(path))
        by_key: dict[str, list[dict]] = {}
        for _, row in frame.iterrows():
            by_key[str(row["question_key"])] = [dict(step) for step in row["steps"]]
        return cls(by_key)

    def n_steps(self, question: str) -> int:
        return len(self._by_key.get(question_key(question), ()))

    def steps(self, question: str) -> list[dict]:
        return list(self._by_key.get(question_key(question), ()))

    def coverage(self) -> tuple[int, int]:
        """Return (questions with a demonstration, total decisions)."""
        return len(self._by_key), sum(len(v) for v in self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)
