"""B5: failure-mode classification over recorded runs must not change.

``classify_failure`` is frozen: the thesis's taxonomy counts come from it.
This fixture is what makes "frozen" enforceable rather than aspirational.

Like B4 this gate is permanent -- it replays over already-recorded runs, so
extending the system cannot invalidate it.
"""

import json
import sys
from collections import Counter

import pytest

from .conftest import assert_matches_fixture
from .replay_corpus import REPLAY_RUNS, REPO, load_rows, missing_runs

# Matches tests/unit/test_analyze_failure_modes.py.  Becomes a package import
# when the analysis code moves into src/, without regenerating the fixture.
sys.path.insert(0, str(REPO / "scripts"))


def test_failure_modes_replay_unchanged(update_fixtures):
    missing = missing_runs()
    if missing:
        pytest.skip(f"replay corpus not present in this checkout: {missing}")

    from failure_modes.analyze_failure_modes import classify_failure

    payload = {}
    for run in REPLAY_RUNS:
        per_question = {}
        for row in load_rows(run):
            evaluation = row.get("evaluation") or {}
            if evaluation.get("correct"):
                continue  # the classifier only runs on failures
            per_question[str(row["question_id"])] = classify_failure(row)

        assert per_question, f"no failures to classify in {run} -- corpus is uninformative"
        payload[run] = {
            "per_question": dict(sorted(per_question.items())),
            "counts": dict(sorted(Counter(per_question.values()).items())),
        }

    assert_matches_fixture(
        "failure_modes_replay.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        update_fixtures,
    )
