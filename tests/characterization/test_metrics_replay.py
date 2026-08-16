"""B4: metrics aggregation over recorded runs must not change.

These are the numbers the thesis reports.  Unlike B1/B2 this gate is permanent:
adding a benchmark cannot change what an already-recorded run scores, so it
stays valid as the system grows.
"""

import json
from types import SimpleNamespace

import pytest

from agent_engine.runner.metrics import compute_metrics

from .conftest import assert_matches_fixture
from .replay_corpus import REPLAY_RUNS, dataset_name_for, load_rows, missing_runs


def test_metrics_replay_unchanged(update_fixtures):
    """The fixture predates the move out of ``scripts/run_experiment.py``.

    It was recorded against ``_compute_metrics`` loaded from the script by file
    path, and is deliberately NOT regenerated now that the function lives in
    ``agent_engine.runner.metrics``: matching the old bytes is the proof that
    relocating the code left the numbers alone.
    """
    missing = missing_runs()
    if missing:
        pytest.skip(f"replay corpus not present in this checkout: {missing}")

    payload = {}
    for run in REPLAY_RUNS:
        rows = load_rows(run)
        # compute_metrics reads example.metadata through its level key, so a
        # namespace with that one attribute is a faithful stand-in.
        examples = [SimpleNamespace(metadata=r.get("metadata") or {}) for r in rows]
        payload[run] = compute_metrics(rows, examples, dataset_name_for(run))

    assert_matches_fixture(
        "metrics_replay.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        update_fixtures,
    )
