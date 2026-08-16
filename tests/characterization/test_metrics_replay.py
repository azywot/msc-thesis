"""B4: metrics aggregation over recorded runs must not change.

These are the numbers the thesis reports.  Unlike B1/B2 this gate is permanent:
adding a benchmark cannot change what an already-recorded run scores, so it
stays valid as the system grows.
"""

import importlib.util
import json
from types import SimpleNamespace

import pytest

from .conftest import assert_matches_fixture
from .replay_corpus import REPLAY_RUNS, REPO, dataset_name_for, load_rows, missing_runs


def _load_compute_metrics():
    """Import ``_compute_metrics`` from the runner script by path.

    The metrics code moves into ``agent_engine.runner.metrics`` later in the
    refactor.  When it does, this helper switches to a package import -- and
    the fixture must NOT be regenerated at that switch.  A relocation that
    changes the numbers is exactly what this gate exists to catch.
    """
    spec = importlib.util.spec_from_file_location(
        "_runexp_for_replay", REPO / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_metrics


def test_metrics_replay_unchanged(update_fixtures):
    missing = missing_runs()
    if missing:
        pytest.skip(f"replay corpus not present in this checkout: {missing}")

    compute_metrics = _load_compute_metrics()

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
