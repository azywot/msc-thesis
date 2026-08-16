"""Characterization-fixture harness.

These tests lock *current* behaviour.  They exist to prove that a refactor
changed nothing: each one recomputes some observable output and compares it
against a committed baseline.

A fixture is regenerated only under an explicit ``--update-fixtures`` run, so
refreshing a baseline is always a deliberate, reviewable act and never a silent
side effect of running the suite.

    # record or refresh every baseline
    pytest tests/characterization -q --update-fixtures
"""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--update-fixtures",
        action="store_true",
        default=False,
        help="Rewrite characterization fixtures from current behaviour.",
    )


@pytest.fixture
def update_fixtures(request):
    return request.config.getoption("--update-fixtures")


def assert_matches_fixture(name: str, actual: str, update: bool) -> None:
    """Compare ``actual`` against the recorded fixture ``name``.

    With ``update=True`` the fixture is (re)written and the check passes.
    """
    path = FIXTURE_DIR / name
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return

    assert path.exists(), (
        f"Missing fixture {path}. Record it with:\n"
        f"    pytest tests/characterization -q --update-fixtures"
    )
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, f"Behaviour changed against fixture {name}"
