"""Unit tests for canonical run-directory resolution.

This rule decides *which* results every fine-tuning analysis reads, so the two
behaviours that previously differed between the duplicated copies -- relative
path handling and missing-folder handling -- are pinned here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts/failure_modes/fine_tuning"))

from runs import latest_run


def _run(parent, name, with_results=True):
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    if with_results:
        (d / "raw_results.json").write_text("[]")
    return d


def test_picks_lexicographically_last_run(tmp_path):
    _run(tmp_path, "2026-01-01_00-00-00_1")
    newest = _run(tmp_path, "2026-05-09_11-22-33_2")
    assert latest_run(tmp_path) == newest


def test_skips_runs_without_raw_results(tmp_path):
    good = _run(tmp_path, "2026-01-01_00-00-00_1")
    _run(tmp_path, "2026-09-30_99-99-99_crashed", with_results=False)
    assert latest_run(tmp_path) == good


def test_returns_none_when_no_run_has_results(tmp_path):
    _run(tmp_path, "2026-01-01_a", with_results=False)
    assert latest_run(tmp_path) is None


def test_returns_none_for_missing_folder(tmp_path):
    """Previously one copy raised IndexError here and the other returned None."""
    assert latest_run(tmp_path / "does_not_exist") is None


def test_returns_none_for_empty_folder(tmp_path):
    assert latest_run(tmp_path) is None


def test_relative_path_resolved_against_root(tmp_path):
    nested = tmp_path / "experiments" / "results" / "aime"
    expected = _run(nested, "2026-05-09_x")
    assert latest_run("experiments/results/aime", root=tmp_path) == expected


def test_absolute_path_used_as_is(tmp_path):
    expected = _run(tmp_path, "2026-05-09_x")
    assert latest_run(tmp_path, root=Path("/nonexistent")) == expected


def test_files_are_not_mistaken_for_runs(tmp_path):
    expected = _run(tmp_path, "2026-01-01_a")
    (tmp_path / "raw_results.json").write_text("[]")  # stray file at the top level
    assert latest_run(tmp_path) == expected
