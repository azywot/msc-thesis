"""Shared run-directory resolution for the fine-tuning analyses.

Both analyses in this package have to answer the same question -- given a results
folder holding several timestamped runs, which one is *the* run? Two independent
copies of that rule would silently disagree about which results get analysed, so
it lives here once.

Rule: the lexicographically last sub-directory containing ``raw_results.json``.
Run directory names are timestamped, so lexicographic order is recency order, and
requiring ``raw_results.json`` skips crashed runs that never wrote results.
"""

from pathlib import Path

from .. import REPO_ROOT

ROOT = REPO_ROOT


def latest_run(folder, root=ROOT):
    """Newest run directory under ``folder``, or ``None`` if there is none.

    ``folder`` may be absolute or relative to the repo root.
    """
    folder = Path(folder)
    if not folder.is_absolute():
        folder = root / folder
    if not folder.is_dir():
        return None
    candidates = [d for d in folder.iterdir() if d.is_dir() and (d / "raw_results.json").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda d: d.name)[-1]
