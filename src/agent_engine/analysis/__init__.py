"""Failure-mode analysis over recorded experiment runs.

Moved here from ``scripts/failure_modes/`` so it can be imported as a package
rather than through ``sys.path`` manipulation.  The old script paths remain as
thin CLI shims.

``classify_failure`` in :mod:`.failure_modes` is **frozen**: the thesis's
taxonomy counts come from it, and a characterization fixture replays it over
recorded runs.  Do not change its body.
"""

from pathlib import Path

# Repo root, used for the default ``--root``/``ROOT`` in the analysis modules.
#
# Defined once, here, on purpose.  Every module below derived this from its own
# ``__file__`` with a hardcoded number of ``parents[]`` levels, so moving the
# package one directory deeper silently repointed all of them at ``src/``.
# One definition means one place to be wrong.
REPO_ROOT = Path(__file__).resolve().parents[3]

__all__ = ["REPO_ROOT"]
