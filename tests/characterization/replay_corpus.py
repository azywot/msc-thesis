"""The recorded runs that B4 and B5 replay over.

``experiments/results/`` is gitignored and multi-gigabyte, so these fixtures
record only *derived* output -- aggregated metrics and failure-mode labels --
never the run data itself.  Committing the rows is not an option either: they
carry ground-truth answers for gated datasets (GAIA, GPQA).

The consequence is that both tests **skip** on a checkout without the results
tree.  That is deliberate, but it means a fresh clone gets no protection from
them: they guard the thesis numbers on the machine where those numbers were
produced.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Chosen to span the three shapes `_compute_metrics` distinguishes:
#   gaia -> stratified, per_level keyed by "level"
#   aime -> stratified, per_level keyed by "year"
#   gpqa -> not stratified, no per_level block at all
REPLAY_RUNS = [
    "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gaia/"
    "qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json",
    "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/aime/"
    "qwen8B_subagent_tools_orchestrator/train_2026-03-15-21-30-27_20752258/raw_results.json",
    "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gpqa/"
    "qwen8B_subagent_tools_orchestrator/diamond_2026-03-15-21-19-20_20752198/raw_results.json",
]

_KNOWN_DATASETS = ("gaia", "aime", "gpqa", "hle", "musique", "math500", "amc", "bigcodebench")


def missing_runs():
    """Repo-relative paths from REPLAY_RUNS that are absent in this checkout."""
    return [r for r in REPLAY_RUNS if not (REPO / r).exists()]


def load_rows(run: str):
    """Load one run's rows.  ``raw_results.json`` is a bare JSON list."""
    rows = json.loads((REPO / run).read_text(encoding="utf-8"))
    assert isinstance(rows, list), f"expected a list of rows in {run}"
    return rows


def dataset_name_for(run: str) -> str:
    """Dataset name is a path component of every results run directory."""
    parts = Path(run).parts
    for known in _KNOWN_DATASETS:
        if known in parts:
            return known
    raise AssertionError(f"cannot infer dataset from run path: {run}")
