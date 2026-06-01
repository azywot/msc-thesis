"""GEPA integration: seed candidate construction and data split generation.

build_seed_candidate() renders the initial two-component candidate from
the PromptBuilder templates.

build_splits() partitions an existing raw_results.json into failure-stratified
train / random val / random test splits, saving them to a JSON file.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Ensure scripts/ is on path for classify_failure import
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
from agent_engine.models.base import ToolCallFormat
from agent_engine.prompts import PromptBuilder


def build_seed_candidate(
    benchmark: str,
    tool_schemas: list[dict],
    direct_tool_call: bool = True,
    tool_call_format: ToolCallFormat = ToolCallFormat.JSON,
    max_search_limit: int = 10,
) -> dict[str, str]:
    """Render the seed GEPA candidate from YAML templates.

    Args:
        benchmark: Dataset name passed to PromptBuilder (e.g. "gaia", "gpqa").
        tool_schemas: List of tool schema dicts (from ToolRegistry.get_all_schemas()).
        direct_tool_call: Whether tools are called directly (vs sub-agent mode).
        tool_call_format: Tool call syntax (JSON for Qwen3, PYTHONIC for OLMo).
        max_search_limit: Passed to PromptBuilder for the search-limit reminder.

    Returns:
        {"system_prompt": str, "planning_suffix": str}
    """
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt(
        dataset_name=benchmark,
        tool_schemas=tool_schemas,
        max_search_limit=max_search_limit,
        direct_tool_call=direct_tool_call,
        baseline=False,
        tool_call_format=tool_call_format,
    )
    return {
        "system_prompt": system_prompt,
        "planning_suffix": _DEFAULT_PLANNING_SUFFIX_TOOLS,
    }


def build_splits(
    raw_results_path: Path,
    train_n: int,
    val_n: int,
    seed: int = 1,
    output_path: Optional[Path] = None,
    test_n: Optional[int] = None,
) -> dict[str, list[int]]:
    """Build stratified train/val/test splits with the same class
    distribution across all three splits.

    Each record is labelled by class — ``"CORRECT"`` for solved questions,
    or one of the six failure modes from
    ``scripts/failure_modes/analyze_failure_modes.py`` for failed ones.
    Within each class, records are allocated to train/val/test in
    proportion to the requested split sizes (relative to the full dataset),
    so each split preserves the natural class distribution. This keeps the
    GEPA-vs-seed comparison apples-to-apples regardless of which split it
    is reported on.

    Args:
        raw_results_path: Path to an existing raw_results.json.
        train_n: Target number of examples in the train set.
        val_n:   Target number of examples in the val (D_pareto) set.
        seed:    Random seed for reproducibility (default 1, distinct from
                 the existing experiment seed 0).
        output_path: If given, write splits dict as JSON to this path.
        test_n:  Optional cap on the test set size. If None, test gets all
                 remaining examples (total - train_n - val_n). If set, test
                 is a stratified sample of this size and the surplus is
                 discarded.

    Returns:
        {"train": [qid, ...], "val": [qid, ...], "test": [qid, ...]}
    """
    from failure_modes.analyze_failure_modes import classify_failure

    rng = random.Random(seed)

    with open(raw_results_path, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    test_remainder = total - train_n - val_n
    if test_remainder < 0:
        raise ValueError(
            f"train_n + val_n ({train_n + val_n}) exceeds dataset size ({total})"
        )
    if test_n is not None and test_n > test_remainder:
        raise ValueError(
            f"test_n ({test_n}) exceeds remaining examples after train+val ({test_remainder})"
        )

    by_class: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        qid = rec["question_id"]
        if rec.get("correct"):
            by_class["CORRECT"].append(qid)
        else:
            by_class[classify_failure(rec)].append(qid)

    test_target = test_n if test_n is not None else test_remainder
    denom = train_n + val_n + test_target

    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    surplus: list[int] = []

    # Per-class proportional allocation with floor; residuals land in the
    # surplus pool below to be redistributed to hit exact target sizes.
    for cls in sorted(by_class):
        shuffled = list(by_class[cls])
        rng.shuffle(shuffled)
        n_class = len(shuffled)
        # Effective per-class allocation honours what fraction of the dataset
        # we are actually keeping: (train_n + val_n + test_target) / total.
        scaled = n_class * denom / total
        n_train = int(scaled * (train_n / denom))
        n_val = int(scaled * (val_n / denom))
        n_test = int(scaled * (test_target / denom))
        train_ids.extend(shuffled[:n_train])
        val_ids.extend(shuffled[n_train:n_train + n_val])
        test_ids.extend(shuffled[n_train + n_val:n_train + n_val + n_test])
        surplus.extend(shuffled[n_train + n_val + n_test:])

    # Top up each split to its exact target by drawing from the surplus.
    # Shuffle first so donations are not biased toward any single class.
    rng.shuffle(surplus)
    while len(train_ids) < train_n and surplus:
        train_ids.append(surplus.pop())
    while len(val_ids) < val_n and surplus:
        val_ids.append(surplus.pop())
    while len(test_ids) < test_target and surplus:
        test_ids.append(surplus.pop())

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

    return splits
