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
) -> dict[str, list[int]]:
    """Build failure-stratified train / random val / random test splits.

    Reads an existing raw_results.json, classifies each failed record using
    the six-mode taxonomy from scripts/failure_modes/analyze_failure_modes.py,
    then samples:
      - train: ~65% failures (proportional across modes) + ~35% successes
      - val:   random sample from remaining questions
      - test:  all remaining questions after train + val

    Args:
        raw_results_path: Path to an existing raw_results.json.
        train_n: Number of examples in the train set.
        val_n:   Number of examples in the val (D_pareto) set.
        seed:    Random seed for reproducibility (default 1, distinct from
                 the existing experiment seed 0).
        output_path: If given, write splits dict as JSON to this path.

    Returns:
        {"train": [qid, ...], "val": [qid, ...], "test": [qid, ...]}
    """
    from failure_modes.analyze_failure_modes import FAILURE_MODES, classify_failure

    rng = random.Random(seed)

    with open(raw_results_path, encoding="utf-8") as f:
        records = json.load(f)

    correct_ids: list[int] = []
    failed_by_mode: dict[str, list[int]] = defaultdict(list)

    for rec in records:
        qid = rec["question_id"]
        if rec.get("correct"):
            correct_ids.append(qid)
        else:
            mode = classify_failure(rec)
            failed_by_mode[mode].append(qid)

    all_failed_ids = [qid for ids in failed_by_mode.values() for qid in ids]
    total_failed = len(all_failed_ids)

    # --- sample failures proportionally across modes ---
    n_failures_train = round(train_n * 0.65)
    train_failed: list[int] = []

    if total_failed > 0:
        for mode in FAILURE_MODES:
            mode_ids = list(failed_by_mode[mode])
            rng.shuffle(mode_ids)
            n_from_mode = round(n_failures_train * len(mode_ids) / total_failed)
            train_failed.extend(mode_ids[:n_from_mode])

        # Fix rounding: top up or trim to exactly n_failures_train
        used_failed = set(train_failed)
        remaining_failed = [qid for qid in all_failed_ids if qid not in used_failed]
        rng.shuffle(remaining_failed)
        if len(train_failed) < n_failures_train:
            train_failed.extend(remaining_failed[: n_failures_train - len(train_failed)])
        train_failed = train_failed[:n_failures_train]

    # --- sample successes ---
    n_success_train = train_n - len(train_failed)
    shuffled_correct = list(correct_ids)
    rng.shuffle(shuffled_correct)
    train_success = shuffled_correct[:n_success_train]

    train_ids = train_failed + train_success
    rng.shuffle(train_ids)

    # --- val + test from the remainder ---
    used = set(train_ids)
    remaining = [rec["question_id"] for rec in records if rec["question_id"] not in used]
    rng.shuffle(remaining)

    val_ids = remaining[:val_n]
    test_ids = remaining[val_n:]

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
