"""GEPA optimization data preparation.

Downloads Search-R1 (HotpotQA + NQ) and DeepMath-103K from HuggingFace,
splits into D_feedback / D_pareto / Test, and saves to disk.

Two presets are supported — configured via the ``--preset`` CLI flag:

  gaia   — 75 % Search-R1 (85 % HotpotQA / 15 % NQ) + 25 % DeepMath (no difficulty filter)
           Targets GAIA / HLE / MuSiQue failure modes: retrieval chaining, single-shot
           tool trust, evidence failure.

  math   — 75 % DeepMath (difficulty >= 5) + 25 % Search-R1
           Targets AIME failure mode: direct reasoning without Coder delegation.

Split sizes (both presets):
  D_feedback (train): 150   D_pareto (val): 50   Test: 100   Total: 300

Outputs (relative to --output-dir):
  all_examples.json     — all 300 examples with sequential question_id 0..299
  (splits written to the path given by --splits-out)

Usage:
  python src/gepa_integration/data/prepare.py \\
      --preset gaia \\
      --output-dir data/gepa/gaia \\
      --splits-out experiments/configs/gepa/splits/gaia_gepa_splits.json \\
      --seed 1

  python src/gepa_integration/data/prepare.py \\
      --preset math \\
      --output-dir data/gepa/math \\
      --splits-out experiments/configs/gepa/splits/math_gepa_splits.json \\
      --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Make fine_tuning importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fine_tuning.data.prepare import (
    _download_deepmath,
    _download_search_r1,
)

# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict] = {
    "gaia": {
        "search": {"feedback": 112, "pareto": 37, "test": 75},
        "math":   {"feedback": 38,  "pareto": 13, "test": 25},
        "deepmath_min_difficulty": 1,   # no meaningful filter (DeepMath min is 1)
        "hotpot_ratio": 0.85,
        "description": "75% Search-R1 (85/15 HotpotQA/NQ) + 25% DeepMath (no difficulty filter)",
    },
    "math": {
        "search": {"feedback": 38,  "pareto": 13, "test": 25},
        "math":   {"feedback": 112, "pareto": 37, "test": 75},
        "deepmath_min_difficulty": 5,
        "hotpot_ratio": 0.85,
        "description": "75% DeepMath (difficulty >= 5) + 25% Search-R1 (85/15 HotpotQA/NQ)",
    },
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def make_gepa_splits(
    all_ids: list[int],
    n_feedback: int,
    n_pareto: int,
    n_test: int,
    seed: int = 1,
) -> dict[str, list[int]]:
    """Randomly partition ``all_ids`` into D_feedback / D_pareto / Test splits.

    Args:
        all_ids:    All example IDs (must have length >= n_feedback + n_pareto + n_test).
        n_feedback: D_feedback size (GEPA ``train`` split).
        n_pareto:   D_pareto size (GEPA ``val`` split).
        n_test:     Held-out test size.
        seed:       Random seed for reproducibility.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` — uses the same key
        names as the thesis-benchmark splits so ``run_gepa.py`` reads them unchanged.

    Raises:
        ValueError: If available IDs are fewer than n_feedback + n_pareto + n_test.
    """
    total = n_feedback + n_pareto + n_test
    if len(all_ids) < total:
        raise ValueError(
            f"n_feedback ({n_feedback}) + n_pareto ({n_pareto}) + n_test ({n_test}) "
            f"= {total} exceeds available IDs ({len(all_ids)})"
        )
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    return {
        "train": sorted(ids[:n_feedback]),
        "val":   sorted(ids[n_feedback : n_feedback + n_pareto]),
        "test":  sorted(ids[n_feedback + n_pareto : total]),
    }


def _norm_to_example(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a fine_tuning normalised row to GEPA example schema (without question_id)."""
    return {
        "question": row["question"],
        "answer": row["result"],
        "answer_aliases": row["extra_info"].get("golden_answers", [row["result"]]),
        "data_source": row["data_source"],
    }


def build_gepa_examples(
    feedback_search: list[dict],
    feedback_math: list[dict],
    pareto_search: list[dict],
    pareto_math: list[dict],
    test_search: list[dict],
    test_math: list[dict],
    seed: int = 1,
) -> list[dict]:
    """Combine and shuffle split buckets, then assign sequential ``question_id``.

    Within each split, search and math rows are shuffled together so the GEPA
    minibatch sampler sees a mixed distribution. Splits are concatenated
    feedback → pareto → test, and global IDs are assigned in that order.

    Args:
        feedback_search / feedback_math: fine_tuning normalised rows for D_feedback
        pareto_search   / pareto_math:   fine_tuning normalised rows for D_pareto
        test_search     / test_math:     fine_tuning normalised rows for Test
        seed: random seed for within-split shuffling

    Returns:
        List of example dicts with ``question_id`` 0..N-1 and fields
        ``question``, ``answer``, ``answer_aliases``, ``data_source``.
    """
    rng = random.Random(seed)

    def _mix(search_rows: list[dict], math_rows: list[dict]) -> list[dict]:
        combined = [_norm_to_example(r) for r in search_rows + math_rows]
        rng.shuffle(combined)
        return combined

    ordered: list[dict] = (
        _mix(feedback_search, feedback_math)
        + _mix(pareto_search, pareto_math)
        + _mix(test_search, test_math)
    )
    for i, ex in enumerate(ordered):
        ex["question_id"] = i
    return ordered


def save_gepa_data(
    examples: list[dict],
    splits: dict[str, list[int]],
    data_file: Path,
    splits_file: Path,
) -> None:
    """Write examples and splits to disk.

    Args:
        examples:    List of example dicts (with ``question_id``).
        splits:      ``{"train": [...], "val": [...], "test": [...]}``
        data_file:   Output path for ``all_examples.json``.
        splits_file: Output path for the splits JSON.
    """
    data_file = Path(data_file)
    splits_file = Path(splits_file)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    splits_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------

def prepare(
    preset_name: str,
    output_dir: Path,
    splits_out: Path,
    seed: int = 1,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Download, split, and save GEPA data for the given preset.

    Args:
        preset_name: ``"gaia"`` or ``"math"``.
        output_dir:  Directory for ``all_examples.json``.
        splits_out:  Path for the splits JSON (``{"train":..., "val":..., "test":...}``).
        seed:        Random seed (default 1 — matches GEPA config default).

    Returns:
        ``(examples, splits)`` tuple for inspection / testing.
    """
    if preset_name not in _PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Choose from: {list(_PRESETS)}")

    p = _PRESETS[preset_name]
    search_cfg = p["search"]
    math_cfg = p["math"]

    print(f"Preparing GEPA data for preset '{preset_name}': {p['description']}")

    print(
        f"  Downloading Search-R1 "
        f"({search_cfg['feedback']} feedback + {search_cfg['pareto']} pareto "
        f"+ {search_cfg['test']} test, hotpot_ratio={p['hotpot_ratio']})..."
    )
    search_feedback, search_pareto, search_test = _download_search_r1(
        n_train=search_cfg["feedback"],
        n_val=search_cfg["pareto"],
        n_test=search_cfg["test"],
        seed=seed,
        search_source="both",
        hotpot_ratio=p["hotpot_ratio"],
    )

    print(
        f"  Downloading DeepMath-103K "
        f"({math_cfg['feedback']} feedback + {math_cfg['pareto']} pareto "
        f"+ {math_cfg['test']} test, "
        f"min_difficulty={p['deepmath_min_difficulty']})..."
    )
    math_feedback, math_pareto, math_test = _download_deepmath(
        n_train=math_cfg["feedback"],
        n_val=math_cfg["pareto"],
        n_test=math_cfg["test"],
        seed=seed,
        min_difficulty=p["deepmath_min_difficulty"],
    )

    examples = build_gepa_examples(
        feedback_search=search_feedback, feedback_math=math_feedback,
        pareto_search=search_pareto,   pareto_math=math_pareto,
        test_search=search_test,       test_math=math_test,
        seed=seed,
    )

    n_feedback = search_cfg["feedback"] + math_cfg["feedback"]
    n_pareto   = search_cfg["pareto"]   + math_cfg["pareto"]
    n_test     = search_cfg["test"]     + math_cfg["test"]

    splits = make_gepa_splits(
        all_ids=[ex["question_id"] for ex in examples],
        n_feedback=n_feedback,
        n_pareto=n_pareto,
        n_test=n_test,
        seed=seed,
    )

    data_file = Path(output_dir) / "all_examples.json"
    save_gepa_data(examples, splits, data_file, Path(splits_out))

    print(f"  Wrote {len(examples)} examples to {data_file}")
    print(f"  D_feedback (train): {len(splits['train'])}")
    print(f"  D_pareto   (val):   {len(splits['val'])}")
    print(f"  Test:               {len(splits['test'])}")
    print(f"  Splits saved to:    {splits_out}")

    return examples, splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and split GEPA optimization data (Search-R1 + DeepMath)."
    )
    parser.add_argument(
        "--preset", choices=list(_PRESETS), required=True,
        help="Data preset: 'gaia' (75%% Search-R1) or 'math' (75%% DeepMath diff>=5)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for all_examples.json",
    )
    parser.add_argument(
        "--splits-out", type=Path, required=True,
        help="Output path for splits JSON (train/val/test question IDs)",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    prepare(
        preset_name=args.preset,
        output_dir=args.output_dir,
        splits_out=args.splits_out,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
