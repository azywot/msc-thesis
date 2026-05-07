"""Data preparation: download Search-R1 + DeepMath and write VERL parquet files.

Usage:
    python src/fine_tuning/data/prepare.py \\
        --n-search 10000 --n-math 10000 \\
        --output-dir data/training --seed 42

Validation sets:
    Both domains get a held-out val split carved out BEFORE the training
    subsample, guaranteeing no overlap with combined_train.parquet.

    Files written under <output-dir>/val/:
        val_search.parquet      — held-out Search-R1 (NQ + HotpotQA)
        val_deepmath.parquet    — held-out DeepMath
        val_combined.parquet    — both merged and shuffled (used by VERL)

    VERL's data.val_files points at val_combined.parquet so the reported
    val/reward_mean covers both domains.  Per-domain breakdown is available
    offline from val_search.parquet / val_deepmath.parquet and from the
    rollout JSON files saved during training (each record includes data_source).

    AIME is an evaluation benchmark and must not be used here —
    see docs/failure_modes_fine_tuning_alignment.md §6.3.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Row normalisers
# ---------------------------------------------------------------------------

def normalise_search_r1_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a Search-R1 row to VERL schema.

    Search-R1 (PeterJinGo/SearchR1-nq_hotpotqa_train) columns:
        question, answer (or answers list), dataset (one of "nq", "hotpotqa")
    """
    answer = raw.get("answer") or raw.get("answers") or ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    answer = str(answer)

    return {
        "data_source": str(raw.get("dataset", "nq")).lower(),
        "question": str(raw.get("question", "")),
        "result": answer,
        "extra_info": {"idx": idx, "groundtruth": answer},
    }


def normalise_deepmath_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a DeepMath-103K row to VERL schema.

    DeepMath (zwhe99/DeepMath-103K) columns: problem, answer, source (optional)
    """
    answer = str(raw.get("answer", ""))
    return {
        "data_source": "deepmath",
        "question": str(raw.get("problem", "")),
        "result": answer,
        "extra_info": {"idx": idx, "groundtruth": answer},
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"data_source", "question", "result", "extra_info"}


def validate_parquet_schema(df) -> None:
    """Raise ValueError if df is missing required VERL columns."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_search_r1(
    n_train: int, n_val: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Download Search-R1 and split into train and val.

    Val rows are taken first (indices 0..n_val-1 after shuffling) so they
    are guaranteed to be excluded from the training set regardless of n_train.
    """
    from datasets import load_dataset  # lazy import — not needed for unit tests
    ds = load_dataset("PeterJinGo/SearchR1-nq_hotpotqa_train", split="train")
    ds = ds.shuffle(seed=seed)

    total_needed = n_val + n_train
    if total_needed < len(ds):
        ds = ds.select(range(total_needed))

    val_rows = [
        normalise_search_r1_row(dict(row), idx=i)
        for i, row in enumerate(ds.select(range(n_val)))
    ]
    train_rows = [
        normalise_search_r1_row(dict(row), idx=i)
        for i, row in enumerate(ds.select(range(n_val, n_val + n_train)))
    ]
    return train_rows, val_rows


def _download_deepmath(
    n_train: int, n_val: int, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Download DeepMath-103K and split into train and val.

    Val rows are taken first (indices 0..n_val-1 after shuffling) so they
    are guaranteed to be excluded from the training set regardless of n_train.
    """
    from datasets import load_dataset
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=seed)

    total_needed = n_val + n_train
    if total_needed < len(ds):
        ds = ds.select(range(total_needed))

    val_rows = [
        normalise_deepmath_row(dict(row), idx=i)
        for i, row in enumerate(ds.select(range(n_val)))
    ]
    train_rows = [
        normalise_deepmath_row(dict(row), idx=i)
        for i, row in enumerate(ds.select(range(n_val, n_val + n_train)))
    ]
    return train_rows, val_rows


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_combined_train(
    search_train_rows: List[Dict[str, Any]],
    math_train_rows: List[Dict[str, Any]],
    output_dir: Path,
    seed: int,
) -> Path:
    """Mix search and math training rows and write combined_train.parquet."""
    import pandas as pd

    combined = search_train_rows + math_train_rows
    rng = random.Random(seed)
    rng.shuffle(combined)

    df = pd.DataFrame(combined)
    validate_parquet_schema(df)

    out_path = output_dir / "train" / "combined_train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def _write_val(rows: List[Dict[str, Any]], path: Path) -> Path:
    import pandas as pd
    df = pd.DataFrame(rows)
    validate_parquet_schema(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    sources = df["data_source"].unique().tolist()
    print(f"Wrote {len(df)} rows to {path}  (sources: {sources})")
    return path


def build_val_files(
    search_val_rows: List[Dict[str, Any]],
    deepmath_val_rows: List[Dict[str, Any]],
    output_dir: Path,
    seed: int,
) -> Dict[str, Path]:
    """Write three val parquet files under <output_dir>/val/.

    Returns a dict with keys "search", "deepmath", "combined".

    val_search.parquet    — Search-R1 held-out (NQ + HotpotQA)
    val_deepmath.parquet  — DeepMath held-out
    val_combined.parquet  — both merged and shuffled; used by VERL's data.val_files
                            so that val/reward_mean covers both domains.
                            Per-domain breakdown available offline from the
                            separate files and from rollout JSONs (data_source field).
    """
    val_dir = output_dir / "val"

    search_path = _write_val(search_val_rows, val_dir / "val_search.parquet")
    deepmath_path = _write_val(deepmath_val_rows, val_dir / "val_deepmath.parquet")

    # Combined: re-index so idx is unique across both sets
    combined = []
    for i, row in enumerate(search_val_rows + deepmath_val_rows):
        row = dict(row)
        row["extra_info"] = {**row["extra_info"], "idx": i}
        combined.append(row)
    rng = random.Random(seed)
    rng.shuffle(combined)
    combined_path = _write_val(combined, val_dir / "val_combined.parquet")

    return {"search": search_path, "deepmath": deepmath_path, "combined": combined_path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare training data for orchestrator fine-tuning.")
    parser.add_argument("--n-search", type=int, default=10_000,
                        help="Number of Search-R1 training rows (default: 10000)")
    parser.add_argument("--n-math", type=int, default=10_000,
                        help="Number of DeepMath training rows (default: 10000)")
    parser.add_argument("--n-val-search", type=int, default=200,
                        help="Number of Search-R1 rows held out for validation (default: 200)")
    parser.add_argument("--n-val-math", type=int, default=200,
                        help="Number of DeepMath rows held out for validation (default: 200)")
    parser.add_argument("--output-dir", type=str, default="data/training",
                        help="Output directory for parquet files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Downloading Search-R1 ({args.n_val_search} val + {args.n_search} train)...")
    search_train_rows, search_val_rows = _download_search_r1(
        args.n_search, args.n_val_search, args.seed
    )

    print(f"Downloading DeepMath ({args.n_val_math} val + {args.n_math} train)...")
    math_train_rows, math_val_rows = _download_deepmath(
        args.n_math, args.n_val_math, args.seed
    )

    build_combined_train(search_train_rows, math_train_rows, output_dir, args.seed)
    build_val_files(search_val_rows, math_val_rows, output_dir, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
