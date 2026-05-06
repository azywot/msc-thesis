"""Data preparation: download Search-R1 + DeepMath and write VERL parquet files.

Usage:
    python src/fine_tuning/data/prepare.py \\
        --n-search 10000 --n-math 10000 \\
        --output-dir data/training --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List


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

def _download_search_r1(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # lazy import — not needed for unit tests
    ds = load_dataset("PeterJinGo/SearchR1-nq_hotpotqa_train", split="train")
    ds = ds.shuffle(seed=seed)
    if n < len(ds):
        ds = ds.select(range(n))
    return [normalise_search_r1_row(dict(row), idx=i) for i, row in enumerate(ds)]


def _download_deepmath(n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=seed)
    if n < len(ds):
        ds = ds.select(range(n))
    return [normalise_deepmath_row(dict(row), idx=i) for i, row in enumerate(ds)]


def _download_aime24() -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    rows = []
    for i, row in enumerate(ds):
        answer = str(row.get("answer", ""))
        rows.append({
            "data_source": "aime",
            "question": str(row.get("problem", "")),
            "result": answer,
            "extra_info": {"idx": i, "groundtruth": answer},
        })
    return rows


# ---------------------------------------------------------------------------
# Main builders
# ---------------------------------------------------------------------------

def build_combined_train(
    n_search: int,
    n_math: int,
    output_dir: Path,
    seed: int,
) -> Path:
    """Download, normalise, mix, and write combined_train.parquet."""
    import pandas as pd

    print(f"Downloading Search-R1 ({n_search} examples)...")
    search_rows = _download_search_r1(n_search, seed)

    print(f"Downloading DeepMath ({n_math} examples)...")
    math_rows = _download_deepmath(n_math, seed)

    combined = search_rows + math_rows
    rng = random.Random(seed)
    rng.shuffle(combined)

    df = pd.DataFrame(combined)
    validate_parquet_schema(df)

    out_path = output_dir / "train" / "combined_train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def build_aime24_val(output_dir: Path) -> Path:
    """Download AIME24 and write aime24.parquet."""
    import pandas as pd

    print("Downloading AIME24 validation set...")
    rows = _download_aime24()
    df = pd.DataFrame(rows)
    validate_parquet_schema(df)

    out_path = output_dir / "val" / "aime24.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for orchestrator fine-tuning.")
    parser.add_argument("--n-search", type=int, default=10_000,
                        help="Number of Search-R1 rows (default: 10000)")
    parser.add_argument("--n-math", type=int, default=10_000,
                        help="Number of DeepMath rows (default: 10000)")
    parser.add_argument("--output-dir", type=str, default="data/training",
                        help="Output directory for parquet files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    build_combined_train(args.n_search, args.n_math, output_dir, args.seed)
    build_aime24_val(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
