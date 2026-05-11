"""Data preparation: download Search-R1 + DeepMath and write VERL parquet files.

Dataset curation rationale
--------------------------
GRPO reward improvements plateau early and adding
more data beyond ~1–3K examples provides diminishing returns.  Small, curated
datasets reduce training time, enable rapid experimentation cycles, and lower
the risk of GRPO collapse (pathological over-searching, reward hacking, looping)
that can emerge from overtraining on narrow distributions.

Search source composition (default: 85% HotpotQA / 15% NQ)
------------------------------------------------------------
HotpotQA requires multi-hop reasoning and deliberate evidence aggregation —
exactly the retrieval-policy behaviour we want to reinforce.  NQ questions are
mostly single-hop factual lookups that produce shallow retrieval trajectories
and provide weaker RL signal for learning *when* and *how* to search.  A small
NQ fraction (~15%) preserves diversity without diluting the multi-hop signal.

DeepMath difficulty filtering (default: difficulty >= 5)
--------------------------------------------------------
Hard mathematical reasoning problems produce cleaner RL signals for GRPO:
the model must reason deeply before converging on an answer, which encourages
long-horizon thinking traces.  Easy problems are solved trivially in one step,
contributing near-zero gradient signal.  The ``difficulty`` field in
zwhe99/DeepMath-103K is an integer 1–9; the default threshold of 5 retains
medium-hard and hard problems.

Usage:
    python src/fine_tuning/data/prepare.py \\
        --n-search 1000 --n-math 1000 \\
        --search-source both --hotpot-ratio 0.85 \\
        --deepmath-min-difficulty 5 \\
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

# HuggingFace dataset id (Search-R1 NQ + HotpotQA). Older Hub name
# PeterJinGo/SearchR1-nq_hotpotqa_train was removed; use nq_hotpotqa_train.
SEARCH_R1_HF_DATASET = "PeterJinGo/nq_hotpotqa_train"


# ---------------------------------------------------------------------------
# Row normalisers
# ---------------------------------------------------------------------------

def normalise_search_r1_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a Search-R1 row to VERL schema.

    Search-R1 (`SEARCH_R1_HF_DATASET`) columns:
        question, golden_answers, data_source (one of "nq", "hotpotqa")

    Legacy / alternate tables may use answer(s), dataset, reward_model, or
    extra_info. Keep those aliases as fallbacks so fixtures and forks work.
    """
    answer_val = (
        raw.get("golden_answers")
        or raw.get("answer")
        or raw.get("answers")
        or raw.get("ground_truth")
        or raw.get("groundtruth")
    )
    if answer_val is None and isinstance(raw.get("reward_model"), dict):
        reward_model = raw["reward_model"]
        answer_val = (
            reward_model.get("ground_truth")
            or reward_model.get("groundtruth")
            or reward_model.get("answer")
        )
    if answer_val is None and isinstance(raw.get("extra_info"), dict):
        extra_info = raw["extra_info"]
        answer_val = (
            extra_info.get("ground_truth")
            or extra_info.get("groundtruth")
            or extra_info.get("answer")
        )

    aliases = answer_val if isinstance(answer_val, list) else []
    if isinstance(answer_val, list):
        answer_val = next((a for a in answer_val if str(a).strip()), "")
    answer = str(answer_val) if answer_val is not None else ""

    data_source = raw.get("data_source") or raw.get("dataset") or "nq"

    return {
        "data_source": str(data_source).lower(),
        "question": str(raw.get("question", "")),
        "result": answer,
        "extra_info": {
            "idx": idx,
            "groundtruth": answer,
            "golden_answers": [str(a) for a in aliases],
        },
    }


def normalise_deepmath_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a DeepMath-103K row to VERL schema.

    zwhe99/DeepMath-103K (HF) uses ``question`` + ``final_answer``.
    Legacy / alternate tables may use ``problem`` + ``answer`` — those are kept as
    fallbacks so unit tests and forks keep working.
    """
    answer_val = raw.get("final_answer")
    if answer_val is None:
        answer_val = raw.get("answer", "")
    answer = str(answer_val) if answer_val is not None else ""

    q_val = raw.get("question")
    if q_val is None or (isinstance(q_val, str) and not q_val.strip()):
        q_val = raw.get("problem") or raw.get("instruction") or ""
    question = str(q_val) if q_val is not None else ""

    return {
        "data_source": "deepmath",
        "question": question,
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

def _is_valid_norm(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("question", "")).strip()) and bool(
        str(row.get("result", "")).strip()
    )


def _download_search_r1(
    n_train: int,
    n_val: int,
    seed: int,
    search_source: str = "both",
    hotpot_ratio: float = 0.85,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Download Search-R1 and split into train and val.

    Val rows are taken first to guarantee no overlap with train regardless
    of n_train.

    search_source controls which sources are included:
      "hotpotqa" — multi-hop questions only (preferred for retrieval-policy learning)
      "nq"       — open-domain factual questions only
      "both"     — mix; hotpot_ratio controls the HotpotQA fraction

    Uses *streaming* loading: the Hub dataset mixes NQ and HotpotQA parquet
    shards with incompatible nested Arrow schemas; a single non-streaming load
    forces a unified cast and fails (DatasetGenerationError / type mismatch).
    """
    from datasets import load_dataset  # lazy import — not needed for unit tests

    # Compute per-source quotas for val and train independently
    hotpot_val_quota, nq_val_quota = _search_source_quotas(n_val, search_source, hotpot_ratio)
    hotpot_train_quota, nq_train_quota = _search_source_quotas(n_train, search_source, hotpot_ratio)

    total_needed = n_val + n_train
    buffer_size = min(max(total_needed * 4, 10_000), 500_000)

    ds = load_dataset(SEARCH_R1_HF_DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    val_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    val_hotpot = val_nq = train_hotpot = train_nq = 0
    skipped = 0
    scanned = 0

    for sample in ds:
        scanned += 1
        if (
            val_hotpot >= hotpot_val_quota
            and val_nq >= nq_val_quota
            and train_hotpot >= hotpot_train_quota
            and train_nq >= nq_train_quota
        ):
            break

        norm = normalise_search_r1_row(dict(sample), idx=0)  # idx patched below
        if not _is_valid_norm(norm):
            skipped += 1
            continue

        src = norm["data_source"]  # "hotpotqa" or "nq"
        is_hotpot = src == "hotpotqa"

        if is_hotpot:
            if val_hotpot < hotpot_val_quota:
                norm["extra_info"]["idx"] = len(val_rows)
                val_rows.append(norm)
                val_hotpot += 1
            elif train_hotpot < hotpot_train_quota:
                norm["extra_info"]["idx"] = len(train_rows)
                train_rows.append(norm)
                train_hotpot += 1
        else:
            if val_nq < nq_val_quota:
                norm["extra_info"]["idx"] = len(val_rows)
                val_rows.append(norm)
                val_nq += 1
            elif train_nq < nq_train_quota:
                norm["extra_info"]["idx"] = len(train_rows)
                train_rows.append(norm)
                train_nq += 1

    if len(val_rows) != n_val or len(train_rows) != n_train:
        raise RuntimeError(
            f"Search-R1: only collected val={len(val_rows)}/{n_val}, "
            f"train={len(train_rows)}/{n_train} valid rows after scanning "
            f"{scanned} shuffled examples (skipped {skipped} with empty "
            f"question/result). "
            f"source={search_source!r}, hotpot_ratio={hotpot_ratio}. "
            f"Dataset may be too small or have changed schema."
        )

    if skipped:
        print(
            f"Search-R1: skipped {skipped} example(s) with empty question or "
            f"answer after normalisation."
        )

    return train_rows, val_rows


def _is_valid_deepmath_norm(row: Dict[str, Any]) -> bool:
    return _is_valid_norm(row)


def _search_source_quotas(n: int, source: str, hotpot_ratio: float) -> tuple:
    """Return (hotpot_quota, nq_quota) for n total Search-R1 rows.

    source must be one of "both", "hotpotqa", "nq".
    hotpot_ratio is only used when source == "both".
    """
    if source == "hotpotqa":
        return n, 0
    if source == "nq":
        return 0, n
    # "both": split by ratio, rounding hotpot up so quotas always sum to n
    hotpot = round(n * hotpot_ratio)
    return hotpot, n - hotpot


def _passes_difficulty_filter(raw: Dict[str, Any], min_difficulty: int) -> bool:
    """Return True if the raw DeepMath row meets the minimum difficulty.

    Rows with no difficulty field are accepted (fail-open) so that forks or
    schema changes do not silently drop all data.
    DeepMath-103K stores difficulty as an integer (1–9); Hub may return it as
    a string, so coerce before comparing.
    """
    difficulty = raw.get("difficulty")
    if difficulty is None:
        return True
    try:
        return int(difficulty) >= min_difficulty
    except (TypeError, ValueError):
        return True


def _download_deepmath(
    n_train: int,
    n_val: int,
    seed: int,
    min_difficulty: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Download DeepMath-103K and split into train and val.

    Val rows are filled first by scanning the shuffled dataset in order, then
    train rows.  Rows failing the difficulty threshold or with empty normalised
    ``question`` / ``result`` are skipped.

    min_difficulty filters on the ``difficulty`` integer field of DeepMath-103K
    (range 1–9).  Default 5 retains medium-hard and hard problems, which
    provide cleaner RL signal for GRPO — harder problems reduce spurious reward
    and encourage longer, more deliberate reasoning trajectories.

    Raises if the filtered dataset cannot yield n_val + n_train valid rows.
    """
    from datasets import load_dataset
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=seed)

    val_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    skipped = 0
    scanned = 0

    for row in ds:
        scanned += 1
        if len(val_rows) >= n_val and len(train_rows) >= n_train:
            break

        if not _passes_difficulty_filter(dict(row), min_difficulty):
            skipped += 1
            continue

        if len(val_rows) < n_val:
            idx = len(val_rows)
            target = val_rows
        else:
            idx = len(train_rows)
            target = train_rows

        norm = normalise_deepmath_row(dict(row), idx=idx)
        if not _is_valid_deepmath_norm(norm):
            skipped += 1
            continue
        target.append(norm)

    if len(val_rows) != n_val or len(train_rows) != n_train:
        raise RuntimeError(
            f"DeepMath: only collected val={len(val_rows)}/{n_val}, "
            f"train={len(train_rows)}/{n_train} valid rows after scanning "
            f"{scanned} examples (skipped {skipped} with difficulty < "
            f"{min_difficulty} or empty question/result). "
            f"Try lowering --deepmath-min-difficulty or the dataset is too small."
        )

    if skipped:
        print(
            f"DeepMath: skipped {skipped} example(s) below difficulty "
            f"{min_difficulty} or with empty question/answer."
        )

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
    parser = argparse.ArgumentParser(
        description=(
            "Prepare training data for orchestrator fine-tuning. "
            "Defaults are tuned for GRPO on Qwen3-8B (~1 epoch, ~2K samples): "
            "small high-signal subsets avoid GRPO saturation and policy collapse."
        ),
    )
    parser.add_argument(
        "--n-search", type=int, default=1_000,
        help=(
            "Number of Search-R1 training rows (default: 1000). "
            "GRPO reward plateaus early — 1K curated examples is typically sufficient."
        ),
    )
    parser.add_argument(
        "--n-math", type=int, default=1_000,
        help=(
            "Number of DeepMath training rows (default: 1000). "
            "Filtered by --deepmath-min-difficulty before sampling."
        ),
    )
    parser.add_argument(
        "--n-val-search", type=int, default=200,
        help="Number of Search-R1 rows held out for validation (default: 200)",
    )
    parser.add_argument(
        "--n-val-math", type=int, default=200,
        help="Number of DeepMath rows held out for validation (default: 200)",
    )
    parser.add_argument(
        "--search-source", choices=["hotpotqa", "nq", "both"], default="both",
        help=(
            "Which Search-R1 source(s) to include (default: both). "
            "hotpotqa: multi-hop questions only — strongest retrieval-policy signal. "
            "nq: open-domain factoid questions only. "
            "both: mix controlled by --hotpot-ratio."
        ),
    )
    parser.add_argument(
        "--hotpot-ratio", type=float, default=0.85,
        help=(
            "Fraction of Search-R1 rows drawn from HotpotQA when --search-source=both "
            "(default: 0.85). Must be in [0, 1]."
        ),
    )
    parser.add_argument(
        "--deepmath-min-difficulty", type=int, default=5,
        help=(
            "Minimum difficulty score for DeepMath rows (default: 5, range 1–9). "
            "Hard problems produce cleaner GRPO signal and encourage long reasoning traces. "
            "Lower this if you cannot collect enough rows after filtering."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/training",
        help="Output directory for parquet files",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.hotpot_ratio <= 1.0:
        parser.error(f"--hotpot-ratio must be in [0, 1], got {args.hotpot_ratio}")

    output_dir = Path(args.output_dir)

    print(
        f"Downloading Search-R1 "
        f"({args.n_val_search} val + {args.n_search} train, "
        f"source={args.search_source!r}, hotpot_ratio={args.hotpot_ratio})..."
    )
    search_train_rows, search_val_rows = _download_search_r1(
        args.n_search,
        args.n_val_search,
        args.seed,
        search_source=args.search_source,
        hotpot_ratio=args.hotpot_ratio,
    )

    print(
        f"Downloading DeepMath "
        f"({args.n_val_math} val + {args.n_math} train, "
        f"min_difficulty={args.deepmath_min_difficulty})..."
    )
    math_train_rows, math_val_rows = _download_deepmath(
        args.n_math,
        args.n_val_math,
        args.seed,
        min_difficulty=args.deepmath_min_difficulty,
    )

    build_combined_train(search_train_rows, math_train_rows, output_dir, args.seed)
    build_val_files(search_val_rows, math_val_rows, output_dir, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
