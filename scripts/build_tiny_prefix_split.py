"""Build the two-question split that 011_tiny_prefix_rft.job trains on.

The questions are chosen, not sampled. To test Prefix-RFT rather than the training
pipeline, a run needs trajectories that contain BOTH replayed and generated tokens,
which means demonstrations long enough for the schedule to cut in the middle:

    k = clamp(floor(l * m), 0, m - 1),   l = 0.95 at global_step 0

At m = 3 that gives k = 2, so the plan and the tool call are replayed and the final
answer is generated on-policy. Rows are taken from the existing smoke split so the
schema is exactly what the trainer already reads, and one goes through web_search
while the other goes through code_generator, because the replay path has to hand a
stored tool result back through the registry for both.

Writes:
    data/training/prefix_rft/tiny/train/tiny_train.parquet   (2 questions)
    data/training/prefix_rft/tiny/val/tiny_val.parquet       (2 questions, no demos needed)

Usage:
    python scripts/build_tiny_prefix_split.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from verl_ext.prefix_rft.demos import DemoStore
from verl_ext.prefix_rft.schedule import PrefixStepSchedule

# Row indices into data/training/smoke/train/combined_train.parquet.
#   62 -> nq,       "ranchi is capital of which state in india?"  (web_search)
#   17 -> deepmath, a limit evaluated with code                   (code_generator)
# Both have three teacher decisions. _pick_rows re-derives the choice if the smoke
# split is ever rebuilt, so these are a starting point rather than a hard dependency.
PREFERRED_TRAIN_ROWS = (62, 17)
WANTED_TOOLS = ("web_search", "code_generator")


def _demo_shape(store: DemoStore, question: str):
    steps = store.steps(question)
    if not steps:
        return 0, None
    tools = [s.get("tool_name") for s in steps if s.get("tool_name")]
    return len(steps), (tools[0] if tools else None)


def _pick_rows(train: pd.DataFrame, store: DemoStore) -> list[int]:
    """Prefer the recorded rows; fall back to the first match per tool."""
    schedule = PrefixStepSchedule(seed=42)

    def usable(idx: int, tool: str) -> bool:
        question = train.iloc[idx]["question"]
        n, first_tool = _demo_shape(store, question)
        if first_tool != tool:
            return False
        k = schedule.sample_k(n, global_step=0)
        return 1 <= k < n

    chosen = []
    for preferred, tool in zip(PREFERRED_TRAIN_ROWS, WANTED_TOOLS):
        if usable(preferred, tool):
            chosen.append(preferred)
            continue
        replacement = next((i for i in range(len(train)) if usable(i, tool)), None)
        if replacement is None:
            sys.exit(
                f"FAIL: no smoke question has a splittable demonstration through {tool}. "
                "Rebuild the demonstration store or widen WANTED_TOOLS."
            )
        print(f"  row {preferred} is no longer usable for {tool}; using {replacement}")
        chosen.append(replacement)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, default=Path("data/training/smoke"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training/prefix_rft/tiny"))
    parser.add_argument(
        "--demos", type=Path, default=Path("data/training/prefix_rft/prefix_demos.parquet")
    )
    args = parser.parse_args()

    store = DemoStore.from_parquet(args.demos)
    train = pd.read_parquet(args.smoke_dir / "train" / "combined_train.parquet")
    val = pd.read_parquet(args.smoke_dir / "val" / "val_combined.parquet")

    rows = _pick_rows(train, store)
    tiny_train = train.iloc[rows].reset_index(drop=True)
    # Validation only has to run; it is never prefixed, so any two rows will do.
    tiny_val = val.iloc[[0, 1]].reset_index(drop=True)

    (args.out_dir / "train").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "val").mkdir(parents=True, exist_ok=True)
    tiny_train.to_parquet(args.out_dir / "train" / "tiny_train.parquet", index=False)
    tiny_val.to_parquet(args.out_dir / "val" / "tiny_val.parquet", index=False)

    schedule = PrefixStepSchedule(seed=42)
    print(f"Wrote {args.out_dir}/train/tiny_train.parquet ({len(tiny_train)} questions):")
    for i, row in tiny_train.iterrows():
        n, tool = _demo_shape(store, row["question"])
        k = schedule.sample_k(n, global_step=0)
        print(
            f"  [{i}] {row['data_source']:<10} n_steps={n} k={k} tool={tool} "
            f"{row['question'][:50]!r}"
        )
    print(f"Wrote {args.out_dir}/val/tiny_val.parquet ({len(tiny_val)} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
