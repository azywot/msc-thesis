"""Preflight gate for the Prefix-RFT demonstration store.

Run before training. Exits non-zero on the first class of defect that would
make replay silently wrong, in the spirit of check_sft_folded_format.py.

Usage:
    python scripts/check_prefix_demos.py \
        --demos data/training/prefix_rft/prefix_demos.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agent_engine.utils.parsing import parse_tool_call


def check_row(row) -> list[str]:
    problems: list[str] = []
    steps = list(row["steps"])
    key = str(row["question_key"])[:8]

    if row["n_steps"] != len(steps):
        problems.append(f"key={key}: n_steps={row['n_steps']} but {len(steps)} steps")

    for i, step in enumerate(steps):
        where = f"key={key} step={i}"
        if not str(step["response"]).strip():
            problems.append(f"{where}: empty response")
        if "<think>" in str(step["response"]):
            problems.append(f"{where}: surviving <think> block")

        is_first = i == 0
        is_last = i == len(steps) - 1

        if step["tool_name"] is not None:
            # A tool step is the only kind replay has to serve a result for, so
            # both the stored result and a parseable call are mandatory.
            if step["tool_result"] is None:
                problems.append(f"{where}: tool step has no stored tool_result")
            if parse_tool_call(str(step["response"])) is None:
                problems.append(f"{where}: tool call does not parse")
            if is_last:
                problems.append(
                    f"{where}: final step is a tool call, so the trajectory has no answer"
                )
        else:
            # A step with no tool call is only legitimate as the planning turn or
            # the final answer. One in the middle means the orchestrator emitted
            # something replay cannot reproduce.
            if not is_first and not is_last:
                problems.append(f"{where}: middle step is neither a tool call nor an answer")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demos",
        type=Path,
        default=Path("data/training/prefix_rft/prefix_demos.parquet"),
    )
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_parquet(args.demos)
    problems: list[str] = []
    for _, row in frame.iterrows():
        problems.extend(check_row(row))

    # The store is looked up by question_key; a duplicate would silently shadow.
    n_dupes = len(frame) - frame["question_key"].nunique()
    if n_dupes:
        problems.append(f"store has {n_dupes} duplicate question_key values")

    n_single = int((frame["n_steps"] == 1).sum())
    print(f"Checked {len(frame)} demonstrations from {args.demos}")
    print(f"  decisions: {int(frame['n_steps'].sum())}, mean {frame['n_steps'].mean():.2f}")
    print(f"  single-decision (never prefixed): {n_single}")
    print(f"  prefixable questions: {len(frame) - n_single}")

    if problems:
        print(f"\nFAILED with {len(problems)} problems, first {args.max_report}:")
        for problem in problems[: args.max_report]:
            print(f"  {problem}")
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
