"""Inspect an SFT training parquet (messages + data_source columns).

Prints summary statistics (row counts, message-role distribution, thinking-token
proportion, tool usage) and optionally exports the data to a human-readable
JSONL file for browsing in an editor (parquet itself isn't directly readable).

Usage:
    # Just print stats
    python scripts/inspect_sft_data.py data/training/sft/sft_train.parquet

    # Also export to JSONL for browsing
    python scripts/inspect_sft_data.py data/training/sft/sft_train.parquet \
        --export-jsonl data/training/sft/sft_train_preview.jsonl
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _think_chars(text: str) -> int:
    return sum(len(t) for t in _THINK_RE.findall(text))


def main():
    parser = argparse.ArgumentParser(description="Inspect an SFT training parquet.")
    parser.add_argument("parquet", help="Path to sft_train.parquet")
    parser.add_argument(
        "--export-jsonl", default=None,
        help="If set, write the rows out as a human-readable JSONL file at this path "
             "(one row per line, indices added) for browsing in an editor/IDE.",
    )
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="Only export/inspect the first N rows (-1 = all).",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.limit > 0:
        df = df.iloc[: args.limit]

    logger.info("Loaded %d rows from %s", len(df), args.parquet)
    logger.info("Columns: %s", list(df.columns))

    if len(df) == 0:
        logger.info("Parquet has no rows; nothing to inspect.")
        return

    # ── Composition ───────────────────────────────────────────────────────────
    by_src = Counter(df["data_source"])
    logger.info("=" * 70)
    logger.info("Rows by data_source:")
    for src, count in sorted(by_src.items()):
        logger.info("  %-15s  %d  (%.1f%%)", src, count, 100.0 * count / len(df))

    # ── Message structure / thinking / tool usage ────────────────────────────
    role_counts = Counter()
    think_chars_total = 0
    non_think_chars_total = 0
    tool_name_counts = Counter()
    n_with_tool_use = 0
    turn_counts = []

    for messages in df["messages"]:
        roles = [m["role"] for m in messages]
        role_counts.update(roles)
        turn_counts.append(len(messages))

        has_tool = False
        for m in messages:
            if m["role"] == "assistant":
                tc = _think_chars(m["content"])
                think_chars_total += tc
                non_think_chars_total += len(m["content"]) - tc
            elif m["role"] == "tool":
                has_tool = True
                tool_name_counts[m.get("tool_name", "?")] += 1
        if has_tool:
            n_with_tool_use += 1

    logger.info("=" * 70)
    logger.info("Message roles (total across all rows): %s", dict(role_counts))
    logger.info("Avg messages per row: %.1f", sum(turn_counts) / len(turn_counts))
    logger.info("Rows using >=1 tool: %d / %d (%.1f%%)", n_with_tool_use, len(df),
                100.0 * n_with_tool_use / len(df))
    logger.info("Tool calls by name: %s", dict(tool_name_counts))

    total_assistant_chars = think_chars_total + non_think_chars_total
    if total_assistant_chars:
        logger.info(
            "Assistant content: %d think chars / %d non-think chars  (%.1f%% thinking)",
            think_chars_total, non_think_chars_total,
            100.0 * think_chars_total / total_assistant_chars,
        )
    logger.info("=" * 70)

    # ── Optional JSONL export for browsing ────────────────────────────────────
    if args.export_jsonl:
        out_path = Path(args.export_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in df.reset_index(drop=True).iterrows():
                record = {
                    "row_index": idx,
                    "data_source": row["data_source"],
                    "n_messages": len(row["messages"]),
                    "messages": list(row["messages"]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Exported %d rows → %s  (open in editor to browse)", len(df), out_path)


if __name__ == "__main__":
    main()
