"""Build the Prefix-RFT demonstration store from collected teacher trajectories.

One row per question, holding the teacher's decisions in order. Prefix-RFT
replays the first ``k`` of them and lets the policy continue from there, so each
decision needs both the assistant response and, where the decision was a tool
call, the tool result that came back.

Reuses ``build_sft_parquet``'s helpers so the two pipelines cannot drift: the
same correctness filter, the same thinking strip, the same positional split of a
stored trajectory into (plan, actions, answer).

Usage:
    python scripts/build_prefix_demos.py \
        data/training/sft/collected_20260605_214650.jsonl \
        --output data/training/prefix_rft/prefix_demos.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from build_sft_parquet import _classify_turns, _strip_thinking

# Single source of truth: the runtime store must hash questions exactly as the
# builder does, so both import the same function.
#
# Not ``extra_info.idx``: prepare.py assigns idx per data source, so it collides
# across them (idx 669 is both a deepmath and a hotpotqa question in
# combined_train.parquet), and keying on it would attach a maths demonstration to
# a search question with nothing downstream to notice. Not ``question_id``
# either: that is the row position in the shuffled parquet, which the rollout
# worker never sees. The question text is unique and both sides hold it verbatim.
from verl_ext.prefix_rft.demos import question_key

logger = logging.getLogger(__name__)


def record_to_steps(record: dict) -> list[dict]:
    """Return the teacher's decisions in order, or [] if the record is unusable."""
    messages = _strip_thinking(record["messages"])
    plan, actions, answer = _classify_turns(messages)

    steps: list[dict] = []
    if plan is not None:
        steps.append({"response": plan, "tool_name": None, "tool_result": None})
    for content, tool_name, tool_result in actions:
        steps.append(
            {"response": content, "tool_name": tool_name, "tool_result": tool_result}
        )
    if answer is not None:
        steps.append({"response": answer, "tool_name": None, "tool_result": None})

    if not steps:
        return []
    if any(not s["response"] or not s["response"].strip() for s in steps):
        return []
    if any("<think>" in s["response"] for s in steps):
        # _strip_thinking matches <think>...</think>, so a surviving open tag means
        # the teacher hit its token limit mid-thought. Both such trajectories in the
        # 2026-06-05 collection are 26k+ character repetition loops. Replaying one
        # would teach the policy to loop.
        return []
    return steps


def records_to_demo_rows(records: list[dict]) -> list[dict]:
    """Filter to correct trajectories, one per question, and build store rows.

    ``question_id`` and ``idx`` are carried for diagnostics only; ``question_key``
    is the lookup key.
    """
    rows: dict[str, dict] = {}
    for record in records:
        if not record.get("correct") or not record.get("messages"):
            continue
        question = str(record.get("question", ""))
        key = question_key(question)
        if key in rows:
            continue
        steps = record_to_steps(record)
        if not steps:
            continue
        rows[key] = {
            "question_key": key,
            "question_id": int(record["question_id"]),
            "data_source": str(record.get("data_source", "")),
            "question": question,
            "n_steps": len(steps),
            "steps": steps,
        }
    return [rows[k] for k in sorted(rows, key=lambda k: rows[k]["question_id"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="collected_<ts>.jsonl from 006")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/prefix_rft/prefix_demos.parquet"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = []
    with args.jsonl.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Read %d records from %s", len(records), args.jsonl)

    rows = records_to_demo_rows(records)
    if not rows:
        raise SystemExit("No usable demonstrations found; store not written.")

    n_correct = len({question_key(str(r.get("question", ""))) for r in records
                     if r.get("correct") and r.get("messages")})
    if n_correct > len(rows):
        logger.info(
            "Dropped %d of %d correct trajectories as unusable "
            "(empty step, or truncated mid-<think>)",
            n_correct - len(rows),
            n_correct,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)

    n_single = sum(1 for r in rows if r["n_steps"] == 1)
    total = sum(r["n_steps"] for r in rows)
    logger.info("Wrote %d demonstrations to %s", len(rows), args.output)
    logger.info(
        "Decisions: total %d, mean %.2f, single-decision questions %d "
        "(these can never carry a prefix, see the spec)",
        total,
        total / len(rows),
        n_single,
    )


if __name__ == "__main__":
    main()
