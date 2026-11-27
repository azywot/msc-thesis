"""Build the arm I training set: arm C plus rows from rescued trajectories.

Arm C is the folded SFT build over the model's own correct trajectories (built by
build_sft_parquet.py, unchanged). Arm I is arm C followed by new rows taken from
trajectories the reflector rescued, starting at the reflector's fixed call. Each new row
is one orchestrator decision from the fix onward, so its target is the corrected call and
the decisions that followed it in the successful rerun. The fixed call's context is
everything before it, which is unchanged, so no earlier decision is supervised.

Usage:
    python scripts/build_arm_i_parquet.py \
        --arm-c-dir data/training/rsi_phase2/arm_c \
        --rescue experiments/results/rsi/phase2/rescue_8b/rescue.jsonl \
        --reference-parquet data/training/train/combined_train.parquet \
        --output-dir data/training/rsi_phase2/arm_i \
        --max-length 16384 --tokenizer Qwen/Qwen3-8B
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_sft_parquet as sft  # noqa: E402

logger = logging.getLogger(__name__)

ARM_C_TRAIN = "arm_c_folded_train.parquet"
ARM_C_VAL = "arm_c_folded_val.parquet"


def _clean(text: str) -> str:
    """Assistant text as it appears after the SFT build strips thinking."""
    return sft._strip_thinking([{"role": "assistant", "content": text}])[0]["content"]


def rescued_rows(row: dict, planning_suffix: str) -> Optional[list]:
    """Folded rows of a rescued trajectory from the fixed call onward.

    Returns None when the fold has no planning row or when the row at the fixed step does
    not carry ``fix_text`` as its target, which means the rescue record and the trajectory
    disagree about where the fix sits. The caller counts that as misaligned.
    """
    messages = sft._strip_thinking(list(row["messages"]))
    if len(messages) < 2:
        return None
    plan, _, _ = sft._classify_turns(messages)
    if plan is None:
        return None
    rows = sft._fold_trajectory(messages, planning_suffix)
    idx = 1 + int(row["step"])
    if idx >= len(rows):
        return None
    if rows[idx][2]["content"] != _clean(row["fix_text"]):
        return None
    return rows[idx:]


def _is_no_tool(messages: list) -> bool:
    return "<tool_call>" not in messages[-1]["content"]


def _arm_stats(df: pd.DataFrame) -> dict:
    cats = [sft._category(s) for s in df["data_source"]]
    return {
        "rows": len(df),
        "questions": int(df["question"].nunique()) if len(df) else 0,
        "math_rows": cats.count("math"),
        "search_rows": cats.count("search"),
        "no_tool_rows": sum(_is_no_tool(list(m)) for m in df["messages"]),
    }


def _idx(extra) -> Optional[int]:
    if isinstance(extra, dict) and extra.get("idx") is not None:
        return int(extra["idx"])
    return None


def build_arm_i(
    arm_c_train: pd.DataFrame,
    rescue_rows: list,
    extra_info: dict,
    planning_suffix: str,
    max_length: Optional[int] = None,
    tokenizer_name: Optional[str] = None,
) -> tuple:
    """Return (arm I dataframe, stats). Arm I is arm C's rows, untouched, then the new rows."""
    arm_c_questions = set(arm_c_train["question"])
    arm_c_idx = {i for i in (_idx(e) for e in arm_c_train["extra_info"]) if i is not None}

    first = {}
    for r in rescue_rows:
        if r.get("status") != "ok" or not r.get("correct"):
            continue
        first.setdefault(r["question_id"], r)

    new_rows, misaligned, by_kind, by_domain = [], 0, Counter(), Counter()
    for qid, r in first.items():
        info = extra_info.get(qid)
        if r["question"] in arm_c_questions or (_idx(info) is not None and _idx(info) in arm_c_idx):
            raise ValueError(f"rescued question {qid} already appears in arm C")
        folded = rescued_rows(r, planning_suffix)
        if folded is None:
            misaligned += 1
            continue
        for msgs in folded:
            new_rows.append(
                {
                    "data_source": r["data_source"],
                    "question": r["question"],
                    "result": r["ground_truth"],
                    "extra_info": info,
                    "messages": msgs,
                    "_kind": r["kind"],
                    "_domain": r["domain"],
                }
            )

    n_before = len(new_rows)
    if max_length and new_rows:
        public = [{k: v for k, v in x.items() if not k.startswith("_")} for x in new_rows]
        kept = sft._drop_over_length(public, max_length, tokenizer_name)
        kept_ids = {id(k) for k in kept}
        new_rows = [x for x, p in zip(new_rows, public) if id(p) in kept_ids]
    dropped = n_before - len(new_rows)

    qids_kept = {x["question"] for x in new_rows}
    seen = set()
    for x in new_rows:
        if x["question"] not in seen:
            seen.add(x["question"])
            by_kind[x["_kind"]] += 1
            by_domain[x["_domain"]] += 1

    cols = list(arm_c_train.columns)
    new_df = pd.DataFrame(
        [{k: v for k, v in x.items() if not k.startswith("_")} for x in new_rows], columns=cols
    )
    arm_i = pd.concat([arm_c_train, new_df], ignore_index=True)

    stats = {
        "arm_c": _arm_stats(arm_c_train),
        "arm_i": {
            **_arm_stats(arm_i),
            "rescued_questions": len(qids_kept),
            "rescued_rows": len(new_rows),
            "by_kind": dict(by_kind),
            "by_domain": dict(by_domain),
            "misaligned": misaligned,
            "dropped_over_length": dropped,
        },
    }
    return arm_i, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build arm I from arm C and rescued trajectories.")
    p.add_argument("--arm-c-dir", required=True)
    p.add_argument("--rescue", required=True)
    p.add_argument("--reference-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    p.add_argument("--planning-suffix", default=None)
    args = p.parse_args()

    arm_c_dir, out = Path(args.arm_c_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    arm_c = pd.read_parquet(arm_c_dir / ARM_C_TRAIN)
    with open(args.rescue, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    suffix = (
        args.planning_suffix if args.planning_suffix is not None else sft._default_planning_suffix()
    )
    extra = sft._extra_info_by_qid(Path(args.reference_parquet))
    arm_i, stats = build_arm_i(arm_c, rows, extra, suffix, args.max_length, args.tokenizer)

    arm_i.to_parquet(out / "arm_i_folded_train.parquet", index=False)
    shutil.copyfile(arm_c_dir / ARM_C_VAL, out / "arm_i_folded_val.parquet")
    (out / "arm_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    logger.info("arm C %d rows, arm I %d rows", len(arm_c), len(arm_i))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
