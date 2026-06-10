"""Build sft_train.parquet from one or more collected_*.jsonl files.

Reads JSONL files produced by collect_sft_data.py, filters to correct
trajectories, and writes a VERL-compatible parquet with `messages` and
`data_source` columns.

Usage:
    # Single file
    python scripts/build_sft_parquet.py data/training/sft/collected_<ts>.jsonl

    # Merge multiple files (e.g. after several partial runs)
    python scripts/build_sft_parquet.py data/training/sft/collected_*.jsonl

    # Custom output directory
    python scripts/build_sft_parquet.py collected.jsonl --output-dir data/training/sft
"""

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _uses_tool(record: dict) -> bool:
    """True if any message in the trajectory is a tool-result turn."""
    return any(m.get("role") == "tool" for m in record.get("messages", []))


def _category(data_source: str) -> str:
    """Bucket a data_source into 'math' or 'search' — mirrors the original
    train/val/test split composition (deepmath/aime vs hotpotqa/nq)."""
    ds = data_source.lower()
    return "math" if (ds == "deepmath" or ds.startswith("aime")) else "search"


def _original_category_counts(parquet_path: Path) -> dict:
    """Read the original training parquet and count rows per category.

    Used to determine the target math:search ratio so the SFT set mirrors
    the composition of combined_train.parquet (e.g. 900 search / 900 math).
    """
    df = pd.read_parquet(parquet_path)
    counts: dict = defaultdict(int)
    for src in df["data_source"]:
        counts[_category(src)] += 1
    return dict(counts)


def _extra_info_by_qid(parquet_path: Path) -> dict:
    """Map question_id → extra_info dict, recovered from the reference parquet.

    collect_sft_data.py assigns question_id = positional index over the reference
    parquet's rows (`rows = [{"_qid": i, **r} for i, r in enumerate(df.to_dict(...))]`),
    so question_id == row position. This lets the SFT output carry the same
    `extra_info` (idx, groundtruth, difficulty, golden_answers) as combined_train —
    not stored in the collected records themselves.
    """
    df = pd.read_parquet(parquet_path)
    return {i: row["extra_info"] for i, row in df.reset_index(drop=True).iterrows()}


def _balance_to_ratio(selected: list, target_counts: dict, seed: int) -> list:
    """Downsample the over-represented category so math:search matches target_counts.

    Keeps all trajectories from the under-represented category and randomly
    subsamples the other (seeded, so the result is reproducible).
    """
    by_cat: dict = defaultdict(list)
    for r in selected:
        by_cat[_category(r["data_source"])].append(r)

    math_n, search_n = len(by_cat.get("math", [])), len(by_cat.get("search", []))
    target_math, target_search = target_counts.get("math", 0), target_counts.get("search", 0)
    if math_n == 0 or search_n == 0 or target_math == 0 or target_search == 0:
        return selected

    target_ratio = target_math / target_search   # math per search
    current_ratio = math_n / search_n

    rng = random.Random(seed)
    if current_ratio > target_ratio:
        # Too much math relative to search — cap math at search_n * target_ratio
        keep_math = min(math_n, round(search_n * target_ratio))
        by_cat["math"] = rng.sample(by_cat["math"], keep_math)
    elif current_ratio < target_ratio:
        # Too much search relative to math — cap search at math_n / target_ratio
        keep_search = min(search_n, round(math_n / target_ratio))
        by_cat["search"] = rng.sample(by_cat["search"], keep_search)

    return by_cat["math"] + by_cat["search"]


def _strip_thinking(messages: list) -> list:
    """Remove <think>...</think> blocks from assistant message content.

    The teacher runs with thinking_mode=ORCHESTRATOR_ONLY, but the student is
    trained with THINKING_MODE: NO during RL — keeping <think> blocks in SFT
    data would teach the student a behavior it never exhibits at inference
    time (and they make up ~82% of assistant content, mostly noise for a
    smaller model to imitate).
    """
    cleaned = []
    for m in messages:
        if m.get("role") == "assistant":
            content = _THINK_RE.sub("", m["content"]).strip()
            cleaned.append({**m, "content": content})
        else:
            cleaned.append(m)
    return cleaned


def _load_system_prompt_suffix(config_path: Path) -> Optional[str]:
    """Read the optional `system_prompt_suffix` field from the collection config YAML.

    Mirrors how collect_sft_data.py reads it (the field isn't in the Pydantic
    schema, so it's read directly from the raw YAML).
    """
    if not config_path or not Path(config_path).exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("system_prompt_suffix") or None


def _strip_system_suffix(messages: list, suffix: str) -> list:
    """Remove a trailing `system_prompt_suffix` from the system message.

    The 32B teacher's system prompt during collection was the GRPO/inference
    prompt + this suffix appended (a collection-time nudge to use tools more).
    Leaving it in the SFT data would create a train/inference mismatch: the
    student would learn to associate tool use with seeing this exact text,
    which GRPO training and final inference never show it. Stripping it makes
    the SFT system prompt byte-identical to what rollout.py builds.
    """
    cleaned = []
    for m in messages:
        if m.get("role") == "system" and suffix and m["content"].endswith(suffix):
            cleaned.append({**m, "content": m["content"][: -len(suffix)]})
        else:
            cleaned.append(m)
    return cleaned


def _split_train_val(selected: list, val_fraction: float, seed: int) -> tuple:
    """Split trajectories into train/val sets, stratified by category (math/search).

    Splitting per-category (rather than globally) keeps the val set's math:search
    ratio close to the train set's, mirroring the balancing already applied above.
    Selection is by question_id (each record is already one-per-question, so this
    is equivalent to a question-level split — train and val never share a question).
    """
    by_cat: dict = defaultdict(list)
    for r in selected:
        by_cat[_category(r["data_source"])].append(r)

    rng = random.Random(seed)
    train, val = [], []
    for cat, records in by_cat.items():
        records = list(records)
        rng.shuffle(records)
        n_val = round(len(records) * val_fraction)
        val.extend(records[:n_val])
        train.extend(records[n_val:])

    # Per-category extend leaves each split grouped by category (math block,
    # then search block) — shuffle so categories are interleaved in the final
    # row order written to parquet/jsonl.
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _select_one_per_question(correct_records: list) -> list:
    """Pick exactly one trajectory per question_id.

    When a question has multiple correct trajectories (from different passes),
    prefer one that used at least one tool — tool-free trajectories from a 32B
    teacher are often unrepresentative of what an 8B student can replicate, so
    they shouldn't crowd out tool-using examples in the SFT set.
    Falls back to a tool-free trajectory only if none of the correct ones used
    a tool. Selection is deterministic (lowest `pass` first) for reproducibility.
    """
    by_qid: dict = defaultdict(list)
    for r in correct_records:
        by_qid[r["question_id"]].append(r)

    selected = []
    for qid, group in by_qid.items():
        group_sorted = sorted(group, key=lambda r: r["pass"])
        tool_using = [r for r in group_sorted if _uses_tool(r)]
        chosen = tool_using[0] if tool_using else group_sorted[0]
        selected.append(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Build sft_train.parquet from collected JSONL files.")
    parser.add_argument(
        "jsonl_files", default=["/projects/0/gusr0608/msc-thesis/data/training/sft/collected_20260605_214650.jsonl"], nargs="*",
        help=(
            "Path(s) to collected_*.jsonl file(s). If omitted, all "
            "collected_*.jsonl files in --output-dir are used."
        ),
    )
    parser.add_argument(
        "--output-dir", default="/projects/0/gusr0608/msc-thesis/data/training/sft",
        help="Directory to read collected_*.jsonl from (when jsonl_files is omitted) "
             "and to write the output parquet to (default: data/training/sft).",
    )
    parser.add_argument(
        "--output-name", default="sft_train.parquet",
        help="Output filename (default: sft_train.parquet).",
    )
    parser.add_argument(
        "--reference-parquet", default="/projects/0/gusr0608/msc-thesis/data/training/train/combined_train.parquet",
        help=(
            "Original training parquet used to determine the target math:search "
            "ratio (default: data/training/train/combined_train.parquet, i.e. "
            "900 search / 900 math). Set to '' to skip balancing."
        ),
    )
    parser.add_argument(
        "--collection-config",
        default="/projects/0/gusr0608/msc-thesis/experiments/configs/fine_tuning/collect_sft_qwen32b.yaml",
        help=(
            "Collection config YAML to read `system_prompt_suffix` from — this text "
            "was appended to every system prompt the teacher saw during collection "
            "(a nudge to use tools more) but is NOT part of the GRPO/inference system "
            "prompt (rollout.py never applies it). It is stripped from the SFT system "
            "messages by default so SFT prompts are byte-identical to GRPO/inference "
            "prompts — pass --keep-system-suffix to retain it, or '' to skip."
        ),
    )
    parser.add_argument(
        "--keep-system-suffix", action="store_true",
        help="Keep the collection-time system_prompt_suffix in SFT system prompts "
             "(default: stripped — see --collection-config).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for downsampling and train/val splitting (default 0).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help=(
            "Fraction of selected trajectories held out for validation, split by "
            "question_id and stratified by category (math/search) so the val set "
            "mirrors the train set's ratio (default 0.1). Set to 0 to skip — only "
            "<output-name> train files are written."
        ),
    )
    parser.add_argument(
        "--keep-thinking", action="store_true",
        help=(
            "Keep <think>...</think> blocks in assistant messages (default: stripped). "
            "Stripping is the default because the student is trained with "
            "THINKING_MODE: NO during RL — keeping them would teach a behavior "
            "the student never uses at inference time."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.jsonl_files:
        jsonl_paths = [Path(p) for p in args.jsonl_files]
    else:
        jsonl_paths = sorted(output_dir.glob("collected_*.jsonl"))
        if not jsonl_paths:
            parser.error(
                f"No collected_*.jsonl files found in {output_dir} and none given explicitly. "
                "Pass file path(s) or check --output-dir."
            )
        logger.info("No jsonl_files given — auto-discovered %d file(s) in %s:",
                    len(jsonl_paths), output_dir)
        for p in jsonl_paths:
            logger.info("  %s", p)

    seen_keys: set = set()  # (question_id, pass) — deduplicate across files
    correct_records = []
    # Track every attempted question per source, and which were ever solved —
    # used to report per-type accuracy (yield) below.
    attempted_by_source: dict = defaultdict(set)
    solved_by_source: dict = defaultdict(set)

    for path in jsonl_paths:
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue
        logger.info("Reading %s", path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed line: %s", e)
                    continue
                key = (record.get("question_id"), record.get("pass"))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                src = record.get("data_source")
                qid = record.get("question_id")
                attempted_by_source[src].add(qid)
                if record.get("correct"):
                    solved_by_source[src].add(qid)
                if record.get("correct") and record.get("messages"):
                    correct_records.append(record)

    logger.info("Total correct trajectories across all files: %d", len(correct_records))

    if not correct_records:
        logger.warning("No correct trajectories found — parquet not written.")
        return

    # One trajectory per question — prefer tool-using ones over tool-free.
    selected = _select_one_per_question(correct_records)
    n_tool = sum(1 for r in selected if _uses_tool(r))
    logger.info(
        "Selected %d trajectories (one per question)  |  %d use tools  |  %d tool-free fallback",
        len(selected), n_tool, len(selected) - n_tool,
    )

    # Balance math:search ratio to mirror the original training composition, and
    # recover `extra_info` so SFT rows carry the same schema as combined_train
    # (data_source, question, result, extra_info) plus the trajectory `messages`.
    extra_info_map: dict = {}
    if args.reference_parquet:
        ref_path = Path(args.reference_parquet)
        if ref_path.exists():
            extra_info_map = _extra_info_by_qid(ref_path)
            target_counts = _original_category_counts(ref_path)
            before = Counter(_category(r["data_source"]) for r in selected)
            selected = _balance_to_ratio(selected, target_counts, seed=args.seed)
            after = Counter(_category(r["data_source"]) for r in selected)
            logger.info(
                "Balanced to original ratio (reference=%s, target math:search = %d:%d): "
                "math %d → %d, search %d → %d",
                ref_path, target_counts.get("math", 0), target_counts.get("search", 0),
                before.get("math", 0), after.get("math", 0),
                before.get("search", 0), after.get("search", 0),
            )
        else:
            logger.warning("Reference parquet not found (%s) — skipping ratio balancing.", ref_path)

    # Stats by source
    by_src = Counter(r["data_source"] for r in selected)
    tool_by_src = Counter(r["data_source"] for r in selected if _uses_tool(r))
    for src, count in sorted(by_src.items()):
        logger.info("  %-20s  %d trajectories  (%d use tools)", src, count, tool_by_src.get(src, 0))

    if not args.keep_thinking:
        for r in selected:
            r["messages"] = _strip_thinking(r["messages"])
        logger.info("Stripped <think>...</think> blocks from assistant messages "
                    "(pass --keep-thinking to retain them).")

    if not args.keep_system_suffix:
        suffix = _load_system_prompt_suffix(Path(args.collection_config)) if args.collection_config else None
        if suffix:
            n_stripped = 0
            for r in selected:
                before = r["messages"][0]["content"]
                r["messages"] = _strip_system_suffix(r["messages"], suffix)
                if r["messages"][0]["content"] != before:
                    n_stripped += 1
            logger.info(
                "Stripped collection-time system_prompt_suffix from %d/%d system prompts "
                "(reference=%s) — SFT prompts now match GRPO/inference exactly "
                "(pass --keep-system-suffix to retain it).",
                n_stripped, len(selected), args.collection_config,
            )
        else:
            logger.info("No system_prompt_suffix found in %s — nothing to strip.", args.collection_config)

    # Split into train/val (stratified by category, by question_id — train and
    # val never share a question) before writing, so VERL's `data.val_files`
    # gets a proper held-out set rather than reusing training rows.
    if args.val_fraction > 0:
        train_records, val_records = _split_train_val(selected, args.val_fraction, seed=args.seed)
        train_cat = Counter(_category(r["data_source"]) for r in train_records)
        val_cat = Counter(_category(r["data_source"]) for r in val_records)
        logger.info(
            "Split into train/val (val_fraction=%.2f, stratified by category): "
            "train %d (math %d, search %d)  |  val %d (math %d, search %d)",
            args.val_fraction, len(train_records), train_cat.get("math", 0), train_cat.get("search", 0),
            len(val_records), val_cat.get("math", 0), val_cat.get("search", 0),
        )
        splits = {"train": train_records, "val": val_records}
    else:
        splits = {"train": selected}

    # Same schema as combined_train (data_source, question, result, extra_info)
    # plus `messages` — so the SFT data traces back to the exact source row and
    # can later be consumed the same way as the original training/inference data.
    def _to_output_record(r: dict) -> dict:
        return {
            "data_source": r["data_source"],
            "question": r["question"],
            "result": r["ground_truth"],
            "extra_info": extra_info_map.get(r["question_id"]),
            "messages": r["messages"],
        }

    stem = Path(args.output_name).stem  # e.g. "sft_train" → base name for both splits
    base_name = stem[:-len("_train")] if stem.endswith("_train") else stem
    for split_name, records in splits.items():
        out_records = [_to_output_record(r) for r in records]

        df = pd.DataFrame(out_records)
        out_path = output_dir / f"{base_name}_{split_name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows → %s", len(df), out_path)

        # Mirror as JSONL (same rows, same order) for easy human verification —
        # parquet isn't directly readable in an editor.
        jsonl_path = out_path.with_suffix(".jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        logger.info("Saved %d rows → %s  (for verification — open in editor)", len(records), jsonl_path)

    # Statistics below report on the full `selected` set (pre-split) by source.

    # ── Final statistics ─────────────────────────────────────────────────────
    original_counts: dict = {}
    if args.reference_parquet:
        ref_path = Path(args.reference_parquet)
        if ref_path.exists():
            ref_df = pd.read_parquet(ref_path)
            original_counts = Counter(ref_df["data_source"])

    filtered_counts = Counter(r["data_source"] for r in selected)
    all_sources = sorted(set(original_counts) | set(attempted_by_source) | set(filtered_counts))

    logger.info("=" * 70)
    logger.info("STATISTICS BY QUESTION TYPE")
    logger.info("=" * 70)
    logger.info("%-12s  %10s  %18s  %16s", "type", "original", "accuracy (solved/attempted)", "filtered (SFT)")
    for src in all_sources:
        orig_n = original_counts.get(src, 0)
        attempted_n = len(attempted_by_source.get(src, set()))
        solved_n = len(solved_by_source.get(src, set()))
        acc_pct = 100.0 * solved_n / attempted_n if attempted_n else 0.0
        filt_n = filtered_counts.get(src, 0)
        logger.info(
            "%-12s  %10d  %5d/%-5d (%5.1f%%)  %16d",
            src, orig_n, solved_n, attempted_n, acc_pct, filt_n,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
