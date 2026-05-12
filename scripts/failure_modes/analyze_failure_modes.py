"""Analyze MAS failure modes from raw_results.json files.

Classifies every failed question-run from the 20 hard-coded MAS run inventory
into one of six mutually exclusive failure modes, then outputs:
  - breakdown.json  (machine-readable counts + question IDs)
  - breakdown.csv   (flat table)
  - console table   (thesis-style pretty print)
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS = ["aime", "gaia", "gpqa", "hle", "musique"]

FAILURE_MODES = [
    "modality_tool_gap",
    "tool_loop_or_empty_final",
    "direct_reasoning_no_action",
    "computational_subgoal_error",
    "retrieval_evidence_failure",
    "single_shot_tool_trust",
]

MODE_LABELS = {
    "modality_tool_gap":          "Modality / tool-coverage gap",
    "tool_loop_or_empty_final":   "Tool loop / empty final answer",
    "direct_reasoning_no_action": "Direct reasoning, no action",
    "computational_subgoal_error": "Computational sub-goal error",
    "retrieval_evidence_failure":  "Retrieval/evidence failure",
    "single_shot_tool_trust":     "Single-shot tool trust",
}

VISUAL_TOOLS = {"video_analysis", "image_inspector"}
VISUAL_KEYWORDS = {
    "image", "photo", "picture", "figure", "diagram",
    "video", "screenshot", "visual", "illustration",
}

MAX_TURNS = 15
MIN_LOOP_REPEATS = 3

# Relative to repo root: experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/
_MAS_BASE = "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow"

_INVENTORY = [
    # (benchmark, variant, relative_path)
    ("aime", "all",          f"{_MAS_BASE}/aime/qwen8B_subagent_tools_all/train_2026-03-15-21-28-24_20752251/raw_results.json"),
    ("aime", "none",         f"{_MAS_BASE}/aime/qwen8B_subagent_tools_none/train_2026-03-15-21-29-20_20752253/raw_results.json"),
    ("aime", "orchestrator", f"{_MAS_BASE}/aime/qwen8B_subagent_tools_orchestrator/train_2026-03-15-21-30-27_20752258/raw_results.json"),
    ("aime", "subagents",    f"{_MAS_BASE}/aime/qwen8B_subagent_tools_subagents/train_2026-03-15-21-31-28_20752265/raw_results.json"),
    ("gaia", "all",          f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_all/all_validation_2026-03-15-20-53-22_20752029/raw_results.json"),
    ("gaia", "none",         f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_none/all_validation_2026-03-15-20-54-46_20752030/raw_results.json"),
    ("gaia", "orchestrator", f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json"),
    ("gaia", "subagents",    f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_subagents/all_validation_2026-03-15-20-56-48_20752056/raw_results.json"),
    ("gpqa", "all",          f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_all/diamond_2026-03-15-21-17-57_20752192/raw_results.json"),
    ("gpqa", "none",         f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_none/diamond_2026-03-15-21-18-51_20752195/raw_results.json"),
    ("gpqa", "orchestrator", f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_orchestrator/diamond_2026-03-15-21-19-20_20752198/raw_results.json"),
    ("gpqa", "subagents",    f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_subagents/diamond_2026-03-15-21-20-26_20752204/raw_results.json"),
    ("hle",  "all",          f"{_MAS_BASE}/hle/qwen8B_subagent_tools_all/test_subset_200_2026-03-15-21-06-19_20752116/raw_results.json"),
    ("hle",  "none",         f"{_MAS_BASE}/hle/qwen8B_subagent_tools_none/test_subset_200_2026-03-15-21-06-56_20752118/raw_results.json"),
    ("hle",  "orchestrator", f"{_MAS_BASE}/hle/qwen8B_subagent_tools_orchestrator/test_subset_200_2026-03-15-21-07-48_20752122/raw_results.json"),
    ("hle",  "subagents",    f"{_MAS_BASE}/hle/qwen8B_subagent_tools_subagents/test_subset_200_2026-03-15-21-08-57_20752132/raw_results.json"),
    ("musique", "all",          f"{_MAS_BASE}/musique/qwen8B_subagent_tools_all/validation_2026-03-15-22-08-22_20752493/raw_results.json"),
    ("musique", "none",         f"{_MAS_BASE}/musique/qwen8B_subagent_tools_none/validation_2026-03-15-22-09-22_20752495/raw_results.json"),
    ("musique", "orchestrator", f"{_MAS_BASE}/musique/qwen8B_subagent_tools_orchestrator/validation_2026-03-15-22-09-55_20752496/raw_results.json"),
    ("musique", "subagents",    f"{_MAS_BASE}/musique/qwen8B_subagent_tools_subagents/validation_2026-03-15-22-10-51_20752501/raw_results.json"),
]

# ---------------------------------------------------------------------------
# Core functions (stubs — filled in subsequent tasks)
# ---------------------------------------------------------------------------

def classify_failure(record: dict) -> str:
    """Classify a single failed record into one of six failure modes.

    Rules are applied in priority order; the first match wins.
    """
    action_history = record.get("action_history") or []
    prediction = str(record.get("prediction") or "")
    turns = record.get("turns") or 0
    tool_counts = record.get("tool_counts") or {}

    tools_used = [s.get("tool_name", "") for s in action_history]
    tool_counter = Counter(tools_used)

    # ── Priority 1: modality / tool-coverage gap ────────────────────────────
    # Signal A: non-existent or image-specific tool was called
    if any(t in VISUAL_TOOLS for t in tools_used):
        return "modality_tool_gap"

    # Signal B: >= 2 text_inspector calls, at least one empty result,
    #           and at least one subgoal mentions an image/visual keyword
    ti_steps = [s for s in action_history if s.get("tool_name") == "text_inspector"]
    if len(ti_steps) >= 2:
        has_empty = any(not str(s.get("result") or "").strip() for s in ti_steps)
        has_keyword = any(
            kw in (s.get("sub_goal") or "").lower()
            for s in ti_steps
            for kw in VISUAL_KEYWORDS
        )
        if has_empty and has_keyword:
            return "modality_tool_gap"

    # ── Priority 2: tool loop / empty final answer ───────────────────────────
    if not prediction.strip():
        return "tool_loop_or_empty_final"
    if turns >= MAX_TURNS:
        return "tool_loop_or_empty_final"
    if tool_counter and max(tool_counter.values()) >= MIN_LOOP_REPEATS:
        return "tool_loop_or_empty_final"

    # ── Priority 3: direct reasoning without action ──────────────────────────
    if not action_history:
        return "direct_reasoning_no_action"

    # ── Priority 4: computational sub-goal error ─────────────────────────────
    if tool_counts.get("code_generator", 0) >= 1 and len(action_history) >= 2:
        return "computational_subgoal_error"

    # ── Priority 5: retrieval / evidence failure ─────────────────────────────
    if tool_counts.get("web_search", 0) >= 1:
        return "retrieval_evidence_failure"

    # ── Priority 6: single-shot tool trust (catch-all) ───────────────────────
    return "single_shot_tool_trust"


def load_run(path: Path) -> list:
    """Load records from a raw_results.json. Returns [] and warns on error."""
    if not path.exists():
        print(f"  [WARN] missing: {path}", file=sys.stderr)
        return []
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON parse error in {path}: {exc}", file=sys.stderr)
            return []


def run_inventory(root: Path) -> list:
    """Return list of (benchmark, variant, resolved_path) from the hard-coded inventory."""
    return [(bench, variant, root / rel) for bench, variant, rel in _INVENTORY]


def analyze(root: Path, output_dir: Path) -> None:
    """Run the full analysis: classify all failures, aggregate, write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # counts[mode][benchmark] = int
    counts: dict = {m: {b: 0 for b in BENCHMARKS} for m in FAILURE_MODES}
    # unique_qids[mode][benchmark] = set of question_ids
    unique_qids: dict = {m: {b: set() for b in BENCHMARKS} for m in FAILURE_MODES}
    # total failures per benchmark
    bench_totals: dict = {b: 0 for b in BENCHMARKS}

    for benchmark, variant, path in run_inventory(root):
        records = load_run(path)
        for rec in records:
            if rec.get("correct"):
                continue
            bench_totals[benchmark] += 1
            mode = classify_failure(rec)
            counts[mode][benchmark] += 1
            unique_qids[mode][benchmark].add(rec.get("question_id"))

    overall_total = sum(bench_totals.values())

    # ── Build structured result ──────────────────────────────────────────────
    result = {"failure_modes": {}, "totals": {**bench_totals, "overall": overall_total}}
    for mode in FAILURE_MODES:
        total = sum(counts[mode].values())
        share = round(total / overall_total * 100, 1) if overall_total else 0.0
        unique = sum(len(unique_qids[mode][b]) for b in BENCHMARKS)
        result["failure_modes"][mode] = {
            "label": MODE_LABELS[mode],
            "benchmarks": {
                b: {
                    "count": counts[mode][b],
                    "question_ids": sorted(unique_qids[mode][b]),
                }
                for b in BENCHMARKS
            },
            "total": total,
            "share_pct": share,
            "unique_questions": unique,
        }

    # ── Write JSON ───────────────────────────────────────────────────────────
    json_path = output_dir / "breakdown.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Written: {json_path}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = output_dir / "breakdown.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_mode", "aime", "gaia", "gpqa", "hle", "musique",
                         "total", "share_pct", "unique_questions"])
        for mode in FAILURE_MODES:
            fm = result["failure_modes"][mode]
            writer.writerow([
                mode,
                fm["benchmarks"]["aime"]["count"],
                fm["benchmarks"]["gaia"]["count"],
                fm["benchmarks"]["gpqa"]["count"],
                fm["benchmarks"]["hle"]["count"],
                fm["benchmarks"]["musique"]["count"],
                fm["total"],
                fm["share_pct"],
                fm["unique_questions"],
            ])
    print(f"Written: {csv_path}")

    # ── Console table ────────────────────────────────────────────────────────
    _print_table(result)


def _print_table(result: dict) -> None:
    """Print a thesis-style breakdown table to stdout."""
    col_w = [38, 6, 6, 6, 6, 9, 7, 7, 6]
    header = ["Failure mode", "AIME", "GAIA", "GPQA", "HLE", "MuSiQue",
              "Total", "Share", "Uniq."]
    sep = "─" * sum(col_w + [2] * len(col_w))

    def row(*vals):
        parts = [str(v).ljust(col_w[i]) if i == 0 else str(v).rjust(col_w[i])
                 for i, v in enumerate(vals)]
        return "  ".join(parts)

    print()
    print(row(*header))
    print(sep)
    for mode in FAILURE_MODES:
        fm = result["failure_modes"][mode]
        print(row(
            fm["label"],
            fm["benchmarks"]["aime"]["count"],
            fm["benchmarks"]["gaia"]["count"],
            fm["benchmarks"]["gpqa"]["count"],
            fm["benchmarks"]["hle"]["count"],
            fm["benchmarks"]["musique"]["count"],
            fm["total"],
            f"{fm['share_pct']}%",
            fm["unique_questions"],
        ))
    print(sep)
    t = result["totals"]
    print(row(
        "Total failures",
        t["aime"], t["gaia"], t["gpqa"], t["hle"], t["musique"],
        t["overall"], "100%", "---",
    ))
    print()


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent.parent
    default_out = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Classify MAS failures into 6 failure modes and output breakdown."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Repo root directory (default: {default_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Directory for breakdown.json and breakdown.csv (default: {default_out})",
    )
    args = parser.parse_args()
    analyze(args.root, args.output_dir)


if __name__ == "__main__":
    main()
