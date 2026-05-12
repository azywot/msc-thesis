"""Analyze MAS failure modes from raw_results.json files.

Classifies every failed question-run from the 20 hard-coded MAS run inventory
into one of six mutually exclusive failure modes, then outputs:
  - breakdown.json  (machine-readable counts + question IDs)
  - breakdown.csv   (flat table)
  - console table   (thesis-style pretty print)

── How the thesis table figures were obtained ────────────────────────────────

Table 5 (failure mode × benchmark counts) in the thesis was produced by
*manual trace-level labelling*, not by this script.  Each of the 2 534 failed
question-run records was read individually — including the full action_history,
tool results, and question text — and assigned a label using the six-mode
taxonomy defined in docs/failure_mode_and_fine_tuning/failure_mode.md.  The
per-question assignments, representative examples, and source file/question-ID
lists are recorded in that document (Appendix A sections).

This script re-derives the same breakdown *automatically* from the structured
fields of raw_results.json (action_history, tool_counts, prediction, turns,
question text) using a priority-rule cascade that mirrors the taxonomy.  It
is intended as a reproducible proxy and a starting point for analysis of new
runs — not as an exact replayer of the manual labels.

Known heuristic gaps (irreducible without reading full trace content):
  - modality_tool_gap is undercounted: many cases are identified manually from
    attachment metadata or question phrasing that does not contain one of the
    VISUAL_KEYWORDS, and from multi-step traces where text_inspector returned
    empty but the subgoal text does not use image/visual language.
  - retrieval_evidence_failure and single_shot_tool_trust share a boundary
    (multi-search vs single-search) that the manual labeller resolved from the
    quality of the evidence, not just the call count.
  - direct_reasoning_no_action is slightly overcounted because some no-tool
    diagram questions lack visual keywords and cannot be promoted to
    modality_tool_gap by Signal C.

The aggregate total (2 534 failures, per-benchmark totals) is exact because it
is read directly from the correct/incorrect flags; only the per-mode split
approximates the manual assignment.
────────────────────────────────────────────────────────────────────────────"""

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

# Thinking-mode variant names as they appear in the run inventory, plus
# their display labels used in table headers and JSON keys.
THINKING_MODES = ["none", "all", "orchestrator", "subagents"]
THINKING_MODE_LABELS = {
    "none":         "NO",
    "all":          "ALL",
    "orchestrator": "ORCH",
    "subagents":    "SUB",
}

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

# Tools that only exist for non-text modalities.  video_analysis was never
# wired into the MAS tool set (its results are always empty); image_inspector
# is the image-specific variant of text_inspector.  Any call to either signals
# a modality gap regardless of the result content (Signal A).
VISUAL_TOOLS = {"video_analysis", "image_inspector"}

# Keywords scanned in text_inspector subgoals (Signal B) and in the question
# text when action_history is empty (Signal C).  These are intentionally
# conservative: broad terms such as "visual" or "video" can appear in
# non-modality contexts (e.g. "visual inspection of the data", "video game").
# The heuristic accepts some false negatives (missing real modality cases) in
# exchange for avoiding false positives that would inflate the count.
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
# Core functions
# ---------------------------------------------------------------------------

def classify_failure(record: dict) -> str:
    """Classify a single failed record into one of six failure modes.

    Rules are applied in priority order; the first match wins.
    All counts are derived from action_history (the ground-truth step list)
    rather than the pre-computed tool_counts field so that a single source
    of truth is used throughout.
    """
    action_history = record.get("action_history") or []
    prediction = str(record.get("prediction") or "")
    turns = record.get("turns") or 0

    tools_used = [s.get("tool_name", "") for s in action_history]
    # Counter({}) is falsy, so `if tool_counter and max(...)` below safely
    # avoids calling max() on an empty sequence when action_history is empty.
    tool_counter = Counter(tools_used)

    question_text = (record.get("question") or "").lower()

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

    # Signal C: no tool calls, non-empty prediction, and the question requires a
    #           visual modality — upstream cause is the missing modality, not a
    #           reasoning choice.  Empty-prediction cases belong in Priority 2
    #           (tool_loop_or_empty_final) regardless of visual content.
    if (not action_history and prediction.strip()
            and any(kw in question_text for kw in VISUAL_KEYWORDS)):
        return "modality_tool_gap"

    # ── Priority 2: tool loop / empty final answer ───────────────────────────
    if not prediction.strip():
        return "tool_loop_or_empty_final"
    if turns >= MAX_TURNS:
        return "tool_loop_or_empty_final"
    # tool_counter is falsy when action_history is empty, guarding max() safely.
    if tool_counter and max(tool_counter.values()) >= MIN_LOOP_REPEATS:
        return "tool_loop_or_empty_final"

    # ── Priority 3: direct reasoning without action ──────────────────────────
    if not action_history:
        return "direct_reasoning_no_action"

    # ── Priority 4: computational sub-goal error ─────────────────────────────
    # Requires >= 2 code_generator calls: the model ran multiple computational
    # sub-goals with wrong quantities/formulas.  A single code_generator call
    # in an otherwise multi-step trace is single-shot trust of the code result.
    if tool_counter.get("code_generator", 0) >= 2:
        return "computational_subgoal_error"

    # ── Priority 5: retrieval / evidence failure ─────────────────────────────
    # Requires >= 2 web_search calls: the agent searched multiple times but
    # failed to reconcile the evidence.  A single web_search that was blindly
    # trusted is single-shot trust (priority 6), not a retrieval failure.
    if tool_counter.get("web_search", 0) >= 2:
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


def _build_result_single_run(counts_m: dict, unique_m: dict, total: int) -> dict:
    """Build a result dict for one (benchmark, thinking_mode) run file.

    Args:
        counts_m: {failure_mode: int}
        unique_m: {failure_mode: set of question_ids}
        total:    total failed question-run cases in this single run
    """
    result: dict = {"failure_modes": {}, "total": total}
    for mode in FAILURE_MODES:
        count = counts_m[mode]
        share = round(count / total * 100, 1) if total else 0.0
        result["failure_modes"][mode] = {
            "label": MODE_LABELS[mode],
            "count": count,
            "share_pct": share,
            # unique_questions equals count for a single run (no cross-condition dedup)
            "unique_questions": len(unique_m[mode]),
            "question_ids": sorted(unique_m[mode]),
        }
    return result


def _build_result_by_benchmark(counts_bm: dict, unique_bm: dict, totals_bm: dict) -> dict:
    """Build a result dict with benchmarks as the inner dimension.

    Used for global view and per-thinking-mode views (same shape as the thesis
    table: rows = failure modes, columns = benchmarks).

    Args:
        counts_bm: {benchmark: {mode: int}}
        unique_bm: {benchmark: {mode: set of question_ids}}
        totals_bm: {benchmark: int}  — total failures per benchmark
    """
    overall_total = sum(totals_bm[b] for b in BENCHMARKS)
    result: dict = {
        "failure_modes": {},
        "totals": {b: totals_bm[b] for b in BENCHMARKS} | {"overall": overall_total},
    }
    for mode in FAILURE_MODES:
        total = sum(counts_bm[b][mode] for b in BENCHMARKS)
        share = round(total / overall_total * 100, 1) if overall_total else 0.0
        # unique_questions = distinct (benchmark, question_id) pairs; IDs are
        # benchmark-local so summing set sizes is equivalent.
        unique = sum(len(unique_bm[b][mode]) for b in BENCHMARKS)
        result["failure_modes"][mode] = {
            "label": MODE_LABELS[mode],
            "benchmarks": {
                b: {"count": counts_bm[b][mode], "question_ids": sorted(unique_bm[b][mode])}
                for b in BENCHMARKS
            },
            "total": total,
            "share_pct": share,
            "unique_questions": unique,
        }
    return result


def _build_result_by_thinking(counts_th: dict, unique_th: dict, totals_th: dict) -> dict:
    """Build a result dict with thinking modes as the inner dimension.

    Used for per-benchmark views (rows = failure modes, columns = thinking
    modes).  unique_questions counts distinct question_ids within this
    benchmark across all thinking conditions for a given mode.

    Args:
        counts_th: {thinking_mode: {mode: int}}
        unique_th: {thinking_mode: {mode: set of question_ids}}
        totals_th: {thinking_mode: int}  — total failures per thinking mode
    """
    overall_total = sum(totals_th[v] for v in THINKING_MODES)
    result: dict = {
        "failure_modes": {},
        "totals": {v: totals_th[v] for v in THINKING_MODES} | {"overall": overall_total},
    }
    for mode in FAILURE_MODES:
        total = sum(counts_th[v][mode] for v in THINKING_MODES)
        share = round(total / overall_total * 100, 1) if overall_total else 0.0
        # Union across thinking conditions: same question solved in 4 runs = 1 unique
        unique = len(set().union(*(unique_th[v][mode] for v in THINKING_MODES)))
        result["failure_modes"][mode] = {
            "label": MODE_LABELS[mode],
            "thinking_modes": {
                v: {"count": counts_th[v][mode], "question_ids": sorted(unique_th[v][mode])}
                for v in THINKING_MODES
            },
            "total": total,
            "share_pct": share,
            "unique_questions": unique,
        }
    return result


def analyze(root: Path, output_dir: Path) -> None:
    """Run the full analysis: classify all failures, aggregate, write outputs.

    Produces three levels of breakdown:
      global          — all 20 runs combined (failure mode × benchmark)
      by_thinking_mode — one table per thinking condition (same shape as global)
      by_benchmark    — one table per benchmark (failure mode × thinking mode)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parse_errors: int = 0

    # Raw counts indexed [variant][benchmark][failure_mode]
    raw_counts = {v: {b: {m: 0  for m in FAILURE_MODES} for b in BENCHMARKS} for v in THINKING_MODES}
    raw_unique  = {v: {b: {m: set() for m in FAILURE_MODES} for b in BENCHMARKS} for v in THINKING_MODES}
    # total failures per (variant, benchmark); question-run cases, not unique questions
    raw_totals  = {v: {b: 0 for b in BENCHMARKS} for v in THINKING_MODES}

    for benchmark, variant, path in run_inventory(root):
        records = load_run(path)
        for rec in records:
            if rec.get("correct"):
                continue
            raw_totals[variant][benchmark] += 1
            try:
                mode = classify_failure(rec)
            except Exception as exc:
                print(f"  [WARN] classify_failure failed for qid={rec.get('question_id')}: {exc}",
                      file=sys.stderr)
                parse_errors += 1
                continue
            raw_counts[variant][benchmark][mode] += 1
            raw_unique[variant][benchmark][mode].add(rec.get("question_id"))

    # ── Aggregate ────────────────────────────────────────────────────────────
    # Global: sum over all thinking-mode variants
    global_result = _build_result_by_benchmark(
        {b: {m: sum(raw_counts[v][b][m] for v in THINKING_MODES) for m in FAILURE_MODES}
         for b in BENCHMARKS},
        {b: {m: set().union(*(raw_unique[v][b][m] for v in THINKING_MODES)) for m in FAILURE_MODES}
         for b in BENCHMARKS},
        {b: sum(raw_totals[v][b] for v in THINKING_MODES) for b in BENCHMARKS},
    )

    # By thinking mode: one global-shaped result per variant
    by_thinking = {
        v: _build_result_by_benchmark(
            {b: {m: raw_counts[v][b][m] for m in FAILURE_MODES} for b in BENCHMARKS},
            {b: {m: raw_unique[v][b][m] for m in FAILURE_MODES} for b in BENCHMARKS},
            {b: raw_totals[v][b] for b in BENCHMARKS},
        )
        for v in THINKING_MODES
    }

    # By benchmark: one thinking-mode-shaped result per benchmark
    by_benchmark = {
        b: _build_result_by_thinking(
            {v: {m: raw_counts[v][b][m] for m in FAILURE_MODES} for v in THINKING_MODES},
            {v: {m: raw_unique[v][b][m] for m in FAILURE_MODES} for v in THINKING_MODES},
            {v: raw_totals[v][b] for v in THINKING_MODES},
        )
        for b in BENCHMARKS
    }

    # By benchmark × thinking mode: individual result for each of the 20 runs
    by_benchmark_and_thinking = {
        b: {
            v: _build_result_single_run(
                {m: raw_counts[v][b][m] for m in FAILURE_MODES},
                {m: raw_unique[v][b][m] for m in FAILURE_MODES},
                raw_totals[v][b],
            )
            for v in THINKING_MODES
        }
        for b in BENCHMARKS
    }

    full_result = {
        "global": global_result,
        "by_thinking_mode": by_thinking,
        "by_benchmark": by_benchmark,
        "by_benchmark_and_thinking": by_benchmark_and_thinking,
        "parse_errors": parse_errors,
    }

    # ── Write JSON (single file, all levels) ─────────────────────────────────
    json_path = output_dir / "breakdown.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2)
    print(f"Written: {json_path}")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    _write_csv_global(output_dir / "breakdown_global.csv", global_result)
    _write_csv_by_thinking(output_dir / "breakdown_by_thinking_mode.csv", by_thinking)
    _write_csv_by_benchmark(output_dir / "breakdown_by_benchmark.csv", by_benchmark)
    _write_csv_by_benchmark_and_thinking(
        output_dir / "breakdown_by_benchmark_and_thinking.csv", by_benchmark_and_thinking
    )

    # ── LaTeX tables ─────────────────────────────────────────────────────────
    _tables_dir = output_dir.parent.parent / "results" / "tables" / "failure_modes"
    _write_latex_global(
        _tables_dir / "failure_modes_global.tex",
        global_result,
    )
    _write_latex_by_thinking(
        _tables_dir / "failure_modes_by_thinking.tex",
        by_thinking,
    )
    _write_latex_thinking_x_benchmark(
        _tables_dir / "total_failures_thinking_x_benchmark.tex",
        by_benchmark_and_thinking,
    )

    # ── Console tables ────────────────────────────────────────────────────────
    _print_table(global_result, title="Global — all 20 runs")
    _print_table_thinking_x_benchmark(by_benchmark_and_thinking)
    for v in THINKING_MODES:
        _print_table(by_thinking[v], title=f"Thinking mode: {THINKING_MODE_LABELS[v]}")
    for b in BENCHMARKS:
        _print_table(
            by_benchmark[b],
            title=f"Benchmark: {b.upper()}",
            cols=THINKING_MODES,
            col_labels=THINKING_MODE_LABELS,
            get_count=lambda fm, c: fm["thinking_modes"][c]["count"],
        )
    for b in BENCHMARKS:
        for v in THINKING_MODES:
            _print_table_single_run(
                by_benchmark_and_thinking[b][v],
                title=f"{b.upper()} × {THINKING_MODE_LABELS[v]}",
            )


def _print_table(
    result: dict,
    title: str = "",
    cols: list = BENCHMARKS,
    col_labels: dict = None,
    get_count=None,
) -> None:
    """Print a thesis-style breakdown table.

    Args:
        result:     result dict from _build_result_by_benchmark or _by_thinking
        title:      optional section heading printed above the table
        cols:       ordered list of column keys (benchmarks or thinking modes)
        col_labels: display label for each column key; defaults to key.upper()
        get_count:  callable(fm, col) → int; defaults to fm["benchmarks"][col]["count"]
    """
    if col_labels is None:
        col_labels = {c: c.upper() for c in cols}
    if get_count is None:
        get_count = lambda fm, c: fm["benchmarks"][c]["count"]

    col_w = [38] + [max(6, len(col_labels[c]) + 1) for c in cols] + [7, 7, 6]
    header = ["Failure mode"] + [col_labels[c] for c in cols] + ["Total", "Share", "Uniq."]
    sep = "─" * (sum(col_w) + 2 * len(col_w))

    def row(*vals):
        parts = [str(v).ljust(col_w[i]) if i == 0 else str(v).rjust(col_w[i])
                 for i, v in enumerate(vals)]
        return "  ".join(parts)

    if title:
        print(f"\n── {title} " + "─" * max(0, len(sep) - len(title) - 4))
    print()
    print(row(*header))
    print(sep)
    for mode in FAILURE_MODES:
        fm = result["failure_modes"][mode]
        print(row(
            fm["label"],
            *[get_count(fm, c) for c in cols],
            fm["total"],
            f"{fm['share_pct']}%",
            fm["unique_questions"],
        ))
    print(sep)
    t = result["totals"]
    print(row(
        "Total failures",
        *[t[c] for c in cols],
        t["overall"], "100%", "---",
    ))
    print()


_BM_DISPLAY = {
    "aime": "AIME", "gaia": "GAIA", "gpqa": "GPQA",
    "hle": "HLE", "musique": "MuSiQue",
}
_TM_DISPLAY = {
    "none": "No thinking", "all": "All thinking",
    "orchestrator": "Orch.\ thinking", "subagents": "Sub.\ thinking",
}


def _write_latex_thinking_x_benchmark(path: Path, by_bm_th: dict) -> None:
    """Write a booktabs LaTeX table: rows = thinking modes, cols = benchmarks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "l" + "r" * len(BENCHMARKS) + "r"
    bm_headers = " & ".join(f"\\textbf{{{_BM_DISPLAY[b]}}}" for b in BENCHMARKS)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Thinking mode}} & {bm_headers} & \\textbf{{Total}} \\\\",
        "\\midrule",
    ]
    col_totals = {b: 0 for b in BENCHMARKS}
    grand_total = 0
    for v in THINKING_MODES:
        counts = [by_bm_th[b][v]["total"] for b in BENCHMARKS]
        row_total = sum(counts)
        for b, c in zip(BENCHMARKS, counts):
            col_totals[b] += c
        grand_total += row_total
        cells = " & ".join(str(c) for c in counts)
        lines.append(f"{_TM_DISPLAY[v]} & {cells} & {row_total} \\\\")
    lines += [
        "\\midrule",
        "Total & " + " & ".join(str(col_totals[b]) for b in BENCHMARKS) + f" & {grand_total} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")


_FM_DISPLAY = {
    "modality_tool_gap":           "Modality / tool-coverage gap",
    "tool_loop_or_empty_final":    "Tool loop / empty final answer",
    "direct_reasoning_no_action":  "Direct reasoning, no action",
    "computational_subgoal_error": "Computational sub-goal error",
    "retrieval_evidence_failure":  "Retrieval / evidence failure",
    "single_shot_tool_trust":      "Single-shot tool trust",
}


def _write_latex_global(path: Path, result: dict) -> None:
    """Booktabs table: rows = failure modes, cols = benchmarks + Total + Share."""
    path.parent.mkdir(parents=True, exist_ok=True)
    col_spec = "l" + "r" * len(BENCHMARKS) + "rr"
    bm_headers = " & ".join(f"\\textbf{{{_BM_DISPLAY[b]}}}" for b in BENCHMARKS)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Failure mode}} & {bm_headers} & \\textbf{{Total}} & \\textbf{{Share}} \\\\",
        "\\midrule",
    ]
    for mode in FAILURE_MODES:
        fm = result["failure_modes"][mode]
        counts = " & ".join(str(fm["benchmarks"][b]["count"]) for b in BENCHMARKS)
        lines.append(
            f"{_FM_DISPLAY[mode]} & {counts} & {fm['total']} & {fm['share_pct']}\\% \\\\"
        )
    t = result["totals"]
    col_totals = " & ".join(str(t[b]) for b in BENCHMARKS)
    lines += [
        "\\midrule",
        f"Total failures & {col_totals} & {t['overall']} & 100\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")


def _write_latex_by_thinking(path: Path, by_thinking: dict) -> None:
    """Booktabs table: rows = failure modes, cols = thinking modes + Total + Share.

    Counts are summed over all benchmarks for each (mode, thinking) pair.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Short column labels for thinking modes
    tm_col_labels = {
        "none": "No think.", "all": "All think.",
        "orchestrator": "Orch.", "subagents": "Sub.",
    }
    col_spec = "l" + "r" * len(THINKING_MODES) + "rr"
    tm_headers = " & ".join(f"\\textbf{{{tm_col_labels[v]}}}" for v in THINKING_MODES)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\textbf{{Failure mode}} & {tm_headers} & \\textbf{{Total}} & \\textbf{{Share}} \\\\",
        "\\midrule",
    ]
    grand_total = sum(by_thinking[v]["totals"]["overall"] for v in THINKING_MODES)
    for mode in FAILURE_MODES:
        counts = [by_thinking[v]["failure_modes"][mode]["total"] for v in THINKING_MODES]
        row_total = sum(counts)
        share = round(row_total / grand_total * 100, 1) if grand_total else 0.0
        cells = " & ".join(str(c) for c in counts)
        lines.append(f"{_FM_DISPLAY[mode]} & {cells} & {row_total} & {share}\\% \\\\")
    tm_totals = " & ".join(str(by_thinking[v]["totals"]["overall"]) for v in THINKING_MODES)
    lines += [
        "\\midrule",
        f"Total failures & {tm_totals} & {grand_total} & 100\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {path}")


def _print_table_thinking_x_benchmark(by_bm_th: dict) -> None:
    """Print a THINKING_MODE × DATASET summary table of total failures."""
    BM_LABELS = {b: b.upper() for b in BENCHMARKS}
    col_w = [8] + [max(7, len(BM_LABELS[b]) + 1) for b in BENCHMARKS] + [7]
    header = [" "] + [BM_LABELS[b] for b in BENCHMARKS] + ["Total"]
    sep = "─" * (sum(col_w) + 2 * len(col_w))

    def row(*vals):
        parts = [str(v).ljust(col_w[i]) if i == 0 else str(v).rjust(col_w[i])
                 for i, v in enumerate(vals)]
        return "  ".join(parts)

    print(f"\n── Total failures: thinking mode × benchmark " + "─" * 20)
    print()
    print(row(*header))
    print(sep)
    col_totals = {b: 0 for b in BENCHMARKS}
    grand_total = 0
    for v in THINKING_MODES:
        counts = [by_bm_th[b][v]["total"] for b in BENCHMARKS]
        row_total = sum(counts)
        for b, c in zip(BENCHMARKS, counts):
            col_totals[b] += c
        grand_total += row_total
        print(row(THINKING_MODE_LABELS[v], *counts, row_total))
    print(sep)
    print(row("Total", *[col_totals[b] for b in BENCHMARKS], grand_total))
    print()


def _print_table_single_run(result: dict, title: str = "") -> None:
    """Print a single-run breakdown table (failure mode × count only)."""
    col_w = [38, 7, 7, 6]
    header = ["Failure mode", "Count", "Share", "Uniq."]
    sep = "─" * (sum(col_w) + 2 * len(col_w))

    def row(*vals):
        parts = [str(v).ljust(col_w[i]) if i == 0 else str(v).rjust(col_w[i])
                 for i, v in enumerate(vals)]
        return "  ".join(parts)

    if title:
        print(f"\n── {title} " + "─" * max(0, len(sep) - len(title) - 4))
    print()
    print(row(*header))
    print(sep)
    for mode in FAILURE_MODES:
        fm = result["failure_modes"][mode]
        print(row(fm["label"], fm["count"], f"{fm['share_pct']}%", fm["unique_questions"]))
    print(sep)
    print(row("Total failures", result["total"], "100%", "---"))
    print()


def _write_csv_global(path: Path, result: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_mode"] + BENCHMARKS + ["total", "share_pct", "unique_questions"])
        for mode in FAILURE_MODES:
            fm = result["failure_modes"][mode]
            writer.writerow(
                [mode]
                + [fm["benchmarks"][b]["count"] for b in BENCHMARKS]
                + [fm["total"], fm["share_pct"], fm["unique_questions"]]
            )
    print(f"Written: {path}")


def _write_csv_by_thinking(path: Path, by_thinking: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["thinking_mode", "failure_mode"] + BENCHMARKS + ["total", "share_pct", "unique_questions"]
        )
        for v in THINKING_MODES:
            for mode in FAILURE_MODES:
                fm = by_thinking[v]["failure_modes"][mode]
                writer.writerow(
                    [v, mode]
                    + [fm["benchmarks"][b]["count"] for b in BENCHMARKS]
                    + [fm["total"], fm["share_pct"], fm["unique_questions"]]
                )
    print(f"Written: {path}")


def _write_csv_by_benchmark_and_thinking(path: Path, by_bm_th: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "thinking_mode", "failure_mode",
                         "count", "share_pct", "unique_questions"])
        for b in BENCHMARKS:
            for v in THINKING_MODES:
                run = by_bm_th[b][v]
                for mode in FAILURE_MODES:
                    fm = run["failure_modes"][mode]
                    writer.writerow([b, v, mode, fm["count"], fm["share_pct"], fm["unique_questions"]])
    print(f"Written: {path}")


def _write_csv_by_benchmark(path: Path, by_benchmark: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["benchmark", "failure_mode"] + THINKING_MODES + ["total", "share_pct", "unique_questions"]
        )
        for b in BENCHMARKS:
            for mode in FAILURE_MODES:
                fm = by_benchmark[b]["failure_modes"][mode]
                writer.writerow(
                    [b, mode]
                    + [fm["thinking_modes"][v]["count"] for v in THINKING_MODES]
                    + [fm["total"], fm["share_pct"], fm["unique_questions"]]
                )
    print(f"Written: {path}")


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent.parent
    default_out = default_root / "data" / "results" / "failure_modes"

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
