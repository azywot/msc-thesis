"""MAS-vs-baseline overlap and the no-action counterfactual (automatic proxy).

This is a NEW analysis script (it does NOT modify analyze_failure_modes.py).
It re-uses the *exact* automatic classifier and MAS run inventory from
``analyze_failure_modes`` so that every per-mode count here is on the same
automatic-proxy basis as Table 6.1 in the thesis (breakdown_global.csv).

It joins each MAS failed question-run to the four selected single-agent
direct-tools baseline runs for the same benchmark (Qwen3-8B/32B x none/orch),
matched on (benchmark, question_id), and computes:

  * per failure mode: how many MAS failures coincide with >=1 baseline failure
    ("any baseline failed"), and how many occur on a question that >=1 baseline
    solved ("any baseline succeeded" = the recoverable subset);
  * MAS-only successes: MAS-correct question-runs where every baseline failed
    the same question (validation target: 105 runs / 51 unique questions in the
    manual labelling, docs/.../failure_mode.md Part 2);
  * the sharper no-action counterfactual (Task 1): among recoverable
    direct_reasoning_no_action failures, the share whose matched baseline solved
    the question WHILE issuing >=1 tool call (tools specifically helped) vs.
    solving from memory (no tool call).

Baseline tool use is read from ``tool_counts`` because baseline records store an
empty ``action_history`` (the vanilla-LLM-with-tools loop keeps a growing
conversation rather than the compact MAS action history).

Outputs: prints a report and writes
``data/results/failure_modes/baseline_counterfactual.json``.
"""

import json
import sys
from pathlib import Path

# Re-use the frozen classifier + MAS inventory (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_failure_modes import (  # noqa: E402
    BENCHMARKS,
    FAILURE_MODES,
    MODE_LABELS,
    classify_failure,
    load_run,
    run_inventory,
)

# ---------------------------------------------------------------------------
# Baseline inventory: the four selected direct-tools runs per benchmark.
# Paths copied verbatim from docs/failure_mode_and_fine_tuning/failure_mode.md
# (the "earliest completed" run per config), so the join matches the manual doc.
# ---------------------------------------------------------------------------
_NB = "experiments/results/NEW_baseline"
_BASELINE_INVENTORY = [
    # aime
    ("aime", "qwen32B_direct_tools_none",         f"{_NB}/aime/qwen32B_direct_tools_none/train_2026-03-22-17-47-51_21054288/raw_results.json"),
    ("aime", "qwen32B_direct_tools_orchestrator", f"{_NB}/aime/qwen32B_direct_tools_orchestrator/train_2026-03-22-17-48-39_21054301/raw_results.json"),
    ("aime", "qwen8B_direct_tools_none",          f"{_NB}/aime/qwen8B_direct_tools_none/train_2026-03-22-17-51-52_21054323/raw_results.json"),
    ("aime", "qwen8B_direct_tools_orchestrator",  f"{_NB}/aime/qwen8B_direct_tools_orchestrator/train_2026-03-22-17-52-56_21054332/raw_results.json"),
    # gaia
    ("gaia", "qwen32B_direct_tools_none",         f"{_NB}/gaia/qwen32B_direct_tools_none/all_validation_2026-03-21-17-26-52_21028281/raw_results.json"),
    ("gaia", "qwen32B_direct_tools_orchestrator", f"{_NB}/gaia/qwen32B_direct_tools_orchestrator/all_validation_2026-03-21-17-27-24_21028326/raw_results.json"),
    ("gaia", "qwen8B_direct_tools_none",          f"{_NB}/gaia/qwen8B_direct_tools_none/all_validation_2026-03-21-23-34-32_21033545/raw_results.json"),
    ("gaia", "qwen8B_direct_tools_orchestrator",  f"{_NB}/gaia/qwen8B_direct_tools_orchestrator/all_validation_2026-03-21-23-35-26_21033549/raw_results.json"),
    # gpqa
    ("gpqa", "qwen32B_direct_tools_none",         f"{_NB}/gpqa/qwen32B_direct_tools_none/diamond_2026-03-22-17-57-06_21054373/raw_results.json"),
    ("gpqa", "qwen32B_direct_tools_orchestrator", f"{_NB}/gpqa/qwen32B_direct_tools_orchestrator/diamond_2026-03-22-17-58-27_21054380/raw_results.json"),
    ("gpqa", "qwen8B_direct_tools_none",          f"{_NB}/gpqa/qwen8B_direct_tools_none/diamond_2026-03-22-18-01-41_21054411/raw_results.json"),
    ("gpqa", "qwen8B_direct_tools_orchestrator",  f"{_NB}/gpqa/qwen8B_direct_tools_orchestrator/diamond_2026-03-22-18-02-39_21054421/raw_results.json"),
    # hle
    ("hle",  "qwen32B_direct_tools_none",         f"{_NB}/hle/qwen32B_direct_tools_none/test_subset_200_2026-03-22-18-32-27_21054871/raw_results.json"),
    ("hle",  "qwen32B_direct_tools_orchestrator", f"{_NB}/hle/qwen32B_direct_tools_orchestrator/test_subset_200_2026-03-22-18-33-34_21054917/raw_results.json"),
    ("hle",  "qwen8B_direct_tools_none",          f"{_NB}/hle/qwen8B_direct_tools_none/test_subset_200_2026-03-22-18-37-00_21054958/raw_results.json"),
    ("hle",  "qwen8B_direct_tools_orchestrator",  f"{_NB}/hle/qwen8B_direct_tools_orchestrator/test_subset_200_2026-03-22-18-38-06_21054966/raw_results.json"),
    # musique
    ("musique", "qwen32B_direct_tools_none",         f"{_NB}/musique/qwen32B_direct_tools_none/validation_subset_200_2026-03-22-18-10-45_21054468/raw_results.json"),
    ("musique", "qwen32B_direct_tools_orchestrator", f"{_NB}/musique/qwen32B_direct_tools_orchestrator/validation_subset_200_2026-03-22-18-11-41_21054491/raw_results.json"),
    ("musique", "qwen8B_direct_tools_none",          f"{_NB}/musique/qwen8B_direct_tools_none/validation_subset_200_2026-03-22-18-15-02_21054539/raw_results.json"),
    ("musique", "qwen8B_direct_tools_orchestrator",  f"{_NB}/musique/qwen8B_direct_tools_orchestrator/validation_subset_200_2026-03-22-18-16-12_21054560/raw_results.json"),
]


def _record_used_tool(rec: dict) -> bool:
    """True if the run issued >=1 tool call. Baseline runs keep counts in
    tool_counts (action_history is empty for the vanilla LLM-with-tools loop)."""
    tc = rec.get("tool_counts") or {}
    if any((v or 0) > 0 for v in tc.values()):
        return True
    # Fallback for any record that only populated action_history.
    return bool(rec.get("action_history"))


def build_baseline_index(root: Path):
    """Return {benchmark: {qid: [(correct, used_tool), ...]}} over the 4
    baseline runs per benchmark."""
    idx = {b: {} for b in BENCHMARKS}
    for bench, _variant, rel in _BASELINE_INVENTORY:
        recs = load_run(root / rel)
        for rec in recs:
            qid = rec.get("question_id")
            idx[bench].setdefault(qid, []).append(
                (bool(rec.get("correct")), _record_used_tool(rec))
            )
    return idx


def analyze(root: Path, out_dir: Path) -> dict:
    baseline = build_baseline_index(root)

    # Per-mode accumulators over MAS failures.
    per_mode = {
        m: {
            "mas_failures": 0,
            "matched": 0,            # qid present in baseline index
            "any_baseline_failed": 0,
            "any_baseline_succeeded": 0,  # recoverable subset
        }
        for m in FAILURE_MODES
    }

    # No-action counterfactual accumulators.
    na_recoverable = 0          # direct_reasoning_no_action with any baseline success
    na_solved_with_tool = 0     # ... where >=1 successful baseline used a tool
    na_solved_memory_only = 0   # ... where every successful baseline used NO tool

    # MAS-only successes: MAS correct & every baseline for that qid failed.
    mas_only_success_runs = 0
    mas_only_success_unique = set()
    mas_correct_total = 0

    for bench, _variant, path in run_inventory(root):
        for rec in load_run(path):
            qid = rec.get("question_id")
            base_runs = baseline[bench].get(qid, [])
            any_b_correct = any(c for c, _ in base_runs)
            any_b_failed = any(not c for c, _ in base_runs)

            if rec.get("correct"):
                mas_correct_total += 1
                # MAS-only success: at least one baseline matched AND none correct
                if base_runs and not any_b_correct:
                    mas_only_success_runs += 1
                    mas_only_success_unique.add((bench, qid))
                continue

            # Failed MAS run -> classify on the frozen automatic basis.
            mode = classify_failure(rec)
            pm = per_mode[mode]
            pm["mas_failures"] += 1
            if base_runs:
                pm["matched"] += 1
                if any_b_failed:
                    pm["any_baseline_failed"] += 1
                if any_b_correct:
                    pm["any_baseline_succeeded"] += 1

            if mode == "direct_reasoning_no_action" and any_b_correct:
                na_recoverable += 1
                # Among the *successful* baseline runs, did any use a tool?
                if any(used for c, used in base_runs if c):
                    na_solved_with_tool += 1
                else:
                    na_solved_memory_only += 1

    # ── Assemble report ───────────────────────────────────────────────────────
    def pct(n, d):
        return round(100.0 * n / d, 1) if d else 0.0

    modes_out = {}
    for m in FAILURE_MODES:
        pm = per_mode[m]
        tot = pm["mas_failures"]
        modes_out[m] = {
            "label": MODE_LABELS[m],
            "mas_failures": tot,
            "matched": pm["matched"],
            "any_baseline_failed": pm["any_baseline_failed"],
            "any_baseline_failed_pct": pct(pm["any_baseline_failed"], tot),
            "any_baseline_succeeded": pm["any_baseline_succeeded"],
            "any_baseline_succeeded_pct": pct(pm["any_baseline_succeeded"], tot),
        }

    result = {
        "basis": "automatic proxy (analyze_failure_modes.classify_failure); "
                 "same basis as Table 6.1 / breakdown_global.csv",
        "per_mode": modes_out,
        "mas_only_successes": {
            "runs": mas_only_success_runs,
            "unique_questions": len(mas_only_success_unique),
            "mas_correct_total": mas_correct_total,
        },
        "no_action_counterfactual": {
            "recoverable": na_recoverable,
            "solved_with_tool": na_solved_with_tool,
            "solved_memory_only": na_solved_memory_only,
            "share_solved_with_tool_pct": pct(na_solved_with_tool, na_recoverable),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_counterfactual.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ── Console ────────────────────────────────────────────────────────────────
    print(f"Basis: {result['basis']}\n")
    hdr = f"{'Failure mode':35s} {'MASfail':>8s} {'anyBfail':>9s} {'anyBsucc(recov)':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for m in FAILURE_MODES:
        o = modes_out[m]
        print(f"{o['label']:35s} {o['mas_failures']:8d} "
              f"{o['any_baseline_failed']:4d} ({o['any_baseline_failed_pct']:4.1f}%) "
              f"{o['any_baseline_succeeded']:4d} ({o['any_baseline_succeeded_pct']:4.1f}%)")
    total_fail = sum(modes_out[m]["mas_failures"] for m in FAILURE_MODES)
    min_anyfail = min(modes_out[m]["any_baseline_failed_pct"] for m in FAILURE_MODES)
    print("-" * len(hdr))
    print(f"Total MAS failures: {total_fail}  |  min any-baseline-failed across modes: {min_anyfail}%")
    print()
    mo = result["mas_only_successes"]
    print(f"MAS-only successes: {mo['runs']} question-runs / {mo['unique_questions']} unique "
          f"(of {mo['mas_correct_total']} MAS-correct runs)")
    print()
    nc = result["no_action_counterfactual"]
    print("No-action counterfactual (direct_reasoning_no_action):")
    print(f"  recoverable (>=1 baseline solved same q): {nc['recoverable']}")
    print(f"  ... solved by a baseline that USED a tool: {nc['solved_with_tool']} "
          f"({nc['share_solved_with_tool_pct']}%)")
    print(f"  ... solved by a baseline from memory only: {nc['solved_memory_only']}")
    print(f"\nWritten: {out_path}")
    return result


def main():
    root = Path(__file__).resolve().parent.parent.parent
    out_dir = root / "data" / "results" / "failure_modes"
    analyze(root, out_dir)


if __name__ == "__main__":
    main()
