#!/usr/bin/env python3
"""
Generate a LaTeX results table for DeepSeek-R1-Distill experiments on GAIA.

Input:  data/results/wandb/DeepSeek/DS_GAIA_results.csv
Output: data/results/tables/ds_table_gaia.tex

Usage:
    python scripts/tables/ds_table.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/results/wandb/DeepSeek/DS_GAIA_results.csv"
OUT_PATH = ROOT / "data/results/tables/ds_table_gaia.tex"

# ─── row specification ────────────────────────────────────────────────────────
# (model_label, tools_label, thinking_label, exact_Name_value)
# Grouped: DS-R1-7B rows first, then DS-R1-32B, separated by \midrule.
CONFIGS: list[tuple[str, str, str, str, str]] = [
    # model_label      tools_label  thinking_label  name
    ("DS-R1-7B",  "Baseline", "—",           "—",           "DS_baseline_gaia_ds7b_no_tools_none"),
    ("DS-R1-7B",  "Baseline", "—",           "Orchestrator","DS_baseline_gaia_ds7b_no_tools_orchestrator"),
    ("DS-R1-7B",  "Baseline", "Direct",      "—",           "DS_baseline_gaia_ds7b_direct_tools_none"),
    ("DS-R1-7B",  "Baseline", "Direct",      "Orchestrator","DS_baseline_gaia_ds7b_direct_tools_orchestrator"),
    ("DS-R1-7B",  "AgentFlow","Sub-agent",   "—",           "DS_AF_gaia_ds7b_subagent_tools_none"),
    ("DS-R1-7B",  "AgentFlow","Sub-agent",   "Sub-agents",  "DS_AF_gaia_ds7b_subagent_tools_subagents"),
    ("DS-R1-7B",  "AgentFlow","Sub-agent",   "Orchestrator","DS_AF_gaia_ds7b_subagent_tools_orchestrator"),
    ("DS-R1-7B",  "AgentFlow","Sub-agent",   "All",         "DS_AF_gaia_ds7b_subagent_tools_all"),
    ("DS-R1-32B", "Baseline", "—",           "—",           "DS_baseline_gaia_ds32b_no_tools_none"),
    ("DS-R1-32B", "Baseline", "—",           "Orchestrator","DS_baseline_gaia_ds32b_no_tools_orchestrator"),
    ("DS-R1-32B", "Baseline", "Direct",      "—",           "DS_baseline_gaia_ds32b_direct_tools_none"),
    ("DS-R1-32B", "Baseline", "Direct",      "Orchestrator","DS_baseline_gaia_ds32b_direct_tools_orchestrator"),
]


def pct(v: str | float | None) -> float | None:
    if v is None or str(v).strip() in ("", "nan"):
        return None
    try:
        return round(float(v) * 100, 1)
    except ValueError:
        return None


def fmt(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def build_table(rows: dict[str, dict]) -> list[dict]:
    records = []
    for model, system, tools, thinking, name in CONFIGS:
        row_data = rows.get(name)
        if row_data is None:
            acc = l1 = l2 = l3 = None
        else:
            acc = pct(row_data.get("accuracy"))
            l1  = pct(row_data.get("L1_accuracy"))
            l2  = pct(row_data.get("L2_accuracy"))
            l3  = pct(row_data.get("L3_accuracy"))
        records.append({
            "model": model, "system": system,
            "tools": tools, "thinking": thinking,
            "acc": acc, "l1": l1, "l2": l2, "l3": l3,
        })
    return records


def best_per_col(records: list[dict], keys: list[str]) -> dict[str, float | None]:
    result = {}
    for k in keys:
        vals = [r[k] for r in records if r[k] is not None]
        result[k] = max(vals) if vals else None
    return result


def render(records: list[dict]) -> str:
    bests = best_per_col(records, ["acc", "l1", "l2", "l3"])

    lines: list[str] = [
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{System} & \textbf{Tools} & \textbf{Thinking}"
        r" & \textbf{Acc.\,(\%)} & \textbf{L1} & \textbf{L2} & \textbf{L3} \\",
        r"\midrule",
    ]

    prev_model: str | None = None
    prev_system: str | None = None

    for r in records:
        # Model-group separator
        if r["model"] != prev_model:
            if prev_model is not None:
                lines.append(r"\midrule")
            model_str = r["model"]
            prev_model = r["model"]
            prev_system = None
        else:
            model_str = ""

        # Light rule between Baseline and AgentFlow sub-groups
        if r["system"] != prev_system and prev_system is not None:
            lines.append(r"\cmidrule{2-8}")
        prev_system = r["system"]

        cells = [
            model_str,
            r["system"],
            r["tools"],
            r["thinking"],
            fmt(r["acc"], bold=(bests["acc"] is not None and r["acc"] is not None
                                and r["acc"] >= bests["acc"] - 1e-9)),
            fmt(r["l1"],  bold=(bests["l1"]  is not None and r["l1"]  is not None
                                and r["l1"]  >= bests["l1"]  - 1e-9)),
            fmt(r["l2"],  bold=(bests["l2"]  is not None and r["l2"]  is not None
                                and r["l2"]  >= bests["l2"]  - 1e-9)),
            fmt(r["l3"],  bold=(bests["l3"]  is not None and r["l3"]  is not None
                                and r["l3"]  >= bests["l3"]  - 1e-9)),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    if not CSV_PATH.is_file():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        raise SystemExit(1)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows_by_name = {row["Name"]: row for row in csv.DictReader(f)}

    records = build_table(rows_by_name)

    # Print summary
    print(f"{'Name':<52} {'Acc':>6} {'L1':>6} {'L2':>6} {'L3':>6}")
    print("-" * 78)
    for r in records:
        name = f"{r['model']} {r['system']} {r['tools']} {r['thinking']}"
        print(f"{name:<52} {r['acc'] or '—':>6} {r['l1'] or '—':>6}"
              f" {r['l2'] or '—':>6} {r['l3'] or '—':>6}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render(records))
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
