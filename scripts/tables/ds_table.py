#!/usr/bin/env python3
"""
Generate a LaTeX results table for DeepSeek-R1-Distill experiments on GAIA.

Input:  data/results/wandb/DeepSeek/DS_GAIA_results.csv
Output: data/results/tables/ds_table_gaia.tex

Tool columns available in the CSV:
  tool/total_tool_calls, tool/text_inspector_total
  (web search + code generator are not logged separately; derived as total - text_inspector)

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
CONFIGS: list[tuple[str, str, str, str, str]] = [
    # model_label   system       tools         thinking        exact_Name
    ("DS-R1-7B",  "Baseline",  "—",          "—",           "DS_baseline_gaia_ds7b_no_tools_none"),
    ("DS-R1-7B",  "Baseline",  "—",          "Orchestrator","DS_baseline_gaia_ds7b_no_tools_orchestrator"),
    ("DS-R1-7B",  "Baseline",  "Direct",     "—",           "DS_baseline_gaia_ds7b_direct_tools_none"),
    ("DS-R1-7B",  "Baseline",  "Direct",     "Orchestrator","DS_baseline_gaia_ds7b_direct_tools_orchestrator"),
    ("DS-R1-7B",  "AgentFlow", "Sub-agent",  "—",           "DS_AF_gaia_ds7b_subagent_tools_none"),
    ("DS-R1-7B",  "AgentFlow", "Sub-agent",  "Sub-agents",  "DS_AF_gaia_ds7b_subagent_tools_subagents"),
    ("DS-R1-7B",  "AgentFlow", "Sub-agent",  "Orchestrator","DS_AF_gaia_ds7b_subagent_tools_orchestrator"),
    ("DS-R1-7B",  "AgentFlow", "Sub-agent",  "All",         "DS_AF_gaia_ds7b_subagent_tools_all"),
    ("DS-R1-32B", "Baseline",  "—",          "—",           "DS_baseline_gaia_ds32b_no_tools_none"),
    ("DS-R1-32B", "Baseline",  "—",          "Orchestrator","DS_baseline_gaia_ds32b_no_tools_orchestrator"),
    ("DS-R1-32B", "Baseline",  "Direct",     "—",           "DS_baseline_gaia_ds32b_direct_tools_none"),
    ("DS-R1-32B", "Baseline",  "Direct",     "Orchestrator","DS_baseline_gaia_ds32b_direct_tools_orchestrator"),
]


def _pct(v) -> float | None:
    if v is None or str(v).strip() in ("", "nan"):
        return None
    try:
        return round(float(v) * 100, 1)
    except ValueError:
        return None


def _int(v) -> int | None:
    if v is None or str(v).strip() in ("", "nan"):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def fmt_f(v: float | None, bold: bool = False) -> str:
    if v is None:
        return "—"
    s = f"{v:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_i(v: int | None, bold: bool = False) -> str:
    if v is None:
        return "—"
    s = str(v)
    return f"\\textbf{{{s}}}" if bold else s


def build_table(rows: dict[str, dict]) -> list[dict]:
    records = []
    for model, system, tools, thinking, name in CONFIGS:
        rd = rows.get(name)
        if rd is None:
            acc = total = search = code = text_insp = None
        else:
            acc = _pct(rd.get("accuracy"))
            if tools == "—":   # no-tools run — counts are meaningless zeros
                total = search = code = text_insp = None
            else:
                total     = _int(rd.get("tool/total_tool_calls"))
                search    = _int(rd.get("tool/search_total"))
                code      = _int(rd.get("tool/code_total"))
                text_insp = _int(rd.get("tool/text_inspector_total"))
        records.append({
            "model": model, "system": system,
            "tools": tools, "thinking": thinking,
            "acc": acc, "total": total,
            "search": search, "code": code, "text_insp": text_insp,
        })
    return records


def best_per_col(records, keys):
    return {
        k: (max(r[k] for r in records if r[k] is not None) or None)
        for k in keys
    }


def render(records: list[dict]) -> str:
    bests_f = best_per_col(records, ["acc"])
    bests_i = best_per_col(records, ["total", "search", "code", "text_insp"])

    def is_best_f(r, k):
        return bests_f[k] is not None and r[k] is not None and r[k] >= bests_f[k] - 1e-9

    def is_best_i(r, k):
        return bests_i[k] is not None and r[k] is not None and r[k] >= bests_i[k] - 0.5

    lines: list[str] = [
        r"\begin{tabular}{lllr|rrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Tools} & \textbf{Thinking}"
        r" & \textbf{Acc.\,(\%)} & \textbf{\shortstack{Total\\Tool Calls}}"
        r" & \textbf{\shortstack{Web\\Searcher}}"
        r" & \textbf{Coder}"
        r" & \textbf{\shortstack{File\\Inspector}} \\",
        r"\midrule",
    ]

    prev_model: str | None = None
    prev_system: str | None = None

    for r in records:
        if r["model"] != prev_model:
            if prev_model is not None:
                lines.append(r"\midrule")
            model_str  = r["model"]
            prev_model = r["model"]
            prev_system = None
        else:
            model_str = ""

        if r["system"] != prev_system and prev_system is not None:
            lines.append(r"\cmidrule{2-8}")
        prev_system = r["system"]

        cells = [
            model_str,
            r["tools"],
            r["thinking"],
            fmt_f(r["acc"],       bold=is_best_f(r, "acc")),
            fmt_i(r["total"],     bold=is_best_i(r, "total")),
            fmt_i(r["search"],    bold=is_best_i(r, "search")),
            fmt_i(r["code"],      bold=is_best_i(r, "code")),
            fmt_i(r["text_insp"], bold=is_best_i(r, "text_insp")),
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

    print(f"{'Name':<52} {'Acc':>6} {'Total':>6} {'Search':>7} {'Code':>6} {'FileInsp':>9}")
    print("-" * 90)
    for r in records:
        name = f"{r['model']} {r['system']} {r['tools']} {r['thinking']}"
        print(f"{name:<52} {str(r['acc'] or '—'):>6} {str(r['total'] or '—'):>6}"
              f" {str(r['search'] or '—'):>7} {str(r['code'] or '—'):>6}"
              f" {str(r['text_insp'] or '—'):>9}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render(records))
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
