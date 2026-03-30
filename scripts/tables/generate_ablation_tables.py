#!/usr/bin/env python3
"""
Generate LaTeX ablation tables for tool and structured-memory ablations.

For each dataset, each table contains two groups:
  1. Full system – all four thinking modes (none / sub-agents / orchestrator / all)
     from AF_no_img_no_mm_{ds}_qwen8B_subagent_tools_{***}
  2. Ablation rows – orchestrator-only thinking only

The reference for Δ is always the full system with orchestrator-only thinking.

Outputs (all datasets concatenated in one file each):
    data/results/ablations/tools.txt
    data/results/ablations/structured_memory.txt

Usage:
    python scripts/tables/generate_ablation_tables.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/results/ablations"

DATASETS = ["gaia", "aime", "gpqa", "hle", "musique"]
DATASET_LABELS = {
    "gaia":    "GAIA",
    "aime":    "AIME",
    "gpqa":    "GPQA",
    "hle":     "HLE",
    "musique": "MuSiQue",
}

# Ordered thinking modes for display
THINKING_MODES = [
    ("none",         "None"),
    ("subagents",    "Sub-agents only"),
    ("orchestrator", "Orchestrator only"),
    ("all",          "All"),
]

# Tool ablations: (display label, name suffix)
TOOL_ABLATIONS = [
    ("Web Search",      "no_web_search"),
    ("Text Inspector",  "no_text_inspector"),
    ("Code Generator",  "no_code_generator"),
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["Name"]] = row
    return rows


def pct(val: str | float) -> float:
    return round(float(val) * 100, 1)


def delta_str(val: float, ref: float) -> str:
    d = round(val - ref, 1)
    return f"+{d}" if d >= 0 else str(d)


def full_key(ds: str, thinking_suffix: str) -> str:
    return f"AF_no_img_no_mm_{ds}_qwen8B_subagent_tools_{thinking_suffix}"


def tool_ablation_key(ds: str, tool_suffix: str) -> str:
    return f"AF_subagent_orch_{ds}_qwen8B_subagent_orch_{tool_suffix}"


def struct_mem_key(ds: str) -> str:
    return f"AF_struct_mem_ablation_{ds}_qwen8B_subagent_orch_baseline_chat"


def full_system_rows(ds: str, rows: dict[str, dict], ref_acc: float) -> list[str]:
    """Render the 'Full system (all tools)' block."""
    lines: list[str] = [r"\multicolumn{4}{l}{\textit{Full system (all tools)}} \\"]
    for suffix, label in THINKING_MODES:
        key = full_key(ds, suffix)
        if key not in rows:
            continue
        acc = pct(rows[key]["accuracy"])
        d   = "—" if suffix == "orchestrator" else delta_str(acc, ref_acc)
        if suffix == "orchestrator":
            lines.append(
                f" & \\textbf{{{label}}} & \\textbf{{{acc:.1f}}} & \\textbf{{{d}}} \\\\"
            )
        else:
            lines.append(f" & {label} & {acc:.1f} & {d} \\\\")
    return lines


# ─── tools ablation ──────────────────────────────────────────────────────────

# Whether each dataset has text-inspector enabled in the full system
_TEXT_INSPECTOR_DATASETS = {"gaia", "hle"}

CMARK = r"\cmark"
XMARK = r"\xmark"
DASH  = r"$-$"

# Tool columns: (display name, tool_suffix for ablation key, CSV field to check)
# Order: Web Search, Code Generator, Text Inspector
_TOOL_COLS = [
    ("Web",  "no_web_search",      "enable_search_tool"),
    ("Code", "no_code_generator",  "enable_code_tool"),
    ("File", "no_text_inspector",  "enable_text_inspector_tool"),
]


def _tool_cell(ds: str, ablated_suffix: str | None, col_suffix: str) -> str:
    r"""Return \cmark, \xmark, or $-$ for one tool column in one data row.

    ablated_suffix -- the tool being removed in this row (None = full-system row).
    col_suffix     -- the ablation suffix that corresponds to this column's tool.
    """
    if ds not in _TEXT_INSPECTOR_DATASETS and col_suffix == "no_text_inspector":
        return DASH
    if ablated_suffix == col_suffix:
        return XMARK
    return CMARK


def tools_table(ds: str, label: str, rows: dict[str, dict]) -> str | None:
    ref_key = full_key(ds, "orchestrator")
    if ref_key not in rows:
        return None
    ref_acc = pct(rows[ref_key]["accuracy"])

    # Column header names
    col_names = [name for name, *_ in _TOOL_COLS]
    # Number of data columns: 3 tool cols + Thinking + Avg + Δ = 6
    N = 6

    lines: list[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{",
        f"  Tool ablation on {label} (Qwen3-8B, sub-agent tools).",
        r"  The upper block shows the full system under each thinking mode;",
        r"  the lower block performs leave-one-out tool ablations with orchestrator-only thinking.",
        r"  $\Delta$ is relative to the full system with orchestrator-only thinking (bold).",
        r"  A dash (${-}$) indicates the tool was not used for this dataset.",
        r"}",
        f"\\label{{tab:tool_ablation_{ds}}}",
        r"\vspace{0.5em}",
        r"\begin{tabular}{ccc l rr}",
        r"\toprule",
    ]

    # Column headers
    header_tools = " & ".join(f"\\textbf{{{n}}}" for n in col_names)
    lines.append(
        f"{header_tools} & \\textbf{{Thinking}} & \\textbf{{Avg.\\,(\\%)}} & \\textbf{{$\\Delta$}} \\\\"
    )
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{N}}}{{l}}{{\\textit{{Full system (all tools)}}}} \\\\"
    )

    # Full-system rows: all tools ✓, varying thinking mode
    for suffix, th_label in THINKING_MODES:
        key = full_key(ds, suffix)
        if key not in rows:
            continue
        acc = pct(rows[key]["accuracy"])
        d   = "—" if suffix == "orchestrator" else delta_str(acc, ref_acc)
        tools_cells = " & ".join(
            _tool_cell(ds, None, col_suffix)
            for _, col_suffix, _ in _TOOL_COLS
        )
        if suffix == "orchestrator":
            lines.append(
                f"{tools_cells} & \\textbf{{{th_label}}} & \\textbf{{{acc:.1f}}} & \\textbf{{{d}}} \\\\"
            )
        else:
            lines.append(f"{tools_cells} & {th_label} & {acc:.1f} & {d} \\\\")

    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{{N}}}{{l}}{{\\textit{{Leave-one-out ablations (orchestrator-only thinking)}}}} \\\\"
    )

    # Ablation rows: one tool ✗ at a time
    has_any = False
    for _, col_suffix, _ in _TOOL_COLS:
        abl_key = tool_ablation_key(ds, col_suffix)
        if abl_key not in rows:
            continue
        has_any = True
        acc = pct(rows[abl_key]["accuracy"])
        d   = delta_str(acc, ref_acc)
        tools_cells = " & ".join(
            _tool_cell(ds, col_suffix, c_suffix)
            for _, c_suffix, _ in _TOOL_COLS
        )
        lines.append(f"{tools_cells} & Orchestrator only & {acc:.1f} & {d} \\\\")

    if not has_any:
        lines.append(
            f"\\multicolumn{{{N}}}{{l}}{{\\quad\\textit{{(no ablation data for this dataset)}}}} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── structured-memory ablation ──────────────────────────────────────────────

def structured_memory_table(ds: str, label: str, rows: dict[str, dict]) -> str | None:
    ref_key = full_key(ds, "orchestrator")
    if ref_key not in rows:
        return None
    ref_acc = pct(rows[ref_key]["accuracy"])

    sm_key = struct_mem_key(ds)

    lines: list[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{",
        f"  Structured-memory ablation on {label} (Qwen3-8B, sub-agent tools).",
        r"  The upper block shows the full system (with query analysis and structured memory)",
        r"  under each thinking mode; the lower block replaces both components with a plain",
        r"  chat baseline, evaluated with orchestrator-only thinking.",
        r"  $\Delta$ is relative to the full system with orchestrator-only thinking (bold).",
        r"}",
        f"\\label{{tab:struct_mem_ablation_{ds}}}",
        r"\vspace{0.5em}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Setup} & \textbf{Thinking} & \textbf{Avg.\ (\%)} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]

    lines += full_system_rows(ds, rows, ref_acc)
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{l}{\textit{Ablation: no structured memory (orchestrator-only thinking)}} \\"
    )

    if sm_key in rows:
        acc = pct(rows[sm_key]["accuracy"])
        d   = delta_str(acc, ref_acc)
        lines.append(
            f"w/o Structured memory & Orchestrator only & {acc:.1f} & {d} \\\\"
        )
    else:
        lines.append(
            r"\multicolumn{4}{l}{\quad\textit{(no ablation data for this dataset)}} \\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, dict[str, dict]] = {}
    for ds in DATASETS:
        csv_path = ROOT / f"data/results/all_results_{ds}.csv"
        all_rows[ds] = read_csv(csv_path)

    # Tools ablation — one table per dataset
    tools_parts: list[str] = []
    for ds in DATASETS:
        t = tools_table(ds, DATASET_LABELS[ds], all_rows[ds])
        if t:
            if tools_parts:
                tools_parts.append("")
            label = DATASET_LABELS[ds]
            sep = f"% {'─' * (len(label) + 2)} {label} {'─' * (len(label) + 2)}"
            tools_parts.append(sep)
            tools_parts.append(t)
    (OUT_DIR / "tools.txt").write_text("\n".join(tools_parts) + "\n")
    print(f"Wrote {OUT_DIR / 'tools.txt'}")

    # Structured-memory ablation — one table per dataset
    sm_parts: list[str] = []
    for ds in DATASETS:
        t = structured_memory_table(ds, DATASET_LABELS[ds], all_rows[ds])
        if t:
            if sm_parts:
                sm_parts.append("")
            label = DATASET_LABELS[ds]
            sep = f"% {'─' * (len(label) + 2)} {label} {'─' * (len(label) + 2)}"
            sm_parts.append(sep)
            sm_parts.append(t)
    (OUT_DIR / "structured_memory.txt").write_text("\n".join(sm_parts) + "\n")
    print(f"Wrote {OUT_DIR / 'structured_memory.txt'}")


if __name__ == "__main__":
    main()
