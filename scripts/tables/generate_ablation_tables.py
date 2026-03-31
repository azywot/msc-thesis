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
    """Render the 'Full system' reference row (orchestrator-only thinking)."""
    key = full_key(ds, "orchestrator")
    if key not in rows:
        return []
    acc = pct(rows[key]["accuracy"])
    return [
        r"\multicolumn{4}{l}{\textit{Full system (orchestrator-only thinking)}} \\",
        f" & Orchestrator only & \\textbf{{{acc:.1f}}} & \\textbf{{—}} \\\\",
    ]


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


_NO_TOOLS = "no_tools"  # sentinel for the all-tools-disabled row


def _tool_cell(ds: str, ablated_suffix: str | None, col_suffix: str) -> str:
    r"""Return \cmark, \xmark, or $-$ for one tool column in one data row.

    ablated_suffix -- tool being removed (None = full system, _NO_TOOLS = all off).
    col_suffix     -- ablation suffix that corresponds to this column's tool.
    """
    if ds not in _TEXT_INSPECTOR_DATASETS and col_suffix == "no_text_inspector":
        return DASH
    if ablated_suffix == _NO_TOOLS or ablated_suffix == col_suffix:
        return XMARK
    return CMARK


def _no_tools_key(ds: str, rows: dict[str, dict]) -> str | None:
    """Return the best-matching no-tools orchestrator key for a dataset."""
    for prefix in (f"AF_no_img_no_mm_{ds}", f"NEW_baseline_{ds}"):
        k = f"{prefix}_qwen8B_no_tools_orchestrator"
        if k in rows:
            return k
    return None


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
        r"  The upper block shows the full system with orchestrator-only thinking (reference);",
        r"  the lower block performs leave-one-out tool ablations, also with orchestrator-only thinking.",
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
        f"\\multicolumn{{{N}}}{{l}}{{\\textit{{Full system (orchestrator-only thinking)}}}} \\\\"
    )

    # Full-system reference row: all tools ✓, orchestrator-only thinking
    tools_cells = " & ".join(
        _tool_cell(ds, None, col_suffix) for _, col_suffix, _ in _TOOL_COLS
    )
    lines.append(
        f"{tools_cells} & Orchestrator only & \\textbf{{{ref_acc:.1f}}} & \\textbf{{—}} \\\\"
    )

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


# ─── shared helper ───────────────────────────────────────────────────────────

def _fmt(val: float | None, is_best: bool) -> str:
    """Format an accuracy cell, bolding the best value per column."""
    if val is None:
        return "—"
    s = f"{val:.1f}"
    return f"\\textbf{{{s}}}" if is_best else s


def _best_per_col(data: list[list[float | None]]) -> list[float | None]:
    """Return the max (ignoring None) for each column across all rows."""
    n_cols = len(data[0])
    bests: list[float | None] = []
    for c in range(n_cols):
        vals = [row[c] for row in data if row[c] is not None]
        bests.append(max(vals) if vals else None)
    return bests


def _combined_table(
    caption_lines: list[str],
    label: str,
    ds_list: list[str],
    row_specs: list[tuple[str, list[float | None]]],
    ref_accs: list[float],
    show_delta: bool = True,
) -> str:
    """Render a compact all-datasets table.

    row_specs  -- [(row_label, [val_ds0, val_ds1, ...]), ...]
    ref_accs   -- reference accuracy per dataset column for the Delta row
    show_delta -- whether to append a midrule + Delta row at the bottom
    """
    ds_labels = [DATASET_LABELS[ds] for ds in ds_list]
    col_spec  = "l " + " ".join(["r"] * len(ds_list))

    # Determine per-column best across all data rows
    bests = _best_per_col([vals for _, vals in row_specs])

    lines: list[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{",
        *[f"  {l}" for l in caption_lines],
        r"}",
        f"\\label{{{label}}}",
        r"\vspace{0.5em}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Setup} & "
        + " & ".join(f"\\textbf{{{lbl}}}" for lbl in ds_labels)
        + r" \\",
        r"\midrule",
    ]

    for row_label, vals in row_specs:
        cells = [
            _fmt(v, bests[c] is not None and v is not None and v >= bests[c] - 1e-9)
            for c, v in enumerate(vals)
        ]
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    if show_delta:
        lines.append(r"\midrule")
        last_vals = row_specs[-1][1]
        delta_cells = [
            "—" if last_vals[c] is None else delta_str(last_vals[c], ref_accs[c])
            for c in range(len(ds_list))
        ]
        lines.append(r"$\Delta$ & " + " & ".join(delta_cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── structured-memory ablation ──────────────────────────────────────────────

def structured_memory_combined_table(all_rows: dict[str, dict[str, dict]]) -> str:
    ds_list  = [ds for ds in DATASETS if full_key(ds, "orchestrator") in all_rows[ds]]
    ref_accs = [pct(all_rows[ds][full_key(ds, "orchestrator")]["accuracy"]) for ds in ds_list]

    sm_vals: list[float | None] = []
    for ds in ds_list:
        k = struct_mem_key(ds)
        sm_vals.append(pct(all_rows[ds][k]["accuracy"]) if k in all_rows[ds] else None)

    row_specs = [
        ("Full system",          ref_accs),
        ("w/o Structured memory", sm_vals),
    ]
    return _combined_table(
        caption_lines=[
            "Structured-memory ablation across all datasets (Qwen3-8B, sub-agent tools,",
            "orchestrator-only thinking). The reference row uses the full system with query",
            "analysis and structured memory; the ablation row replaces both with a plain chat",
            "baseline. Bold denotes the better result per dataset.",
            "$\\Delta$ is relative to the full system.",
        ],
        label="tab:struct_mem_ablation",
        ds_list=ds_list,
        row_specs=row_specs,
        ref_accs=ref_accs,
        show_delta=True,
    )


# ─── tools ablation (combined) ───────────────────────────────────────────────

def tools_combined_table(all_rows: dict[str, dict[str, dict]]) -> str:
    """Combined table: tool-tick columns (Web/Code/File) + dataset accuracy columns."""
    ds_list   = [ds for ds in DATASETS if full_key(ds, "orchestrator") in all_rows[ds]]
    ds_labels = [DATASET_LABELS[ds] for ds in ds_list]

    ref_accs: list[float] = [
        pct(all_rows[ds][full_key(ds, "orchestrator")]["accuracy"]) for ds in ds_list
    ]

    # ── collect all data rows first so we can compute bests in one pass ───────
    # Each entry: (ablated_suffix, vals, add_midrule_before)
    data_rows: list[tuple[str | None, list[float | None], bool]] = [
        (None, list(ref_accs), False),
    ]
    for _, col_suffix, _ in _TOOL_COLS:
        vals: list[float | None] = [
            pct(all_rows[ds][k]["accuracy"])
            if (k := tool_ablation_key(ds, col_suffix)) in all_rows[ds] else None
            for ds in ds_list
        ]
        if any(v is not None for v in vals):
            data_rows.append((col_suffix, vals, False))

    no_tools_vals: list[float | None] = [
        pct(all_rows[ds][k]["accuracy"])
        if (k := _no_tools_key(ds, all_rows[ds])) is not None else None
        for ds in ds_list
    ]
    if any(v is not None for v in no_tools_vals):
        data_rows.append((_NO_TOOLS, no_tools_vals, True))   # midrule before

    bests = _best_per_col([vals for _, vals, _ in data_rows])

    def acc_row(ablated_suffix: str | None, vals: list[float | None]) -> str:
        tick_cells = " & ".join(
            _tool_cell(ds_list[0], ablated_suffix, col_suffix)
            for _, col_suffix, _ in _TOOL_COLS
        )
        acc_cells = " & ".join(
            _fmt(v, bests[c] is not None and v is not None and v >= bests[c] - 1e-9)
            for c, v in enumerate(vals)
        )
        return f"{tick_cells} & {acc_cells} \\\\"

    # ── build LaTeX ───────────────────────────────────────────────────────────
    col_spec = "ccc|" + "r" * len(ds_list)
    lines: list[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{",
        r"  Tool ablation across all datasets (Qwen3-8B, sub-agent tools,",
        r"  orchestrator-only thinking). Each row removes one tool from the full system;",
        r"  the last row disables all tools (no-tools baseline).",
        r"  A dash ($-$) indicates the tool was not used for that dataset.",
        r"  Bold denotes the best result per dataset.",
        r"}",
        r"\label{tab:tool_ablation}",
        r"\vspace{0.5em}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\multicolumn{3}{c|}{\textbf{Tools}} & "
        + f"\\multicolumn{{{len(ds_list)}}}{{c}}{{\\textbf{{Accuracy (\\%)}}}} \\\\",
        r"\textbf{Web} & \multirow{2}{*}{\textbf{Coder}} & \textbf{File} & "
        + " & ".join(f"\\multirow{{2}}{{*}}{{\\textbf{{{lbl}}}}}" for lbl in ds_labels)
        + r" \\",
        r"\textbf{Searcher} & & \textbf{Inspector} & "
        + " & ".join("" for _ in ds_labels)
        + r" \\",
        r"\midrule",
    ]

    for ablated_suffix, vals, add_midrule in data_rows:
        if add_midrule:
            lines.append(r"\midrule")
        lines.append(acc_row(ablated_suffix, vals))

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, dict[str, dict]] = {}
    for ds in DATASETS:
        csv_path = ROOT / f"data/results/all_results_{ds}.csv"
        all_rows[ds] = read_csv(csv_path)

    # Tools ablation — single combined table across all datasets
    (OUT_DIR / "tools.txt").write_text(tools_combined_table(all_rows) + "\n")
    print(f"Wrote {OUT_DIR / 'tools.txt'}")

    # Structured-memory ablation — single combined table across all datasets
    sm_table = structured_memory_combined_table(all_rows)
    (OUT_DIR / "structured_memory.txt").write_text(sm_table + "\n")
    print(f"Wrote {OUT_DIR / 'structured_memory.txt'}")


if __name__ == "__main__":
    main()
