#!/usr/bin/env python3
"""
Orchestrator capacity – grouped bar chart (per dataset).

Auto-discovers datasets from data/results/wandb/orchestrator_capabilities/
by matching files named ``{DATASET}_orchestrator_capabilities.csv``.

For each dataset the script writes:
    data/results/plots/orchestrator_capabilities/<dataset>/figure.png
    data/results/tables/orchestrator_capabilities_<dataset>.tex

Bars are grouped first by orchestrator size (8B / 32B), then by thinking
mode within each group.  Bar colour indicates sub-agent size (1.7B / 8B / 32B).
"""
from __future__ import annotations

import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
OC_CSV_DIR  = ROOT / "data/results/wandb/orchestrator_capabilities"
PLOTS_DIR   = ROOT / "data/results/plots/orchestrator_capabilities"
TABLES_DIR  = ROOT / "data/results/tables"

# ─── display config ───────────────────────────────────────────────────────────
THINKING_MODES = [
    ("NO",                "No Think"),
    ("ORCHESTRATOR_ONLY", "Orch. Think"),
    ("ALL",               "All Think"),
]
THINKING_KEYS = [k for k, _ in THINKING_MODES]
THINKING_LBLS = {k: lbl for k, lbl in THINKING_MODES}

SUB_SIZES  = ["1.7B", "8B", "32B"]
SUB_COLORS = {"1.7B": "#4C72B0", "8B": "#DD8452", "32B": "#55A868"}
SUB_LABELS = {"1.7B": "1.7B",  "8B": "8B",      "32B": "32B"}

ORCH_GROUPS = [("8B", "8B Orchestrator"), ("32B", "32B Orchestrator")]

BAR_W   = 0.18
SUB_GAP = 0.12
GRP_GAP = 0.55


# ─── per-dataset data helpers ─────────────────────────────────────────────────

def parse_config(name: str, dataset: str) -> tuple[str, str] | tuple[None, None]:
    """Return (orch_size, sub_size) from experiment name, or (None, None)."""
    ds = dataset.lower()
    # AF baseline → treated as 8B orchestrator / 8B sub-agent
    if f"AF_no_img_no_mm_{ds}_qwen8B" in name:
        return "8B", "8B"
    if not name.startswith(f"OC_{ds}_"):
        return None, None
    orch = "8B" if "orch8b" in name else "32B"
    if   "sub1_7b" in name: sub = "1.7B"
    elif "sub8b"   in name: sub = "8B"
    elif "sub32b"  in name: sub = "32B"
    else: return None, None
    return orch, sub


def build_data(rows: list[dict[str, str]], dataset: str) -> dict:
    """Return data[orch][thinking_key][sub_size] = accuracy fraction."""
    d: dict = {
        orch: {t: {s: None for s in SUB_SIZES} for t in THINKING_KEYS}
        for orch, _ in ORCH_GROUPS
    }
    for row in rows:
        orch, sub = parse_config(row["Name"], dataset)
        if orch is None:
            continue
        tkey = row["thinking_mode"]
        if tkey not in THINKING_KEYS:
            continue
        d[orch][tkey][sub] = float(row["accuracy"])
    return d


# ─── figure ──────────────────────────────────────────────────────────────────

def draw(data: dict) -> plt.Figure:
    matplotlib.rcParams.update({
        "font.family":  "serif",
        "font.size":    10.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    n_sub = len(SUB_SIZES)
    subw  = n_sub * BAR_W

    bar_x:       dict = {}
    thinking_cx: dict = {}
    orch_cx:     dict = {}
    sep_x:       float = 0.0

    x = 0.0
    for g_idx, (orch, _) in enumerate(ORCH_GROUPS):
        grp_start = x
        for t_idx, (tkey, _) in enumerate(THINKING_MODES):
            for s_idx, sub in enumerate(SUB_SIZES):
                bar_x[(orch, tkey, sub)] = x + s_idx * BAR_W
            thinking_cx[(orch, tkey)] = x + subw / 2
            x += subw + SUB_GAP
        x -= SUB_GAP
        orch_cx[orch] = (grp_start + x) / 2
        if g_idx == 0:
            sep_x = x + GRP_GAP / 2
        x += GRP_GAP

    x_max = x - GRP_GAP

    fig, ax = plt.subplots(figsize=(10.0, 4.5))

    for (orch, tkey, sub), xpos in bar_x.items():
        val = data[orch][tkey][sub]
        if val is None:
            continue
        ax.bar(xpos, val, width=BAR_W * 0.92,
               align="edge",
               color=SUB_COLORS[sub], edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(xpos + BAR_W / 2, val + 0.004,
                f"{val * 100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, zorder=4)

    all_vals = [
        data[orch][tkey][sub]
        for orch, _ in ORCH_GROUPS
        for tkey in THINKING_KEYS
        for sub in SUB_SIZES
        if data[orch][tkey][sub] is not None
    ]
    y_max = (max(all_vals) * 1.18) if all_vals else 0.5

    ax.set_xlim(-BAR_W * 0.5, x_max + BAR_W * 0.5)
    ax.set_ylim(0, y_max)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Avg. Accuracy [%]", fontsize=12)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="gray", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tick_positions = [thinking_cx[(orch, tkey)]
                      for orch, _ in ORCH_GROUPS
                      for tkey, _ in THINKING_MODES]
    tick_labels    = [THINKING_LBLS[tkey]
                      for _, _ in ORCH_GROUPS
                      for tkey, _ in THINKING_MODES]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10.5)
    ax.tick_params(axis="x", length=0)

    for orch, orch_lbl in ORCH_GROUPS:
        ax.text(orch_cx[orch], -0.075, orch_lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12, fontweight="bold")

    ax.axvline(sep_x, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    handles = [
        mpatches.Patch(facecolor=SUB_COLORS[s], edgecolor="none",
                       label=SUB_LABELS[s])
        for s in SUB_SIZES
    ]
    fig.legend(
        handles=handles,
        title="Sub-agent size",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(SUB_SIZES),
        framealpha=0.9,
        fontsize=10.5,
        title_fontsize=11.5,
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.0,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, top=0.82)
    return fig


# ─── LaTeX table ─────────────────────────────────────────────────────────────

def generate_latex_table(data: dict, out_tex: Path) -> None:
    sub_headers = " & ".join(
        f"\\textbf{{Sub-{s}}}" for s in SUB_SIZES
    )
    lines: list[str] = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Orchestrator} & \textbf{Thinking} & "
        + sub_headers + r" \\",
        r"\midrule",
    ]

    for g_idx, (orch, orch_lbl) in enumerate(ORCH_GROUPS):
        if g_idx > 0:
            lines.append(r"\midrule")
        for t_idx, (tkey, tlbl) in enumerate(THINKING_MODES):
            orch_str = orch_lbl if t_idx == 0 else ""
            vals = [data[orch][tkey][s] for s in SUB_SIZES]
            cells = " & ".join(
                f"{v * 100:.1f}" if v is not None else "—" for v in vals
            )
            lines.append(f"{orch_str} & {tlbl} & {cells} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"  Saved → {out_tex}")


# ─── per-dataset entry point ──────────────────────────────────────────────────

def process_dataset(csv_path: Path) -> int:
    # Derive dataset name from filename: "{DATASET}_orchestrator_capabilities.csv"
    dataset = csv_path.stem.replace("_orchestrator_capabilities", "")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    data = build_data(rows, dataset)

    # Print summary
    print(f"\n  Dataset: {dataset.upper()}")
    for orch, _ in ORCH_GROUPS:
        print(f"  {orch} Orchestrator")
        for tkey, tlbl in THINKING_MODES:
            vals = {s: data[orch][tkey][s] for s in SUB_SIZES}
            row  = "  ".join(
                f"{SUB_LABELS[s]}: {v*100:.1f}%" if v is not None else f"{SUB_LABELS[s]}: —"
                for s, v in vals.items()
            )
            print(f"    [{tlbl:9s}]  {row}")

    # Outputs
    out_dir = PLOTS_DIR / dataset.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "orchestrator_capabilities_figure.png"
    out_tex = TABLES_DIR / f"orchestrator_capabilities_{dataset.lower()}.tex"

    fig = draw(data)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_png}")

    generate_latex_table(data, out_tex)
    return 0


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not OC_CSV_DIR.is_dir():
        print(
            "Skipping orchestrator capabilities: input directory not found.\n"
            f"  Expected: {OC_CSV_DIR}",
            file=sys.stderr,
        )
        return 0

    csv_files = sorted(OC_CSV_DIR.glob("*_orchestrator_capabilities.csv"))
    if not csv_files:
        print(
            "Skipping orchestrator capabilities: no CSV files found in\n"
            f"  {OC_CSV_DIR}",
            file=sys.stderr,
        )
        return 0

    print(f"Found {len(csv_files)} orchestrator-capabilities dataset(s).")
    for csv_path in csv_files:
        process_dataset(csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
