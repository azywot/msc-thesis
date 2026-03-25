#!/usr/bin/env python3
"""
Orchestrator capacity – grouped bar chart.

Bars are grouped first by orchestrator size (8B / 32B), then by thinking
mode within each group. Bar colour indicates sub-agent size (1.7B / 8B / 32B).

Output:
    data/results/orchestrator_capabilities_figure.{pdf,png}
"""
from __future__ import annotations

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "data/results/orchestrator_capabilities_results.csv"
OUT_PDF  = ROOT / "data/results/orchestrator_capabilities_figure.pdf"
OUT_PNG  = ROOT / "data/results/orchestrator_capabilities_figure.png"

# ─── display config ───────────────────────────────────────────────────────────
# thinking modes in display order
THINKING_MODES = [
    ("NO",                "None"),
    ("ORCHESTRATOR_ONLY", "Orch."),
    ("ALL",               "All"),
]
THINKING_KEYS = [k for k, _ in THINKING_MODES]
THINKING_LBLS = {k: lbl for k, lbl in THINKING_MODES}

# sub-agent sizes and their colours (matches example figure palette)
SUB_SIZES  = ["1.7B", "8B", "32B"]
# Seaborn "deep" palette – standard in NeurIPS/ICML papers
SUB_COLORS = {"1.7B": "#4C72B0", "8B": "#DD8452", "32B": "#55A868"}
SUB_LABELS = {"1.7B": "1.7–2B",  "8B": "8B",      "32B": "32B"}

ORCH_GROUPS = [("8B", "8B Orchestrator"), ("32B", "32B Orchestrator")]

# layout
BAR_W   = 0.18   # width of one bar
SUB_GAP = 0.12   # gap between thinking-mode sub-groups within an orch group
GRP_GAP = 0.55   # gap between the two major orch groups


# ─── data ────────────────────────────────────────────────────────────────────

def parse_config(name: str) -> tuple[str, str] | tuple[None, None]:
    """Return (orch_size, sub_size) from experiment name, or (None, None)."""
    # AF baseline → treated as 8B orchestrator / 8B sub-agent
    if "AF_no_img_no_mm_gaia_qwen8B" in name:
        return "8B", "8B"
    if not name.startswith("OC_gaia_"):
        return None, None
    orch = "8B" if "orch8b" in name else "32B"
    if   "sub1_7b" in name: sub = "1.7B"
    elif "sub8b"   in name: sub = "8B"
    elif "sub32b"  in name: sub = "32B"
    else: return None, None
    return orch, sub


def build_data(df: pd.DataFrame) -> dict:
    """Return data[orch][thinking_key][sub_size] = accuracy fraction."""
    d: dict = {
        orch: {t: {s: None for s in SUB_SIZES} for t in THINKING_KEYS}
        for orch, _ in ORCH_GROUPS
    }
    for _, row in df.iterrows():
        orch, sub = parse_config(row["Name"])
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
        "font.size":    9,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    n_sub  = len(SUB_SIZES)
    subw   = n_sub * BAR_W   # x-width of one thinking-mode sub-group

    # ── pre-compute x positions ───────────────────────────────────────────────
    bar_x:        dict = {}   # (orch, tkey, sub) → left edge of bar
    thinking_cx:  dict = {}   # (orch, tkey)      → x centre of sub-group
    orch_cx:      dict = {}   # orch              → x centre of orch group
    sep_x:        float = 0.0 # x of separator between the two groups

    x = 0.0
    for g_idx, (orch, _) in enumerate(ORCH_GROUPS):
        grp_start = x
        for t_idx, (tkey, _) in enumerate(THINKING_MODES):
            for s_idx, sub in enumerate(SUB_SIZES):
                bar_x[(orch, tkey, sub)] = x + s_idx * BAR_W
            thinking_cx[(orch, tkey)] = x + subw / 2
            x += subw + SUB_GAP
        x -= SUB_GAP   # trim trailing gap
        orch_cx[orch] = (grp_start + x) / 2
        if g_idx == 0:
            sep_x = x + GRP_GAP / 2
        x += GRP_GAP

    x_max = x - GRP_GAP   # rightmost bar edge

    # ── draw ─────────────────────────────────────────────────────────────────
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
                ha="center", va="bottom", fontsize=6.5, zorder=4)

    # ── axes styling ─────────────────────────────────────────────────────────
    ax.set_xlim(-BAR_W * 0.5, x_max + BAR_W * 0.5)
    ax.set_ylim(0, 0.34)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Avg. Accuracy [%]", fontsize=9.5)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="gray", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── x ticks: thinking-mode labels ────────────────────────────────────────
    tick_positions = [thinking_cx[(orch, tkey)]
                      for orch, _ in ORCH_GROUPS
                      for tkey, _ in THINKING_MODES]
    tick_labels    = [THINKING_LBLS[tkey]
                      for _, _ in ORCH_GROUPS
                      for tkey, _ in THINKING_MODES]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.tick_params(axis="x", length=0)

    # ── orchestrator group labels (below x ticks) ─────────────────────────────
    for orch, orch_lbl in ORCH_GROUPS:
        ax.text(orch_cx[orch], -0.075, orch_lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold")

    # ── vertical separator between groups ────────────────────────────────────
    ax.axvline(sep_x, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    # ── legend ───────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=SUB_COLORS[s], edgecolor="none",
                       label=SUB_LABELS[s])
        for s in SUB_SIZES
    ]
    ax.legend(handles=handles, title="Sub-agent size",
              loc="upper right", framealpha=0.9,
              fontsize=8.5, title_fontsize=9,
              handlelength=1.2, handleheight=0.9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)   # room for the group labels
    return fig


# ─── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    df   = pd.read_csv(CSV_PATH)
    data = build_data(df)

    # print summary
    for orch, _ in ORCH_GROUPS:
        print(f"\n{orch} Orchestrator")
        for tkey, tlbl in THINKING_MODES:
            vals = {s: data[orch][tkey][s] for s in SUB_SIZES}
            row  = "  ".join(
                f"{SUB_LABELS[s]}: {v*100:.1f}%" if v is not None else f"{SUB_LABELS[s]}: —"
                for s, v in vals.items()
            )
            print(f"  [{tlbl:9s}]  {row}")

    fig = draw(data)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved → {OUT_PDF}")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
