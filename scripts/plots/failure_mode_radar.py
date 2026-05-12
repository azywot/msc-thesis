#!/usr/bin/env python3
"""
Radar (spider) plots of failure-mode distributions per benchmark.

One subplot per benchmark (5 total), laid out 3+2 in a 2-row figure.
Each coloured line represents a thinking-mode configuration.
Axes = failure-mode share (%).

Usage:
    python scripts/plots/failure_mode_radar.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────── paths ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FILE = ROOT / "data/results/failure_modes/breakdown_by_benchmark_and_thinking.csv"
OUT_DIR = ROOT / "data/results/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── style (NeurIPS-ish) ─────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.size":         8.5,
    "axes.titlesize":    9.5,
    "axes.labelsize":    8.5,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.dpi":        150,
})

# ─────────────────────────── constants ───────────────────────────────────────
BENCHMARKS = ["gaia", "gpqa", "aime", "musique", "hle"]
BM_LABELS = {
    "gaia": "GAIA",
    "gpqa": "GPQA",
    "aime": "AIME",
    "musique": "MuSiQue",
    "hle": "HLE",
}

FAILURE_MODES = [
    "modality_tool_gap",
    "tool_loop_or_empty_final",
    "direct_reasoning_no_action",
    "computational_subgoal_error",
    "retrieval_evidence_failure",
    "single_shot_tool_trust",
]
FM_LABELS = {
    "modality_tool_gap":          "Modality /\nTool Gap",
    "tool_loop_or_empty_final":   "Tool Loop /\nEmpty Final",
    "direct_reasoning_no_action": "Direct\nReasoning",
    "computational_subgoal_error":"Compute\nError",
    "retrieval_evidence_failure":  "Retrieval\nFailure",
    "single_shot_tool_trust":     "Single-Shot\nTool Trust",
}

THINKING_MODES = ["none", "orchestrator", "subagents", "all"]
TM_LABELS = {
    "none":        "No thinking",
    "orchestrator":"Orch. thinking",
    "subagents":   "Sub. thinking",
    "all":         "All thinking",
}

# Per-mode visual style: Okabe–Ito palette + distinct linestyle + marker
# so all four series remain legible even in greyscale print.
TM_STYLE: dict[str, dict] = {
    "none": {
        "color":  "#222222",          # near-black
        "ls":     "-",                # solid
        "marker": "o",
        "lw":     1.8,
        "ms":     4.5,
        "zorder": 2,
    },
    "orchestrator": {
        "color":  "#0072B2",          # Okabe-Ito blue
        "ls":     "--",               # dashed
        "marker": "s",
        "lw":     1.8,
        "ms":     4.5,
        "zorder": 3,
    },
    "subagents": {
        "color":  "#D55E00",          # Okabe-Ito vermilion
        "ls":     (0, (4, 1.5)),      # loosely dashed
        "marker": "^",
        "lw":     1.8,
        "ms":     4.5,
        "zorder": 4,
    },
    "all": {
        "color":  "#009E73",          # Okabe-Ito bluish-green
        "ls":     (0, (1, 1)),        # dotted
        "marker": "D",
        "lw":     1.8,
        "ms":     3.8,
        "zorder": 5,
    },
}

# ─────────────────────────── helpers ─────────────────────────────────────────

def make_radar_axes(fig, spec) -> plt.Axes:
    """Create a polar axis from a GridSpec cell."""
    ax = fig.add_subplot(spec, projection="polar")
    return ax


_NICE_STEPS = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]


def _scale_for(df_bm: pd.DataFrame) -> tuple[int, list[int]]:
    """Return (r_max, tick_positions) scaled to count data, with clean integer ticks."""
    raw_max = int(df_bm["count"].max())
    # Pick the smallest 'nice' step s.t. 4 × step >= raw_max
    step = next(s for s in _NICE_STEPS if s * 4 >= raw_max)
    r_max = step * 4
    ticks = [step * i for i in range(1, 5)]
    return r_max, ticks


def draw_radar(ax: plt.Axes, df_bm: pd.DataFrame, title: str) -> None:
    """Draw one radar chart for a single benchmark."""
    n = len(FAILURE_MODES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    r_max, ticks = _scale_for(df_bm)

    # ── tick rings ──────────────────────────────────────────────────────────
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, r_max)
    ax.set_rlabel_position(22.5)   # place ring labels in the gap between spokes 0 and 1
    ax.set_rticks(ticks)
    tick_labels = [str(t) for t in ticks]
    tick_labels[-1] = "count\n\n" + tick_labels[-1]   # annotate unit on outermost ring only
    ax.set_yticklabels(tick_labels, fontsize=6, color="#aaaaaa")
    ax.yaxis.set_tick_params(pad=1)

    # ── axis labels ──────────────────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [FM_LABELS[fm] for fm in FAILURE_MODES],
        fontsize=6.5,
        ha="center",
    )

    # ── grid style ───────────────────────────────────────────────────────────
    ax.grid(color="#e0e0e0", linewidth=0.6, linestyle="-")
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.6)

    # ── plot each thinking mode ──────────────────────────────────────────────
    for tm in THINKING_MODES:
        sub = df_bm[df_bm["thinking_mode"] == tm]
        if sub.empty:
            continue
        vals = [
            sub.loc[sub["failure_mode"] == fm, "count"].values[0]
            if fm in sub["failure_mode"].values else 0.0
            for fm in FAILURE_MODES
        ]
        vals += vals[:1]  # close

        st = TM_STYLE[tm]
        ax.plot(
            angles, vals,
            color=st["color"],
            linewidth=st["lw"],
            linestyle=st["ls"],
            zorder=st["zorder"],
            label=TM_LABELS[tm],
            clip_on=False,
        )

    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=16)


# ─────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(DATA_FILE)

    # ── figure: 2×6 GridSpec so bottom two plots can be centred ──────────────
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(
        2, 6,
        hspace=0.55,
        wspace=0.55,
    )

    # Row 0: three equally spaced subplots
    specs_row0 = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6]]
    # Row 1: two centred subplots
    specs_row1 = [gs[1, 1:3], gs[1, 3:5]]

    all_specs = specs_row0 + specs_row1

    axes: list[plt.Axes] = []
    for i, bm in enumerate(BENCHMARKS):
        ax = make_radar_axes(fig, all_specs[i])
        df_bm = df[df["benchmark"] == bm]
        draw_radar(ax, df_bm, BM_LABELS[bm])
        axes.append(ax)

    # ── shared legend (bottom centre, outside subplots) ──────────────────────
    handles = [
        matplotlib.lines.Line2D(
            [], [],
            color=TM_STYLE[tm]["color"],
            linewidth=TM_STYLE[tm]["lw"],
            linestyle=TM_STYLE[tm]["ls"],
            label=TM_LABELS[tm],
        )
        for tm in THINKING_MODES
    ]
    fig.suptitle(
        "Failure-mode distribution by benchmark and thinking mode",
        fontsize=10,
        fontweight="bold",
        y=1.05,
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.025),
        framealpha=0.92,
        edgecolor="#cccccc",
        fontsize=8,
    )

    out_png = OUT_DIR / "failure_mode_radar.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=400)
    print(f"Saved: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
