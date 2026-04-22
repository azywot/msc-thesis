#!/usr/bin/env python3
"""
Generate results table from main_results.csv.

Output:
    data/results/main_results_table.png

Usage:
    python scripts/generate_results_table.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

# ─────────────────────────── paths ───────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "data/results/wandb/main_results.csv"
OUT_PNG  = ROOT / "data/results/plots/main_results_table.png"

# ─────────────────────────── table structure ─────────────────────────────────
DATASETS       = ["gaia", "gpqa", "aime", "musique", "hle"]
DATASET_LABELS = ["GAIA", "GPQA", "AIME", "MuSiQue", "HLE"]

# Each entry: (model_name, tools_key, thinking_mode, tools_label, thinking_label)
# tools_key is matched as a substring in the experiment Name column.
CONFIGS: list[tuple[str, str, str, str, str]] = [
    # ── Qwen3-8B direct (baseline row first: no tools, no thinking) ──────────
    ("Qwen3-8B",  "no_tools",       "NO",                "—",         "—"),
    ("Qwen3-8B",  "no_tools",       "ORCHESTRATOR_ONLY", "—",         "Orchestrator"),
    ("Qwen3-8B",  "direct_tools",   "NO",                "Direct",    "—"),
    ("Qwen3-8B",  "direct_tools",   "ORCHESTRATOR_ONLY", "Direct",    "Orchestrator"),
    # ── Qwen3-8B AgentFlow (subagent mode) ───────────────────────────────────
    ("Qwen3-8B",  "subagent_tools", "NO",                "Sub-agent", "—"),
    ("Qwen3-8B",  "subagent_tools", "SUBAGENTS_ONLY",    "Sub-agent", "Sub-agents"),
    ("Qwen3-8B",  "subagent_tools", "ORCHESTRATOR_ONLY", "Sub-agent", "Orchestrator"),
    ("Qwen3-8B",  "subagent_tools", "ALL",               "Sub-agent", "All"),
    # ── Qwen3-32B (direct mode only; after 8B blocks) ─────────────────────────
    ("Qwen3-32B", "no_tools",       "NO",                "—",         "—"),
    ("Qwen3-32B", "no_tools",       "ORCHESTRATOR_ONLY", "—",         "Orchestrator"),
    ("Qwen3-32B", "direct_tools",   "NO",                "Direct",    "—"),
    ("Qwen3-32B", "direct_tools",   "ORCHESTRATOR_ONLY", "Direct",    "Orchestrator"),
]

# ─────────────────────────── layout constants ─────────────────────────────────
# (left-edge x, width) in inches for each column
COL_LAYOUT = [
    (0.08, 1.10),   # 0 Model
    (1.18, 1.00),   # 1 Tools
    (2.18, 1.15),   # 2 Thinking
    (3.33, 0.97),   # 3 GAIA
    (4.30, 0.97),   # 4 GPQA
    (5.27, 0.97),   # 5 AIME
    (6.24, 0.97),   # 6 MuSiQue
    (7.21, 0.96),   # 7 HLE  (ends at 8.17; + 0.08 pad = 8.25)
]
FIG_W   = 8.25   # total figure width
ROW_H   = 0.400  # height of each data row (two sub-lines: value + delta)
HDR_H   = 0.330  # height of the header row
LEG_H   = 0.40   # height of the legend area
LR_PAD  = 0.08   # left/right padding for horizontal rules

FONT_SZ  = 8.5
DELTA_SZ = 7.0   # font size for the delta sub-line
SMALL_SZ = 7.5

# Row index of the reference baseline (Qwen3-8B, no tools, no thinking) — first row
BASELINE_IDX = 0

CMAP    = LinearSegmentedColormap.from_list("tbl", ["#f0f9ff", "#74c0fc"], N=256)

COL_POS  = "#276221"   # dark green – positive delta
COL_NEG  = "#b71c1c"   # dark red   – negative delta
COL_ZERO = "#666666"   # grey       – zero / baseline row


# ─────────────────────────── data helpers ────────────────────────────────────

def lookup(
    df: pd.DataFrame,
    model: str,
    tools_key: str,
    thinking: str,
    dataset: str,
) -> float | None:
    """Return accuracy (0–100, 1 d.p.) for the given config, or None if absent."""
    mask = (
        (df["model_name"] == model)
        & df["Name"].str.contains(f"_{tools_key}_", regex=False)
        & df["Name"].str.contains(f"_{dataset}_", regex=False)
        & (df["thinking_mode"] == thinking)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return round(float(rows.iloc[0]["accuracy"]) * 100, 1)


def build_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, tools_key, thinking, t_lbl, th_lbl in CONFIGS:
        rec: dict = {"model": model, "tools": t_lbl, "thinking": th_lbl}
        for ds in DATASETS:
            rec[ds] = lookup(df_raw, model, tools_key, thinking, ds)
        records.append(rec)
    return pd.DataFrame(records)


# ─────────────────────────── rendering ───────────────────────────────────────

def _col_cx(col: int) -> float:
    """Center x of column `col`."""
    x, w = COL_LAYOUT[col]
    return x + w / 2


def _col_lx(col: int, pad: float = 0.04) -> float:
    """Left-aligned x of column `col` (with small indent)."""
    return COL_LAYOUT[col][0] + pad


def draw(data: pd.DataFrame) -> plt.Figure:
    # ── best-value bookkeeping ────────────────────────────────────────────────
    best_all = {ds: data[ds].max() for ds in DATASETS}
    norms = {ds: Normalize(vmin=0, vmax=best_all[ds]) for ds in DATASETS}
    # Best value strictly below the overall best (runner-up threshold).
    # All rows tied at this value get underlined (any model).
    best_runnerup: dict = {}
    for ds in DATASETS:
        sub = data.loc[
            data[ds].notna() & (data[ds] < best_all[ds] - 1e-9), ds
        ]
        best_runnerup[ds] = sub.max() if not sub.empty else None

    # baseline values (row BASELINE_IDX: Qwen3-8B, no tools, no thinking)
    baseline = {ds: data.iloc[BASELINE_IDX][ds] for ds in DATASETS}

    n_rows = len(data)
    fig_h  = HDR_H + n_rows * ROW_H + LEG_H

    matplotlib.rcParams.update({
        "font.family":  "serif",
        "font.size":    FONT_SZ,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # texts whose bounding boxes we need after canvas.draw() for underlines
    _ul_queue: list[matplotlib.text.Text] = []

    # ── local helpers ─────────────────────────────────────────────────────────
    def hline(y: float, lw: float = 0.6, **kw) -> None:
        ax.plot([LR_PAD, FIG_W - LR_PAD], [y, y], "k-", lw=lw, **kw)

    def txt(x, y, s, underline=False, **kw):
        t = ax.text(x, y, s, **kw)
        if underline:
            _ul_queue.append(t)
        return t

    def apply_underlines(items: list[matplotlib.text.Text]) -> None:
        """Draw underlines under listed Text objects (requires canvas.draw() first)."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        for t in items:
            bb = t.get_window_extent(renderer=renderer)
            x0d, y0d = inv.transform((bb.x0, bb.y0))
            x1d, _   = inv.transform((bb.x1, bb.y0))
            gap = ROW_H * 0.02
            ax.plot([x0d, x1d], [y0d - gap, y0d - gap], "k-", lw=0.75, zorder=5)

    # ── header ────────────────────────────────────────────────────────────────
    y_top = fig_h - 0.04
    hline(y_top, lw=0.9)

    hy = y_top - HDR_H / 2
    for i, lbl in enumerate(["Model", "Tools", "Thinking"] + DATASET_LABELS):
        if i < 3:
            ax.text(_col_lx(i), hy, lbl, ha="left", va="center", fontsize=FONT_SZ)
        else:
            ax.text(_col_cx(i), hy, lbl, ha="center", va="center", fontsize=FONT_SZ)

    y_mid = y_top - HDR_H
    hline(y_mid, lw=0.5)

    # ── data rows ─────────────────────────────────────────────────────────────
    y           = y_mid
    prev_model: str | None = None
    prev_tools: str | None = None

    for row_idx, row in data.iterrows():
        y  -= ROW_H
        yc  = y + ROW_H / 2

        # Separator between model groups (e.g. 8B → 32B)
        if row["model"] != prev_model:
            if prev_model is not None:
                hline(y + ROW_H, lw=0.8, alpha=0.75)
            prev_model = row["model"]
            ax.text(_col_lx(0), yc, row["model"],
                    ha="left", va="center", fontsize=FONT_SZ)

        # Separator between tool groups within the same model (e.g. Direct → Sub-agent)
        elif row["tools"] != prev_tools and prev_tools is not None:
            hline(y + ROW_H, lw=0.3, alpha=0.45)

        prev_tools = row["tools"]

        ax.text(_col_lx(1), yc, row["tools"],    ha="left", va="center", fontsize=FONT_SZ)
        ax.text(_col_lx(2), yc, row["thinking"], ha="left", va="center", fontsize=FONT_SZ)

        is_baseline_row = (row_idx == BASELINE_IDX)

        for j, ds in enumerate(DATASETS):
            ci  = j + 3          # column index in COL_LAYOUT
            val = row[ds]
            xc  = _col_cx(ci)
            cx, cw = COL_LAYOUT[ci]

            if val is None:
                ax.text(xc, yc, "—", ha="center", va="center", fontsize=FONT_SZ)
                continue

            # shaded background
            color = CMAP(norms[ds](val))
            ax.add_patch(mpatches.Rectangle(
                (cx, y), cw, ROW_H,
                facecolor=color, edgecolor="none", zorder=0,
            ))

            # bold      → tied for best overall (any model)
            # underline → tied for runner-up (second best, any model)
            is_best       = val >= best_all[ds] - 1e-9
            is_best_8b    = (
                not is_best
                and best_runnerup[ds] is not None
                and val >= best_runnerup[ds] - 1e-9
            )

            # upper sub-line: accuracy value
            y_val   = y + ROW_H * 0.67
            y_delta = y + ROW_H * 0.20

            txt(xc, y_val, f"{val:.1f}",
                underline=is_best_8b,
                ha="center", va="center",
                fontsize=FONT_SZ,
                fontweight="bold" if is_best else "normal",
                zorder=2)

            # lower sub-line: delta vs baseline (omit for baseline row itself)
            if not is_baseline_row and baseline[ds] is not None:
                delta  = round(val - baseline[ds], 1)
                sign   = "+" if delta >= 0 else ""
                d_col  = COL_POS if delta > 0 else (COL_NEG if delta < 0 else COL_ZERO)
                ax.text(xc, y_delta, f"({sign}{delta:.1f})",
                        ha="center", va="center",
                        fontsize=DELTA_SZ, color=d_col, zorder=2)

    y_bot = y
    hline(y_bot, lw=0.9)

    # underlines require a canvas draw pass
    apply_underlines(_ul_queue)

    # ── legend ────────────────────────────────────────────────────────────────
    # vertically centred in the legend strip, all drawn on the main axes
    yl    = y_bot - LEG_H / 2   # data-coord y (same units as everything else)
    BAR_W = 1.20
    BAR_H = 0.10
    bar_x = LR_PAD + 1.06      # x start of bar, after label

    ax.text(LR_PAD, yl, "Accuracy: Low", ha="left", va="center", fontsize=SMALL_SZ)

    # draw gradient bar directly in data coordinates — no separate axes
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient,
              extent=[bar_x, bar_x + BAR_W, yl - BAR_H / 2, yl + BAR_H / 2],
              aspect="auto", cmap=CMAP, origin="lower", zorder=1)
    ax.add_patch(mpatches.Rectangle(
        (bar_x, yl - BAR_H / 2), BAR_W, BAR_H,
        linewidth=0.4, edgecolor="black", facecolor="none", zorder=2,
    ))

    ax.text(bar_x + BAR_W + 0.05, yl, "High", ha="left", va="center", fontsize=SMALL_SZ)

    return fig


# ─────────────────────────── entry point ─────────────────────────────────────

def main() -> None:
    df_raw = pd.read_csv(CSV_PATH)
    data   = build_table(df_raw)

    print("Table data:")
    print(data.to_string(index=False))
    print()

    fig = draw(data)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
