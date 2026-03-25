#!/usr/bin/env python3
"""
Efficiency plots for AgentFlow experiments — NeurIPS style.

Three categories compared:
  • Qwen3-32B  Direct   (large baseline, single-agent)
  • Qwen3-8B   Direct   (small baseline, single-agent)
  • Qwen3-8B   AgentFlow (proposed multi-agent system)

Outputs (data/results/plots/):
  token_breakdown.png       — avg tokens/query, grouped by category
  tool_calls_breakdown.png  — total tool calls by type, grouped by category
  pareto_per_benchmark.png  — accuracy vs tokens/query, one panel per benchmark

Usage:
    python scripts/plots/efficiency_plots.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────── paths ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

RESULTS_SUBAGENT = ROOT / "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow"
RESULTS_BASELINE = ROOT / "experiments/results/NEW_baseline"

OUT_DIR = ROOT / "data/results/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── NeurIPS RC ──────────────────────────────────────
# Target: ~6.75" wide (NeurIPS double-column), Computer Modern–like serif,
# clean spines, tight layout, 8–9 pt labels.
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          8.5,
    "axes.titlesize":     9.5,
    "axes.labelsize":     8.5,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "-",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "figure.dpi":         150,
})

# ─────────────────────────── categories & configs ────────────────────────────
BENCHMARKS = ["gaia", "gpqa", "aime", "musique", "hle"]
BM_LABELS  = {"gaia": "GAIA", "gpqa": "GPQA", "aime": "AIME",
               "musique": "MuSiQue", "hle": "HLE"}

# Wong (2011) colorblind-safe palette — standard in NeurIPS/ICLR accessibility guidelines.
# Blue=#0072B2, Dark-red=#990000, Bluish-green=#009E73
#
# Direct categories carry 4 configs (no_tools + direct_tools × no-think/+think) so
# the token-breakdown plot shows the full picture.
# The Pareto scatter filters to direct_tools only (2 dots per model).
# "Sub. Think" and "All Think" are exclusive to the MAS category.
CATEGORIES: dict[str, dict] = {
    "32B Direct": {
        "label":  "Qwen3-32B (Direct)",
        "color":  "#0072B2",                                     # Wong blue
        "shades": ["#B3D7F0", "#66AEDE", "#0072B2", "#004F7D"],
        "configs": [
            ("qwen32B", "no_tools",     "none"),
            ("qwen32B", "no_tools",     "orchestrator"),
            ("qwen32B", "direct_tools", "none"),
            ("qwen32B", "direct_tools", "orchestrator"),
        ],
        "config_labels": ["No tools", "No tools + Think", "Direct", "Direct + Think"],
    },
    "8B Direct": {
        "label":  "Qwen3-8B (Direct)",
        "color":  "#990000",                                     # dark red
        "shades": ["#FFAAAA", "#EE6666", "#CC0000", "#880000"],
        "configs": [
            ("qwen8B", "no_tools",     "none"),
            ("qwen8B", "no_tools",     "orchestrator"),
            ("qwen8B", "direct_tools", "none"),
            ("qwen8B", "direct_tools", "orchestrator"),
        ],
        "config_labels": ["No tools", "No tools + Think", "Direct", "Direct + Think"],
    },
    "8B MAS": {
        "label":  "Qwen3-8B (MAS)",
        "color":  "#009E73",                                     # Wong bluish-green
        "shades": ["#A3DED0", "#47C7A8", "#009E73", "#006B4F"],
        "configs": [
            ("qwen8B", "subagent_tools", "none"),
            ("qwen8B", "subagent_tools", "subagents"),
            ("qwen8B", "subagent_tools", "orchestrator"),
            ("qwen8B", "subagent_tools", "all"),
        ],
        "config_labels": ["No think", "Sub. Think", "Orch. Think", "All Think"],
    },
}

ALL_CONFIGS: list[tuple[str, str, str, str]] = [
    (cat_key, *cfg)
    for cat_key, cat in CATEGORIES.items()
    for cfg in cat["configs"]
]

# Marker style per thinking mode — keyed by the `think` config value so that
# the same mode (e.g. "none" / "orchestrator") gets an identical marker across
# all categories (Direct and MAS alike).
# Order: No Think, Orch. Think, Sub. Think (MAS-only), All Think (MAS-only).
THINK_STYLE: dict[str, tuple[str, str]] = {
    "none":         ("o",  "No Think"),
    "orchestrator": ("^",  "Orch. Think"),
    "subagents":    ("s",  "Sub. Think"),
    "all":          ("D",  "All Think"),
}

# ─────────────────────────── data loading ────────────────────────────────────

# Explicit routing table — subagent_tools ALWAYS come from the milestone run;
# no_tools / direct_tools ALWAYS come from the dedicated baseline run.
# "subagents" and "all" thinking modes exist only in the MAS category.
_RESULTS_ROOT = {
    "subagent_tools": RESULTS_SUBAGENT,
    "no_tools":       RESULTS_BASELINE,
    "direct_tools":   RESULTS_BASELINE,
}


def find_metrics(model: str, tools: str, thinking: str, benchmark: str
                 ) -> tuple[dict[str, Any], Path] | tuple[None, None]:
    """Return (metrics_dict, metrics_file_path) or (None, None) if not found."""
    root   = _RESULTS_ROOT[tools]
    folder = root / benchmark
    if not folder.exists():
        return None, None
    pattern    = f"{model}_{tools}_{thinking}"
    candidates = [d for d in folder.iterdir() if d.is_dir() and d.name == pattern]
    if not candidates:
        return None, None
    for run in sorted(candidates[0].iterdir(), reverse=True):
        mf = run / "metrics.json"
        if mf.exists():
            return json.loads(mf.read_text()), mf
    return None, None


def num_samples(s: str) -> int:
    m = re.search(r"of\s+(\d+)", s)
    return int(m.group(1)) if m else 1


def load_all() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {k: [] for k in CATEGORIES}
    for cat_key, model, tools, think in ALL_CONFIGS:
        cat     = CATEGORIES[cat_key]
        cfg_idx = cat["configs"].index((model, tools, think))

        per_bm:      dict[str, dict] = {}
        source_files: dict[str, Path] = {}

        for bm in BENCHMARKS:
            m, mf = find_metrics(model, tools, think, bm)
            if m is None:
                continue
            ov = m["overall"]
            n  = num_samples(ov["num_correct"])
            tu = ov["token_usage"]
            per_bm[bm] = {
                "accuracy":          ov["accuracy"] * 100,
                "tokens_per_query":  tu["total_tokens"] / n,
                "prompt_tokens":     tu["prompt_tokens"] / n,
                "completion_tokens": tu["completion_tokens"] / n,
                "n":                 n,
                "tool_usage":        m.get("tool_usage", {}),
            }
            source_files[bm] = mf

        if not per_bm:
            continue

        tool_agg: dict[str, int] = {}
        for v in per_bm.values():
            for k, cnt in v["tool_usage"].items():
                tool_agg[k] = tool_agg.get(k, 0) + int(cnt)

        data[cat_key].append({
            "model":  model, "tools": tools, "think": think,
            "cat_key":      cat_key,
            "config_label": cat["config_labels"][cfg_idx],
            "think_idx":    cfg_idx,
            "avg_tokens_per_query": np.mean([v["tokens_per_query"]  for v in per_bm.values()]),
            "avg_prompt_per_query": np.mean([v["prompt_tokens"]     for v in per_bm.values()]),
            "avg_compl_per_query":  np.mean([v["completion_tokens"] for v in per_bm.values()]),
            "tool_calls":    tool_agg,
            "per_benchmark": per_bm,
            "source_files":  source_files,
        })
    return data


def verify_data_sources(data: dict[str, list[dict]]) -> None:
    """
    Print a full audit table: for every (category, config, benchmark) triple,
    show which results directory was used and the loaded accuracy value.
    Cross-check these against your paper table.
    """
    bm_cols = "  ".join(f"{BM_LABELS[b]:>8}" for b in BENCHMARKS)
    hdr = f"{'Category':<22} {'Config':<20} {'Source dir':<12}  {bm_cols}"
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print("DATA SOURCE AUDIT  (verify accuracy values against paper table)")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for cat_key, recs in data.items():
        for r in recs:
            # source directory tag
            src_tag = "SUBAGENT" if r["tools"] == "subagent_tools" else "BASELINE"
            # per-benchmark accuracy (or "—" if missing)
            acc_cols = "  ".join(
                f"{r['per_benchmark'][b]['accuracy']:>8.1f}"
                if b in r["per_benchmark"] else f"{'—':>8}"
                for b in BENCHMARKS
            )
            lbl = r["config_label"].replace("\n", " ")
            print(f"{cat_key:<22} {lbl:<20} {src_tag:<12}  {acc_cols}")

            # print the actual file paths at a lower indent
            for bm, mf in r["source_files"].items():
                # show path relative to ROOT for readability
                try:
                    rel = mf.relative_to(ROOT)
                except ValueError:
                    rel = mf
                print(f"  {'':22} {'':20} {BM_LABELS[bm]:>8}: {rel}")
        print()
    print(sep)


def print_summary(data: dict[str, list[dict]]) -> None:
    hdr = f"{'Category':<22} {'Config':<20} {'Tok/Q':>8}  {'Tool calls':>10}"
    sep = "=" * len(hdr)
    print(f"\n{sep}\nEFFICIENCY SUMMARY\n{sep}\n{hdr}\n{'-'*len(hdr)}")
    for cat_key, recs in data.items():
        for r in recs:
            print(f"{cat_key:<22} {r['config_label'].replace(chr(10),' '):<20}"
                  f" {r['avg_tokens_per_query']:8.0f}  {sum(r['tool_calls'].values()):10d}")
        print()
    print(sep)


# ─────────────────────────── shared helpers ───────────────────────────────────

def _category_underline(ax, positions: list[float], bar_w: float,
                         cat_key: str, y_frac: float = -0.50) -> None:
    """Draw a coloured horizontal rule + label below a group of bars."""
    cat   = CATEGORIES[cat_key]
    x0    = positions[0]  - bar_w * 0.5
    x1    = positions[-1] + bar_w * 0.5
    xm    = (x0 + x1) / 2
    trans = ax.get_xaxis_transform()   # x in data, y in axes fraction
    ax.plot([x0, x1], [y_frac, y_frac], transform=trans,
            color=cat["color"], lw=2.0, solid_capstyle="round",
            clip_on=False)
    ax.text(xm, y_frac - 0.055, cat["label"], transform=trans,
            ha="center", va="top", fontsize=7.5,
            color=cat["color"], fontweight="bold", clip_on=False)


def _category_sidebar(ax, y_positions: list[float], bar_h: float,
                       cat_key: str, x_frac: float = 1.02) -> None:
    """Draw a coloured vertical rule + label to the right of a group of bars."""
    cat   = CATEGORIES[cat_key]
    y0    = min(y_positions) - bar_h * 0.5
    y1    = max(y_positions) + bar_h * 0.5
    ym    = (y0 + y1) / 2
    trans = ax.get_yaxis_transform()   # y in data, x in axes fraction
    ax.plot([x_frac, x_frac], [y0, y1], transform=trans,
            color=cat["color"], lw=2.0, solid_capstyle="round",
            clip_on=False)
    ax.text(x_frac + 0.015, ym, cat["label"], transform=trans,
            ha="left", va="center", fontsize=7.5,
            color=cat["color"], fontweight="bold", clip_on=False)


# ─────────────────────────── Figure 1: Token breakdown ───────────────────────

def plot_token_breakdown(data: dict[str, list[dict]]) -> None:
    BAR_W     = 0.60   # individual bar width
    BAR_GAP   = 0.10   # gap between bars within the same group
    GROUP_GAP = 1.20   # gap between category groups

    # Compute x-positions with intra-group spacing
    positions: dict[str, list[float]] = {}
    x = 0.0
    for cat_key, recs in data.items():
        positions[cat_key] = [x + i * (BAR_W + BAR_GAP) for i in range(len(recs))]
        x += len(recs) * (BAR_W + BAR_GAP) - BAR_GAP + GROUP_GAP

    fig, ax = plt.subplots(figsize=(6.75, 4.6))

    for cat_key, recs in data.items():
        base_color = CATEGORIES[cat_key]["color"]
        for r, xi in zip(recs, positions[cat_key]):
            p = r["avg_prompt_per_query"] / 1_000
            c = r["avg_compl_per_query"]  / 1_000
            ax.bar(xi, p, BAR_W, color=base_color, zorder=2)
            ax.bar(xi, c, BAR_W, bottom=p,
                   color=base_color, hatch="///", alpha=0.40, zorder=2)
            total_k = p + c
            ax.text(xi, total_k + 0.3, f"{total_k:.1f}K",
                    ha="center", va="bottom", fontsize=6.5, color="#333333")

    # x-ticks
    all_x, all_lbl = [], []
    for cat_key, recs in data.items():
        for r, xi in zip(recs, positions[cat_key]):
            all_x.append(xi)
            all_lbl.append(r["config_label"])

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_lbl, fontsize=7.0, rotation=45, ha="right", va="top")

    # Axis styling — light theme
    ax.set_ylabel("Avg. tokens per query (K)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}K"))
    ax.set_xlim(-0.55, all_x[-1] + 0.55)
    ax.set_ylim(bottom=0, top=22)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    # Category underlines — placed via fixed figure-space y so they never
    # collide with rotated tick labels regardless of axes height.
    # We draw them AFTER subplots_adjust so figure coords are stable.
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.38, top=0.88)

    for cat_key, recs in data.items():
        _category_underline(ax, positions[cat_key], BAR_W, cat_key, y_frac=-0.42)

    # Title and legend in the space above the axes
    fig.text(0.5, 0.97, "Token usage per configuration",
             ha="center", va="top", fontsize=9.5,
             transform=fig.transFigure)

    fig.legend(handles=[
        mpatches.Patch(facecolor="#888888",                          label="Prompt tokens"),
        mpatches.Patch(facecolor="#888888", hatch="///", alpha=0.55, label="Completion tokens"),
    ], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.92),
       facecolor="white", edgecolor="#cccccc", fontsize=7.5)

    fig.savefig(OUT_DIR / "token_breakdown.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/token_breakdown.png")


# ─────────────────────────── Figure 2: Tool-call breakdown ───────────────────

TOOL_KEYS   = ["web_search", "code_generator", "text_inspector",
               "image_inspector", "mind_map"]
TOOL_LABELS = ["Web search", "Code gen.", "Text insp.", "Image insp.", "Mind map"]
TOOL_COLORS = ["#1976D2", "#388E3C", "#F57F17", "#7B1FA2", "#E53935"]


def plot_tool_calls(data: dict[str, list[dict]]) -> None:
    BAR_H    = 0.60
    GRP_GAP  = 0.55

    # Collect only configs with at least one tool call
    rows, row_cats = [], []
    for cat_key, recs in data.items():
        for r in recs:
            if sum(r["tool_calls"].values()) > 0:
                rows.append(r); row_cats.append(cat_key)

    if not rows:
        print("  (No tool calls; skipping)")
        return

    # Assign y-positions, inserting gaps between categories
    y_pos: list[float] = []
    cat_ys: dict[str, list[float]] = {}
    y = 0.0
    prev = None
    for r, ck in zip(rows, row_cats):
        if prev is not None and ck != prev:
            y += GRP_GAP
        y_pos.append(y)
        cat_ys.setdefault(ck, []).append(y)
        y += BAR_H
        prev = ck

    y_arr = np.array(y_pos)
    fig, ax = plt.subplots(figsize=(6.75, max(3.5, y + 0.4)))

    lefts = np.zeros(len(rows))
    for tk, tl, tc in zip(TOOL_KEYS, TOOL_LABELS, TOOL_COLORS):
        vals = np.array([r["tool_calls"].get(tk, 0) for r in rows], dtype=float)
        ax.barh(y_arr, vals, BAR_H, left=lefts, label=tl, color=tc, alpha=0.85, zorder=2)
        lefts += vals

    ax.set_yticks(y_arr)
    ax.set_yticklabels([r["config_label"] for r in rows])
    for tick, ck in zip(ax.get_yticklabels(), row_cats):
        tick.set_color(CATEGORIES[ck]["color"])

    for ck, ys in cat_ys.items():
        _category_sidebar(ax, ys, BAR_H, ck)

    ax.set_xlabel("Total tool calls (all benchmarks)")
    ax.set_title("Tool-Call Breakdown by Configuration", pad=6)
    ax.legend(loc="lower right", ncol=2)
    ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.set_ylim(-0.4, y_arr[-1] + BAR_H * 0.7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.68)
    fig.savefig(OUT_DIR / "tool_calls_breakdown.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/tool_calls_breakdown.png")


# ─────────────────────────── Figure 3: Per-benchmark Pareto ──────────────────

def plot_per_benchmark_pareto(data: dict[str, list[dict]]) -> None:
    """
    1 × 5 small-multiples.  Each panel = one benchmark.
    x = tokens/query,  y = accuracy (%).
    Colour = category; marker = thinking level.
    The Pareto frontier across all plotted configs is drawn as a solid step line.
    """
    fig, axes = plt.subplots(1, len(BENCHMARKS),
                             figsize=(15.0, 3.0), sharey=False)
    # Reserve a small right margin for the legend (placed just outside the panels).
    fig.subplots_adjust(wspace=0.30, right=0.88)

    for ax, bm in zip(axes, BENCHMARKS):
        pts: list[tuple[str, str, float, float]] = []  # (cat, think, x, y)
        for cat_key, recs in data.items():
            for r in recs:
                if bm not in r["per_benchmark"]:
                    continue
                # Direct categories: direct_tools only (no no_tools)
                if cat_key != "8B MAS" and r["tools"] == "no_tools":
                    continue
                bmd = r["per_benchmark"][bm]
                pts.append((cat_key, r["think"],
                             bmd["tokens_per_query"], bmd["accuracy"]))

        if not pts:
            ax.set_visible(False)
            continue

        # Scatter — marker determined by think value, consistent across categories
        for cat_key, think_val, x, y in pts:
            marker, _ = THINK_STYLE.get(think_val, ("o", ""))
            ax.scatter(x, y,
                       color=CATEGORIES[cat_key]["color"],
                       marker=marker,
                       s=90, zorder=4,
                       edgecolors="white", linewidths=0.8)

        # Pareto frontier: point i is non-dominated if no other point j has
        # (tokens_j <= tokens_i AND accuracy_j >= accuracy_i) with at least
        # one strict inequality  →  min tokens, max accuracy.
        pareto = []
        for i, (_, _, xi, yi) in enumerate(pts):
            dominated = any(
                pts[j][2] <= xi and pts[j][3] >= yi
                and (pts[j][2] < xi or pts[j][3] > yi)
                for j in range(len(pts)) if j != i
            )
            if not dominated:
                pareto.append((xi, yi))
        if pareto:
            # Sort by tokens ascending; accuracy must be strictly increasing
            # (any point with same/more tokens but same/lower accuracy is dominated).
            pareto.sort()
            # Staircase: vertical-then-horizontal steps (┐─) so each plateau
            # extends rightward from the point where accuracy improves.
            sx, sy = [pareto[0][0]], [pareto[0][1]]
            for i in range(1, len(pareto)):
                # jump up to new accuracy at previous x, then extend right
                sx += [pareto[i - 1][0], pareto[i][0]]
                sy += [pareto[i][1],     pareto[i][1]]
            ax.plot(sx, sy, color="#555555", lw=1.3, ls="-",
                    alpha=0.75, zorder=2)

        ax.set_title(BM_LABELS.get(bm, bm), fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlabel("Tokens / query", labelpad=2)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy (%)", labelpad=2)
        ax.tick_params(pad=2)
        # Consistent headroom: top point always sits at ~77% of panel height.
        y_max = max(y for *_, y in pts)
        ax.set_ylim(0, min(100.0, y_max * 1.30))
        ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K"))
        ax.yaxis.grid(True); ax.xaxis.grid(False)

    # ── single-column legend on the right, two visual groups ──────────────────
    # Group 1: Pareto frontier line + one colour patch per category (setup)
    group1 = [
        plt.Line2D([0], [0], color="#555555", lw=1.3, label="Pareto frontier"),
    ] + [
        mpatches.Patch(color=cat["color"], label=cat["label"])
        for cat in CATEGORIES.values()
    ]

    # blank separator between groups
    separator = [plt.Line2D([0], [0], color="none", label="")]

    # Group 2: marker shapes = thinking modes (keyed by think value)
    group2 = [
        plt.Line2D([0], [0], marker=marker, color="#444444",
                   linestyle="none", markersize=6, label=lbl)
        for marker, lbl in THINK_STYLE.values()
    ]

    all_handles = group1 + separator + group2

    fig.legend(
        handles=all_handles,
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(0.885, 0.992),
        handlelength=1.25,
        handletextpad=0.5,
        borderpad=0.6,
        labelspacing=0.35,
    )
    fig.suptitle("Accuracy vs. Token Cost per Benchmark", y=1.02, fontsize=9.5)
    fig.savefig(OUT_DIR / "pareto_per_benchmark.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/pareto_per_benchmark.png")


# ─────────────────────────── entry point ─────────────────────────────────────

def main() -> None:
    print("Loading metrics…")
    data = load_all()
    n    = sum(len(v) for v in data.values())
    print(f"  Loaded {n} configurations across {len(data)} categories.")
    if n == 0:
        print("ERROR: No metrics found."); return

    verify_data_sources(data)
    print_summary(data)
    print("\nGenerating plots…")
    plot_token_breakdown(data)
    plot_tool_calls(data)
    plot_per_benchmark_pareto(data)
    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
