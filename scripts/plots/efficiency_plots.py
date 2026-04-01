#!/usr/bin/env python3
"""
Efficiency plots for AgentFlow experiments.

Three categories compared:
  • Qwen3-32B  Direct   (large baseline, single-agent)
  • Qwen3-8B   Direct   (small baseline, single-agent)
  • Qwen3-8B   AgentFlow (proposed multi-agent system)

Outputs (data/results/plots/):
  token_breakdown.png       — avg tokens/query, grouped by category
  tool_calls_breakdown.png  — total tool calls by type, grouped by category

Usage:
    python scripts/plots/efficiency_plots.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

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
# "Sub. Think" and "All Think" are exclusive to the MAS category.
CATEGORIES: dict[str, dict] = {
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
        "config_labels": ["No Think", "Sub. Think", "Orch. Think", "All Think"],
    },
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
}

ALL_CONFIGS: list[tuple[str, str, str, str]] = [
    (cat_key, *cfg)
    for cat_key, cat in CATEGORIES.items()
    for cfg in cat["configs"]
]

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
    """
    Load per-run metrics and compute all reported aggregate values.

    Computation notes (source: metrics.json):
      - n = parsed from overall.num_correct string (e.g., "23 of 200")
      - accuracy (%) = overall.accuracy * 100
      - tokens_per_query = token_usage.total_tokens / n
      - prompt_tokens_per_query = token_usage.prompt_tokens / n
      - completion_tokens_per_query = token_usage.completion_tokens / n
      - wall_sec_per_q = (end_time - start_time).total_seconds() / n
        This is throughput latency (batched run wall-time / questions), not
        isolated single-query sequential latency.
      - tool_calls[tool] = sum_benchmarks int(tool_usage[tool])
      - avg_* values = arithmetic mean across available benchmarks in per_bm
    """
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
            # Throughput-based latency: total wall-clock time for the entire
            # benchmark run (all questions batched in parallel) divided by the
            # number of questions.  Not equivalent to sequential single-query
            # latency, but a fair apples-to-apples comparison across configs.
            wall_sec = (
                datetime.fromisoformat(m["end_time"]) -
                datetime.fromisoformat(m["start_time"])
            ).total_seconds()
            per_bm[bm] = {
                "accuracy":          ov["accuracy"] * 100,
                "tokens_per_query":  tu["total_tokens"] / n,
                "prompt_tokens":     tu["prompt_tokens"] / n,
                "completion_tokens": tu["completion_tokens"] / n,
                "wall_sec_per_q":    wall_sec / n,  # seconds per question
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
            "avg_sec_per_query":    np.mean([v["wall_sec_per_q"]    for v in per_bm.values()]),
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
                         cat_key: str, y_frac: float = -0.50,
                         fontsize: float = 7.5) -> None:
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
            ha="center", va="top", fontsize=fontsize,
            color=cat["color"], fontweight="bold", clip_on=False)


def _category_sidebar(ax, y_positions: list[float], bar_h: float,
                       cat_key: str, x_frac: float = 1.07,
                       align: str = "right") -> None:
    """Draw a coloured vertical rule + label beside a group of bars.

    align="right"  — rule + label to the right of the axes (default).
    align="left"   — rule + label to the left  of the axes; text rotates
                     bottom-to-top so it reads naturally on a left margin.
    """
    cat   = CATEGORIES[cat_key]
    y0    = min(y_positions) - bar_h * 0.5
    y1    = max(y_positions) + bar_h * 0.5
    ym    = (y0 + y1) / 2
    trans = ax.get_yaxis_transform()   # y in data, x in axes fraction
    ax.plot([x_frac, x_frac], [y0, y1], transform=trans,
            color=cat["color"], lw=2.0, solid_capstyle="round",
            clip_on=False)
    # Split "(Direct)" labels onto two lines; keep "(MAS)" on one line
    raw   = cat["label"]
    label = raw.replace(" (", "\n(")
    rot = 0
    ax.text(x_frac + 0.2, ym, label, transform=trans,
            ha="center", va="center", fontsize=10.5, rotation=rot,
            rotation_mode="anchor", linespacing=1.2,
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
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    # x-ticks
    all_x, all_lbl = [], []
    for cat_key, recs in data.items():
        for r, xi in zip(recs, positions[cat_key]):
            all_x.append(xi)
            all_lbl.append(r["config_label"])

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_lbl, fontsize=9.0, rotation=45, ha="right", va="top")

    # Axis styling — light theme
    ax.set_ylabel("Avg. tokens per query (K)", fontsize=11.0)
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
        _category_underline(ax, positions[cat_key], BAR_W, cat_key, y_frac=-0.38,
                            fontsize=11.5)

    # Title and legend in the space above the axes
    fig.text(0.5, 0.97, "Token usage per configuration",
             ha="center", va="top", fontsize=11.5,
             transform=fig.transFigure)

    fig.legend(handles=[
        mpatches.Patch(facecolor="#888888",                          label="Prompt tokens"),
        mpatches.Patch(facecolor="#888888", hatch="///", alpha=0.55, label="Completion tokens"),
    ], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.938),
       facecolor="white", edgecolor="#cccccc", fontsize=9.5)

    fig.savefig(OUT_DIR / "token_breakdown.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/token_breakdown.png")


# ─────────────────────────── Figure 2: Latency breakdown ────────────────────

_LATENCY_FOOTNOTE = (
    "Each bar = (end\_time \u2212 start\_time) / n\_samples, where start/end are taken from "
    "metrics.json and n\_samples is the number of evaluation questions in that benchmark run. "
    "All questions are processed in a single batched pass on the cluster (parallel execution), "
    "so this figure reflects \u2018throughput latency\u2019 \u2014 average cluster wall-time consumed per "
    "question \u2014 rather than the sequential latency of a single isolated query. "
    "Values are averaged over the five benchmarks (GAIA, GPQA, AIME, MuSiQue, HLE)."
)


def plot_latency_breakdown(data: dict[str, list[dict]],
                           with_footnote: bool = False,
                           benchmark: str | None = None) -> None:
    """
    Vertical bar chart: wall-clock seconds per query.

    How "seconds per query" is computed
    ------------------------------------
    Each metrics.json records start_time and end_time for the whole benchmark
    run.  All questions in a run are processed in a single batched pass on the
    cluster (questions run in parallel), so:

        wall_sec_per_q = (end_time - start_time).total_seconds() / n_samples

    This is a *throughput-based* latency figure — how many seconds of cluster
    wall-time were consumed per question on average.  It is NOT the sequential
    latency for a single isolated query.

    Parameters
    ----------
    with_footnote : bool
        If True, adds a detailed explanatory footnote and saves as
        latency_breakdown_footnote.png (or latency_breakdown_<bm>_footnote.png).
    benchmark : str | None
        If None (default), bar heights are averaged over all five benchmarks.
        If a benchmark key (e.g. "gaia"), only that dataset's latency is shown.

    Style mirrors plot_token_breakdown exactly (spacing, colours, underlines).
    """
    BAR_W     = 0.60
    BAR_GAP   = 0.10
    GROUP_GAP = 1.20

    positions: dict[str, list[float]] = {}
    x = 0.0
    for cat_key, recs in data.items():
        positions[cat_key] = [x + i * (BAR_W + BAR_GAP) for i in range(len(recs))]
        x += len(recs) * (BAR_W + BAR_GAP) - BAR_GAP + GROUP_GAP

    fig_h   = 5.4 if with_footnote else 4.6
    bot_adj = 0.42 if with_footnote else 0.38
    fig, ax = plt.subplots(figsize=(6.75, fig_h))

    for cat_key, recs in data.items():
        base_color = CATEGORIES[cat_key]["color"]
        for r, xi in zip(recs, positions[cat_key]):
            if benchmark is None:
                # Average wall-clock seconds/query across all available benchmarks.
                val = r["avg_sec_per_query"]
            else:
                # Single-dataset latency; skip bar if this benchmark wasn't run.
                if benchmark not in r["per_benchmark"]:
                    continue
                val = r["per_benchmark"][benchmark]["wall_sec_per_q"]
            ax.bar(xi, val, BAR_W, color=base_color, zorder=2)
            ax.text(xi, val + 0.4, f"{val:.1f}s",
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    all_x, all_lbl = [], []
    for cat_key, recs in data.items():
        for r, xi in zip(recs, positions[cat_key]):
            all_x.append(xi)
            all_lbl.append(r["config_label"])

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_lbl, fontsize=9.0, rotation=45, ha="right", va="top",
                       color="black")
    ax.set_ylabel("Wall-clock seconds per query", fontsize=11.0)
    ax.set_xlim(-0.55, all_x[-1] + 0.55)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.5)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    fig.subplots_adjust(left=0.10, right=0.97, bottom=bot_adj, top=0.91)

    for cat_key, recs in data.items():
        _category_underline(ax, positions[cat_key], BAR_W, cat_key, y_frac=-0.36,
                            fontsize=11.5)

    bm_label = BM_LABELS.get(benchmark, benchmark) if benchmark else "average over all benchmarks"
    fig.text(0.5, 0.97, f"Latency per configuration  —  {bm_label}",
             ha="center", va="top", fontsize=11.5, transform=fig.transFigure)

    if with_footnote:
        fig.text(0.5, 0.01, _LATENCY_FOOTNOTE,
                 ha="center", va="bottom", fontsize=8.0, color="#555555",
                 style="italic", transform=fig.transFigure,
                 multialignment="center", wrap=True)

    # File naming: latency_breakdown[_<bm>][_footnote].png
    bm_suffix  = f"_{benchmark}" if benchmark else ""
    fn_suffix  = "_footnote"     if with_footnote else ""
    fname      = f"latency_breakdown{bm_suffix}{fn_suffix}.png"
    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/{fname}")


# ─────────────────────────── Figure 3: Tool-call breakdown ───────────────────

TOOL_KEYS   = ["web_search", "code_generator", "text_inspector",
               "image_inspector", "mind_map"]
TOOL_LABELS = ["Web search", "Code generator", "Text inspector", "Image insp.", "Mind map"]
# Wong orange · Tol reddish-purple · Tol teal — none overlaps the config blues/reds/greens
TOOL_COLORS = ["#AA4499", "#44AA99", "#E69F00", "#7B1FA2", "#E53935"]

TOOL_CALL_BAR_H = 0.30
TOOL_CALL_ROW_GAP = 0.06
TOOL_CALL_GROUP_GAP = 0.18
TOOL_CALL_XMAX = 1300
TOOL_CALL_TOTAL_XPAD = 10
TOOL_CALL_TITLE_Y = 0.965
TOOL_CALL_LEGEND_Y = 0.928
TOOL_CALL_SUBPLOT = dict(left=0.23, right=0.80, top=0.875, bottom=0.12)


def plot_tool_calls(data: dict[str, list[dict]]) -> None:
    """
    Per-benchmark small-multiples stacked horizontal bar chart.

    Layout: 2 × 3 grid — five panels (one per benchmark) + one legend panel.

    Each panel shows tool-call counts for every configuration, stacked by
    tool type (web_search, code_generator, text_inspector).  The shared
    x-axis is scaled to the per-benchmark maximum so that within-benchmark
    proportions are clear and cross-benchmark differences are visible.

    Y-tick labels are coloured by category and appear only on the left-most
    column; a subtle background band per category group aids scanning.

    Reported values:
      - Each stacked segment = tool calls for one tool for that (config, bm).
      - Number label at bar end = total tool calls for that (config, bm).
      - `image_inspector` and `mind_map` are intentionally excluded.
    """
    # ── collect rows ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    row_cats: list[str] = []
    for cat_key, recs in data.items():
        for r in recs:
            if any(
                sum(int(v) for v in
                    r["per_benchmark"].get(bm, {}).get("tool_usage", {}).values()) > 0
                for bm in BENCHMARKS
            ):
                rows.append(r)
                row_cats.append(cat_key)

    if not rows:
        print("  (No tool calls; skipping)")
        return

    # ── active tools ──────────────────────────────────────────────────────────
    excluded = {"image_inspector", "mind_map"}
    all_counts: dict[str, int] = {}
    for r in rows:
        for bm_data in r["per_benchmark"].values():
            for k, v in bm_data.get("tool_usage", {}).items():
                all_counts[k] = all_counts.get(k, 0) + int(v)
    active_tools = [
        (tk, tl, tc)
        for tk, tl, tc in zip(TOOL_KEYS, TOOL_LABELS, TOOL_COLORS)
        if all_counts.get(tk, 0) > 0 and tk not in excluded
    ]
    if not active_tools:
        print("  (No active tools after filtering; skipping)")
        return

    # ── y-positions ───────────────────────────────────────────────────────────
    BAR_H = TOOL_CALL_BAR_H
    y_pos: list[float] = []
    cat_ys: dict[str, list[float]] = {}
    y, prev = 0.0, None
    for r, ck in zip(rows, row_cats):
        if prev is not None and ck != prev:
            y += TOOL_CALL_GROUP_GAP
        y_pos.append(y)
        cat_ys.setdefault(ck, []).append(y)
        y += BAR_H + TOOL_CALL_ROW_GAP
        prev = ck
    y_arr = np.array(y_pos)
    y_lo  = y_arr[-1] + BAR_H * 0.9   # bottom of inverted axis
    y_hi  = -0.40                       # top of inverted axis

    # ── global x-max (across all per-benchmark totals) ────────────────────────
    per_bm_max = max(
        sum(int(r["per_benchmark"].get(bm, {}).get("tool_usage", {}).get(tk, 0))
            for tk, _, _ in active_tools)
        for r in rows for bm in BENCHMARKS
    )
    x_max = int(per_bm_max * 1.22) + 15

    # ── figure: 1 row × 5 cols ───────────────────────────────────────────────
    fig_h = max(3.2, (y_arr[-1] + BAR_H * 1.2) + 1.0)
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(13.5, fig_h), squeeze=True)

    for i, bm in enumerate(BENCHMARKS):
        ax = axes[i]

        # subtle category background bands — symmetric half-row padding
        half_row = (BAR_H + TOOL_CALL_ROW_GAP) / 2
        for ck, ys in cat_ys.items():
            ax.axhspan(min(ys) - half_row, max(ys) + half_row,
                       color=CATEGORIES[ck]["color"], alpha=0.06, zorder=0)

        # stacked bars
        lefts = np.zeros(len(rows))
        for tk, tl, tc in active_tools:
            vals = np.array([
                int(r["per_benchmark"].get(bm, {}).get("tool_usage", {}).get(tk, 0))
                for r in rows
            ], dtype=float)
            ax.barh(y_arr, vals, BAR_H, left=lefts, label=tl,
                    color=tc, alpha=0.92, edgecolor="white", linewidth=0.4, zorder=2)
            lefts += vals

        # total labels at bar end (show 0 explicitly when no tool calls)
        for yi, total in zip(y_arr, lefts):
            ax.text(total + x_max * 0.02, yi, str(int(total)),
                    va="center", ha="left", fontsize=6.2,
                    color="#333333", clip_on=False)

        # axes formatting
        ax.set_title(BM_LABELS[bm], fontsize=8.5, fontweight="bold", pad=3)
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.5, zorder=1)
        ax.yaxis.grid(False)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7.0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

        # y-tick labels: only leftmost panel, black font
        ax.set_yticks(y_arr)
        if i == 0:
            ax.set_yticklabels([r["config_label"] for r in rows],
                               fontsize=7.3, color="black")
        else:
            ax.set_yticklabels([])

    # ── category sidebars on the leftmost panel ──────────────────────────────
    for ck, ys in cat_ys.items():
        _category_sidebar(axes[0], ys, BAR_H, ck, x_frac=-0.80, align="left")

    # ── legend (tools only) just below title ─────────────────────────────────
    tool_handles = [mpatches.Patch(facecolor=tc, alpha=0.92, label=tl)
                    for _, tl, tc in active_tools]

    SA_LEFT, SA_RIGHT = 0.17, 0.97
    # The sidebar labels extend ~0.10 left of SA_LEFT in figure coords, so
    # the visual centre of the saved image sits left of the subplot midpoint.
    subplot_cx = (SA_LEFT + SA_RIGHT) / 2   # centre of bar area (0.57)
    content_cx = (SA_LEFT - 0.10 + SA_RIGHT) / 2  # centre of all content (~0.52)

    fig.subplots_adjust(top=0.86, bottom=0.10, left=SA_LEFT, right=SA_RIGHT, wspace=0.05)
    fig.text(content_cx, 0.02, "Tool calls",
             ha="center", va="bottom", fontsize=8.0)

    fig.suptitle("Tool-call breakdown per benchmark",
                 fontsize=9.5, x=content_cx, y=0.980, va="bottom")
    fig.legend(
        handles=tool_handles,
        loc="upper center",
        bbox_to_anchor=(content_cx, 0.973),
        ncol=len(tool_handles),
        frameon=True, framealpha=0.95, edgecolor="#cccccc",
        borderpad=0.45, handlelength=1.2, handletextpad=0.55,
        columnspacing=0.8, fontsize=7.5,
    )
    fig.savefig(OUT_DIR / "tool_calls_breakdown.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/tool_calls_breakdown.png")


# ─────────────────────────── Figure 4: Latency heatmap table ─────────────────

# Orange colormap: lighter = lower latency (faster / better),
#                  darker  = higher latency (slower / worse).
_LAT_CMAP = LinearSegmentedColormap.from_list(
    "lat_orange", ["#fff7ed", "#ea580c"], N=256
)

# Table geometry (inches) — mirrors generate_results_table.py column layout.
_TBL_COL = [
    (0.08, 1.10),   # 0  Model
    (1.18, 1.00),   # 1  Tools      (+0.20 wider)
    (2.18, 1.15),   # 2  Thinking
    (3.33, 0.82),   # 3  GAIA
    (4.15, 0.82),   # 4  GPQA
    (4.97, 0.82),   # 5  AIME
    (5.79, 0.92),   # 6  MuSiQue
    (6.71, 0.82),   # 7  HLE
    (7.53, 0.82),   # 8  Avg
]
_TBL_FIG_W = 8.43
_ROW_H     = 0.400   # matches main results table ROW_H
_HDR_H     = 0.33
_LEG_H     = 0.40
_LR_PAD    = 0.08
_FONT_SZ   = 10.0
_SMALL_SZ  = 9.0


def plot_latency_heatmap(data: dict[str, list[dict]]) -> None:
    """
    Heatmap table of wall-clock seconds per query.

    Rows  = configurations (grouped by category with horizontal rules).
    Cols  = GAIA | GPQA | AIME | MuSiQue | HLE | Avg.
    Color = orange, globally normalised: lighter → faster (lower latency).
    """
    # ── build flat row list ───────────────────────────────────────────────────
    # Labels match the main results table (generate_results_table.py) exactly:
    #   Tools:    "—" (no_tools) | "Direct" (direct_tools) | "Sub-agent" (subagent_tools)
    #   Thinking: "—" (none) | "Orchestrator" (orchestrator) | "Sub-agents" (subagents) | "All" (all)
    #   Model:    shown only on the first row of each model group (blank thereafter)
    _TOOLS_LBL   = {"no_tools": "—", "direct_tools": "Direct", "subagent_tools": "Sub-agent"}
    _THINK_LBL   = {"none": "—", "orchestrator": "Orchestrator",
                    "subagents": "Sub-agents", "all": "All"}
    # Canonical model name per category (same as main table)
    _MODEL_LBL   = {"32B Direct": "Qwen3-32B", "8B Direct": "Qwen3-8B", "8B MAS": "Qwen3-8B"}

    rows = []
    prev_model_name: str | None = None
    for cat_key, recs in data.items():
        model_name = _MODEL_LBL[cat_key]
        for r in recs:
            # Show the model name only on the first row where it changes
            show_model  = model_name != prev_model_name
            prev_model_name = model_name
            row: dict = {
                "cat_key":   cat_key,
                "model_lbl": model_name if show_model else "",
                "tools_lbl": _TOOLS_LBL[r["tools"]],
                "think_lbl": _THINK_LBL[r["think"]],
            }
            for bm in BENCHMARKS:
                row[bm] = (r["per_benchmark"][bm]["wall_sec_per_q"]
                           if bm in r["per_benchmark"] else None)
            vals = [v for v in (row[bm] for bm in BENCHMARKS) if v is not None]
            row["avg"] = float(np.mean(vals)) if vals else None
            rows.append(row)

    # ── global normalisation across all latency values ───────────────────────
    all_vals = [row[col] for row in rows
                for col in BENCHMARKS + ["avg"] if row[col] is not None]
    norm = Normalize(vmin=min(all_vals), vmax=max(all_vals))

    # ── figure setup ─────────────────────────────────────────────────────────
    n_rows = len(rows)
    fig_h  = _HDR_H + n_rows * _ROW_H + _LEG_H
    matplotlib.rcParams.update({"font.family": "serif", "font.size": _FONT_SZ,
                                 "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(_TBL_FIG_W, fig_h))
    ax.set_xlim(0, _TBL_FIG_W)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    def hline(y, lw=0.6, **kw):
        ax.plot([_LR_PAD, _TBL_FIG_W - _LR_PAD], [y, y], "k-", lw=lw, **kw)

    def cx(ci):   return _TBL_COL[ci][0] + _TBL_COL[ci][1] / 2
    def lx(ci):   return _TBL_COL[ci][0] + 0.04

    # ── header ───────────────────────────────────────────────────────────────
    col_hdrs = ["Model", "Tools", "Thinking"] + \
               [BM_LABELS[bm] for bm in BENCHMARKS] + ["Avg."]
    y_top = fig_h - 0.04
    hline(y_top, lw=0.9)
    hy = y_top - _HDR_H / 2
    for ci, lbl in enumerate(col_hdrs):
        hdr_kw: dict = {"fontsize": _FONT_SZ}
        if ci == 8:  # Avg. column — emphasise header
            hdr_kw["fontweight"] = "bold"
        if ci < 3:
            ax.text(lx(ci), hy, lbl, ha="left", va="center", **hdr_kw)
        else:
            ax.text(cx(ci), hy, lbl, ha="center", va="center", **hdr_kw)
    hline(y_top - _HDR_H, lw=0.5)

    # ── data rows ─────────────────────────────────────────────────────────────
    bm_cols = BENCHMARKS + ["avg"]
    y = y_top - _HDR_H
    prev_cat = None
    for row in rows:
        y  -= _ROW_H
        yc  = y + _ROW_H / 2

        # Separator between category groups; stronger at the 8B → 32B model boundary
        if row["cat_key"] != prev_cat:
            if prev_cat is not None:
                cross_model = _MODEL_LBL[row["cat_key"]] != _MODEL_LBL[prev_cat]
                hline(y + _ROW_H, lw=0.8 if cross_model else 0.3,
                      alpha=0.75 if cross_model else 0.45)
            prev_cat = row["cat_key"]

        # Plain black text — no RGB category colouring
        if row["model_lbl"]:
            ax.text(lx(0), yc, row["model_lbl"],
                    ha="left", va="center", fontsize=_FONT_SZ)
        ax.text(lx(1), yc, row["tools_lbl"], ha="left", va="center",
                fontsize=_FONT_SZ)
        ax.text(lx(2), yc, row["think_lbl"], ha="left", va="center",
                fontsize=_FONT_SZ)

        for ci, col in enumerate(bm_cols, start=3):
            val = row[col]
            xc_pos      = cx(ci)
            col_x, col_w = _TBL_COL[ci]
            cell_txt_kw: dict = {"fontsize": _FONT_SZ}
            if ci == 8:  # Avg. column — bold figures
                cell_txt_kw["fontweight"] = "bold"

            if val is None:
                ax.text(xc_pos, yc, "—", ha="center", va="center",
                        **cell_txt_kw)
                continue

            # coloured cell background
            ax.add_patch(mpatches.Rectangle(
                (col_x, y), col_w, _ROW_H,
                facecolor=_LAT_CMAP(norm(val)),
                edgecolor="none", zorder=0,
            ))
            # cell value
            ax.text(xc_pos, yc, f"{val:.1f}",
                    ha="center", va="center", zorder=2, **cell_txt_kw)

    y_bot = y
    # Bold outline around the average column (header + all data rows)
    _ci_avg = 8
    ax_x, ax_w = _TBL_COL[_ci_avg]
    ax.add_patch(mpatches.Rectangle(
        (ax_x, y_bot), ax_w, y_top - y_bot,
        fill=False, edgecolor="#1a1a1a", linewidth=2.0,
        joinstyle="miter", zorder=6,
    ))
    hline(y_bot, lw=0.9)

    # ── legend (gradient bar) ─────────────────────────────────────────────────
    yl    = y_bot - _LEG_H / 2
    bar_w = 1.20
    bar_h = 0.10
    bar_x = _LR_PAD + 1.70
    ax.text(_LR_PAD, yl, "Latency (s/query):", ha="left", va="center",
            fontsize=_SMALL_SZ)
    ax.text(bar_x - 0.07, yl, "Fast", ha="right", va="center",
            fontsize=_SMALL_SZ)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient,
              extent=[bar_x, bar_x + bar_w, yl - bar_h / 2, yl + bar_h / 2],
              aspect="auto", cmap=_LAT_CMAP, origin="lower", zorder=1)
    ax.add_patch(mpatches.Rectangle(
        (bar_x, yl - bar_h / 2), bar_w, bar_h,
        linewidth=0.4, edgecolor="black", facecolor="none", zorder=2,
    ))
    ax.text(bar_x + bar_w + 0.05, yl, "Slow", ha="left", va="center",
            fontsize=_SMALL_SZ)

    fig.savefig(OUT_DIR / "latency_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved → {OUT_DIR}/latency_heatmap.png")


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
    plot_latency_breakdown(data, with_footnote=False)   # avg, clean
    plot_latency_breakdown(data, with_footnote=True)    # avg, annotated
    plot_latency_heatmap(data)                          # per-dataset heatmap table
    plot_tool_calls(data)
    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
