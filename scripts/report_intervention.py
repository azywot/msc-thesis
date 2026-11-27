"""Figures report for the tool-call intervention.

Reads the collected trajectories and the rescue folder (rescue.jsonl, gate.json, optional
blind.jsonl), plus optionally comparison.json from compare_arms.py, and writes report.md,
report.json and figures/*.png and *.pdf. Works on a subset trial run and on the full run.

    python scripts/report_intervention.py --collected data/.../collected_*.jsonl \
        --rescue-dir experiments/results/rsi/phase2/rescue_8b --out-dir <dir> \
        [--comparison comparison.json] [--examples 8]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_engine.utils.parsing import parse_tool_call  # noqa: E402
from rsi.intervention.records import domain_of, load_collected  # noqa: E402
from rsi.intervention.report import summarize  # noqa: E402

# Same rcParams as scripts/plots/efficiency_plots.py.
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e0e0e0",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
    }
)

# Wong palette, one colour per role, the same in every figure.
RESCUE, CONTROL = "#0072B2", "#E69F00"
ARM_C, ARM_I, BASE = "#009E73", "#0072B2", "#999999"
KIND_COLOURS = {"rewrite": "#0072B2", "follow_up": "#E69F00", "insert": "#009E73"}
KINDS = tuple(KIND_COLOURS)
PASS1, PASS3, NET, GROSS = "#999999", "#56B4E9", "#0072B2", "#CC79A7"
NOT_RESCUED, ORIGINAL = "#BBBBBB", "#999999"
TEXT = "#222222"
DOMAINS = ("search", "math")
BENCH_LABELS = {"gaia": "GAIA", "gpqa": "GPQA", "aime": "AIME", "musique": "MuSiQue", "hle": "HLE"}
NO_DATA = "no data"


# ───────────────────────────── numbers ──────────────────────────────────────


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for k of n, as fractions; (0, 0) when n is 0."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def net_interval(both: int, rescue_only: int, control_only: int, neither: int):
    """Newcombe's paired interval (method 10) for rescue rate minus control rate, as fractions.

    The 2x2 table is over the same questions: solved by both draws, by the rescue only, by the
    control only, by neither. Returns (0, 0) for an empty table."""
    n = both + rescue_only + control_only + neither
    if n <= 0:
        return 0.0, 0.0
    k1, k2 = both + rescue_only, both + control_only
    p1, p2 = k1 / n, k2 / n
    l1, u1 = wilson(k1, n)
    l2, u2 = wilson(k2, n)
    margins = (k1 * (n - k1) * k2 * (n - k2)) ** 0.5
    c = both * neither - rescue_only * control_only
    c = max(c - n / 2, 0.0) if c > 0 else c  # Newcombe corrects only a positive AD - BC
    phi = c / margins if margins else 0.0
    d = p1 - p2
    lo = d - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    hi = d + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return lo, hi


def net_label(v: dict) -> str:
    """Text written above a domain in rescue_vs_control."""
    return "no data" if not v["zero_of_n"] else f"net {v['net_points']:+.1f} points"


def ceiling_label(v: dict) -> str:
    return "no data" if not v["questions"] else ""


def _pct(k: int, n: int) -> float:
    return 100.0 * k / n if n else 0.0


def _read_rows(path: Path) -> List[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _call_names_from_history(rec: dict) -> List[str]:
    names = []
    for action in rec.get("action_history") or []:
        try:
            names.append(json.loads(action["command"]).get("name"))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
    return names


def _call_names_from_messages(messages: Sequence[dict], plan: Optional[str]) -> List[str]:
    names = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        if i == 2 and plan and m.get("content") == plan:
            continue  # the planning turn is not a tool call
        call = parse_tool_call(m.get("content") or "")
        if call:
            names.append(call.get("name"))
    return names


def _rescued_rows(rows: List[dict]) -> Dict[object, dict]:
    """First fix row (by attempt) that rescued each question, as the gate counts it."""
    out: Dict[object, dict] = {}
    for r in sorted(rows, key=lambda r: r.get("attempt") or 0):
        if r.get("status") == "ok" and r.get("correct"):
            out.setdefault(r["question_id"], r)
    return out


def compute(
    records: List[dict], rows: List[dict], gate: dict, pool=None, candidate_ids=None
) -> dict:
    if pool is not None:
        records = [r for r in records if r.get("question_id") in pool]
    by_q: Dict[object, List[dict]] = {}
    for rec in records:
        by_q.setdefault(rec.get("question_id"), []).append(rec)
    first = {q: next((r for r in rs if r.get("pass") == 1), rs[0]) for q, rs in by_q.items()}
    rescued_row = _rescued_rows(rows)
    out: dict = {"threshold_points": gate["gate"]["threshold_points"], "domains": {}}
    for domain in DOMAINS:
        g = gate.get(domain)
        if g is None:
            continue
        qs = [q for q, rec in first.items() if domain_of(rec["data_source"]) == domain]
        n_q = len(qs)
        pass1 = sum(1 for q in qs if first[q].get("correct"))
        pass3 = sum(1 for q in qs if any(r.get("correct") for r in by_q[q]))
        zero, rescued, control = g["zero_of_n"], g["rescued"], g["control_solved"]
        drows = [r for r in rows if r.get("domain") == domain]
        ok_rows = [r for r in drows if r.get("status") == "ok"]
        ok_questions = {r["question_id"] for r in ok_rows}
        kinds = {}
        for kind in KINDS:
            k_ok = [r for r in ok_rows if r.get("kind") == kind]
            k_res = [r for r in k_ok if r.get("correct")]
            kinds[kind] = {
                "ok": len(k_ok),
                "rescued": len(k_res),
                "rescue_rate": _pct(len(k_res), len(k_ok)),
            }
        steps = {"rescued": [], "not_rescued": []}
        for r in ok_rows:
            steps["rescued" if r.get("correct") else "not_rescued"].append(r.get("step"))
        # behaviour: original pass-1 trajectory vs the rescuing trajectory
        orig, resc = [], []
        for q, r in rescued_row.items():
            if r.get("domain") != domain or q not in first:
                continue
            rec = first[q]
            orig.append(_call_names_from_history(rec))
            resc.append(
                _call_names_from_messages(r.get("messages") or [], rec.get("raw_query_analysis"))
            )
        if domain == "math":
            tool, measure = "code_generator", "share_with_code_call"
            f = lambda lists: _pct(sum(1 for ns in lists if tool in ns), len(lists))  # noqa: E731
        else:
            tool, measure = "web_search", "mean_web_search_calls"
            f = lambda lists: (  # noqa: E731
                sum(ns.count(tool) for ns in lists) / len(lists) if lists else 0.0
            )
        lo1, hi1 = wilson(rescued, zero)
        loc, hic = wilson(control, zero)
        # Same set the gate counts over: the domain's candidates.
        cands = (
            set(candidate_ids[domain])
            if candidate_ids is not None
            else {r["question_id"] for r in drows}
        )
        res_ids = {q for q, r in rescued_row.items() if r.get("domain") == domain and q in cands}
        ctl_ids = {r["question_id"] for r in drows if r.get("control_correct")} & cands
        if len(res_ids) != rescued or len(ctl_ids) != control:
            raise ValueError(
                f"{domain}: rescue.jsonl gives {len(res_ids)} rescued and {len(ctl_ids)} "
                f"control-solved candidates, gate.json says {rescued} and {control}; "
                "the rescue folder is inconsistent"
            )
        both = len(res_ids & ctl_ids)
        r_only, c_only = len(res_ids) - both, len(ctl_ids) - both
        lon, hin = net_interval(both, r_only, c_only, zero - both - r_only - c_only)
        out["domains"][domain] = {
            "questions": n_q,
            "pass_at_1": pass1,
            "pass_at_1_pct": _pct(pass1, n_q),
            "pass_at_3": pass3,
            "pass_at_3_pct": _pct(pass3, n_q),
            "ceiling_net_pct": _pct(pass3 + rescued - control, n_q),
            "ceiling_gross_pct": _pct(pass3 + rescued, n_q),
            "zero_of_n": zero,
            "skipped": g["zero_of_n"] - g["candidates"],
            "candidates": g["candidates"],
            "ok_questions": len(ok_questions),
            "rescued": rescued,
            "rescue_rate": g["rescue_rate"],
            "rescue_ci": [100 * lo1, 100 * hi1],
            "control_solved": control,
            "control_rate": g["control_rate"],
            "control_ci": [100 * loc, 100 * hic],
            "net_points": g["net_points"],
            "net_ci": [100 * lon, 100 * hin],
            "rescue_rate_candidates": g["rescue_rate_candidates"],
            "control_rate_candidates": g["control_rate_candidates"],
            "net_points_candidates": g["net_points_candidates"],
            "kinds": kinds,
            "steps": steps,
            "behaviour": {
                "measure": measure,
                "tool": tool,
                "n_rescued": len(orig),
                "original": f(orig),
                "rescued": f(resc),
            },
        }
    if "blind" in gate:
        b = gate["blind"]
        n = b["n"]
        out["blind"] = {
            "n": n,
            "sighted_rate": 100.0 if n else 0.0,
            "blind_rescued": b["rescued"],
            "blind_rate": b["rate"],
            "blind_ci": [100 * x for x in wilson(b["rescued"], n)],
            "control_rescued": b["control_rescued"],
            "control_rate": b["control_rate"],
            "control_ci": [100 * x for x in wilson(b["control_rescued"], n)],
        }
    return out


# ───────────────────────────── figures ──────────────────────────────────────


def _save(fig, fig_dir: Path, name: str) -> None:
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _no_data(ax, note: str = NO_DATA) -> None:
    ax.text(0.5, 0.5, note, ha="center", va="center", transform=ax.transAxes, color="#777777")
    ax.set_xticks([])
    ax.set_yticks([])


def _err(lo_hi, value):
    return [[max(0.0, value - lo_hi[0])], [max(0.0, lo_hi[1] - value)]]


def fig_rescue_vs_control(data: dict, fig_dir: Path) -> None:
    doms = list(data["domains"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.0), gridspec_kw={"width_ratios": [2.2, 1]})
    w, gap = 0.17, 0.02
    top = 5.0
    for i, d in enumerate(doms):
        v = data["domains"][d]
        if not v["zero_of_n"]:
            continue
        for j, (key, ci, colour) in enumerate(
            (("rescue_rate", "rescue_ci", RESCUE), ("control_rate", "control_ci", CONTROL))
        ):
            x = i + (j - 0.5) * (w + gap)
            ax.bar(x, v[key], w, color=colour, label=None)
            ax.errorbar(x, v[key], yerr=_err(v[ci], v[key]), color="#333333", lw=0.8, capsize=2)
            top = max(top, v[ci][1])
    for i, d in enumerate(doms):
        v = data["domains"][d]
        ax.text(
            i,
            top + 3 if v["zero_of_n"] else top * 0.3,
            net_label(v),
            ha="center",
            color=TEXT if v["zero_of_n"] else "#777777",
        )
    ax.set_xlim(-0.6, len(doms) - 0.4)
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels([f"{d}\n(0-of-3: {data['domains'][d]['zero_of_n']})" for d in doms])
    ax.set_ylabel("share of 0-of-3 questions solved (%)")
    ax.set_ylim(0, (top + 8) * 1.3)
    ax.set_title("Rescue vs no-fix control")
    ax.legend(
        handles=[
            Patch(color=RESCUE, label="rescue (fixed call)"),
            Patch(color=CONTROL, label="no-fix control"),
        ],
        loc="upper left",
        ncol=2,
    )
    thr_y = data["threshold_points"] * 0.5
    for i, d in enumerate(doms):
        v = data["domains"][d]
        if not v["zero_of_n"]:
            ax2.text(i, thr_y, net_label(v), ha="center", color="#777777")
            continue
        ax2.bar(i, v["net_points"], 0.4, color=RESCUE)
        ax2.errorbar(
            i,
            v["net_points"],
            yerr=[[v["net_points"] - v["net_ci"][0]], [v["net_ci"][1] - v["net_points"]]],
            color="#333333",
            lw=0.8,
            capsize=2,
        )
    thr = data["threshold_points"]
    ax2.axhline(thr, color="#555555", ls="--", lw=0.9, label=f"gate threshold ({thr:g})")
    ax2.axhline(0, color="#999999", lw=0.6)
    ax2.set_xlim(-0.6, len(doms) - 0.4)
    ax2.set_xticks(range(len(doms)))
    ax2.set_xticklabels(doms)
    ax2.set_ylabel("net rescue (percentage points)")
    measured = [d for d in doms if data["domains"][d]["zero_of_n"]]
    lows = [data["domains"][d]["net_ci"][0] for d in measured] + [0]
    highs = [data["domains"][d]["net_ci"][1] for d in measured] + [thr]
    ax2.set_ylim(min(0, min(lows)) - 4, max(highs) * 1.2 + 4)
    ax2.set_title("Net vs gate")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, fig_dir, "rescue_vs_control")


def fig_ceiling(data: dict, fig_dir: Path) -> None:
    doms = list(data["domains"])
    series = [
        ("pass_at_1_pct", "pass@1", PASS1),
        ("pass_at_3_pct", "pass@3", PASS3),
        ("ceiling_net_pct", "pass@3 + net rescued", NET),
        ("ceiling_gross_pct", "pass@3 + gross rescued", GROSS),
    ]
    fig, ax = plt.subplots(figsize=(6.75, 3.2))
    w, gap = 0.16, 0.02
    for j, (key, label, colour) in enumerate(series):
        xs = [i + (j - 1.5) * (w + gap) for i in range(len(doms))]
        vals = [data["domains"][d][key] for d in doms]
        empty = [not data["domains"][d]["questions"] for d in doms]
        ax.bar(
            [x for x, e in zip(xs, empty) if not e],
            [v for v, e in zip(vals, empty) if not e],
            w,
            color=colour,
            label=label,
        )
        for x, v, e in zip(xs, vals, empty):
            if not e:
                ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=6.5, color=TEXT)
    for i, d in enumerate(doms):
        if ceiling_label(data["domains"][d]):
            ax.text(i, 8, ceiling_label(data["domains"][d]), ha="center", color="#777777")
    ax.set_xlim(-0.6, len(doms) - 0.4)
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels([f"{d}\n(n={data['domains'][d]['questions']})" for d in doms])
    ax.set_ylabel("share of all collected questions (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Ceiling on the training questions")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    _save(fig, fig_dir, "ceiling")


def fig_funnel(data: dict, fig_dir: Path) -> None:
    doms = list(data["domains"])
    stages = [
        ("0-of-3 questions", "zero_of_n"),
        ("skipped (planning answer or error)", "skipped"),
        ("candidates", "candidates"),
        ("fix parsed and passed leakage", "ok_questions"),
        ("rescued", "rescued"),
    ]
    fig, axes = plt.subplots(1, len(doms), figsize=(6.75, 2.8), squeeze=False)
    for ax, d in zip(axes[0], doms):
        v = data["domains"][d]
        if not v["zero_of_n"]:
            _no_data(ax, "no data: no 0-of-3 questions")
            ax.set_title(d)
            continue
        vals = [v[k] for _, k in stages]
        ys = list(range(len(stages)))[::-1]
        ax.barh(ys, vals, 0.55, color=RESCUE)
        for y, val in zip(ys, vals):
            ax.text(val + max(vals) * 0.02, y, str(val), va="center", color=TEXT)
        ax.set_yticks(ys)
        ax.set_yticklabels([s for s, _ in stages] if d == doms[0] else [""] * len(stages))
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_xlabel("questions")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(d)
    fig.tight_layout()
    _save(fig, fig_dir, "funnel")


def fig_fix_kinds(data: dict, fig_dir: Path) -> None:
    doms = list(data["domains"])
    fig, axes = plt.subplots(1, len(doms), figsize=(6.75, 2.9), squeeze=False)
    for ax, d in zip(axes[0], doms):
        kinds = data["domains"][d]["kinds"]
        if not any(k["ok"] for k in kinds.values()):
            _no_data(ax, "no data: no fixes passed")
            ax.set_title(d)
            continue
        w, gap = 0.3, 0.03
        for i, kind in enumerate(KINDS):
            k = kinds[kind]
            ax.bar(i - (w + gap) / 2, k["ok"], w, color=KIND_COLOURS[kind], alpha=0.35)
            ax.bar(i + (w + gap) / 2, k["rescued"], w, color=KIND_COLOURS[kind])
            ax.text(
                i + (w + gap) / 2,
                k["rescued"] + 0.3,
                f"{k['rescue_rate']:.0f}%" if k["ok"] else "",
                ha="center",
                fontsize=7,
                color=TEXT,
            )
        ax.set_xticks(range(len(KINDS)))
        ax.set_xticklabels(KINDS)
        ax.set_ylabel("fixes (count)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, max(k["ok"] for k in kinds.values()) * 1.5 + 1)
        ax.set_title(d)
        ax.legend(
            handles=[
                Patch(color="#888888", alpha=0.35, label="fixes that ran (ok)"),
                Patch(color="#888888", label="rescued (label: rescue rate)"),
            ],
            loc="upper right",
        )
    fig.tight_layout()
    _save(fig, fig_dir, "fix_kinds")


def fig_fix_step(data: dict, fig_dir: Path) -> None:
    doms = list(data["domains"])
    fig, axes = plt.subplots(1, len(doms), figsize=(6.75, 2.9), squeeze=False)
    for ax, d in zip(axes[0], doms):
        steps = data["domains"][d]["steps"]
        allsteps = [s for s in steps["rescued"] + steps["not_rescued"] if s is not None]
        if not allsteps:
            _no_data(ax, "no data: no fixes passed")
            ax.set_title(d)
            continue
        xs = list(range(min(allsteps), max(allsteps) + 1))
        w, gap = 0.38, 0.03
        ax.bar(
            [x - (w + gap) / 2 for x in xs],
            [steps["rescued"].count(x) for x in xs],
            w,
            color=RESCUE,
            label="rescued",
        )
        ax.bar(
            [x + (w + gap) / 2 for x in xs],
            [steps["not_rescued"].count(x) for x in xs],
            w,
            color=NOT_RESCUED,
            label="not rescued",
        )
        ax.set_xticks(xs)
        ax.set_xlim(min(xs) - 2, max(xs) + 2)
        top_count = max(max(steps["rescued"].count(x), steps["not_rescued"].count(x)) for x in xs)
        ax.set_ylim(0, top_count * 1.35 + 0.5)
        ax.set_xlabel("step of the fixed call (0 = first call)")
        ax.set_ylabel("fixes (count)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(d)
        ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, fig_dir, "fix_step")


def fig_behaviour(data: dict, fig_dir: Path) -> None:
    doms = list(data["domains"])
    fig, axes = plt.subplots(1, len(doms), figsize=(6.75, 2.9), squeeze=False)
    for ax, d in zip(axes[0], doms):
        b = data["domains"][d]["behaviour"]
        if not b["n_rescued"]:
            _no_data(ax, "no data: no rescued questions")
            ax.set_title(d)
            continue
        w = 0.35
        vals = [b["original"], b["rescued"]]
        ax.bar([-w / 2 - 0.02, w / 2 + 0.02], vals, w, color=[ORIGINAL, RESCUE])
        for x, v in zip([-w / 2 - 0.02, w / 2 + 0.02], vals):
            ax.text(
                x,
                v,
                f"{v:.1f}" if d == "search" else f"{v:.0f}%",
                ha="center",
                va="bottom",
                color=TEXT,
            )
        ax.set_xticks([-w / 2 - 0.02, w / 2 + 0.02])
        ax.set_xticklabels(["original", "rescued"])
        ax.set_ylim(0, max(vals) * 1.25 + 0.5)
        ax.set_xlim(-1, 1)
        ax.set_ylabel(
            "questions with a code_generator call (%)"
            if d == "math"
            else "web_search calls per question (mean)"
        )
        ax.set_title(f"{d} (n={b['n_rescued']} rescued)")
    fig.tight_layout()
    _save(fig, fig_dir, "behaviour")


def fig_blind(data: dict, fig_dir: Path) -> None:
    b = data["blind"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    items = [
        ("sighted\n(100% by construction)", b["sighted_rate"], None, RESCUE),
        ("blind rescue", b["blind_rate"], b["blind_ci"], RESCUE),
        ("blind no-fix control", b["control_rate"], b["control_ci"], CONTROL),
    ]
    for i, (_, v, ci, colour) in enumerate(items):
        ax.bar(i, v, 0.5, color=colour)
        if ci is not None:
            ax.errorbar(i, v, yerr=_err(ci, v), color="#333333", lw=0.8, capsize=2)
        ax.text(i, (ci[1] if ci is not None else v) + 2, f"{v:.0f}%", ha="center", color=TEXT)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([s for s, *_ in items])
    ax.set_ylabel(f"share of the {b['n']} blind-sample questions (%)")
    ax.set_ylim(0, 118)
    ax.set_title("Blind check (reflector without the gold answer)")
    fig.tight_layout()
    _save(fig, fig_dir, "blind")


def fig_benchmarks(comp: dict, fig_dir: Path) -> None:
    per = comp["benchmarks"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.1), gridspec_kw={"width_ratios": [4, 1]})
    arms = [("base", "base", BASE), ("arm_c", "arm C", ARM_C), ("arm_i", "arm I", ARM_I)]
    w, gap = 0.24, 0.02
    for j, (key, label, colour) in enumerate(arms):
        xs = [i + (j - 1) * (w + gap) for i in range(len(per))]
        ax.bar(xs, [100 * p["accuracy"][key] for p in per], w, color=colour, label=label)
        mean = comp.get("mean_accuracy", {}).get(key, 0.0)
        ax2.bar(j, 100 * mean, 0.6, color=colour)
        ax2.text(j, 100 * mean + 0.5, f"{100 * mean:.1f}", ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(range(len(per)))
    ax.set_xticklabels(
        [f"{BENCH_LABELS.get(p['benchmark'], p['benchmark'])}\np={p['mcnemar_p']:.3f}" for p in per]
    )
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy per benchmark (p: McNemar, arm I vs arm C)")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, max(100 * p["accuracy"][k] for p in per for k, _, _ in arms) * 1.25 + 1)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([a[1] for a in arms])
    ax2.set_ylabel("mean accuracy (%)")
    ax2.set_title("Five-benchmark mean")
    fig.tight_layout()
    _save(fig, fig_dir, "benchmarks")


BEHAVIOUR_LABELS = {
    "code_call_rate": "code_generator call",
    "multi_search_rate": "2+ web_search calls",
}


def behaviour_groups(comp: dict) -> List[Tuple[str, dict]]:
    groups = []
    for p in comp["benchmarks"]:
        for name, v in p.get("behaviour", {}).items():
            label = f"{BENCH_LABELS.get(p['benchmark'], p['benchmark'])}: {BEHAVIOUR_LABELS.get(name, name)}"
            groups.append((label, v))
    return groups


def fig_benchmark_behaviour(comp: dict, fig_dir: Path) -> None:
    groups = behaviour_groups(comp)
    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    if not groups:
        _no_data(ax)
    else:
        arms = [("base", "base", BASE), ("arm_c", "arm C", ARM_C), ("arm_i", "arm I", ARM_I)]
        w, gap = 0.24, 0.02
        for j, (key, label, colour) in enumerate(arms):
            xs = [i + (j - 1) * (w + gap) for i in range(len(groups))]
            ax.bar(xs, [100 * v[key] for _, v in groups], w, color=colour, label=label)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g for g, _ in groups], fontsize=7)
        ax.set_ylabel("questions with the behaviour (%)")
        ax.set_ylim(0, max(100 * v[k] for _, v in groups for k, _, _ in arms) * 1.25 + 1)
        ax.legend(loc="upper right", ncol=3)
    ax.set_title("Behaviour rates on the benchmarks")
    fig.tight_layout()
    _save(fig, fig_dir, "benchmark_behaviour")


# ───────────────────────────── examples ─────────────────────────────────────


def _cell(text, limit: Optional[int] = None) -> str:
    s = " ".join(str(text if text is not None else "").split())
    if limit is not None and len(s) > limit:
        s = s[: limit - 3] + "..."
    return s.replace("|", "\\|")


def _call_text(call: Optional[dict]) -> str:
    return json.dumps(call, ensure_ascii=False) if call else "none"


def select_examples(rows: List[dict], firsts: Dict[object, dict], n: int) -> List[dict]:
    """Rescued questions, balanced across domains (round robin), sorted by question_id."""
    rescued = _rescued_rows(rows)
    pools = {
        d: [rescued[q] for q in sorted(rescued, key=str) if rescued[q].get("domain") == d]
        for d in DOMAINS
    }
    out, i = [], 0
    while len(out) < n and any(i < len(p) for p in pools.values()):
        for d in DOMAINS:
            if i < len(pools[d]) and len(out) < n:
                out.append(pools[d][i])
        i += 1
    examples = []
    for r in out:
        rec = firsts.get(r["question_id"]) or {}
        history = rec.get("action_history") or []
        original = None
        if r.get("kind") == "rewrite" and r.get("step") is not None and r["step"] < len(history):
            try:
                original = json.loads(history[r["step"]]["command"])
            except (TypeError, ValueError, KeyError):
                original = None
        examples.append(
            {
                "domain": r["domain"],
                "question_id": r["question_id"],
                "question": r.get("question") or rec.get("question"),
                "kind": r.get("kind"),
                "step": r.get("step"),
                "original_call": _call_text(original),
                "fixed_call": _call_text(parse_tool_call(r.get("fix_text") or "")),
                "original_prediction": rec.get("prediction"),
                "rescued_prediction": r.get("prediction"),
                "gold": r.get("ground_truth"),
            }
        )
    return examples


def examples_table(examples: List[dict]) -> List[str]:
    if not examples:
        return ["No rescued questions to show.", ""]
    lines = [
        "| domain | question | kind, step | original call | fixed call | original prediction "
        "| rescued prediction | gold |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in examples:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(e["domain"]),
                    _cell(e["question"], 200),
                    _cell(f"{e['kind']}, {e['step']}"),
                    _cell(e["original_call"], 200),
                    _cell(e["fixed_call"], 200),
                    _cell(e["original_prediction"], 120),
                    _cell(e["rescued_prediction"], 120),
                    _cell(e["gold"], 120),
                ]
            )
            + " |"
        )
    return lines + [""]


# ───────────────────────────── markdown ─────────────────────────────────────


def _table(header: List[str], body: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return lines + [""]


def _fig(name: str) -> List[str]:
    return [f"![{name}](figures/{name}.png)", ""]


def render_md(data: dict, examples: List[dict], comp: Optional[dict], has_blind: bool) -> str:
    D = data["domains"]
    doms = list(D)
    L = ["# Tool-call intervention report", ""]
    L += [
        "Rates over the 0-of-3 questions (zero_of_n) are the gate's rates. Rate intervals are 95% "
        "Wilson intervals; the net interval is Newcombe's paired interval (method 10) over the "
        "same questions. On a small subset the intervals are wide: read the figures as a "
        "first look at the pipeline, not as the gate decision.",
        "",
        "## Rescue versus no-fix control",
        "",
    ]
    for d in doms:
        if not D[d]["zero_of_n"]:
            L += [f"Domain {d}: no data, there are no 0-of-3 questions in this run.", ""]
    L += _fig("rescue_vs_control")
    L += [
        "Left: for each domain, the share of 0-of-3 questions the orchestrator solved when the "
        "reflector's fixed call was injected (blue) and when the same step was simply resumed "
        "without a fix (orange), with 95% intervals; the net points are written above each "
        "domain. Right: the net (rescue minus control) on its own axis in percentage points, "
        "with the gate threshold dashed. The gate passes when the bar reaches the "
        "line; an interval wholly above it means the pass is unlikely to be noise.",
        "",
    ]
    L += _table(
        [
            "domain",
            "0-of-3",
            "rescued",
            "rescue %",
            "control solved",
            "control %",
            "net points",
            "net 95% interval",
        ],
        [
            [
                d,
                D[d]["zero_of_n"],
                D[d]["rescued"],
                f"{D[d]['rescue_rate']:.1f}",
                D[d]["control_solved"],
                f"{D[d]['control_rate']:.1f}",
                f"{D[d]['net_points']:+.1f}",
                f"{D[d]['net_ci'][0]:+.1f} to {D[d]['net_ci'][1]:+.1f}",
            ]
            for d in doms
        ],
    )
    L += ["## Ceiling on the training questions", ""] + _fig("ceiling")
    L += [
        "Per domain, the share of all collected questions solved at pass@1 (first pass), pass@3 "
        "(any of three passes), and pass@3 plus the questions the fixed call rescued, net of the "
        "no-fix control (blue) or gross (pink). This is a ceiling on what the decision changes "
        "could add on the training questions, not a benchmark gain: the reflector saw the gold "
        "answer, and the trained model still has to learn the decisions without it.",
        "",
    ]
    L += _table(
        ["domain", "questions", "pass@1 %", "pass@3 %", "pass@3 + net %", "pass@3 + gross %"],
        [
            [
                d,
                D[d]["questions"],
                f"{D[d]['pass_at_1_pct']:.1f}",
                f"{D[d]['pass_at_3_pct']:.1f}",
                f"{D[d]['ceiling_net_pct']:.1f}",
                f"{D[d]['ceiling_gross_pct']:.1f}",
            ]
            for d in doms
        ],
    )
    L += ["## Funnel", ""] + _fig("funnel")
    L += [
        "How many questions survive each stage, from the 0-of-3 questions down to the rescued "
        "ones. Skipped questions had an error record or were answered at the planning turn, so "
        "there is no tool call to fix. The next bar counts questions with at least one fix that "
        "parsed and passed the leakage guard.",
        "",
    ]
    L += _table(
        ["domain", "0-of-3", "skipped", "candidates", "ok (questions)", "rescued"],
        [
            [
                d,
                D[d]["zero_of_n"],
                D[d]["skipped"],
                D[d]["candidates"],
                D[d]["ok_questions"],
                D[d]["rescued"],
            ]
            for d in doms
        ],
    )
    L += ["## Fix kinds", ""] + _fig("fix_kinds")
    L += [
        "Fixes that ran (pale bars) and fixes that rescued the question (solid bars) by kind: "
        "rewrite replaces an existing call, follow_up adds a call after the last one, insert "
        "adds a call to a trajectory that made none. The label is the rescue rate for that kind.",
        "",
    ]
    rows_k = []
    for d in doms:
        for k in KINDS:
            v = D[d]["kinds"][k]
            rows_k.append([d, k, v["ok"], v["rescued"], f"{v['rescue_rate']:.1f}"])
    L += _table(["domain", "kind", "ok fixes", "rescued fixes", "rescue %"], rows_k)
    L += ["## Where in the trajectory the fix lands", ""] + _fig("fix_step")
    L += [
        "Histogram of the step of each fixed call (0 is the first call), rescued fixes in blue "
        "and fixes that did not rescue in grey.",
        "",
    ]
    rows_s = []
    for d in doms:
        for name in ("rescued", "not_rescued"):
            st = [s for s in D[d]["steps"][name] if s is not None]
            rows_s.append([d, name, len(st), f"{sum(st) / len(st):.2f}" if st else "-"])
    L += _table(["domain", "fixes", "count", "mean step"], rows_s)
    L += ["## Behaviour of rescued trajectories", ""] + _fig("behaviour")
    L += [
        "For rescued questions only, the original pass-1 trajectory (grey) against the rescued "
        "trajectory (blue). Math: share of questions with at least one code_generator call. "
        "Search: mean number of web_search calls per question.",
        "",
    ]
    L += _table(
        ["domain", "measure", "rescued questions", "original", "rescued"],
        [
            [
                d,
                D[d]["behaviour"]["measure"],
                D[d]["behaviour"]["n_rescued"],
                f"{D[d]['behaviour']['original']:.2f}",
                f"{D[d]['behaviour']['rescued']:.2f}",
            ]
            for d in doms
        ],
    )
    L += ["## Blind check", ""]
    if "blind" in data:
        b = data["blind"]
        L += _fig("blind")
        L += [
            "The blind check reruns the reflector without the gold answer on a sample of "
            "questions it rescued while sighted, so the sighted bar is 100% by construction and "
            "has no interval. "
            "The blind rescue rate against the blind no-fix control shows how much of the "
            "rescue survives without the answer.",
            "",
        ]
        L += _table(
            ["n", "sighted %", "blind rescued", "blind %", "control rescued", "control %"],
            [
                [
                    b["n"],
                    f"{b['sighted_rate']:.0f}",
                    b["blind_rescued"],
                    f"{b['blind_rate']:.1f}",
                    b["control_rescued"],
                    f"{b['control_rate']:.1f}",
                ]
            ],
        )
    else:
        L += [
            "No blind check in this rescue folder (no `blind.jsonl`), so there is no blind figure.",
            "",
        ]
    L += ["## Benchmarks after training", ""]
    if comp is not None:
        L += _fig("benchmarks")
        L += [
            "Accuracy per benchmark for the base model (grey), arm C (green) and arm I (blue). "
            "The McNemar p under each benchmark compares arm I with arm C on the shared "
            "questions. The small panel is the five-benchmark mean.",
            "",
        ]
        L += _table(
            ["benchmark", "base %", "arm C %", "arm I %", "McNemar p"],
            [
                [p["benchmark"]]
                + [f"{100 * p['accuracy'][k]:.1f}" for k in ("base", "arm_c", "arm_i")]
                + [f"{p['mcnemar_p']:.4f}"]
                for p in comp["benchmarks"]
            ],
        )
        L += ["### Behaviour rates", ""] + _fig("benchmark_behaviour")
        L += [
            "Share of questions with the behaviour the intervention targets, as compare_arms "
            "reports it: a code_generator call on AIME, two or more web_search calls on MuSiQue "
            "and GAIA.",
            "",
        ]
        L += _table(
            ["behaviour", "base %", "arm C %", "arm I %"],
            [
                [g] + [f"{100 * v[k]:.1f}" for k in ("base", "arm_c", "arm_i")]
                for g, v in behaviour_groups(comp)
            ],
        )
    else:
        L += ["No comparison.json given, so there are no benchmark figures yet.", ""]
    L += ["## Examples of rescued questions", ""] + examples_table(examples)
    return "\n".join(L)


# ───────────────────────────── main ─────────────────────────────────────────


def build_report(collected, rescue_dir, out_dir, comparison=None, n_examples=8) -> dict:
    rescue_dir, out_dir = Path(rescue_dir), Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads((rescue_dir / "gate.json").read_text(encoding="utf-8"))
    rows = _read_rows(rescue_dir / "rescue.jsonl")
    blind_rows = (
        _read_rows(rescue_dir / "blind.jsonl") if (rescue_dir / "blind.jsonl").exists() else None
    )
    if blind_rows is not None and "blind" not in gate:
        gate = {**gate, "blind": summarize([], [], {}, blind_rows=blind_rows)["blind"]}
    records = load_collected([Path(p) for p in collected])
    pool, candidate_ids = None, None
    cand_path = rescue_dir / "candidates.json"
    if cand_path.exists():
        cj = json.loads(cand_path.read_text(encoding="utf-8"))
        ids = cj.get("pool_question_ids")
        candidate_ids = {d: [] for d in DOMAINS}
        for c in cj.get("candidates", []):
            candidate_ids.setdefault(c["domain"], []).append(c["question_id"])
        if ids is not None:
            pool = set(ids)
    data = compute(records, rows, gate, pool, candidate_ids)
    comp = json.loads(Path(comparison).read_text(encoding="utf-8")) if comparison else None

    fig_rescue_vs_control(data, fig_dir)
    fig_ceiling(data, fig_dir)
    fig_funnel(data, fig_dir)
    fig_fix_kinds(data, fig_dir)
    fig_fix_step(data, fig_dir)
    fig_behaviour(data, fig_dir)
    if "blind" in data:
        fig_blind(data, fig_dir)
    if comp is not None:
        fig_benchmarks(comp, fig_dir)
        fig_benchmark_behaviour(comp, fig_dir)

    firsts = {r["question_id"]: r for r in records if r.get("pass") == 1}
    examples = select_examples(rows, firsts, n_examples)
    (out_dir / "report.md").write_text(
        render_md(data, examples, comp, "blind" in data), encoding="utf-8"
    )
    full = {**data, "examples": examples}
    if comp is not None:
        full["comparison"] = comp
    (out_dir / "report.json").write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    return full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--collected", required=True, nargs="+")
    ap.add_argument("--rescue-dir", required=True)
    ap.add_argument("--comparison", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args(argv)
    build_report(args.collected, args.rescue_dir, args.out_dir, args.comparison, args.examples)
    print(f"report: {Path(args.out_dir) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
