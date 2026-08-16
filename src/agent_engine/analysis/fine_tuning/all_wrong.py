"""All-wrong / all-correct analysis of the Flow-GRPO training rollouts.

Source of record for the failure analysis of the RL run. Two analyses, kept
separate because they use two *different* definitions of "all-wrong" and the
thesis contrasts them explicitly:

1. ``composition`` -- whole-group reward composition. Per domain and
   overall, the share of complete groups that are all-wrong, all-correct or
   mixed. The binary reward makes a group informative only when its rollouts
   disagree: if every rollout in a group receives the same reward the
   group-relative advantage is zero and the group contributes no gradient. all-wrong + all-correct is
   therefore dead weight; the mixed share is the fraction of groups that
   actually carry a learning signal.

2. ``axpo`` -- all-wrong *tool-using subgroup* (Kang et al., AXPO,
   arXiv:2605.28774). AXPO looks only at the rollouts in a group that call a
   tool and calls the group all-wrong when none of *those* is correct, even if
   other rollouts in the group succeed. A group can therefore be mixed overall
   yet have every tool-using rollout fail, which suppresses the learning signal
   on exactly the tool-call tokens that need it. We report the two quantities
   AXPO reports -- tool-use attempt rate, and all-wrong rate of the tool-using
   subgroup -- plus the same rate for the no-tool subgroup, which is what
   exposes the domain-dependent asymmetry discussed in the thesis.

Group reconstruction and the rollout schema are documented in
``rollout_groups.py``.

Usage::

    python scripts/failure_modes/fine_tuning/all_wrong.py
    python scripts/failure_modes/fine_tuning/all_wrong.py --section composition --latex
    python scripts/failure_modes/fine_tuning/all_wrong.py --rollout-dir <path>

Prints the tables and writes ``data/results/failure_modes/all_wrong.json``.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .rollout_groups import (
    DEFAULT_TRAIN_RUN,
    ROOT,
    domain_order,
    fmt_pct,
    infer_group_size,
    iter_groups,
    rollout_counts,
    label,
    pct,
    resolve_rollout_dir,
)

DEFAULT_OUT = ROOT / "data/results/failure_modes/all_wrong.json"

# Reported by AXPO (Qwen3-VL-Thinking, multimodal), for context in the printout.
AXPO_REFERENCE = (
    "AXPO reference (Qwen3-VL-Thinking, multimodal): tool use attempted on ~30% of "
    "rollouts; tool-using subgroup all-wrong on ~40% of questions vs ~25% for the "
    "no-tool subgroup."
)


def _blank():
    return {
        "n_rollouts": 0,
        "n_tool_rollouts": 0,
        "n_groups": 0,
        # whole-group composition
        "all_wrong": 0,
        "all_correct": 0,
        "mixed": 0,
        # AXPO-aligned: groups whose tool-using subgroup is non-empty
        "groups_with_tool_subgroup": 0,
        "tool_subgroup_all_wrong": 0,
        # the no-tool subgroup, for the asymmetry check
        "groups_with_notool_subgroup": 0,
        "notool_subgroup_all_wrong": 0,
    }


def collect(rollout_dir, group_size):
    """Accumulate per-domain and overall (``ALL``) group counters."""
    stats = defaultdict(_blank)
    for domain, group in iter_groups(rollout_dir, group_size):
        n_correct = sum(r.correct for r in group)
        tool_sub = [r for r in group if r.tool]
        notool_sub = [r for r in group if not r.tool]
        for key in (domain, "ALL"):
            s = stats[key]
            s["n_groups"] += 1
            s["n_rollouts"] += len(group)
            s["n_tool_rollouts"] += len(tool_sub)
            if n_correct == 0:
                s["all_wrong"] += 1
            elif n_correct == len(group):
                s["all_correct"] += 1
            else:
                s["mixed"] += 1
            if tool_sub:
                s["groups_with_tool_subgroup"] += 1
                if not any(r.correct for r in tool_sub):
                    s["tool_subgroup_all_wrong"] += 1
            if notool_sub:
                s["groups_with_notool_subgroup"] += 1
                if not any(r.correct for r in notool_sub):
                    s["notool_subgroup_all_wrong"] += 1
    return stats


def with_rates(stats):
    """Counters plus the derived percentages the thesis quotes."""
    out = {}
    for key, s in stats.items():
        g = s["n_groups"]
        out[key] = dict(
            s,
            all_wrong_pct=pct(s["all_wrong"], g),
            all_correct_pct=pct(s["all_correct"], g),
            mixed_pct=pct(s["mixed"], g),
            dead_pct=pct(s["all_wrong"] + s["all_correct"], g),
            tool_use_rate_pct=pct(s["n_tool_rollouts"], s["n_rollouts"]),
            tool_subgroup_all_wrong_pct=pct(
                s["tool_subgroup_all_wrong"], s["groups_with_tool_subgroup"]
            ),
            notool_subgroup_all_wrong_pct=pct(
                s["notool_subgroup_all_wrong"], s["groups_with_notool_subgroup"]
            ),
        )
    return out


def print_composition(stats, keys):
    print("== Whole-group reward composition (dead-group breakdown) ==")
    header = f"{'domain':18s} {'rollouts':>9} {'groups':>7} {'all-wrong':>10} {'all-correct':>12} {'mixed':>8}"
    print(header)
    for key in keys:
        s = stats[key]
        g = s["n_groups"]
        print(
            f"{label(key):18s} {s['n_rollouts']:>9} {g:>7} "
            f"{fmt_pct(s['all_wrong'], g):>10} {fmt_pct(s['all_correct'], g):>12} "
            f"{fmt_pct(s['mixed'], g):>8}"
        )
    s = stats["ALL"]
    dead = s["all_wrong"] + s["all_correct"]
    print(
        f"\nDead weight (all-wrong + all-correct) overall: {fmt_pct(dead, s['n_groups'])} "
        f"of {s['n_groups']} groups; mixed (learning signal): "
        f"{fmt_pct(s['mixed'], s['n_groups'])}"
    )


def print_axpo(stats, keys):
    print("== All-wrong tool-using subgroup (AXPO-aligned) ==")
    print(
        f"{'domain':18s} {'tool-use rate':>14} {'tool-subgrp all-wrong':>22} "
        f"{'no-tool-subgrp all-wrong':>24}"
    )
    for key in keys:
        s = stats[key]
        print(
            f"{label(key):18s} {fmt_pct(s['n_tool_rollouts'], s['n_rollouts']):>14} "
            f"{fmt_pct(s['tool_subgroup_all_wrong'], s['groups_with_tool_subgroup']):>22} "
            f"{fmt_pct(s['notool_subgroup_all_wrong'], s['groups_with_notool_subgroup']):>24}"
        )
    print(
        "\n(denominators: tool-subgrp rate over groups with a non-empty tool-using "
        "subgroup; no-tool-subgrp rate over groups with a non-empty no-tool subgroup)"
    )
    print(AXPO_REFERENCE)


def latex_composition(stats, keys, group_size):
    """Reward composition as a LaTeX tabular."""
    n_groups = stats["ALL"]["n_groups"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Reward composition of the Flow-GRPO training rollouts, over all "
        f"{n_groups:,} complete {group_size}-rollout groups. Only mixed groups carry a "
        f"learning signal; all-wrong and all-correct groups have zero group-relative "
        f"advantage.}}",
        "\\label{tab:reward-composition}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Domain & All-wrong (\\%) & All-correct (\\%) & Mixed (\\%) \\\\",
        "\\midrule",
    ]
    for key in keys:
        s = stats[key]
        g = s["n_groups"]
        if key == "ALL":
            lines.append("\\midrule")
        cells = [
            f"{pct(s['all_wrong'], g):.1f}",
            f"{pct(s['all_correct'], g):.1f}",
            f"{pct(s['mixed'], g):.1f}",
        ]
        lines.append(f"{label(key)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rollout-dir",
        default=DEFAULT_TRAIN_RUN,
        help="Training-run root, or a rollout_data/train directory "
        f"(default: {DEFAULT_TRAIN_RUN.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--section",
        choices=["all", "composition", "axpo"],
        default="all",
        help="Which analysis to report (default: all)",
    )
    parser.add_argument("--latex", action="store_true", help="Also emit the reward-composition table as LaTeX")
    parser.add_argument(
        "--group-size",
        default="auto",
        help="GRPO group size (rollout.n). 'auto' infers it from the per-question "
        "rollout counts (default: auto)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"JSON output (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    try:
        rollout_dir = resolve_rollout_dir(args.rollout_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(str(exc))

    if str(args.group_size).lower() == "auto":
        group_size, evidence = infer_group_size(rollout_dir)
    else:
        try:
            group_size = int(args.group_size)
        except ValueError:
            raise SystemExit(f"--group-size must be an integer or 'auto', got {args.group_size!r}")
        if group_size < 1:
            raise SystemExit(f"--group-size must be >= 1, got {group_size}")
        _, evidence = infer_group_size(rollout_dir)
        evidence["method"] = "override"
        evidence["chosen"] = group_size
        evidence["dirs_not_multiple"] = sum(
            1 for c in rollout_counts(rollout_dir) if c and c % group_size
        )

    stats = collect(rollout_dir, group_size)
    if not stats:
        raise SystemExit(
            f"No complete {group_size}-rollout groups found in {rollout_dir} "
            f"(per-question rollout counts: {evidence['count_histogram']}). "
            f"Pass --group-size to override the inferred value."
        )
    keys = domain_order(stats)

    try:
        shown = rollout_dir.relative_to(ROOT)
    except ValueError:
        shown = rollout_dir
    print(f"Source: {shown}")
    print(
        f"Group size n={group_size} ({evidence['method']}; "
        f"gcd={evidence['gcd']}, most common count={evidence['most_common_count']}, "
        f"over {evidence['n_question_dirs']} question dirs / "
        f"{evidence['total_rollouts']} rollouts)"
    )
    if evidence["dirs_not_multiple"]:
        print(
            f"WARNING: {evidence['dirs_not_multiple']} question dir(s) hold a rollout count "
            f"that is not a multiple of {group_size}; those remainders are dropped. "
            f"Counts seen: {evidence['count_histogram']}"
        )
    print("Complete groups only; partial trailing blocks dropped.\n")

    if args.section in ("all", "composition"):
        print_composition(stats, keys)
        print()
    if args.section in ("all", "axpo"):
        print_axpo(stats, keys)
        print()
    if args.latex:
        print(latex_composition(stats, keys, group_size))
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(shown),
        "group_size": group_size,
        "group_size_evidence": evidence,
        "domains": with_rates(stats),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
