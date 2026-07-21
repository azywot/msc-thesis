"""All-wrong / dead-group analysis of the v2 GRPO training rollouts.

NEW analysis script (thesis SQ4, Chapter 7). It is the source of record for two
sets of figures used in Section 7.4:

  1. The GRPO reward-composition / "dead group" breakdown (report Part II Sec. 10):
     per domain and overall, the share of complete n=8 groups that are all-wrong,
     all-correct, or mixed. A mixed group is the only kind that carries a non-zero
     group-relative advantage, so the mixed share = the fraction of groups that
     actually carry a learning signal.

  2. The AXPO-aligned all-wrong tool-using-subgroup metric
     (Kang et al., AXPO, arXiv:2605.28774). AXPO measures, conditional on a tool
     call being attempted, how often the tool-using rollouts within a group are
     all-wrong (so the tool-call tokens receive non-positive advantage). We compute
     the same two quantities on our orchestrator rollouts:
       - tool-use attempt rate: fraction of rollouts that emit >=1 tool call;
       - all-wrong tool-using-subgroup rate: among groups whose tool-using subgroup
         is non-empty, the fraction in which no tool-using rollout is correct.

Source data (per-group rollout outcomes), stated explicitly so the path can be
verified:
  experiments/results/fine_tuning/qwen3-8b-grpo-search-math-v2/
      29-05-2026_11-36-23210365/rollout_data/train/idx_<domain>_<n>/<k>_rollout_*.json

Each rollout JSON carries: data_source, reward (binary 0.0/1.0 on answer
correctness), and output_messages. A rollout is "tool-using" iff its
output_messages contain at least one role=="tool" message (the orchestrator
dispatched a sub-agent / tool); a lone assistant message is the
"direct reasoning, no action" mode.

Group reconstruction: within one idx_<domain>_<n> directory the rollout files are
named <k>_rollout_<hash>.json with k = 0,1,2,... A GRPO group is the n=8 rollouts
sampled for that question at one optimizer step; a question visited at several
steps yields several groups, so we sort by k and chunk into consecutive blocks of
8. A "complete" group has exactly 8 rollouts; trailing partial blocks are dropped
from the group-level statistics (they are not used for advantage normalisation in
a full n=8 group). Running this script reproduces the report's 2,106 complete
groups and 50.6% dead-weight headline, which validates the reconstruction.

Usage:
  python scripts/failure_modes/axpo_all_wrong_analysis.py
Prints the tables and writes data/results/failure_modes/axpo_all_wrong.json.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ROLLOUT_DIR = (
    ROOT
    / "experiments/results/fine_tuning/qwen3-8b-grpo-search-math-v2"
    / "29-05-2026_11-36-23210365/rollout_data/train"
)
GROUP_SIZE = 8
DOMAIN_ORDER = ["deepmath", "hotpotqa", "nq"]

_IDX_RE = re.compile(r"^idx_(?P<domain>.+)_\d+$")
_PREFIX_RE = re.compile(r"^(?P<k>\d+)_rollout_")


def is_correct(reward):
    """Binary outcome reward; treat >=0.5 as correct to be robust to float noise."""
    try:
        return float(reward) >= 0.5
    except (TypeError, ValueError):
        return False


def uses_tool(rec):
    """A rollout is tool-using iff it has at least one role=='tool' message."""
    return any(m.get("role") == "tool" for m in (rec.get("output_messages") or []))


def load_rollout(path):
    d = json.load(open(path))
    return {"reward": is_correct(d.get("reward")), "tool": uses_tool(d)}


def iter_groups():
    """Yield (domain, [rollout, ...]) for every complete n=8 group."""
    for idx_dir in sorted(ROLLOUT_DIR.iterdir()):
        if not idx_dir.is_dir():
            continue
        m = _IDX_RE.match(idx_dir.name)
        if not m:
            continue
        domain = m.group("domain")
        files = []
        for f in idx_dir.glob("*.json"):
            km = _PREFIX_RE.match(f.name)
            if km:
                files.append((int(km.group("k")), f))
        files.sort()
        rollouts = [load_rollout(f) for _, f in files]
        for start in range(0, len(rollouts) - GROUP_SIZE + 1, GROUP_SIZE):
            yield domain, rollouts[start : start + GROUP_SIZE]


def blank():
    return {
        "n_rollouts": 0,
        "n_tool_rollouts": 0,
        "n_groups": 0,
        "all_wrong": 0,
        "all_correct": 0,
        "mixed": 0,
        # AXPO-aligned: groups whose tool-using subgroup is non-empty
        "groups_with_tool_subgroup": 0,
        "tool_subgroup_all_wrong": 0,
        # AXPO Figure 3b parallel: the no-tool subgroup, for the asymmetry check
        "groups_with_notool_subgroup": 0,
        "notool_subgroup_all_wrong": 0,
    }


def main():
    stats = defaultdict(blank)

    for domain, group in iter_groups():
        for key in (domain, "ALL"):
            s = stats[key]
            s["n_groups"] += 1
            s["n_rollouts"] += len(group)
            s["n_tool_rollouts"] += sum(r["tool"] for r in group)
            n_correct = sum(r["reward"] for r in group)
            if n_correct == 0:
                s["all_wrong"] += 1
            elif n_correct == len(group):
                s["all_correct"] += 1
            else:
                s["mixed"] += 1
            tool_sub = [r for r in group if r["tool"]]
            if tool_sub:
                s["groups_with_tool_subgroup"] += 1
                if not any(r["reward"] for r in tool_sub):
                    s["tool_subgroup_all_wrong"] += 1
            notool_sub = [r for r in group if not r["tool"]]
            if notool_sub:
                s["groups_with_notool_subgroup"] += 1
                if not any(r["reward"] for r in notool_sub):
                    s["notool_subgroup_all_wrong"] += 1

    # ---- console report ----
    def pct(num, den):
        return f"{100.0 * num / den:5.1f}%" if den else "   -- "

    print(f"Source: {ROLLOUT_DIR.relative_to(ROOT)}")
    print(f"Complete n={GROUP_SIZE} groups; partial trailing blocks dropped.\n")

    print("== GRPO reward composition (dead-group breakdown) ==")
    print(f"{'domain':10s} {'rollouts':>9} {'groups':>7} {'all-wrong':>10} "
          f"{'all-correct':>12} {'mixed':>8}")
    for key in DOMAIN_ORDER + ["ALL"]:
        s = stats[key]
        g = s["n_groups"]
        print(f"{key:10s} {s['n_rollouts']:>9} {g:>7} "
              f"{pct(s['all_wrong'], g):>10} {pct(s['all_correct'], g):>12} "
              f"{pct(s['mixed'], g):>8}")
    s = stats["ALL"]
    dead = s["all_wrong"] + s["all_correct"]
    print(f"\nDead weight (all-wrong + all-correct) overall: "
          f"{pct(dead, s['n_groups'])} of {s['n_groups']} groups; "
          f"mixed (learning signal): {pct(s['mixed'], s['n_groups'])}")

    print("\n== AXPO-aligned tool-call metrics ==")
    print(f"{'domain':10s} {'tool-use rate':>14} {'tool-subgrp all-wrong':>22} "
          f"{'no-tool-subgrp all-wrong':>24}")
    for key in DOMAIN_ORDER + ["ALL"]:
        s = stats[key]
        print(f"{key:10s} {pct(s['n_tool_rollouts'], s['n_rollouts']):>14} "
              f"{pct(s['tool_subgroup_all_wrong'], s['groups_with_tool_subgroup']):>22} "
              f"{pct(s['notool_subgroup_all_wrong'], s['groups_with_notool_subgroup']):>24}")
    print("\n(denominators: tool-subgrp rate over groups with a non-empty tool-using "
          "subgroup; no-tool-subgrp rate over groups with a non-empty no-tool subgroup)")
    print("AXPO reference (Qwen3-VL-Thinking, multimodal): tool use attempted on "
          "~30% of rollouts; tool-using subgroup all-wrong on ~40% of questions vs "
          "~25% for the no-tool subgroup.")

    outdir = ROOT / "data" / "results" / "failure_modes"
    outdir.mkdir(parents=True, exist_ok=True)
    out = {k: dict(v) for k, v in stats.items()}
    (outdir / "axpo_all_wrong.json").write_text(json.dumps(out, indent=2))
    print(f"\nWritten: {(outdir / 'axpo_all_wrong.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
