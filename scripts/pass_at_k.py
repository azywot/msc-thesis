"""Report pass@k over several seeded runs of one configuration.

Usage:
    python scripts/pass_at_k.py RUN [RUN ...] [--k 1 3 5] [--by-level] [--out FILE]
                                [--allow-mismatch]
"""

import argparse
import json
import sys
from pathlib import Path

from agent_engine.analysis.pass_at_k import (
    PassAtKInputError,
    compute_pass_at_k,
    load_run,
    validate_runs,
)


def _fmt(v):
    return "n/a" if v is None else f"{100 * v:.2f}"


def _row(label, block, ks):
    cells = "  ".join(f"{_fmt(block[f'pass@{k}']):>9}" for k in ks)
    excl = ",".join(str(block["excluded_short_of_k"][k]) for k in ks)
    return f"{label:<14}{block['num_questions']:>6}  {cells}   excluded: {excl}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path, help="run directories or raw_results.json paths")
    p.add_argument("--k", dest="ks", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--by-level", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--allow-mismatch", action="store_true")
    args = p.parse_args(argv)
    if any(k < 1 for k in args.ks):
        p.error("--k values must be at least 1")

    try:
        runs = [load_run(r) for r in args.runs]
        warnings = validate_runs(runs, allow_mismatch=args.allow_mismatch)
        out = compute_pass_at_k(runs, args.ks, by_level=args.by_level)
    except PassAtKInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out["warnings"] = warnings

    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for row in out["runs"]:
        if "warning" in row:
            print(f"warning: {row['run_dir']}: {row['warning']}", file=sys.stderr)
    print(
        f"runs: {out['n_runs']}  accuracy: {_fmt(out['accuracy_mean'])} "
        f"+/- {_fmt(out['accuracy_std'])} (std over seeds)"
    )
    header = f"{'':<14}{'n_q':>6}  " + "  ".join(f"{'pass@' + str(k):>9}" for k in args.ks)
    print(header)
    print(_row("overall", out["overall"], args.ks))
    for lvl, block in out.get("per_level", {}).items():
        print(_row(f"level {lvl}", block, args.ks))

    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
