"""Compare the base, arm C and arm I runs over the five evaluation benchmarks.

Usage:
    python scripts/compare_arms.py --base-root R --arm-c-root R --arm-i-root R [--out-dir D]

Each root holds ``<root>/<benchmark>/<variant>/<run>/raw_results.json``; the newest run
(by directory name) of the first variant found is used. Writes ``comparison.json`` and
``comparison.md`` to ``--out-dir`` (default: current directory).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rsi.confirm import mcnemar_p  # noqa: E402

BENCHMARKS = ["gaia", "hle", "gpqa", "aime", "musique"]
ARMS = ("base", "arm_c", "arm_i")


def load_run(path: Path) -> dict:
    """Per-question records of one ``raw_results.json``, keyed by integer question_id."""
    rows = json.loads(Path(path).read_text())
    return {int(r["question_id"]): r for r in rows}


def _accuracy(run: dict) -> float:
    return sum(bool(r["correct"]) for r in run.values()) / len(run) if run else 0.0


def _rate(run: dict, tool: str, at_least: int) -> float:
    if not run:
        return 0.0
    return sum((r.get("tool_counts") or {}).get(tool, 0) >= at_least for r in run.values()) / len(
        run
    )


def compare(base: dict, arm_c: dict, arm_i: dict, benchmark: str) -> dict:
    runs = {"base": base, "arm_c": arm_c, "arm_i": arm_i}
    shared = sorted(set(arm_c) & set(arm_i))
    i_only = sum(1 for q in shared if arm_i[q]["correct"] and not arm_c[q]["correct"])
    c_only = sum(1 for q in shared if arm_c[q]["correct"] and not arm_i[q]["correct"])
    result = {
        "benchmark": benchmark,
        "accuracy": {k: _accuracy(v) for k, v in runs.items()},
        "n": {k: len(v) for k, v in runs.items()},
        "n_shared": len(shared),
        "discordant": {"arm_i_only": i_only, "arm_c_only": c_only},
        "mcnemar_p": mcnemar_p(i_only, c_only),
        "behaviour": {},
    }
    if benchmark == "aime":
        result["behaviour"]["code_call_rate"] = {
            k: _rate(v, "code_generator", 1) for k, v in runs.items()
        }
    if benchmark in ("musique", "gaia"):
        result["behaviour"]["multi_search_rate"] = {
            k: _rate(v, "web_search", 2) for k, v in runs.items()
        }
    return result


def newest_raw_results(root: Path, benchmark: str) -> Path:
    found = sorted(
        Path(root).glob(f"{benchmark}/*/*/raw_results.json"), key=lambda p: p.parent.name
    )
    if not found:
        raise FileNotFoundError(f"no raw_results.json under {root}/{benchmark}")
    return found[-1]


def build(base_root, arm_c_root, arm_i_root) -> dict:
    per = []
    for b in BENCHMARKS:
        runs = [load_run(newest_raw_results(r, b)) for r in (base_root, arm_c_root, arm_i_root)]
        per.append(compare(*runs, b))
    mean = {k: sum(p["accuracy"][k] for p in per) / len(per) for k in ARMS}
    return {
        "benchmarks": per,
        "mean_accuracy": mean,
        "verdict": f"arm I mean > arm C mean: {mean['arm_i'] > mean['arm_c']}",
    }


def render_md(data: dict) -> str:
    lines = [
        "| benchmark | base | arm C | arm I | McNemar p (I vs C) | behaviour (base / C / I) |",
        "|---|---|---|---|---|---|",
    ]
    for p in data["benchmarks"]:
        a = p["accuracy"]
        beh = "; ".join(
            f"{name} {v['base']:.2f} / {v['arm_c']:.2f} / {v['arm_i']:.2f}"
            for name, v in p["behaviour"].items()
        )
        lines.append(
            f"| {p['benchmark']} | {a['base']:.3f} | {a['arm_c']:.3f} | {a['arm_i']:.3f} "
            f"| {p['mcnemar_p']:.4f} | {beh or '-'} |"
        )
    m = data["mean_accuracy"]
    lines.append(f"| mean | {m['base']:.3f} | {m['arm_c']:.3f} | {m['arm_i']:.3f} | | |")
    return "\n".join(lines) + f"\n\nVerdict: {data['verdict']}\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--arm-c-root", type=Path, required=True)
    ap.add_argument("--arm-i-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    data = build(args.base_root, args.arm_c_root, args.arm_i_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(data, indent=2))
    (args.out_dir / "comparison.md").write_text(render_md(data))
    print(render_md(data))
    return data


if __name__ == "__main__":
    main()
