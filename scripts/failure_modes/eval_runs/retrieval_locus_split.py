"""Rule-5 split: orchestrator-locus vs sub-agent-locus retrieval failures.

NEW analysis script (does NOT modify analyze_failure_modes.py). It re-uses the
frozen automatic classifier so the 227 ``retrieval_evidence_failure`` records it
inspects are exactly the ones counted in breakdown_global.csv.

Every retrieval_evidence_failure record has >=2 web_search calls (that is the
detection rule). The split asks *why* the multi-search run still failed:

  * sub-agent / tool locus (true retrieval ceiling): every web_search call came
    back empty. The Web Searcher emits the fixed string "No helpful information
    found." on a dead query (see src/agent_engine/tools/web_search.py and the
    appendix prompt template), so an all-dead trace means the tool genuinely
    returned nothing for the orchestrator to use.

  * orchestrator locus (early-commit / no-follow-through): at least one
    web_search returned actual content, i.e. a usable lead existed, yet the run
    still failed -- the orchestrator over-committed to a candidate or failed to
    reconcile / follow the retrieved evidence. This is the orchestrator-fixable
    slice WITHIN the (sub-agent-locus) retrieval mode.

This is an automatic PROXY for the manual conceptual definition; it labels the
locus from the dead-vs-content signal in the action history, not from reading
each summary against ground truth.

Outputs: prints a report and writes
``data/results/failure_modes/retrieval_locus_split.json``.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_failure_modes import (  # noqa: E402
    BENCHMARKS,
    classify_failure,
    load_run,
    run_inventory,
)

DEAD_MARKER = "no helpful information found"


def _is_dead_search(step: dict) -> bool:
    """A web_search step is 'dead' if it returned nothing usable: empty result,
    or the Web Searcher's fixed dead-query string."""
    result = str(step.get("result") or "").strip()
    if not result:
        return True
    return DEAD_MARKER in result.lower()


def analyze(root: Path, out_dir: Path) -> dict:
    by_bench = {b: {"orchestrator_locus": 0, "sub_agent_locus": 0} for b in BENCHMARKS}
    # finer diagnostics
    n_all_dead = 0      # every search dead -> ceiling (sub-agent locus)
    n_mixed = 0         # some dead, some content -> orchestrator locus
    n_all_content = 0   # every search returned content -> orchestrator locus
    total = 0
    examples = []

    for bench, _variant, path in run_inventory(root):
        for rec in load_run(path):
            if rec.get("correct"):
                continue
            if classify_failure(rec) != "retrieval_evidence_failure":
                continue
            total += 1
            searches = [s for s in (rec.get("action_history") or [])
                        if s.get("tool_name") == "web_search"]
            n_search = len(searches)
            n_dead = sum(1 for s in searches if _is_dead_search(s))
            n_content = n_search - n_dead

            if n_content == 0:
                locus = "sub_agent_locus"   # retrieval ceiling
                n_all_dead += 1
            else:
                locus = "orchestrator_locus"
                if n_dead == 0:
                    n_all_content += 1
                else:
                    n_mixed += 1
            by_bench[bench][locus] += 1

            if len(examples) < 6:
                examples.append({
                    "benchmark": bench, "qid": rec.get("question_id"),
                    "n_search": n_search, "n_dead": n_dead, "n_content": n_content,
                    "locus": locus,
                })

    orch = sum(by_bench[b]["orchestrator_locus"] for b in BENCHMARKS)
    sub = sum(by_bench[b]["sub_agent_locus"] for b in BENCHMARKS)

    def pct(n):
        return round(100.0 * n / total, 1) if total else 0.0

    result = {
        "basis": "automatic PROXY over retrieval_evidence_failure records "
                 "(analyze_failure_modes.classify_failure)",
        "total_retrieval_failures": total,
        "orchestrator_locus": orch,
        "orchestrator_locus_pct": pct(orch),
        "sub_agent_locus": sub,
        "sub_agent_locus_pct": pct(sub),
        "diagnostics": {
            "all_dead_searches_ceiling": n_all_dead,
            "mixed_dead_and_content": n_mixed,
            "all_content_searches": n_all_content,
        },
        "by_benchmark": by_bench,
        "examples": examples,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_locus_split.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Basis: {result['basis']}\n")
    print(f"Total retrieval_evidence_failure records: {total}")
    print(f"  orchestrator-locus (>=1 search returned content): {orch} ({pct(orch)}%)")
    print(f"  sub-agent-locus    (all searches dead/ceiling)  : {sub} ({pct(sub)}%)")
    print(f"\n  diagnostics: all-dead={n_all_dead}  mixed={n_mixed}  all-content={n_all_content}")
    print("\n  by benchmark (orchestrator-locus / sub-agent-locus):")
    for b in BENCHMARKS:
        o, s = by_bench[b]["orchestrator_locus"], by_bench[b]["sub_agent_locus"]
        if o or s:
            print(f"    {b:8s}: {o:4d} / {s:4d}")
    print(f"\nWritten: {out_path}")
    return result


def main():
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "data" / "results" / "failure_modes"
    analyze(root, out_dir)


if __name__ == "__main__":
    main()
