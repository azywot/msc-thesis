"""Base vs LoRA (v1/v2) behavioural + failure-mode analysis for the FT report.

NEW analysis script. Re-uses the frozen automatic classifier ``classify_failure``
from ``analyze_failure_modes`` so per-mode counts are on the same automatic-proxy
basis as the reported failure-mode breakdown (breakdown_global.csv).

For each (dataset, thinking-regime, model) it loads the canonical eval run and
computes, per question:
  correct, n_tool_calls (= len(action_history)), turns, tool sequence,
  extracted prediction, no_action flag, completion tokens, output char length.

It then reports:
  * accuracy, tool-use, turn, verbosity aggregates per cell;
  * failure-mode breakdown per cell (classify_failure on incorrect records);
  * base->v1 / base->v2 transition matrices (right<->wrong), flip rate,
    prediction agreement, and action-sequence agreement (policy similarity).

Canonical run per (dataset, variant) = latest timestamped sub-dir containing
raw_results.json (see runs.latest_run). This drops the superseded broken v1
re-runs; verified to match the CSV / report numbers.

Output: prints a report and writes data/results/failure_modes/ft_base_vs_lora.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_failure_modes import classify_failure, FAILURE_MODES, MODE_LABELS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs import ROOT, latest_run  # noqa: E402
DATASETS = ["aime", "gpqa", "gaia", "hle", "musique"]
# regime -> (baseline variant suffix, lora variant suffix)
REGIMES = {
    "off": ("none", "none"),            # THINKING_MODE NO (the trained regime)
    "on":  ("orchestrator", "orchestrator"),  # ORCHESTRATOR_ONLY (deployment)
}
MODELS = ["base", "v1", "v2"]


def run_dir(ds: str, regime: str, model: str):
    base_suf, lora_suf = REGIMES[regime]
    if model == "base":
        return ROOT / f"experiments/results/orchestrator_capacity/{ds}/orch8b_sub1_7b_{base_suf}"
    if model == "v1":
        return ROOT / f"experiments/results/lora_inference/{ds}/qwen8B_sub1_7b_{lora_suf}"
    if model == "v2":
        return ROOT / f"experiments/results/lora_inference/{ds}/qwen8B_sub1_7b_{lora_suf}_v2"


def tool_seq(rec):
    return tuple(s.get("tool_name", "") for s in (rec.get("action_history") or []))


def load_cell(ds, regime, model):
    folder = run_dir(ds, regime, model)
    rd = latest_run(folder)
    if rd is None:
        return None
    recs = json.load(open(rd / "raw_results.json"))
    per_q = {}
    for r in recs:
        qid = r.get("question_id")
        ah = r.get("action_history") or []
        tools = [s.get("tool_name", "") for s in ah]
        tu = r.get("token_usage") or {}
        comp = tu.get("completion_tokens") or tu.get("total_tokens") or 0
        out_chars = len(json.dumps(r.get("output_messages") or []))
        per_q[qid] = {
            "correct": bool(r.get("correct")),
            "n_tools": len(ah),
            "turns": r.get("turns") or 0,
            "tool_seq": tuple(tools),
            "tool_counter": Counter(tools),
            "no_action": len(ah) == 0,
            "pred": (r.get("prediction") or "").strip(),
            "comp_tokens": comp,
            "out_chars": out_chars,
            "fmode": None if r.get("correct") else classify_failure(r),
        }
    return {"run": str(rd.relative_to(ROOT)), "n": len(recs), "per_q": per_q}


def agg(cell):
    pq = cell["per_q"]
    n = len(pq)
    acc = sum(v["correct"] for v in pq.values()) / n
    tool_total = Counter()
    for v in pq.values():
        tool_total += v["tool_counter"]
    return {
        "run": cell["run"], "n": n, "acc": round(acc, 4),
        "n_correct": sum(v["correct"] for v in pq.values()),
        "avg_tools": round(sum(v["n_tools"] for v in pq.values()) / n, 3),
        "avg_turns": round(sum(v["turns"] for v in pq.values()) / n, 3),
        "no_action_rate": round(sum(v["no_action"] for v in pq.values()) / n, 3),
        "max_turn_rate": round(sum(v["turns"] >= 15 for v in pq.values()) / n, 3),
        "avg_comp_tokens": round(sum(v["comp_tokens"] for v in pq.values()) / n, 1),
        "avg_out_chars": round(sum(v["out_chars"] for v in pq.values()) / n, 0),
        "tool_totals": dict(tool_total),
    }


def fmode_breakdown(cell):
    c = Counter(v["fmode"] for v in cell["per_q"].values() if v["fmode"])
    return {m: c.get(m, 0) for m in FAILURE_MODES}


def transitions(base, other):
    """base -> other correctness transition + behavioural agreement on shared qids."""
    shared = set(base["per_q"]) & set(other["per_q"])
    rr = rw = wr = ww = 0
    pred_agree = act_agree = ntool_agree = 0
    for q in shared:
        b, o = base["per_q"][q], other["per_q"][q]
        if b["correct"] and o["correct"]:
            rr += 1
        elif b["correct"] and not o["correct"]:
            rw += 1
        elif not b["correct"] and o["correct"]:
            wr += 1
        else:
            ww += 1
        if b["pred"] and b["pred"] == o["pred"]:
            pred_agree += 1
        if b["tool_seq"] == o["tool_seq"]:
            act_agree += 1
        if b["n_tools"] == o["n_tools"]:
            ntool_agree += 1
    ns = len(shared)
    return {
        "n_shared": ns,
        "right_right": rr, "right_wrong": rw, "wrong_right": wr, "wrong_wrong": ww,
        "net_delta": wr - rw,
        "flip_rate": round((rw + wr) / ns, 3) if ns else 0,
        "pred_agree_rate": round(pred_agree / ns, 3) if ns else 0,
        "toolseq_agree_rate": round(act_agree / ns, 3) if ns else 0,
        "ntool_agree_rate": round(ntool_agree / ns, 3) if ns else 0,
    }


def main():
    out = {"cells": {}, "fmodes": {}, "transitions": {}}
    for regime in REGIMES:
        for ds in DATASETS:
            cells = {}
            for model in MODELS:
                c = load_cell(ds, regime, model)
                if c is None:
                    continue
                cells[model] = c
                key = f"{regime}/{ds}/{model}"
                out["cells"][key] = agg(c)
                out["fmodes"][key] = fmode_breakdown(c)
            if "base" in cells:
                for m in ("v1", "v2"):
                    if m in cells:
                        out["transitions"][f"{regime}/{ds}/base->{m}"] = transitions(cells["base"], cells[m])

    outdir = ROOT / "data" / "results" / "failure_modes"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ft_base_vs_lora.json").write_text(json.dumps(out, indent=2))

    # ---- console report ----
    def line(*x): print("  ".join(str(v) for v in x))
    for regime in REGIMES:
        print(f"\n########## THINKING {'OFF (trained regime)' if regime=='off' else 'ON (deployment)'} ##########")
        line(f"{'cell':22s}", f"{'acc':>6}", f"{'nC':>4}", f"{'tools':>6}", f"{'turns':>6}",
             f"{'noact':>6}", f"{'maxT':>5}", f"{'ctok':>7}", f"{'ochar':>7}")
        for ds in DATASETS:
            for model in MODELS:
                k = f"{regime}/{ds}/{model}"
                if k not in out["cells"]:
                    continue
                a = out["cells"][k]
                line(f"{ds+'/'+model:22s}", f"{a['acc']:>6}", f"{a['n_correct']:>4}",
                     f"{a['avg_tools']:>6}", f"{a['avg_turns']:>6}", f"{a['no_action_rate']:>6}",
                     f"{a['max_turn_rate']:>5}", f"{a['avg_comp_tokens']:>7.0f}", f"{a['avg_out_chars']:>7.0f}")
            print()
        print("---- transitions (base -> model) ----")
        line(f"{'cell':26s}", "shr", "RR", "RW", "WR", "WW", "net", "flip", "predAgr", "seqAgr", "ntoolAgr")
        for ds in DATASETS:
            for m in ("v1", "v2"):
                k = f"{regime}/{ds}/base->{m}"
                if k not in out["transitions"]:
                    continue
                t = out["transitions"][k]
                line(f"{ds+' base->'+m:26s}", t["n_shared"], t["right_right"], t["right_wrong"],
                     t["wrong_right"], t["wrong_wrong"], f"{t['net_delta']:+d}", t["flip_rate"],
                     t["pred_agree_rate"], t["toolseq_agree_rate"], t["ntool_agree_rate"])
    print("\n---- failure modes per cell (incorrect only) ----")
    short = {m: m[:14] for m in FAILURE_MODES}
    for regime in REGIMES:
        print(f"\n== thinking {regime} ==")
        line(f"{'cell':22s}", *[f"{short[m]:>15}" for m in FAILURE_MODES], f"{'tot':>5}")
        for ds in DATASETS:
            for model in MODELS:
                k = f"{regime}/{ds}/{model}"
                if k not in out["fmodes"]:
                    continue
                fm = out["fmodes"][k]
                line(f"{ds+'/'+model:22s}", *[f"{fm[m]:>15}" for m in FAILURE_MODES], f"{sum(fm.values()):>5}")
            print()
    print(f"\nWritten: {outdir/'ft_base_vs_lora.json'}")


if __name__ == "__main__":
    main()
