"""Go/no-go gate for the tool-call intervention: per-domain net rescue rate and blind check."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

KINDS = ("rewrite", "follow_up", "insert")


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 6) if d else 0.0


def _domain_summary(cands: List[dict], rows: List[dict], skips: dict) -> dict:
    by_q: Dict[object, List[dict]] = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r)
    rescued, control = 0, 0
    attempts, kinds = Counter(), Counter()
    for c in cands:
        qrows = sorted(by_q.get(c["question_id"], []), key=lambda r: r["attempt"])
        hit = next((r for r in qrows if r.get("correct")), None)
        if hit is not None:
            rescued += 1
            attempts[str(hit["attempt"])] += 1
            kinds[hit.get("kind")] += 1
        if any(r.get("control_correct") for r in qrows):
            control += 1
    n = len(cands)
    domain_rows = [r for r in rows if r["question_id"] in {c["question_id"] for c in cands}]
    reasons = Counter(
        reason for r in domain_rows if r.get("status") == "leakage" for reason in r["reasons"]
    )
    rescue_rate, control_rate = _pct(rescued, n), _pct(control, n)
    out = {
        "candidates": n,
        "rescued": rescued,
        "rescue_rate": rescue_rate,
        "control_solved": control,
        "control_rate": control_rate,
        "net_points": round(rescue_rate - control_rate, 6),
        "attempts_used": {k: attempts[k] for k in sorted(attempts)},
        "parse_errors": sum(1 for r in domain_rows if r.get("status") == "parse_error"),
        "leakage_rejections": sum(1 for r in domain_rows if r.get("status") == "leakage"),
        "leakage_by_reason": dict(reasons),
        "kinds": {k: kinds[k] for k in KINDS if kinds[k]},
    }
    for key, value in skips.items():
        if key != "candidates":
            out[key] = value
    return out


def summarize(
    candidates: List[dict],
    rows: List[dict],
    counts: dict,
    threshold_points: float = 10.0,
    blind_rows: Optional[List[dict]] = None,
) -> dict:
    from rsi.intervention.records import domain_of

    summary: dict = {}
    for domain, skips in counts.items():
        cands = [c for c in candidates if domain_of(c["data_source"]) == domain]
        summary[domain] = _domain_summary(cands, [r for r in rows if r["domain"] == domain], skips)
    passing = [d for d in counts if summary[d]["net_points"] >= threshold_points]
    summary["gate"] = {
        "threshold_points": threshold_points,
        "pass": bool(passing),
        "passing_domains": passing,
    }
    if blind_rows is not None:
        questions = {r["question_id"] for r in blind_rows}
        rescued = {r["question_id"] for r in blind_rows if r.get("correct")}
        summary["blind"] = {
            "n": len(questions),
            "rescued": len(rescued),
            "rate": _pct(len(rescued), len(questions)),
        }
    return summary


def blind_sample(rows: List[dict], n: int, seed: int = 0) -> List[int]:
    rescued = sorted({r["question_id"] for r in rows if r.get("correct")})
    if n >= len(rescued):
        return rescued
    return sorted(random.Random(seed).sample(rescued, n))


def _domain_table(name: str, d: dict) -> List[str]:
    lines = [f"## {name}", "", "| metric | value |", "| --- | --- |"]
    for key, value in d.items():
        lines.append(f"| {key} | {json.dumps(value) if isinstance(value, dict) else value} |")
    return lines + [""]


def write_gate(out_dir: Path, summary: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gate = summary["gate"]
    verdict = "PASS" if gate["pass"] else "FAIL"
    lines = [
        f"# Gate: {verdict}",
        "",
        f"Net rescue of at least {gate['threshold_points']} points in at least one domain. "
        f"Passing domains: {', '.join(gate['passing_domains']) or 'none'}.",
        "",
    ]
    for name, d in summary.items():
        if name not in ("gate", "blind"):
            lines += _domain_table(name, d)
    if "blind" in summary:
        b = summary["blind"]
        lines += [
            "## Blind check (reflector does not see the gold answer)",
            "",
            f"{b['rescued']} of {b['n']} previously rescued questions rescued again "
            f"({b['rate']}%).",
            "",
        ]
    (out_dir / "gate.md").write_text("\n".join(lines), encoding="utf-8")
