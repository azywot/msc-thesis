"""Aggregation of per-example results into experiment metrics.

Moved verbatim from ``scripts/run_experiment.py``; the two functions were
renamed from ``_level_key``/``_compute_metrics`` to drop the private prefix now
that they are a package API. The bodies are unchanged, and a characterization
fixture replays them over recorded runs to prove the numbers did not move.
"""

from typing import Any, Dict, List


def level_key(example, dataset_name: str) -> str:
    if dataset_name == "gaia":
        return str(example.metadata.get("level", "unknown"))
    if dataset_name == "hle":
        # HLE's "category" field is treated as its level.
        return str(example.metadata.get("category", "unknown"))
    if dataset_name in ("math500", "amc"):
        # Math500's "difficulty" field is treated as its level.
        return str(example.metadata.get("difficulty", example.metadata.get("year", "unknown")))
    if dataset_name == "aime":
        # AIME's "year" field is treated as its level.
        return str(example.metadata.get("year", "unknown"))
    return "all"


def compute_metrics(
    results: List[Dict[str, Any]],
    examples,
    dataset_name: str,
) -> Dict[str, Any]:
    """Aggregate per-example results into overall + per-level metrics.

    Structure:
        overall:    accuracy, em, f1, num_correct, token_usage
        tool_usage: per-tool total counts (overall)
        per_level:  (for stratified datasets) same scores + tool_usage + token_usage per level
    """
    _STRATIFIED = {"gaia", "math500", "aime", "amc", "hle"}

    per_level_rows: Dict[str, List[Dict]] = {}

    all_gaia: List[float] = []
    all_em: List[float] = []
    all_f1: List[float] = []
    all_tools: Dict[str, int] = {}
    all_token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for idx, example in enumerate(examples):
        r = results[idx] if idx < len(results) else {}
        evaluation = r.get("evaluation") or {}
        tc = r.get("tool_counts") or {}
        tu = r.get("token_usage") or {}

        gs = float(evaluation.get("accuracy", 0.0))
        em = float(evaluation.get("em", float(evaluation.get("correct", gs > 0))))
        f1 = float(evaluation.get("f1", gs))

        all_gaia.append(gs)
        all_em.append(em)
        all_f1.append(f1)
        for tool, count in tc.items():
            all_tools[tool] = all_tools.get(tool, 0) + int(count or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            all_token_usage[key] = all_token_usage.get(key, 0) + int(tu.get(key, 0))

        if dataset_name in _STRATIFIED:
            lk = level_key(example, dataset_name)
            per_level_rows.setdefault(lk, []).append({"gs": gs, "em": em, "f1": f1, "tc": tc, "tu": tu})

    total = len(examples)

    def _agg(rows: List[Dict]) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"accuracy": 0.0, "em": 0.0, "f1": 0.0, "num_correct": "0 of 0"}
        n_correct = sum(1 for r in rows if r["gs"] > 0)
        level_tools: Dict[str, int] = {}
        level_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for row in rows:
            for tool, count in row["tc"].items():
                level_tools[tool] = level_tools.get(tool, 0) + int(count or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                level_tokens[key] = level_tokens.get(key, 0) + int((row.get("tu") or {}).get(key, 0))
        result: Dict[str, Any] = {
            "accuracy": sum(r["gs"] for r in rows) / n,
            "em":         sum(r["em"] for r in rows) / n,
            "f1":         sum(r["f1"] for r in rows) / n,
            "num_correct": f"{n_correct} of {n}",
        }
        if level_tools:
            result["tool_usage"] = level_tools
        if any(level_tokens.values()):
            result["token_usage"] = level_tokens
        return result

    n_correct_overall = sum(1 for g in all_gaia if g > 0)
    overall: Dict[str, Any] = {
        "accuracy": sum(all_gaia) / total if total else 0.0,
        "em":         sum(all_em) / total if total else 0.0,
        "f1":         sum(all_f1) / total if total else 0.0,
        "num_correct": f"{n_correct_overall} of {total}",
    }
    if any(all_token_usage.values()):
        overall["token_usage"] = all_token_usage

    metrics: Dict[str, Any] = {"overall": overall}
    if all_tools:
        metrics["tool_usage"] = all_tools
    if dataset_name in _STRATIFIED and per_level_rows:
        metrics["per_level"] = {k: _agg(v) for k, v in sorted(per_level_rows.items())}

    return metrics
