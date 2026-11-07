"""Unbiased pass@k over several seeded runs of one configuration.

Given n attempts per question of which c are correct, the unbiased estimator
(Chen et al., 2021) is ``1 - C(n - c, k) / C(n, k)``.  Each run directory is one
attempt per question; questions are averaged after the per-question estimate.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

from agent_engine.runner.metrics import level_key

_METRICS_TOL = 1e-9


class PassAtKInputError(Exception):
    """The supplied runs cannot be combined into a valid pass@k estimate."""


@dataclass
class RunData:
    run_dir: Path
    config: dict
    results: List[dict]
    metrics: Optional[dict]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n attempts with c correct."""
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run(path: Path) -> RunData:
    """Load a run from its directory or from its ``raw_results.json`` path."""
    path = Path(path)
    run_dir = path.parent if path.is_file() else path
    raw = run_dir / "raw_results.json"
    if not raw.exists():
        hint = (
            " (only raw_results.partial.json found; the run did not finish)"
            if (run_dir / "raw_results.partial.json").exists()
            else ""
        )
        raise PassAtKInputError(f"{raw} is missing{hint}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise PassAtKInputError(f"{config_path} is missing")
    metrics_path = run_dir / "metrics.json"
    return RunData(
        run_dir=run_dir,
        config=_read_json(config_path),
        results=_read_json(raw),
        metrics=_read_json(metrics_path) if metrics_path.exists() else None,
    )


def _has_zero_temperature(model_cfg: dict) -> bool:
    base = model_cfg.get("temperature")
    by_thinking = model_cfg.get("sampling_by_thinking")
    if by_thinking:
        temps = []
        for block in by_thinking.values():
            t = (block or {}).get("temperature")
            temps.append(base if t is None else t)
        return any(t == 0 for t in temps)
    return base == 0


def _check_run(run: RunData) -> None:
    """Structural checks that corrupt the estimator if skipped. Never waivable."""
    if not isinstance(run.results, list) or not run.results:
        raise PassAtKInputError(f"{run.run_dir}: raw_results.json holds no results")
    seen = set()
    for i, res in enumerate(run.results):
        if isinstance(res, dict) and "error" in res:
            raise PassAtKInputError(f"run {run.run_dir} errored on result #{i}: {res['error']}")
        for field in ("question_id", "correct"):
            if not isinstance(res, dict) or field not in res:
                raise PassAtKInputError(f"{run.run_dir}: result #{i} is missing '{field}'")
        qid = res["question_id"]
        if qid in seen:
            raise PassAtKInputError(f"{run.run_dir}: duplicate question_id {qid} within one run")
        seen.add(qid)
    models = run.config.get("models")
    if not models:
        raise PassAtKInputError(
            f"{run.run_dir}: config.json has no 'models'; cannot verify sampling temperature"
        )


def _level_of(res: dict, dataset_name: str) -> str:
    # Same semantics as runner.metrics.level_key (reused, not copied).
    return level_key(SimpleNamespace(metadata=res.get("metadata") or {}), dataset_name)


def _diff_fields(runs: List[RunData]) -> List[str]:
    """Names of visible config fields that differ between runs (seed excluded)."""
    ignore = {"seed", "config_fingerprint", "config_path", "output_dir"}
    keys = sorted({k for r in runs for k in r.config} - ignore)
    return [k for k in keys if len({json.dumps(r.config.get(k), sort_keys=True) for r in runs}) > 1]


def validate_runs(runs: List[RunData], allow_mismatch: bool = False) -> List[str]:
    """Check that runs are comparable; return warnings (empty when clean)."""
    # Structural problems, greedy runs, repeated seeds or run dirs and a single run are never
    # waived: they make the estimator meaningless (greedy or same-seed attempts coincide).
    if len(runs) < 2:
        raise PassAtKInputError(f"need at least 2 runs, got {len(runs)}")
    dirs = [str(Path(r.run_dir).resolve()) for r in runs]
    dup_dirs = sorted({d for d in dirs if dirs.count(d) > 1})
    if dup_dirs:
        raise PassAtKInputError(f"the same run directory was given twice: {dup_dirs}")
    seeds = [r.config.get("seed") for r in runs]
    dupes = sorted({s for s in seeds if seeds.count(s) > 1}, key=str)
    if dupes:
        raise PassAtKInputError(f"duplicate seeds: {dupes}")
    all_models = [list((r.config.get("models") or {}).values()) for r in runs]
    if all(ms and all(m.get("seed") is not None for m in ms) for ms in all_models):
        keys = [
            tuple(sorted((r.config["models"][role]["seed"], role) for role in r.config["models"]))
            for r in runs
        ]
        if len(set(keys)) < len(keys):
            raise PassAtKInputError("duplicate per-model seeds across runs")
    for r in runs:
        _check_run(r)
        for role, model_cfg in (r.config.get("models") or {}).items():
            if _has_zero_temperature(model_cfg):
                raise PassAtKInputError(
                    f"{r.run_dir}: role '{role}' has effective temperature 0 "
                    "(greedy); pass@k needs sampled runs"
                )

    problems: List[str] = []

    fps = [r.config.get("config_fingerprint") for r in runs]
    if any(fp is None for fp in fps):
        missing = [str(r.run_dir) for r, fp in zip(runs, fps) if fp is None]
        problems.append(f"runs without config_fingerprint: {missing}")
    elif len(set(fps)) > 1:
        problems.append(
            "config_fingerprint differs between runs; differing config fields: "
            f"{_diff_fields(runs) or '(none visible in config.json)'}"
        )

    sets = [{res["question_id"] for res in r.results} for r in runs]
    union = set().union(*sets) if sets else set()
    if any(s != union for s in sets):
        counts = [len(union - s) for s in sets]
        problems.append(
            f"question sets differ ({len(union)} distinct ids; missing per run: {counts})"
        )

    texts: Dict[int, set] = defaultdict(set)
    for r in runs:
        for res in r.results:
            texts[res["question_id"]].add(res.get("question"))
    changed = sorted(q for q, t in texts.items() if len(t) > 1)
    if changed:
        problems.append(f"question text differs across runs for ids: {changed[:10]}")

    if problems and not allow_mismatch:
        raise PassAtKInputError("; ".join(problems))

    # Behavioural guard: identical predictions everywhere suggest sampling ignores the seed.
    # Only a warning, since tiny smoke splits can legitimately coincide.
    preds = [{res["question_id"]: res.get("prediction") for res in r.results} for r in runs]
    shared = set.intersection(*(set(p) for p in preds))
    if shared and all(all(p[q] == preds[0][q] for q in shared) for p in preds[1:]):
        problems.append(
            "all runs have identical predictions on every shared question; "
            "sampling may not be varying with the seed"
        )
    return problems


def _summarise(per_q: Dict[int, List[bool]], ks: List[int]) -> dict:
    out: dict = {"num_questions": len(per_q), "excluded_short_of_k": {}}
    for k in ks:
        scores = [pass_at_k(len(v), sum(v), k) for v in per_q.values() if len(v) >= k]
        out[f"pass@{k}"] = sum(scores) / len(scores) if scores else None
        out["excluded_short_of_k"][k] = len(per_q) - len(scores)
    return out


def compute_pass_at_k(runs: List[RunData], ks: List[int], by_level: bool = False) -> dict:
    """pass@k for each k, overall and optionally per level."""
    if any(k < 1 for k in ks):
        raise PassAtKInputError(f"k must be at least 1, got {sorted(ks)}")
    per_q: Dict[int, List[bool]] = defaultdict(list)
    levels: Dict[int, str] = {}
    dataset_name = (runs[0].config.get("dataset") or {}).get("name", "") if runs else ""

    run_rows = []
    for r in runs:
        _check_run(r)
        for res in r.results:
            qid = res["question_id"]
            per_q[qid].append(bool(res["correct"]))
            if qid not in levels:
                levels[qid] = _level_of(res, dataset_name)
        acc = sum(bool(x["correct"]) for x in r.results) / len(r.results) if r.results else 0.0
        row = {"run_dir": str(r.run_dir), "seed": r.config.get("seed"), "accuracy": acc}
        if r.metrics is not None:
            m_acc = (r.metrics.get("overall") or {}).get("accuracy")
            if m_acc is not None and abs(m_acc - acc) > _METRICS_TOL:
                row["warning"] = f"metrics.json accuracy {m_acc} differs from mean(correct) {acc}"
        run_rows.append(row)

    accs = [row["accuracy"] for row in run_rows]
    out = {
        "ks": list(ks),
        "n_runs": len(runs),
        "runs": run_rows,
        "accuracy_mean": statistics.mean(accs) if accs else None,
        "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else None,
        "overall": _summarise(per_q, ks),
    }
    if by_level:
        grouped: Dict[str, Dict[int, List[bool]]] = defaultdict(dict)
        for qid, v in per_q.items():
            grouped[levels[qid]][qid] = v
        out["per_level"] = {lvl: _summarise(g, ks) for lvl, g in sorted(grouped.items())}
    return out
