#!/usr/bin/env python
"""Tool-call intervention: rescue failed trajectories with one reflector-corrected call.

    python scripts/run_call_intervention.py --config <yaml> --collected <jsonl...> \
        --out-dir <dir> [--subset N] [--report-only] [--blind]

The YAML is an ExperimentConfig plus an ``intervention:`` block parsed into InterventionConfig.
Writes candidates.json, rescue.jsonl, gate.json and gate.md to --out-dir. ``--blind`` re-runs
one attempt with the gold answer hidden from the reflector on a seeded sample of the rescued
questions (blind.jsonl) and rewrites the gate files. ``--report-only`` loads no models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_prompt_search import _api_keys, _build_tools  # noqa: E402

from agent_engine.caching import CacheManager  # noqa: E402
from agent_engine.config import load_experiment_config  # noqa: E402
from agent_engine.core import AgenticOrchestrator  # noqa: E402
from agent_engine.runner.providers import setup_model_provider  # noqa: E402
from agent_engine.utils import set_seed, setup_logging  # noqa: E402
from rsi.intervention.records import (  # noqa: E402
    domain_of,
    intervention_candidates,
    load_collected,
    subset_pool,
)
from rsi.intervention.report import blind_sample, summarize, write_gate  # noqa: E402
from rsi.intervention.rescue import RescueSettings, _read_rows, run_rescue  # noqa: E402
from rsi.leakage import LeakageIndex  # noqa: E402
from rsi.reflector import ReflectorSettings, build_reflector  # noqa: E402

logger = logging.getLogger("run_call_intervention")


class InterventionConfig(BaseModel):
    data_path: str
    n_passes: int = 3
    rescue: RescueSettings = RescueSettings()
    threshold_points: float = 10.0
    blind_n: int = 100
    ngram: int = 8
    reflector: ReflectorSettings


def load_intervention_config(path):
    path = Path(path)
    exp = load_experiment_config(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "intervention" not in raw:
        raise ValueError(f"{path} has no 'intervention:' block")
    if exp.tools.direct_tool_call:
        raise ValueError(f"{path}: intervention needs tools.direct_tool_call: false")
    if exp.baseline:
        raise ValueError(f"{path}: intervention needs baseline: false")
    return exp, InterventionConfig.model_validate(raw["intervention"])


def messages_token_counter(text_counter):
    """Adapt a string token counter to the message-list callable build_fix_messages expects.

    The message contents are joined with newlines; role markers and chat-template tokens are not
    counted, which is a small undercount covered by the budget's headroom (14000 of 32768).
    """

    def count(messages):
        return text_counter("\n".join(str(m.get("content", "")) for m in messages))

    return count


def build_runtime(exp, icfg):
    """Load providers, tools, orchestrator, reflector and token counter. Needs GPUs."""
    from transformers import AutoTokenizer

    from gepa_integration.reflection import make_token_counter

    cache_manager = CacheManager(
        exp.cache_dir, web_tool_provider=exp.tools.web_tool_provider, dataset_name="rsi_rescue"
    )
    api_keys = _api_keys(exp, False)
    model_cache = {}
    orchestrator_model = setup_model_provider(exp.get_model("orchestrator"), api_keys, model_cache)
    providers = {}
    for name in exp.tools.enabled_tools:
        cfg = exp.get_model(name)
        providers[name] = (
            setup_model_provider(cfg, api_keys, model_cache)
            if cfg is not None
            else orchestrator_model
        )
    registry = _build_tools(exp, cache_manager, api_keys, providers, orchestrator_model)
    orchestrator = AgenticOrchestrator(
        model_provider=orchestrator_model,
        tool_registry=registry,
        max_turns=exp.max_turns,
        tool_limits={"web_search": exp.tools.max_search_limit},
        use_thinking=exp.use_orchestrator_thinking(),
        cache_manager=cache_manager,
        baseline=False,
    )
    reflector = build_reflector(icfg.reflector, orchestrator_model)
    tokenizer = AutoTokenizer.from_pretrained(icfg.reflector.model)
    count_tokens = messages_token_counter(make_token_counter(tokenizer))
    seeded = [orchestrator_model, *providers.values(), *model_cache.values()]
    return orchestrator, seeded, reflector, count_tokens, cache_manager


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tool-call intervention rescue and gate report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--collected", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subset", type=int, default=None, metavar="N")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args(argv)

    setup_logging()
    exp, icfg = load_intervention_config(args.config)
    set_seed(exp.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_collected([Path(p) for p in args.collected])
    pool_ids = None
    if args.subset is not None:
        # Restrict the pool before counting, so zero_of_n, the skip counts and the candidates
        # all describe the same questions.
        records, pool_ids = subset_pool(records, icfg.n_passes, args.subset)
    candidates, counts = intervention_candidates(records, icfg.n_passes)
    _write_json(
        out_dir / "candidates.json",
        {
            "counts": counts,
            **({"pool_question_ids": pool_ids} if pool_ids is not None else {}),
            "candidates": [
                {
                    "question_id": c["question_id"],
                    "data_source": c["data_source"],
                    "domain": domain_of(c["data_source"]),
                }
                for c in candidates
            ],
        },
    )
    rescue_path, blind_path = out_dir / "rescue.jsonl", out_dir / "blind.jsonl"

    if not args.report_only and candidates:
        import pandas as pd

        index = LeakageIndex(
            [str(q) for q in pd.read_parquet(icfg.data_path)["question"]], n=icfg.ngram
        )
        orchestrator, seeded, reflector, count_tokens, cache_manager = build_runtime(exp, icfg)

        try:
            rows = run_rescue(
                candidates,
                orchestrator=orchestrator,
                providers=seeded,
                reflector=reflector,
                index=index,
                settings=icfg.rescue,
                out_path=rescue_path,
                max_turns=exp.max_turns,
                count_tokens=count_tokens,
            )
            if args.blind:
                ids = set(blind_sample(rows, icfg.blind_n))
                blind_settings = icfg.rescue.model_copy(
                    update={
                        "max_attempts": 1,
                        # run_rescue seeds attempt k with base + k, so this gives base + 99
                        "attempt_seed_base": icfg.rescue.attempt_seed_base + 98,
                    }
                )
                run_rescue(
                    [c for c in candidates if c["question_id"] in ids],
                    orchestrator=orchestrator,
                    providers=seeded,
                    reflector=reflector,
                    index=index,
                    settings=blind_settings,
                    out_path=blind_path,
                    max_turns=exp.max_turns,
                    count_tokens=count_tokens,
                    show_gold=False,
                )
        finally:
            cache_manager.save_caches()

    rows = _read_rows(rescue_path)
    blind_rows = _read_rows(blind_path) if blind_path.exists() else None
    summary = summarize(candidates, rows, counts, icfg.threshold_points, blind_rows)
    write_gate(out_dir, summary)
    gate = summary["gate"]
    print(f"gate: {'PASS' if gate['pass'] else 'FAIL'} passing={gate['passing_domains']}")
    print(f"report: {out_dir / 'gate.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
