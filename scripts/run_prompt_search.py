#!/usr/bin/env python
"""Run the MASS-style prompt search (RSI phase 1) for one domain.

    python scripts/run_prompt_search.py --config <yaml> --domain search|math \
        [--run-name NAME] [--dry-run] [--report-only]

The YAML is an ExperimentConfig plus a ``search:`` block parsed into SearchConfig. Outputs go to
``<search.output_root>/<run_name>/<domain>/``.
"""

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent_engine.caching import CacheManager  # noqa: E402
from agent_engine.config import load_experiment_config  # noqa: E402
from agent_engine.models.base import get_tool_call_format  # noqa: E402
from agent_engine.runner.providers import setup_model_provider  # noqa: E402
from agent_engine.runner.tools import build_tool_registry  # noqa: E402
from agent_engine.tools.registry import ToolDeps  # noqa: E402
from agent_engine.utils import set_seed, setup_logging  # noqa: E402
from rsi.bundle import seed_bundle  # noqa: E402
from rsi.data import load_domain_examples, make_pools  # noqa: E402
from rsi.executor import BundleExecutor  # noqa: E402
from rsi.leakage import LeakageIndex  # noqa: E402
from rsi.reflector import build_reflector  # noqa: E402
from rsi.report import write_report  # noqa: E402
from rsi.search import SearchConfig, _write_json_atomic, run_search, write_run_header  # noqa: E402

logger = logging.getLogger("run_prompt_search")


class _StubProvider:
    """Stands in for a model in --dry-run so sub-agent-mode tool schemas can be built."""

    def generate(self, prompts):
        raise RuntimeError("stub provider must never be called")

    apply_chat_template = generate


class _SavingExecutor:
    """Wraps a BundleExecutor and flushes the web caches after every run, so a SIGTERM keeps them."""

    def __init__(self, inner, cache_manager):
        self._inner = inner
        self._cache = cache_manager

    @property
    def rollouts(self):
        return self._inner.rollouts

    @rollouts.setter
    def rollouts(self, value):
        self._inner.rollouts = value

    def run(self, bundle, examples):
        try:
            return self._inner.run(bundle, examples)
        finally:
            self._cache.save_caches()


def _raise_on_sigterm(signum, frame):
    raise SystemExit(143)


def load_search_config(path):
    path = Path(path)
    exp = load_experiment_config(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if "search" not in raw:
        raise ValueError(f"{path} has no 'search:' block")
    return exp, SearchConfig.model_validate(raw["search"])


def _api_keys(exp, dry_run):
    keys = {
        "serper": os.getenv("SERPER_API_KEY"),
        "tavily": os.getenv("TAVILY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }
    if dry_run and not keys.get(exp.tools.web_tool_provider):
        keys[exp.tools.web_tool_provider] = "dry-run"
    return keys


def _build_tools(exp, cache_manager, api_keys, model_providers, orchestrator_model):
    return build_tool_registry(
        ToolDeps(
            config=exp,
            cache_manager=cache_manager,
            api_keys=api_keys,
            model_providers=model_providers,
            orchestrator_model=orchestrator_model,
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="MASS-style prompt search (RSI phase 1)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--domain", required=True, choices=["search", "math"])
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)

    setup_logging()
    exp, search = load_search_config(args.config)
    run_name = args.run_name or Path(args.config).stem
    out_dir = Path(search.output_root) / run_name / args.domain
    family = exp.get_model("orchestrator").family
    tool_call_format = get_tool_call_format(family)
    set_seed(exp.seed)

    cache_manager = CacheManager(
        exp.cache_dir,
        web_tool_provider=exp.tools.web_tool_provider,
        dataset_name="rsi_" + args.domain,
    )
    api_keys = _api_keys(exp, args.dry_run or args.report_only)

    if args.dry_run or args.report_only:
        stub = _StubProvider()
        providers = (
            {t: stub for t in exp.tools.enabled_tools} if not exp.tools.direct_tool_call else {}
        )
        registry = _build_tools(exp, cache_manager, api_keys, providers, stub)
        tool_schemas = registry.get_all_schemas()
        if args.report_only:
            path = write_report(out_dir, tool_schemas, tool_call_format)
            print(f"report: {path}")
            return 0
        examples = load_domain_examples(search.data_path, args.domain)
        pools = make_pools(examples, search.pools, search.pool_seed)
        seed = seed_bundle(args.domain, tool_schemas, exp.tools.direct_tool_call)
        write_run_header(search, pools, out_dir)
        _write_json_atomic(out_dir / "bundles" / "v0.json", seed.to_dict())
        print(f"shown={len(pools.shown)} screen={len(pools.screen)} heldout={len(pools.heldout)}")
        return 0

    orch_cfg = exp.get_model("orchestrator")
    if search.reflector.backend == "in_process":
        if search.reflector.model != orch_cfg.path_or_id:
            raise ValueError(
                "in_process reflector must be the executor model: reflector "
                f"{search.reflector.model!r} != orchestrator {orch_cfg.path_or_id!r}"
            )

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    model_cache = {}
    orchestrator_model = setup_model_provider(orch_cfg, api_keys, model_cache)
    providers = {}
    if not exp.tools.direct_tool_call:
        for name in exp.tools.enabled_tools:
            cfg = exp.get_model(name)
            providers[name] = (
                setup_model_provider(cfg, api_keys, model_cache)
                if cfg is not None
                else orchestrator_model
            )
    registry = _build_tools(exp, cache_manager, api_keys, providers, orchestrator_model)
    tool_schemas = registry.get_all_schemas()

    examples_list = load_domain_examples(search.data_path, args.domain)
    examples = {e.question_id: e for e in examples_list}
    pools = make_pools(examples_list, search.pools, search.pool_seed)
    seed = seed_bundle(args.domain, tool_schemas, exp.tools.direct_tool_call)

    import pandas as pd

    all_questions = [str(q) for q in pd.read_parquet(search.data_path)["question"]]
    leakage = LeakageIndex(all_questions, n=search.ngram)

    executor = _SavingExecutor(
        BundleExecutor(
            orchestrator_model,
            registry,
            tool_call_format,
            max_turns=exp.max_turns,
            tool_limits={"web_search": exp.tools.max_search_limit},
            use_thinking=exp.use_orchestrator_thinking(),
        ),
        cache_manager,
    )
    reflector = build_reflector(search.reflector, orchestrator_model)

    from transformers import AutoTokenizer

    from gepa_integration.reflection import make_token_counter

    count_tokens = make_token_counter(AutoTokenizer.from_pretrained(search.reflector.model))

    try:
        run_search(
            search,
            args.domain,
            executor,
            reflector,
            examples,
            pools,
            seed,
            leakage,
            count_tokens,
            tool_schemas,
            tool_call_format,
            out_dir,
        )
    finally:
        cache_manager.save_caches()
    path = write_report(out_dir, tool_schemas, tool_call_format)
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
