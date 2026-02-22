"""Example: web_search tool — sub-agent mode.

The model is asked a factual question about a recent event that it cannot
answer from training data alone, so it must use web_search to find the answer.

Run from msc-thesis/:
    python examples/example_web_search.py
"""

from pathlib import Path

from _common import (
    DEFAULT_CONFIG,
    build_model_providers,
    build_orchestrator,
    build_system_prompt,
    build_tools,
    print_summary,
    save_result,
)
from agent_engine.caching import CacheManager
from agent_engine.config import load_experiment_config
from agent_engine.utils import set_seed, setup_logging

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/web_search"

# The model cannot answer this reliably from weights alone — it must search.
QUESTION = (
    "Find the most recent FIFA Men's World Cup (after 2018), state who won it, "
    "what was the final match score, and where it was hosted. "
    "You MUST use the web_search tool to retrieve this information."
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: web_search (sub-agent mode) ===")

    config = load_experiment_config(DEFAULT_CONFIG)
    set_seed(config.seed)

    cache_manager = CacheManager(config.cache_dir)
    enabled = ["web_search"]
    orchestrator_model, providers, _ = build_model_providers(config, required_roles=enabled)
    tools = build_tools(config, cache_manager, providers, enabled_tools=enabled)
    system_prompt = build_system_prompt(config, tools)
    orchestrator = build_orchestrator(config, orchestrator_model, tools)

    logger.info(f"Question: {QUESTION}")
    try:
        state = orchestrator.run(
            question=QUESTION,
            question_id=0,
            system_prompt=system_prompt,
        )
        result_path = save_result(OUTPUT_DIR, state, config)
        print_summary(logger, state, config, result_path)
    finally:
        cache_manager.save_caches()
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
