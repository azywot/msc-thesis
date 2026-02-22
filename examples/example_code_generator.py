"""Example: code_generator tool — sub-agent mode.

The model is asked a computation-heavy question that requires writing and
running Python code; the answer is impossible to produce accurately without
actually executing the computation.

Run from msc-thesis/:
    python examples/example_code_generator.py
"""

from pathlib import Path

from _common import (
    DEFAULT_CONFIG,
    build_model_providers,
    build_orchestrator,
    build_system_prompt,
    build_tools,
    print_summary,
    roles_for_tools,
    save_result,
)
from agent_engine.caching import CacheManager
from agent_engine.config import load_experiment_config
from agent_engine.utils import set_seed, setup_logging

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/code_generator"

# Multi-part maths question — precise answer requires code execution.
QUESTION = (
    "Use the code_generator tool to write and execute Python code that does the following:\n"
    "1. Compute all prime numbers up to 1000 using the Sieve of Eratosthenes.\n"
    "2. Find the sum of those primes.\n"
    "3. Find the largest prime below 1000.\n"
    "4. Determine how many of those primes are palindromes (read the same forwards and backwards).\n"
    "Report all four results clearly."
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: code_generator (sub-agent mode) ===")

    config = load_experiment_config(DEFAULT_CONFIG)
    set_seed(config.seed)

    cache_manager = CacheManager(config.cache_dir)
    enabled = ["code_generator"]
    orchestrator_model, providers, _ = build_model_providers(config, required_roles=roles_for_tools(enabled))
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
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
