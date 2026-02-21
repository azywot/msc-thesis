"""Example: mind_map tool — sub-agent mode.

The model is given a multi-hop research task that naturally produces
several intermediate facts.  It must use mind_map to store each fact
as it discovers it, then query the mind map to synthesise a final answer.
This verifies both the write (store) and read (query) paths of the tool.

NOTE: In sub-agent mode mind_map uses keyword/GraphRAG retrieval for queries
instead of exposing raw file operations.

Run from msc-thesis/:
    python examples/example_mind_map.py
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

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/mind_map"

# The orchestrator indexes your assistant output into the mind map (GraphRAG) before
# running any mind_map tool call. So: reason in full, then call mind_map with a query
# to retrieve from that reasoning; use the retrieval to give the final answer next turn.
QUESTION = (
    "Answer the following in two steps. You must use the mind_map tool.\n"
    "\n"
    "Step 1 — In this response: (a) Reason step by step about the Apollo 11 mission: "
    "when it landed, who was on board, how long they stayed on the Moon, and what "
    "Neil Armstrong said when he stepped onto the surface. (b) Then call the mind_map "
    "tool with a natural-language query to retrieve your reasoning, e.g. query: \"key "
    "facts and quote from Apollo 11 mission\" or \"summary of Apollo 11 landing and "
    "Armstrong quote\". The system will index your reasoning and run your query.\n"
    "\n"
    "Step 2 — In your next response: Using only what the mind_map tool returned, "
    "write exactly one sentence that could go in an encyclopaedia (who, when, what, "
    "and the famous quote). Your final answer is that one sentence only."
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: mind_map (sub-agent mode) ===")

    config = load_experiment_config(DEFAULT_CONFIG)
    set_seed(config.seed)

    cache_manager = CacheManager(config.cache_dir)
    enabled = ["mind_map"]
    planner, providers, _ = build_model_providers(config, required_roles=roles_for_tools(enabled))
    tools = build_tools(config, cache_manager, providers, enabled_tools=enabled)
    system_prompt = build_system_prompt(config, tools)
    orchestrator = build_orchestrator(config, planner, tools)

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
