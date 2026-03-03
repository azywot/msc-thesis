"""Example: context_manager tool — sub-agent mode (with web_search to exercise GraphRAG indexing).

The model is given a multi-hop research task that requires web_search to find
facts, then context_manager to store and query them. This verifies that the orchestrator
indexes reasoning before BOTH web_search and context_manager calls into the GraphRAG
knowledge base (matching multi-agent-tools behavior).

NOTE: In sub-agent mode context_manager uses GraphRAG for retrieval instead of raw file ops.

Run from msc-thesis/:
    python examples/example_context_manager.py
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

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/context_manager"

# The orchestrator indexes reasoning before web_search, code_generator, and context_manager
# into the GraphRAG. This example uses web_search + context_manager to verify that indexing
# happens for both tools — run it and inspect the context_manager cache to confirm writes.
QUESTION = (
    "Answer the following in two steps. You must use both web_search and context_manager.\n"
    "\n"
    "Step 1 — In this response: (a) Use the web_search tool to look up key facts "
    "about the Apollo 11 mission: when it landed, who was on board, and what "
    "Neil Armstrong said when he stepped onto the Moon. (b) Then call the context_manager "
    "tool with a query to store or retrieve that information, e.g. query: \"key "
    "facts and quote from Apollo 11 mission\". The system indexes your reasoning "
    "before each tool call into the GraphRAG.\n"
    "\n"
    "Step 2 — In your next response: Using what the context_manager tool returned, "
    "write exactly one sentence that could go in an encyclopaedia (who, when, "
    "what, and the famous quote). Your final answer is that one sentence only."
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: context_manager + web_search (sub-agent mode, GraphRAG indexing) ===")

    config = load_experiment_config(DEFAULT_CONFIG)
    set_seed(config.seed)

    cache_manager = CacheManager(config.cache_dir, web_tool_provider=config.tools.web_tool_provider)
    enabled = ["web_search", "context_manager"]
    orchestrator, providers, _ = build_model_providers(config, required_roles=enabled)
    tools = build_tools(config, cache_manager, providers, enabled_tools=enabled, orchestrator_model=orchestrator)
    system_prompt = build_system_prompt(config, tools)
    orchestrator_instance = build_orchestrator(config, orchestrator, tools, cache_manager=cache_manager)

    logger.info(f"Question: {QUESTION}")
    try:
        state = orchestrator_instance.run(
            question=QUESTION,
            question_id=0,
            system_prompt=system_prompt,
        )
        result_path = save_result(OUTPUT_DIR, state, config)
        print_summary(logger, state, config, result_path)
    finally:
        orchestrator_instance.cleanup()


if __name__ == "__main__":
    main()
