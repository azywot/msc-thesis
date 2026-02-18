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
    save_result,
)
from agent_engine.caching import CacheManager
from agent_engine.config import load_experiment_config
from agent_engine.utils import set_seed, setup_logging

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/mind_map"

# A multi-step task: the model should store intermediate reasoning steps
# in the mind map and then query it to compose the final answer.
QUESTION = (
    "You are researching the Apollo moon-landing programme to write a short summary.\n"
    "Follow these steps strictly:\n"
    "\n"
    "Step 1 — Store the basic facts: use mind_map to record that Apollo 11 landed on "
    "the Moon on 20 July 1969 with astronauts Neil Armstrong and Buzz Aldrin.\n"
    "\n"
    "Step 2 — Store the duration: use mind_map to record that Armstrong and Aldrin "
    "spent approximately 2 hours 31 minutes on the lunar surface.\n"
    "\n"
    "Step 3 — Store the famous quote: use mind_map to record the exact words spoken "
    "by Neil Armstrong upon stepping onto the Moon: "
    "'That's one small step for man, one giant leap for mankind.'\n"
    "\n"
    "Step 4 — Query your mind map: use mind_map to retrieve everything you stored "
    "about Apollo 11, then use that information to write a two-sentence summary "
    "of the mission suitable for an encyclopaedia entry.\n"
    "\n"
    "Your final answer should be ONLY the two-sentence encyclopaedia summary."
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: mind_map (sub-agent mode) ===")

    config = load_experiment_config(DEFAULT_CONFIG)
    set_seed(config.seed)

    cache_manager = CacheManager(config.cache_dir)
    planner, providers, _ = build_model_providers(config)
    tools = build_tools(config, cache_manager, providers, enabled_tools=["mind_map"])
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
