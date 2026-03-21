"""Example: text_inspector tool — sub-agent mode.

The model is asked specific questions about a local document file.
The answers can only be extracted by reading and analysing the file contents,
so the model must call text_inspector.

The fixture document (examples/fixtures/sample_document.txt) is an invented
annual report for a fictional renewable-energy cooperative.

Run from msc-thesis/:
    python examples/example_text_inspector.py
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
from agent_engine.utils import set_seed, setup_logging

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/text_inspector"

DOCUMENT_PATH = Path(__file__).parent / "fixtures" / "sample_document.txt"

# Questions that require reading and reasoning over the document.
QUESTION = (
    f"The file at '{DOCUMENT_PATH}' is an annual report for Greenleaf Renewable Energy Cooperative.\n"
    "Use the text_inspector tool to read the file and then answer ALL of the following questions:\n"
    "1. What was the total revenue in FY 2023, and by what percentage did it grow year-on-year?\n"
    "2. Which energy source produced the most electricity, and how many GWh did it generate?\n"
    "3. How many new member co-operatives joined during FY 2023 compared to FY 2022?\n"
    "4. What is the combined capacity (MW) of the three pipeline projects planned by end of 2026?\n"
    "5. What was the net profit after tax?"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: text_inspector (sub-agent mode) ===")
    logger.info(f"Document: {DOCUMENT_PATH}")

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Fixture document not found: {DOCUMENT_PATH}\n"
            "It should have been created along with this example file."
        )

    config = DEFAULT_CONFIG
    set_seed(config.seed)

    cache_manager = CacheManager(
        config.cache_dir,
        web_tool_provider=config.tools.web_tool_provider,
        dataset_name=config.dataset.name,
    )
    enabled = ["text_inspector"]
    orchestrator_model, providers, _ = build_model_providers(config, required_roles=enabled)
    tools = build_tools(config, cache_manager, providers, enabled_tools=enabled)
    system_prompt = build_system_prompt(config, tools)
    orchestrator = build_orchestrator(config, orchestrator_model, tools, cache_manager=cache_manager)

    logger.info(f"Question: {QUESTION}")
    try:
        state = orchestrator.run(
            question=QUESTION,
            question_id=0,
            system_prompt=system_prompt,
            attachments=[str(DOCUMENT_PATH)],
        )
        result_path = save_result(OUTPUT_DIR, state, config)
        print_summary(logger, state, config, result_path)
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
