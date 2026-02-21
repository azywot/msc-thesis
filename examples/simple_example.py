"""Simple example demonstrating the agent_engine system.

This script shows how to:
1. Load a configuration from YAML
2. Initialize a model provider
3. Register tools
4. Create an orchestrator
5. Run agentic reasoning on a question
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_engine.config import load_experiment_config
from agent_engine.models.vllm_provider import VLLMProvider
from agent_engine.core import ToolRegistry, AgenticOrchestrator
from agent_engine.utils import setup_logging, set_seed
from agent_engine.tools import WebSearchTool, CodeGeneratorTool


def main():
    """Run a simple example."""
    parser = argparse.ArgumentParser(description="Run agent_engine simple example")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "experiments/configs/gaia/baseline.yaml"),
        help="Path to config YAML (default: experiments/configs/gaia/baseline.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override (default: config.output_dir)",
    )
    args = parser.parse_args()

    def _save_json(path: Path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    # Basic console logging until we know output_dir.
    logger = setup_logging()
    logger.info("Starting simple example")

    # Load configuration
    config_path = Path(args.config)
    logger.info(f"Loading config from: {config_path}")
    config = load_experiment_config(config_path)

    # Set random seed for reproducibility (use config.seed like run_experiment.py)
    set_seed(config.seed)

    # Resolve output directory early (to match scripts/run_experiment.py behavior)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconfigure logging to also write to file (same as experiments)
    logger = setup_logging(log_file=output_dir / "experiment.log")
    logger.info("=" * 80)
    logger.info(f"Starting simple example: {datetime.now()}")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output dir: {output_dir}")

    # Initialize model
    logger.info(f"Initializing model: {config.get_model('planner').name}")
    model = VLLMProvider(config.get_model("planner"))

    # Setup tools
    logger.info("Setting up tools")
    tools = ToolRegistry()

    # Add web search tool (requires SERPER_API_KEY environment variable)
    serper_key = os.getenv("SERPER_API_KEY")
    if serper_key:
        tools.register(WebSearchTool(
            serper_api_key=serper_key,
            top_k=config.tools.top_k_results,
            max_doc_len=config.tools.max_doc_len
        ))
        logger.info("Registered web_search tool")
    else:
        logger.warning("SERPER_API_KEY not found, skipping web search tool")

    # Add code generator tool
    tools.register(CodeGeneratorTool(
        timeout_seconds=60,
        temp_dir=str(config.cache_dir / "code_temp")
    ))
    logger.info("Registered code_generator tool")

    # Build simple system prompt
    system_prompt = """You are a helpful AI assistant with access to tools.

You can use the following tools:
- web_search: Search the web for information
- code_generator: Execute Python code for calculations

When you want to use a tool, format your response as:
<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>

When you have the final answer, state it clearly."""

    # Example question
    question = "What is the capital of France? Use web search to verify."
    logger.info(f"Question: {question}")

    # Run agentic reasoning
    orchestrator = None
    try:
        logger.info("Creating orchestrator")
        orchestrator = AgenticOrchestrator(
            model_provider=model,
            tool_registry=tools,
            max_turns=config.max_turns,
            tool_limits={'web_search': config.tools.max_search_limit},
            use_thinking=config.use_planner_thinking(),
        )

        state = orchestrator.run(
            question=question,
            question_id=0,
            system_prompt=system_prompt
        )

        # Persist outputs in the same directory/shape as experiments
        _save_json(output_dir / "state.json", state.to_dict())

        output_text = "\n".join(
            (msg.get("content") or "")
            for msg in state.messages
            if msg.get("role") in ("assistant", "tool") and (msg.get("content") or "")
        ).strip()

        results = [{
            "question_id": state.question_id,
            "question": state.question,
            "prediction": state.answer,
            "ground_truth": None,
            "correct": None,
            "evaluation": {},
            "output_text": output_text,
            "turns": state.turn,
            "tool_counts": state.tool_counts,
            "metadata": state.metadata,
        }]
        _save_json(output_dir / "results.json", results)

        # Display results
        logger.info("\n" + "="*80)
        logger.info("RESULTS")
        logger.info("="*80)
        logger.info(f"Question: {state.question}")
        logger.info(f"Answer: {state.answer}")
        logger.info(f"Turns used: {state.turn}/{config.max_turns}")
        logger.info(f"Finished: {state.finished}")
        logger.info(f"Tool calls: {len(state.tool_calls)}")
        logger.info(f"Tool usage: {state.tool_counts}")
        logger.info("="*80)
        logger.info(f"Saved: {output_dir / 'experiment.log'}")
        logger.info(f"Saved: {output_dir / 'results.json'}")
        logger.info(f"Saved: {output_dir / 'state.json'}")

        # Show conversation history
        logger.info("\nConversation History:")
        for i, msg in enumerate(state.messages):
            role = msg["role"]
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            logger.info(f"  [{i+1}] {role}: {content}")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)

    finally:
        # Cleanup
        logger.info("Cleaning up resources")
        if orchestrator is not None:
            orchestrator.cleanup()


if __name__ == "__main__":
    main()
