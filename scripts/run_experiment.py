"""Main experiment runner script.

This script loads a configuration file and runs a complete experiment,
processing all examples from a dataset and saving results.
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_engine.config import load_experiment_config
from agent_engine.models.base import ModelFamily
from agent_engine.core import ToolRegistry, AgenticOrchestrator, ExecutionState
from agent_engine.utils import setup_logging, set_seed
from agent_engine.utils.wandb_logging import log_results_wandb
from agent_engine.tools import (
    WebSearchTool,
    CodeGeneratorTool,
    MindMapTool,
    TextInspectorTool,
    ImageInspectorTool,
)
from agent_engine.datasets import DatasetRegistry
from agent_engine.prompts import PromptBuilder
from agent_engine.caching import CacheManager


logger = logging.getLogger(__name__)


def setup_model_provider(model_config, api_keys: Dict[str, str], model_cache: Optional[Dict[str, Any]] = None):
    """Initialize model provider with instance caching for memory efficiency.
    
    This function implements model instance reuse: if the same model path is
    requested multiple times (e.g., for different roles like planner/search/code),
    it returns the cached instance instead of loading a new one. This is critical
    for memory efficiency when using the same model for multiple roles.
    
    Thread safety is handled via locks in VLLMProvider, so multiple roles can
    safely share the same model instance.

    Args:
        model_config: ModelConfig instance
        api_keys: Dictionary of API keys
        model_cache: Optional cache dict for model instances (keyed by path_or_id)

    Returns:
        Model provider instance (new or cached)
    """
    # Check cache for local models (API models are lightweight, no need to cache)
    cache_key = model_config.path_or_id
    if model_cache is not None and cache_key in model_cache:
        if model_config.family not in (ModelFamily.GPT4, ModelFamily.CLAUDE):
            cached_provider = model_cache[cache_key]
            logger.info(f"♻️  Reusing cached model instance for: {model_config.name} (role: {model_config.role})")
            return cached_provider
    
    # Lazy imports so this script can run with only API deps installed.
    if model_config.family in [ModelFamily.GPT4]:
        from agent_engine.models.api_provider import OpenAIProvider
        provider = OpenAIProvider(model_config, api_key=api_keys.get("openai"))
    elif model_config.family == ModelFamily.CLAUDE:
        from agent_engine.models.api_provider import AnthropicProvider
        provider = AnthropicProvider(model_config, api_key=api_keys.get("anthropic"))
    else:
        # Local vLLM model - cache these to avoid duplicate loading
        from agent_engine.models.vllm_provider import VLLMProvider
        provider = VLLMProvider(model_config)
        
        # Cache local model instances for reuse
        if model_cache is not None:
            model_cache[cache_key] = provider
            logger.info(f"💾 Cached model instance: {cache_key}")
    
    return provider


def setup_tools(config, cache_manager, api_keys: Dict[str, str], model_providers: Dict[str, Any] = None) -> ToolRegistry:
    """Set up tools based on configuration.

    Args:
        config: Experiment configuration
        cache_manager: Cache manager instance
        api_keys: Dictionary of API keys
        model_providers: Dictionary of model providers for sub-agent mode (optional)

    Returns:
        ToolRegistry with registered tools
    """
    tools = ToolRegistry()

    # Determine if we use sub-agent thinking
    use_subagent_thinking = config.use_subagent_thinking()
    direct_mode = config.tools.direct_tool_call

    for tool_name in config.tools.enabled_tools:
        if tool_name == "web_search":
            # Get model provider for sub-agent mode
            search_model = model_providers.get("search") if not direct_mode and model_providers else None

            tools.register(WebSearchTool(
                serper_api_key=api_keys.get("serper"),
                search_cache=cache_manager.search_cache,
                url_cache=cache_manager.url_cache,
                top_k=config.tools.top_k_results,
                max_doc_len=config.tools.max_doc_len,
                model_provider=search_model,
                fetch_urls=True,  # Enable URL fetching (batched in run_batch)
                use_thinking=use_subagent_thinking
            ))
        elif tool_name == "code_generator":
            # Get model provider for sub-agent mode
            coding_model = model_providers.get("coding") if not direct_mode and model_providers else None

            tools.register(CodeGeneratorTool(
                timeout_seconds=60,
                temp_dir=str(config.cache_dir / "code_temp"),
                model_provider=coding_model,
                use_thinking=use_subagent_thinking
            ))
        elif tool_name == "mind_map":
            # Mind map has different behavior in direct vs non-direct mode
            # In non-direct mode, uses GraphRAG for intelligent retrieval
            storage_path = str(config.cache_dir / "mind_map")
            tools.register(MindMapTool(
                direct_mode=direct_mode,
                storage_path=storage_path,
                use_graphrag=True  # Enable GraphRAG in non-direct mode
            ))
        elif tool_name == "text_inspector":
            # Get model provider for sub-agent mode (optional for text inspector)
            text_inspector_model = model_providers.get("text_inspector") if not direct_mode and model_providers else None

            tools.register(TextInspectorTool(
                max_chars=50000,
                model_provider=text_inspector_model,
                use_thinking=use_subagent_thinking
            ))
        elif tool_name == "image_inspector":
            # Image inspector requires a VLM, so only enable in non-direct mode
            if not direct_mode:
                # Get VLM model provider for image analysis (required)
                vlm_model = model_providers.get("image_inspector") if model_providers else None

                if vlm_model is None:
                    logger.warning("image_inspector enabled but no VLM model provider configured - tool will fail at runtime")

                tools.register(ImageInspectorTool(
                    model_provider=vlm_model,
                    use_thinking=use_subagent_thinking
                ))
            else:
                logger.warning("image_inspector is disabled in direct mode (requires VLM)")

    return tools


def run_experiment(args):
    """Run complete experiment.

    Args:
        args: Command-line arguments
    """
    # Basic console logging until we know output_dir.
    logger = setup_logging()

    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = load_experiment_config(Path(args.config))

    # Resolve output directory early (used for partial + legacy saves)
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconfigure logging to also write to file
    logger = setup_logging(log_file=output_dir / "experiment.log")
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {datetime.now()}")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Description: {config.description}")

    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")

    # Get API keys from environment
    api_keys = {
        "serper": os.getenv("SERPER_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }

    # Initialize cache manager
    logger.info(f"Initializing cache at: {config.cache_dir}")
    cache_manager = CacheManager(config.cache_dir)

    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.name}")
    dataset = DatasetRegistry.get(config.dataset)
    examples = dataset.get_subset(config.dataset.subset_num)
    logger.info(f"Loaded {len(examples)} examples")

    # Initialize model cache for instance reuse
    # This allows sharing the same vLLM instance across multiple roles (memory efficient)
    model_cache: Dict[str, Any] = {}
    
    # Initialize models
    logger.info(f"Initializing planner model: {config.get_model('planner').name}")
    planner_model = setup_model_provider(config.get_model("planner"), api_keys, model_cache)

    # Initialize sub-agent models if not in direct mode (may reuse planner instance)
    model_providers = {}
    if not config.tools.direct_tool_call:
        logger.info("Sub-agent mode enabled - initializing tool models")

        # Initialize search model if configured (will reuse if same path as planner)
        if config.has_model("search"):
            logger.info(f"Initializing search model: {config.get_model('search').name}")
            model_providers["search"] = setup_model_provider(config.get_model("search"), api_keys, model_cache)

        # Initialize coding model if configured (will reuse if same path as planner)
        if config.has_model("coding"):
            logger.info(f"Initializing coding model: {config.get_model('coding').name}")
            model_providers["coding"] = setup_model_provider(config.get_model("coding"), api_keys, model_cache)

        # Optional: model-backed text inspector in sub-agent mode
        if config.has_model("text_inspector"):
            logger.info(f"Initializing text_inspector model: {config.get_model('text_inspector').name}")
            model_providers["text_inspector"] = setup_model_provider(config.get_model("text_inspector"), api_keys, model_cache)

        # Optional: VLM model for image inspection in sub-agent mode
        if config.has_model("image_inspector"):
            logger.info(f"Initializing image_inspector model: {config.get_model('image_inspector').name}")
            model_providers["image_inspector"] = setup_model_provider(config.get_model("image_inspector"), api_keys, model_cache)

        logger.info(f"Thinking mode: {config.thinking_mode.value}")
        logger.info(f"Planner uses thinking: {config.use_planner_thinking()}")
        logger.info(f"Sub-agents use thinking: {config.use_subagent_thinking()}")
    else:
        logger.info("Direct tool mode enabled - no sub-agent models needed")

    # Setup tools
    logger.info(f"Setting up tools: {config.tools.enabled_tools}")
    tools = setup_tools(config, cache_manager, api_keys, model_providers)

    # Initialize prompt builder
    prompt_builder = PromptBuilder()

    orchestrator = None
    results: List[Dict[str, Any]] = []
    correct_count = 0

    logger.info("="*80)
    logger.info("Starting evaluation")
    logger.info("="*80)

    try:
        # Create orchestrator
        orchestrator = AgenticOrchestrator(
            model_provider=planner_model,
            tool_registry=tools,
            max_turns=config.max_turns,
            tool_limits={'web_search': config.tools.max_search_limit},
            use_thinking=config.use_planner_thinking(),
        )

        # Batch size (default: 8 for higher throughput on vLLM; set to 1 to disable batching)
        batch_size = int(getattr(args, "batch_size", 8) or 8)
        batch_size = max(1, batch_size)

        tool_schemas = tools.get_all_schemas()  # schema is constant across the run

        def _chunks(seq, size: int):
            for i in range(0, len(seq), size):
                yield i, seq[i : i + size]

        for base_idx, batch in _chunks(examples, batch_size):
            logger.info(f"\nProcessing batch {base_idx + 1}-{base_idx + len(batch)} / {len(examples)}")

            # Build per-example system prompts (include attachments)
            system_prompts = []
            for ex in batch:
                system_prompts.append(
                    prompt_builder.build_system_prompt(
                        dataset_name=config.dataset.name,
                        tool_schemas=tool_schemas,
                        attachments=ex.get_attachments(),
                        max_search_limit=config.tools.max_search_limit,
                        direct_tool_call=config.tools.direct_tool_call,
                    )
                )

            # Run batched orchestrator
            try:
                states = orchestrator.run_batch(
                    questions=[ex.question for ex in batch],
                    question_ids=[ex.question_id for ex in batch],
                    system_prompts=system_prompts,
                    attachments=[ex.get_attachments() for ex in batch],
                )
            except Exception as e:
                # Catastrophic failure for the whole batch
                logger.error(f"Error processing batch starting at {base_idx}: {e}", exc_info=True)
                for ex in batch:
                    results.append({"question_id": ex.question_id, "question": ex.question, "error": str(e)})
                continue

            # Evaluate + store results
            for ex, state in zip(batch, states):
                prediction = state.answer or ""
                eval_result = dataset.evaluate(
                    prediction=prediction,
                    ground_truth=ex.answer,
                    metadata=ex.metadata,
                )
                if eval_result.get("correct", False):
                    correct_count += 1

                results.append({
                    "question_id": ex.question_id,
                    "question": ex.question,
                    "prediction": prediction,
                    "ground_truth": ex.answer,
                    "correct": eval_result.get("correct", False),
                    "evaluation": eval_result,
                    "output_text": _state_to_legacy_output_text(state),
                    "turns": state.turn,
                    "tool_counts": state.tool_counts,
                    "metadata": ex.metadata,
                })

            # Save intermediate results
            if (base_idx + len(batch)) % 10 == 0:
                save_results(results, output_dir / "results_partial.json")
                cache_manager.save_caches()

    finally:
        # Persist whatever we have (even if interrupted)
        save_results(results, output_dir / "results.json")
        cache_manager.save_caches()

        # Also save legacy-format outputs + metrics (multi-agent-tools compatible)
        legacy = save_legacy_results_and_metrics(
            config=config,
            examples=examples,
            results=results,
            output_dir=output_dir,
        )

        # W&B logging (only if configured in YAML)
        if getattr(config, "use_wandb", False):
            project = getattr(config, "wandb_project", None)
            if not project:
                logger.warning("use_wandb=true but wandb_project is not set; skipping W&B logging.")
            else:
                planner = config.get_model("planner")
                model_name = planner.name if planner else "unknown"
                mode = "direct" if config.tools.direct_tool_call else "subagent"
                enabled = set(config.tools.enabled_tools or [])
                log_results_wandb(
                    project=str(project),
                    run_name=str(config.name),
                    dataset_name=str(config.dataset.name),
                    dataset_split=str(config.dataset.split),
                    model_name=str(model_name),
                    mode=mode,
                    thinking_mode=str(config.thinking_mode.value),
                    direct_tool_call=bool(config.tools.direct_tool_call),
                    enable_search_tool=("web_search" in enabled),
                    enable_code_tool=("code_generator" in enabled),
                    mind_map=("mind_map" in enabled),
                    enable_text_inspector_tool=("text_inspector" in enabled),
                    enable_image_inspector_tool=("image_inspector" in enabled),
                    final_metrics=legacy.get("final_metrics") if isinstance(legacy, dict) else None,
                    tool_stats=legacy.get("tool_stats") if isinstance(legacy, dict) else None,
                    metrics_path=str(legacy.get("metrics_path")) if isinstance(legacy, dict) and legacy.get("metrics_path") else None,
                    config_summary={
                        "experiment": config.name,
                        "dataset": config.dataset.name,
                        "split": config.dataset.split,
                        "direct_tool_call": config.tools.direct_tool_call,
                        "thinking_mode": config.thinking_mode.value,
                        "seed": config.seed,
                    },
                )

        if orchestrator is not None:
            orchestrator.cleanup()

    # Print summary
    accuracy = correct_count / len(examples) if examples else 0
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Total examples: {len(examples)}")
    logger.info(f"Correct: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.2%}")
    logger.info(f"Results saved to: {output_dir}")

    # Cleanup
    orchestrator.cleanup()


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _state_to_legacy_output_text(state: ExecutionState) -> str:
    """Flatten state.messages into a single string containing tool I/O.

    multi-agent-tools stores a single `Output` field which includes the
    model's <tool_call> and the subsequent <tool_response> blocks.
    """
    parts: List[str] = []
    for msg in state.messages:
        role = msg.get("role")
        if role in ("assistant", "tool"):
            content = msg.get("content") or ""
            if content:
                parts.append(content)
    return "\n".join(parts).strip()


def save_legacy_results_and_metrics(
    config,
    examples,
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Write multi-agent-tools-style `<split>.<ts>.json` + `.metrics.json`.

    The goal is not to perfectly replicate every legacy field, but to keep the
    **shape** (list of per-example dicts with `Answer` + `Output`) and always
    produce a metrics JSON alongside.
    """
    if not getattr(config, "dataset", None):
        return {}

    dataset_name = config.dataset.name
    split = config.dataset.split

    t = time.localtime()
    result_name = f"{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json"
    metrics_name = f"{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json"

    legacy_items: List[Dict[str, Any]] = []
    per_level: Dict[str, List[int]] = defaultdict(list)

    total = len(examples)
    correct_flags: List[int] = []

    for idx, example in enumerate(examples):
        r = results[idx] if idx < len(results) else {}
        prediction = r.get("prediction", "") or ""
        output_text = r.get("output_text", "") or prediction
        evaluation = r.get("evaluation", {}) or {}
        correct = bool(evaluation.get("correct", False))

        item: Dict[str, Any] = {
            "Question": example.question,
            "Answer": example.answer,
            "Output": output_text,
            "Pred_Answer": prediction,
        }

        # Dataset-specific fields (minimal subset for legacy tooling)
        if dataset_name == "gaia":
            level = example.metadata.get("level", 1)
            item["Level"] = level
            item["file_name"] = example.metadata.get("file_name", "")
            item["file_path"] = example.metadata.get("file_path", "")
            attachments = example.get_attachments()
            item["local_file_path"] = attachments[0] if attachments else None
            item["input_output"] = json.dumps({"inputs": [example.question], "outputs": [example.answer]})
            level_key = str(level)
        elif dataset_name == "gpqa":
            # Legacy evaluator expects `Choices` for GPQA.
            item["Choices"] = example.metadata.get("choices", [])
            level_key = "all"
        elif dataset_name in ("math500", "amc"):
            level = example.metadata.get("difficulty", example.metadata.get("year", "unknown"))
            item["Level"] = level
            item["type"] = example.metadata.get("problem_type", example.metadata.get("competition", "unknown"))
            level_key = str(level)
        elif dataset_name == "aime":
            year = example.metadata.get("year", "unknown")
            item["Year"] = year
            item["Level"] = year  # legacy evaluator mostly buckets by `Level`
            level_key = str(year)
        else:
            level_key = "all"

        # Always attach a metrics dict (legacy evaluate.py also adds this).
        # We use correctness-based placeholders to keep a stable schema.
        tool_counts = r.get("tool_counts", {}) or {}
        metrics: Dict[str, Any] = {
            "is_valid_answer": bool(prediction),
            "em": int(correct),
            "acc": int(correct),
            "f1": float(correct),
            "math_equal": int(correct),
            "search_total": int(tool_counts.get("web_search", 0)),
            "code_total": int(tool_counts.get("code_generator", 0)),
            "mind_map_total": int(tool_counts.get("mind_map", 0)),
            "text_inspector_total": int(tool_counts.get("text_inspector", 0)),
        }
        if dataset_name == "gaia":
            metrics["gaia_score"] = int(correct)

        item["Metrics"] = metrics

        legacy_items.append(item)
        correct_flags.append(int(correct))
        per_level[level_key].append(int(correct))

    overall_acc = sum(correct_flags) / total if total else 0.0
    overall_metrics: Dict[str, Any] = {
        "em": overall_acc,
        "acc": overall_acc,
        "f1": overall_acc,
        "math_equal": overall_acc,
        "num_valid_answer": f"{sum(1 for x in correct_flags if x)} of {total}",
    }
    if dataset_name == "gaia":
        overall_metrics["gaia_score"] = overall_acc

    per_level_metrics: Dict[str, Any] = {}
    for level_key, flags in per_level.items():
        if not flags:
            continue
        acc = sum(flags) / len(flags)
        entry = {
            "em": acc,
            "acc": acc,
            "f1": acc,
            "math_equal": acc,
            "num_valid_answer": f"{sum(1 for x in flags if x)} of {len(flags)}",
        }
        if dataset_name == "gaia":
            entry["gaia_score"] = acc
        per_level_metrics[level_key] = entry

    final_metrics: Dict[str, Any] = {"overall": overall_metrics}
    if dataset_name in ("gaia", "math500", "aime", "hle"):
        final_metrics["per_level"] = per_level_metrics

    # Save files
    results_path = output_dir / result_name
    metrics_path = output_dir / metrics_name

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(legacy_items, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    # Tool stats (useful for W&B logging; mirrors multi-agent-tools shape)
    tool_stats: Dict[str, Any] = {"overall": {}, "per_question": []}
    overall_tools = defaultdict(int)
    for r in results:
        tc = (r.get("tool_counts") or {}) if isinstance(r, dict) else {}
        per_q = {
            "search_total": int(tc.get("web_search", 0) or 0),
            "code_total": int(tc.get("code_generator", 0) or 0),
            "mind_map_total": int(tc.get("mind_map", 0) or 0),
            "text_inspector_total": int(tc.get("text_inspector", 0) or 0),
            "image_inspector_total": int(tc.get("image_inspector", 0) or 0),
        }
        tool_stats["per_question"].append(per_q)
        for k, v in per_q.items():
            overall_tools[k] += int(v)

    tool_stats["overall"] = dict(overall_tools)

    return {
        "results_path": results_path,
        "metrics_path": metrics_path,
        "final_metrics": final_metrics,
        "tool_stats": tool_stats,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run agent_engine experiment")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment config YAML file")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory (overrides config)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Number of questions to batch for model generation (default: 8). Set 1 to disable.")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
