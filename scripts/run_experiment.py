"""Main experiment runner script.

This script loads a configuration file and runs a complete experiment,
processing all examples from a dataset and saving results.
"""

import argparse
import json
import os
import sys
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

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
    requested multiple times (e.g., for different roles like orchestrator/web_search/code_generator),
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
            logger.info("♻️ Reusing cached model instance for: %s (role: %s)", model_config.name, model_config.role)
            return cached_provider
    
    # Lazy imports so this script can run with only API deps installed.
    backend = getattr(model_config, "backend", "vllm")

    if backend == "openai" or model_config.family in [ModelFamily.GPT4]:
        from agent_engine.models.api_provider import OpenAIProvider
        provider = OpenAIProvider(model_config, api_key=api_keys.get("openai"))
    elif backend == "anthropic" or model_config.family == ModelFamily.CLAUDE:
        from agent_engine.models.api_provider import AnthropicProvider
        provider = AnthropicProvider(model_config, api_key=api_keys.get("anthropic"))
    elif backend == "mlx":
        from agent_engine.models.mlx_provider import MLXProvider
        provider = MLXProvider(model_config)

        if model_cache is not None:
            model_cache[cache_key] = provider
            logger.info(f"💾 Cached MLX model instance: {cache_key}")
    else:
        # Local vLLM model - cache these to avoid duplicate loading
        from agent_engine.models.vllm_provider import VLLMProvider
        provider = VLLMProvider(model_config)

        # Cache local model instances for reuse
        if model_cache is not None:
            model_cache[cache_key] = provider
            logger.info(f"💾 Cached model instance: {cache_key}")

    return provider


def setup_tools(
    config,
    cache_manager,
    api_keys: Dict[str, str],
    model_providers: Optional[Dict[str, Any]] = None,
    orchestrator_model=None,
    mind_map_storage_path: Optional[Path] = None,
) -> ToolRegistry:
    """Set up tools based on configuration.

    Args:
        config: Experiment configuration
        cache_manager: Cache manager instance
        api_keys: Dictionary of API keys
        model_providers: Dictionary of model providers for sub-agent mode (optional)
        mind_map_storage_path: Optional run-specific path for mind_map cache
            (e.g. cache_dir/mind_map/gaia/all_validation/2026-02-22_123abc).
            If None, falls back to config.cache_dir / "mind_map".

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
            search_model = model_providers.get("web_search") if not direct_mode and model_providers else None
            # Select the correct API key based on provider
            provider = config.tools.web_tool_provider
            api_key = api_keys.get(provider)
            if not api_key:
                raise RuntimeError(f"{provider.upper()}_API_KEY environment variable is required for web_tool_provider='{provider}'")
            tools.register(WebSearchTool(
                api_key=api_key,
                provider=provider,
                search_cache=cache_manager.search_cache,
                url_cache=cache_manager.url_cache,
                top_k=config.tools.top_k_results,
                max_doc_len=config.tools.max_doc_len,
                model_provider=search_model,
                fetch_urls=True,
                use_thinking=use_subagent_thinking,
                cache_manager=cache_manager,
            ))
        elif tool_name == "code_generator":
            # Get model provider for sub-agent mode
            coding_model = model_providers.get("code_generator") if not direct_mode and model_providers else None
            tools.register(CodeGeneratorTool(
                timeout_seconds=60,
                temp_dir=str(config.cache_dir / "code_temp"),
                model_provider=coding_model,
                use_thinking=use_subagent_thinking
            ))
        elif tool_name == "mind_map":
            # In sub-agent mode, GraphRAG runs with a local model — no OpenAI key needed (mirrors MAT).
            # Use run-specific path when provided (mirrors results: cache/mind_map/{dataset}/{split}/{date}_{job_id})
            if mind_map_storage_path is not None:
                mind_map_storage_path.mkdir(parents=True, exist_ok=True)
                storage_path = str(mind_map_storage_path)
            else:
                storage_path = str(config.cache_dir / "mind_map")
            mind_map_model = model_providers.get("mind_map") if not direct_mode else None
            tools.register(MindMapTool(
                direct_mode=direct_mode,
                storage_path=storage_path,
                use_graphrag=True,
                model_provider=mind_map_model,
                use_thinking=use_subagent_thinking,
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
    logger = setup_logging()

    logger.info(f"Loading config from: {args.config}")
    config = load_experiment_config(Path(args.config))

    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir, date_str, job_id = _make_run_dir(output_dir, config.dataset.split)

    # Mind map cache path mirrors results: cache/mind_map/{dataset}/{split}/{date}_{job_id}
    mind_map_storage = (
        Path(config.cache_dir) / "mind_map" / config.dataset.name / config.dataset.split / f"{date_str}_{job_id}"
    )

    logger = setup_logging(log_file=run_dir / "experiment.log")
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {datetime.now()}")
    logger.info("=" * 80)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Description: {config.description}")

    set_seed(config.seed)
    start_time = datetime.now().isoformat()

    logger.info("Seed: %s  dataset: %s/%s  thinking: %s  direct_tool_call: %s",
                config.seed, config.dataset.name, config.dataset.split,
                getattr(config.thinking_mode, "value", "?"), config.tools.direct_tool_call)
    logger.info("Enabled tools: %s", list(config.tools.enabled_tools))

    api_keys = {
        "serper": os.getenv("SERPER_API_KEY"),
        "tavily": os.getenv("TAVILY_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }

    _provider = config.tools.web_tool_provider
    _dataset = config.dataset.name
    _cache_root = f"{config.cache_dir}/{_provider}/{_dataset}"
    logger.info("Initializing web cache at: %s  (provider: %s, dataset: %s)", _cache_root, _provider, _dataset)
    cache_manager = CacheManager(
        config.cache_dir,
        web_tool_provider=_provider,
        dataset_name=_dataset,
    )

    logger.info(f"Loading dataset: {config.dataset.name}")
    dataset = DatasetRegistry.get(config.dataset)
    examples = dataset.get_subset(config.dataset.subset_num)
    logger.info(f"Loaded {len(examples)} examples")

    model_cache: Dict[str, Any] = {}

    # Resolve GPU assignments before loading any model so utilization and GPU pinning
    # are set automatically (mirrors multi-agent-tools behaviour).
    # Skip entirely when all models use the MLX backend (no GPU management needed).
    _all_mlx = all(
        getattr(cfg, "backend", "vllm") == "mlx"
        for cfg in config.models.values()
    )
    try:
        if _all_mlx:
            raise RuntimeError("skip — all models use MLX backend")
        from agent_engine.models.vllm_provider import resolve_gpu_assignments
        gpu_assignments = resolve_gpu_assignments(config)

        def _apply_gpu_assignment(model_cfg) -> None:
            if model_cfg is None:
                return
            path = model_cfg.path_or_id
            if path in gpu_assignments:
                util, gpu_ids = gpu_assignments[path]
                if model_cfg.gpu_memory_utilization is None:
                    model_cfg.gpu_memory_utilization = util
                if model_cfg.gpu_ids is None and gpu_ids is not None:
                    model_cfg.gpu_ids = gpu_ids

        _apply_gpu_assignment(config.get_model("orchestrator"))
        for _tool in config.tools.enabled_tools:
            _apply_gpu_assignment(config.get_model(_tool))
    except Exception as _e:
        logger.warning("GPU assignment resolution skipped (%s); using per-model defaults.", _e)

    logger.info(f"Initializing orchestrator model: {config.get_model('orchestrator').name}")
    orchestrator_model = setup_model_provider(config.get_model("orchestrator"), api_keys, model_cache)

    # For each enabled tool: use models.<tool> if specified, else fall back to orchestrator.
    model_providers = {}
    if not config.tools.direct_tool_call:
        logger.info("Sub-agent mode enabled - initializing tool models")

        for tool_name in config.tools.enabled_tools:
            model_cfg = config.get_model(tool_name)
            if model_cfg is not None:
                logger.info(f"Initializing {tool_name} model: {model_cfg.name}")
                model_providers[tool_name] = setup_model_provider(model_cfg, api_keys, model_cache)
            else:
                logger.info(f"No models.{tool_name} in config — using orchestrator model as fallback")
                model_providers[tool_name] = orchestrator_model

        logger.info(f"Thinking mode: {config.thinking_mode.value}")
        logger.info(f"Orchestrator uses thinking: {config.use_orchestrator_thinking()}")
        logger.info(f"Sub-agents use thinking: {config.use_subagent_thinking()}")
    else:
        logger.info("Direct tool mode enabled - no sub-agent models needed")

    # Setup tools
    logger.info(f"Setting up tools: {config.tools.enabled_tools}")
    tools = setup_tools(
        config, cache_manager, api_keys, model_providers,
        orchestrator_model=orchestrator_model,
        mind_map_storage_path=mind_map_storage,
    )

    # Initialize prompt builder
    prompt_builder = PromptBuilder()

    logger.info("Run output directory: %s", run_dir)

    orchestrator = None
    results: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    system_prompt_for_config: Optional[str] = None

    logger.info("="*80)
    logger.info("Starting evaluation")
    logger.info("="*80)

    try:
        # Capture the common system prompt once for this run (same for all questions).
        tool_schemas = tools.get_all_schemas()
        system_prompt_for_config = prompt_builder.build_system_prompt(
            dataset_name=config.dataset.name,
            tool_schemas=tool_schemas,
            max_search_limit=config.tools.max_search_limit,
            direct_tool_call=config.tools.direct_tool_call,
        )

        orchestrator = AgenticOrchestrator(
            model_provider=orchestrator_model,
            tool_registry=tools,
            max_turns=config.max_turns,
            tool_limits={'web_search': config.tools.max_search_limit},
            use_thinking=config.use_orchestrator_thinking(),
            cache_manager=cache_manager,
        )

        raw_batch = getattr(config, "batch_size", -1) or -1
        batch_size = len(examples) if raw_batch <= 0 else max(1, int(raw_batch))

        def _chunks(seq, size: int):
            for i in range(0, len(seq), size):
                yield i, seq[i : i + size]

        for base_idx, batch in _chunks(examples, batch_size):
            logger.info(f"\nProcessing batch {base_idx + 1}-{base_idx + len(batch)} / {len(examples)}")

            # Reuse the common system prompt for all questions in the batch
            system_prompts = [system_prompt_for_config] * len(batch)

            try:
                states = orchestrator.run_batch(
                    questions=[ex.question for ex in batch],
                    question_ids=[ex.question_id for ex in batch],
                    system_prompts=system_prompts,
                    attachments=[ex.get_attachments() for ex in batch],
                )
            except Exception as e:
                logger.error(f"Error processing batch starting at {base_idx}: {e}", exc_info=True)
                for ex in batch:
                    results.append({"question_id": ex.question_id, "question": ex.question, "error": str(e)})
                continue

            for ex, state in zip(batch, states):
                prediction = state.answer or ""
                eval_result = dataset.evaluate(
                    prediction=prediction,
                    ground_truth=ex.answer,
                    metadata=ex.metadata,
                )
                token_usage = state.metadata.get("token_usage", {})
                results.append({
                    "question_id": ex.question_id,
                    "question": ex.question,
                    "prediction": prediction,
                    "ground_truth": ex.answer,
                    "correct": bool(eval_result.get("correct", False)),
                    "evaluation": eval_result,
                    "output_messages": _collect_output_messages(state),
                    "turns": state.turn,
                    "tool_counts": state.tool_counts,
                    "token_usage": token_usage,
                    "query_analysis": state.query_analysis,
                    "action_history": state.action_history,
                    "metadata": ex.metadata,
                })
                logger.info(
                    "Q%s tokens — prompt: %d  completion: %d  total: %d",
                    ex.question_id,
                    token_usage.get("prompt_tokens", 0),
                    token_usage.get("completion_tokens", 0),
                    token_usage.get("total_tokens", 0),
                )

            # Flush intermediate raw results every 10 examples
            if (base_idx + len(batch)) % 10 == 0:
                _write_json(run_dir / "raw_results.partial.json", results)
                cache_manager.save_caches()

    finally:
        cache_manager.save_caches()

        # ── raw_results.json ────────────────────────────────────────────────
        _write_json(run_dir / "raw_results.json", results)
        # Remove partial file if it exists
        (run_dir / "raw_results.partial.json").unlink(missing_ok=True)

        # ── metrics.json ────────────────────────────────────────────────────
        metrics = _compute_metrics(results, examples, config.dataset.name)
        metrics["start_time"] = start_time
        metrics["end_time"] = datetime.now().isoformat()
        _write_json(run_dir / "metrics.json", metrics)

        # ── config.json ─────────────────────────────────────────────────────
        config_dict = _config_to_dict(config)
        if system_prompt_for_config is not None:
            config_dict["system_prompt"] = system_prompt_for_config
        config_dict["config_path"] = str(args.config)
        _write_json(run_dir / "config.json", config_dict)

        logger.info("Results saved to: %s", run_dir)

        # ── W&B logging ─────────────────────────────────────────────────────
        if getattr(config, "use_wandb", False):
            project = getattr(config, "wandb_project", None)
            if not project:
                logger.warning("use_wandb=true but wandb_project is not set; skipping W&B logging.")
            else:
                orchestrator_model_config = config.get_model("orchestrator")
                model_name = orchestrator_model_config.name if orchestrator_model_config else "unknown"
                mode = "direct" if config.tools.direct_tool_call else "subagent"
                enabled = set(config.tools.enabled_tools or [])
                log_results_wandb(
                    project=str(project),
                    run_name=str(config.name),
                    dataset_name=str(config.dataset.name),
                    dataset_split=str(config.dataset.split),
                    subset_num=int(getattr(config.dataset, "subset_num", -1)),
                    model_name=str(model_name),
                    mode=mode,
                    thinking_mode=str(config.thinking_mode.value),
                    direct_tool_call=bool(config.tools.direct_tool_call),
                    enable_search_tool=("web_search" in enabled),
                    enable_code_tool=("code_generator" in enabled),
                    mind_map=("mind_map" in enabled),
                    enable_text_inspector_tool=("text_inspector" in enabled),
                    enable_image_inspector_tool=("image_inspector" in enabled),
                    final_metrics=metrics,
                    tool_stats=metrics.get("tool_usage"),
                    metrics_path=str(run_dir / "metrics.json"),
                    config_summary=config_dict,
                    config_path=str(run_dir / "config.json"),
                )

        if orchestrator is not None:
            orchestrator.cleanup()

    overall = metrics.get("overall", {})
    accuracy = overall.get("accuracy", 0.0)
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info("Total examples : %d", len(examples))
    logger.info("Accuracy: %.2f%%", accuracy * 100)
    logger.info("Results saved to: %s", run_dir)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _collect_output_messages(state: ExecutionState) -> List[Dict[str, str]]:
    """Return the orchestrator message history (assistant + tool messages).

    Skips the initial system and user messages (indices 0 and 1) since those
    are already captured by ``question``, ``query_analysis``, and the config.
    """
    msgs: List[Dict[str, str]] = []
    for msg in state.messages[2:]:
        role = msg.get("role", "")
        if role in ("assistant", "tool"):
            msgs.append({"role": role, "content": msg.get("content", "")})
    return msgs


def _make_run_dir(output_dir: Path, split: str) -> Tuple[Path, str, str]:
    """Create and return (run_dir, date_str, job_id). run_dir = {split}_YYYY-MM-DD-HH-MM-SS-{job_id}/."""
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_id = os.environ.get("SLURM_JOB_ID") or _short_id()
    run_dir = output_dir / f"{split}_{date_str}_{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, date_str, job_id


def _short_id() -> str:
    """Generate a short random hex ID for non-SLURM runs."""
    import secrets
    return secrets.token_hex(3)


def _config_to_dict(config) -> Dict[str, Any]:
    """Serialise config to a dict, stripping sensitive fields."""
    _SENSITIVE = {"openai_api_key", "anthropic_api_key", "serper_api_key"}

    def _model_cfg(m) -> Dict[str, Any]:
        if m is None:
            return {}
        d = {
            "name": m.name,
            "family": m.family.value if hasattr(m.family, "value") else str(m.family),
            "path_or_id": m.path_or_id,
            "role": m.role,
            "backend": getattr(m, "backend", "vllm"),
            "max_model_len": m.max_model_len,
            "max_tokens": m.max_tokens,
            "temperature": m.temperature,
            "top_p": m.top_p,
            "top_k": m.top_k,
            "repetition_penalty": m.repetition_penalty,
            "supports_thinking": m.supports_thinking,
        }
        return d

    d: Dict[str, Any] = {
        "name": config.name,
        "description": getattr(config, "description", "") or "",
        "seed": config.seed,
        "use_wandb": getattr(config, "use_wandb", False),
        "wandb_project": getattr(config, "wandb_project", None),
        "thinking_mode": getattr(config.thinking_mode, "value", str(config.thinking_mode)),
        "max_turns": getattr(config, "max_turns", None),
        "batch_size": getattr(config, "batch_size", -1),
        "dataset": {
            "name": config.dataset.name,
            "split": config.dataset.split,
            "subset_num": getattr(config.dataset, "subset_num", None),
        },
        "tools": {
            "enabled_tools": list(config.tools.enabled_tools),
            "direct_tool_call": config.tools.direct_tool_call,
            "max_search_limit": getattr(config.tools, "max_search_limit", None),
            "top_k_results": getattr(config.tools, "top_k_results", None),
        },
        "models": {},
    }
    for role in ("orchestrator", "web_search", "code_generator", "text_inspector", "mind_map", "image_inspector"):
        if config.has_model(role):
            d["models"][role] = _model_cfg(config.get_model(role))

    # Drop anything sensitive just in case
    for key in _SENSITIVE:
        d.pop(key, None)

    return d


def _level_key(example, dataset_name: str) -> str:
    if dataset_name == "gaia":
        return str(example.metadata.get("level", "unknown"))
    if dataset_name == "hle":
        # HLE's "category" field is treated as its level.
        return str(example.metadata.get("category", "unknown"))
    if dataset_name in ("math500", "amc"):
        # Math500's "difficulty" field is treated as its level.
        return str(example.metadata.get("difficulty", example.metadata.get("year", "unknown")))
    if dataset_name == "aime":
        # AIME's "year" field is treated as its level.
        return str(example.metadata.get("year", "unknown"))
    return "all"


def _compute_metrics(
    results: List[Dict[str, Any]],
    examples,
    dataset_name: str,
) -> Dict[str, Any]:
    """Aggregate per-example results into overall + per-level metrics.

    Structure:
        overall:    accuracy, em, f1, num_correct, token_usage
        tool_usage: per-tool total counts (overall)
        per_level:  (for stratified datasets) same scores + tool_usage + token_usage per level
    """
    _STRATIFIED = {"gaia", "math500", "aime", "amc", "hle"}

    per_level_rows: Dict[str, List[Dict]] = {}

    all_gaia: List[float] = []
    all_em: List[float] = []
    all_f1: List[float] = []
    all_tools: Dict[str, int] = {}
    all_token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for idx, example in enumerate(examples):
        r = results[idx] if idx < len(results) else {}
        evaluation = r.get("evaluation") or {}
        tc = r.get("tool_counts") or {}
        tu = r.get("token_usage") or {}

        gs = float(evaluation.get("accuracy", 0.0))
        em = float(evaluation.get("em", float(evaluation.get("correct", gs > 0))))
        f1 = float(evaluation.get("f1", gs))

        all_gaia.append(gs)
        all_em.append(em)
        all_f1.append(f1)
        for tool, count in tc.items():
            all_tools[tool] = all_tools.get(tool, 0) + int(count or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            all_token_usage[key] = all_token_usage.get(key, 0) + int(tu.get(key, 0))

        if dataset_name in _STRATIFIED:
            lk = _level_key(example, dataset_name)
            per_level_rows.setdefault(lk, []).append({"gs": gs, "em": em, "f1": f1, "tc": tc, "tu": tu})

    total = len(examples)

    def _agg(rows: List[Dict]) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {"accuracy": 0.0, "em": 0.0, "f1": 0.0, "num_correct": "0 of 0"}
        n_correct = sum(1 for r in rows if r["gs"] > 0)
        level_tools: Dict[str, int] = {}
        level_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for row in rows:
            for tool, count in row["tc"].items():
                level_tools[tool] = level_tools.get(tool, 0) + int(count or 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                level_tokens[key] = level_tokens.get(key, 0) + int((row.get("tu") or {}).get(key, 0))
        result: Dict[str, Any] = {
            "accuracy": sum(r["gs"] for r in rows) / n,
            "em":         sum(r["em"] for r in rows) / n,
            "f1":         sum(r["f1"] for r in rows) / n,
            "num_correct": f"{n_correct} of {n}",
        }
        if level_tools:
            result["tool_usage"] = level_tools
        if any(level_tokens.values()):
            result["token_usage"] = level_tokens
        return result

    n_correct_overall = sum(1 for g in all_gaia if g > 0)
    overall: Dict[str, Any] = {
        "accuracy": sum(all_gaia) / total if total else 0.0,
        "em":         sum(all_em) / total if total else 0.0,
        "f1":         sum(all_f1) / total if total else 0.0,
        "num_correct": f"{n_correct_overall} of {total}",
    }
    if any(all_token_usage.values()):
        overall["token_usage"] = all_token_usage

    metrics: Dict[str, Any] = {"overall": overall}
    if all_tools:
        metrics["tool_usage"] = all_tools
    if dataset_name in _STRATIFIED and per_level_rows:
        metrics["per_level"] = {k: _agg(v) for k, v in sorted(per_level_rows.items())}

    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run agent_engine experiment")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment config YAML file")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory (overrides config)")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
