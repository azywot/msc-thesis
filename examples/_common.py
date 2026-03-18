"""Shared setup helpers for the per-tool sub-agent mode examples.

Each example imports from here to avoid duplicating the boilerplate that
mirrors scripts/run_experiment.py (model caching, tool wiring, prompt building).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path so examples can be run from anywhere inside msc-thesis/.
import sys
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_engine.caching import CacheManager
from agent_engine.config import load_experiment_config
from agent_engine.core import AgenticOrchestrator, ToolRegistry
from agent_engine.models.base import ModelFamily
from agent_engine.prompts import PromptBuilder
from agent_engine.tools import (
    CodeGeneratorTool,
    ImageInspectorTool,
    MindMapTool,
    TextInspectorTool,
    WebSearchTool,
)
from agent_engine.utils import set_seed, setup_logging
from agent_engine.models.vllm_provider import resolve_gpu_assignments

DEFAULT_CONFIG = (
    Path(__file__).parent.parent / "experiments/configs/gaia/test_subagent.yaml"
)


# ---------------------------------------------------------------------------
# Model initialisation (with instance caching, same logic as run_experiment.py)
# ---------------------------------------------------------------------------

def _get_model_provider(model_config, model_cache: Dict[str, Any]):
    """Return (possibly cached) model provider for *model_config*."""
    cache_key = model_config.path_or_id

    if model_config.family not in (ModelFamily.GPT4, ModelFamily.CLAUDE):
        if cache_key in model_cache:
            return model_cache[cache_key]

    if model_config.family == ModelFamily.GPT4:
        from agent_engine.models.api_provider import OpenAIProvider
        return OpenAIProvider(model_config, api_key=os.getenv("OPENAI_API_KEY"))
    elif model_config.family == ModelFamily.CLAUDE:
        from agent_engine.models.api_provider import AnthropicProvider
        return AnthropicProvider(model_config, api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        from agent_engine.models.vllm_provider import VLLMProvider
        provider = VLLMProvider(model_config)
        model_cache[cache_key] = provider
        return provider


def build_model_providers(
    config,
    model_cache: Optional[Dict[str, Any]] = None,
    required_roles: Optional[List[str]] = None,
):
    """Initialise orchestrator and only the model roles needed for the example.

    When *required_roles* is set (the list of enabled tool names),
    only the orchestrator and those roles are loaded. This avoids loading the VLM
    (image_inspector) for examples that do not use it, preventing OOM on a
    single GPU.

    When the only required role is image_inspector, the VLM is used as the
    orchestrator as well (single model load) so the example fits on one GPU.

    Returns:
        (orchestrator, providers_by_role, model_cache)
        where *providers_by_role* maps role name → provider for tool sub-agents.
    """
    if model_cache is None:
        model_cache = {}

    roles_to_init = (
        required_roles
        if required_roles is not None
        else ["web_search", "code_generator", "text_inspector", "image_inspector"]  # model roles = tool names
    )

    # Auto-resolve gpu_memory_utilization and GPU pinning before loading any model.
    try:
        gpu_assignments = resolve_gpu_assignments(config)

        def _apply(cfg) -> None:
            if cfg is None:
                return
            path = cfg.path_or_id
            if path in gpu_assignments:
                util, gpu_ids = gpu_assignments[path]
                if cfg.gpu_memory_utilization is None:
                    cfg.gpu_memory_utilization = util
                if cfg.gpu_ids is None and gpu_ids is not None:
                    cfg.gpu_ids = gpu_ids

        _apply(config.get_model("orchestrator"))
        for _role in roles_to_init:
            _apply(config.get_model(_role))
    except Exception:
        pass  # fall back to per-model defaults

    orchestrator = _get_model_provider(config.get_model("orchestrator"), model_cache)
    providers: Dict[str, Any] = {}
    for role in roles_to_init:
        model_cfg = config.get_model(role)
        providers[role] = _get_model_provider(model_cfg, model_cache) if model_cfg is not None else orchestrator

    return orchestrator, providers, model_cache


# ---------------------------------------------------------------------------
# Tool registry builder
# ---------------------------------------------------------------------------

def build_tools(
    config,
    cache_manager: CacheManager,
    model_providers: Dict[str, Any],
    enabled_tools: List[str],
    orchestrator_model: Optional[Any] = None,
) -> ToolRegistry:
    """Register exactly the tools listed in *enabled_tools* (sub-agent mode).

    Args:
        config:           Loaded ExperimentConfig (used for limits, paths …).
        cache_manager:    Cache manager (supplies search_cache / url_cache).
        model_providers:  Role → provider dict from build_model_providers().
        enabled_tools:    Subset of tool names to register for this example.
        orchestrator_model: Optional orchestrator provider for mind_map (GraphRAG needs local model).
    """
    tools = ToolRegistry()
    use_thinking = config.use_subagent_thinking()

    for name in enabled_tools:
        if name == "web_search":
            provider = config.tools.web_tool_provider
            # Get the appropriate API key based on provider
            api_key = os.getenv(f"{provider.upper()}_API_KEY")
            if not api_key:
                raise RuntimeError(f"{provider.upper()}_API_KEY env var is required when web_tool_provider='{provider}'")

            tools.register(WebSearchTool(
                api_key=api_key,
                provider=provider,
                search_cache=cache_manager.search_cache,
                url_cache=cache_manager.url_cache,
                top_k=config.tools.top_k_results,
                max_doc_len=config.tools.max_doc_len,
                model_provider=model_providers.get("web_search"),
                fetch_urls=True,
                use_thinking=use_thinking,
                cache_manager=cache_manager,
            ))

        elif name == "code_generator":
            tools.register(CodeGeneratorTool(
                timeout_seconds=60,
                temp_dir=str(config.cache_dir / "code_temp"),
                model_provider=model_providers.get("code_generator"),
                use_thinking=use_thinking,
            ))

        elif name == "text_inspector":
            tools.register(TextInspectorTool(
                max_chars=50_000,
                model_provider=model_providers.get("text_inspector"),
                use_thinking=use_thinking,
            ))

        elif name == "image_inspector":
            vlm = model_providers.get("image_inspector")
            tools.register(ImageInspectorTool(
                model_provider=vlm,
                use_thinking=use_thinking,
            ))

        elif name == "mind_map":
            mind_map_model = model_providers.get("mind_map") or orchestrator_model
            tools.register(MindMapTool(
                direct_mode=False,  # sub-agent mode
                storage_path=str(config.cache_dir / "mind_map"),
                use_graphrag=True,
                model_provider=mind_map_model,  # local model for GraphRAG — no OpenAI key
            ))

    return tools


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def build_orchestrator(config, orchestrator_model, tools: ToolRegistry, cache_manager=None) -> AgenticOrchestrator:
    return AgenticOrchestrator(
        model_provider=orchestrator_model,
        tool_registry=tools,
        max_turns=config.max_turns,
        tool_limits={"web_search": config.tools.max_search_limit},
        use_thinking=config.use_orchestrator_thinking(),
        cache_manager=cache_manager,
    )


def build_system_prompt(config, tools: ToolRegistry) -> str:
    return PromptBuilder().build_system_prompt(
        dataset_name="gaia",
        tool_schemas=tools.get_all_schemas(),
        direct_tool_call=False,
    )


# ---------------------------------------------------------------------------
# Result persistence and execution trace
# ---------------------------------------------------------------------------

# Max characters to log per message in execution trace (avoid huge dumps).
_TRACE_MSG_MAX_LEN = 10000


def save_result(output_dir: Path, state, config) -> Path:
    """Persist state + pretty result to *output_dir*; also write full trace."""
    out = {
        "question": state.question,
        "answer": state.answer,
        "turns": state.turn,
        "max_turns": config.max_turns,
        "tool_counts": state.tool_counts,
        "finished": state.finished,
    }
    path = output_dir / "result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Full execution trace (messages + tool_calls) for debugging.
    trace_path = output_dir / "trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        state_dict = state.model_dump() if hasattr(state, "model_dump") else state.dict()
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    return path


def log_execution_trace(logger, state):
    """Log a readable execution trace: every message and every tool call with arguments."""
    logger.info("")
    logger.info("--- Execution trace (messages) ---")
    for i, msg in enumerate(state.messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if len(content) > _TRACE_MSG_MAX_LEN:
            content = content[: _TRACE_MSG_MAX_LEN] + "\n... [truncated]"
        logger.info("[%d] %s: %s", i + 1, role, content)
    logger.info("--- Tool calls ---")
    for j, tc in enumerate(state.tool_calls):
        name = tc.get("name", "?")
        args = tc.get("arguments", {}) or {}
        args_str = json.dumps(args, indent=2, ensure_ascii=False)
        if len(args_str) > 2000:
            args_str = args_str[:2000] + "\n... [truncated]"
        logger.info("Tool call #%d: %s\nArguments:\n%s", j + 1, name, args_str)
    logger.info("--- End trace ---")
    logger.info("")


def print_summary(logger, state, config, result_path):
    log_execution_trace(logger, state)
    logger.info("=" * 60)
    logger.info(f"Answer  : {state.answer}")
    logger.info(f"Turns   : {state.turn}/{config.max_turns}")
    logger.info(f"Tools   : {state.tool_counts}")
    logger.info(f"Saved   : {result_path}")
    logger.info(f"Trace   : {result_path.parent / 'trace.json'}")
    logger.info("=" * 60)
