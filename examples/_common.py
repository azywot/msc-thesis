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


def build_model_providers(config, model_cache: Optional[Dict[str, Any]] = None):
    """Initialise all model roles defined in *config*.

    Returns:
        (planner, providers_by_role, model_cache)
        where *providers_by_role* maps role name → provider for tool sub-agents.
    """
    if model_cache is None:
        model_cache = {}

    planner = _get_model_provider(config.get_model("planner"), model_cache)

    providers: Dict[str, Any] = {}
    for role in ("search", "coding", "text_inspector", "image_inspector"):
        if config.has_model(role):
            providers[role] = _get_model_provider(config.get_model(role), model_cache)

    return planner, providers, model_cache


# ---------------------------------------------------------------------------
# Tool registry builder
# ---------------------------------------------------------------------------

def build_tools(
    config,
    cache_manager: CacheManager,
    model_providers: Dict[str, Any],
    enabled_tools: List[str],
) -> ToolRegistry:
    """Register exactly the tools listed in *enabled_tools* (sub-agent mode).

    Args:
        config:           Loaded ExperimentConfig (used for limits, paths …).
        cache_manager:    Cache manager (supplies search_cache / url_cache).
        model_providers:  Role → provider dict from build_model_providers().
        enabled_tools:    Subset of tool names to register for this example.
    """
    tools = ToolRegistry()
    use_thinking = config.use_subagent_thinking()

    for name in enabled_tools:
        if name == "web_search":
            serper_key = os.getenv("SERPER_API_KEY")
            if not serper_key:
                raise RuntimeError("SERPER_API_KEY env var is required for web_search")
            tools.register(WebSearchTool(
                serper_api_key=serper_key,
                search_cache=cache_manager.search_cache,
                url_cache=cache_manager.url_cache,
                top_k=config.tools.top_k_results,
                max_doc_len=config.tools.max_doc_len,
                model_provider=model_providers.get("search"),
                fetch_urls=True,
                use_thinking=use_thinking,
            ))

        elif name == "code_generator":
            tools.register(CodeGeneratorTool(
                timeout_seconds=60,
                temp_dir=str(config.cache_dir / "code_temp"),
                model_provider=model_providers.get("coding"),
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
            tools.register(MindMapTool(
                direct_mode=False,  # sub-agent mode
                storage_path=str(config.cache_dir / "mind_map"),
                use_graphrag=True,
            ))

    return tools


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def build_orchestrator(config, planner, tools: ToolRegistry) -> AgenticOrchestrator:
    return AgenticOrchestrator(
        model_provider=planner,
        tool_registry=tools,
        max_turns=config.max_turns,
        tool_limits={"web_search": config.tools.max_search_limit},
        use_thinking=config.use_planner_thinking(),
    )


def build_system_prompt(config, tools: ToolRegistry, attachments=None) -> str:
    return PromptBuilder().build_system_prompt(
        dataset_name="gaia",
        tool_schemas=tools.get_all_schemas(),
        attachments=attachments,
        max_search_limit=config.tools.max_search_limit,
        direct_tool_call=False,
    )


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(output_dir: Path, state, config) -> Path:
    """Persist state + pretty result to *output_dir*."""
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
    return path


def print_summary(logger, state, config, result_path):
    logger.info("=" * 60)
    logger.info(f"Answer  : {state.answer}")
    logger.info(f"Turns   : {state.turn}/{config.max_turns}")
    logger.info(f"Tools   : {state.tool_counts}")
    logger.info(f"Saved   : {result_path}")
    logger.info("=" * 60)
