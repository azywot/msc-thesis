"""Collect SFT training trajectories using Qwen3-32B as teacher.

Reads data/training/train/combined_train.parquet, runs each question through
the full AgentFlow pipeline, evaluates correctness, and writes successful
trajectories to data/training/sft/ in VERL MultiTurnSFTDataset format.

Usage:
    python scripts/collect_sft_data.py \
        --config experiments/configs/fine_tuning/collect_sft_qwen32b.yaml \
        [--parquet data/training/train/combined_train.parquet] \
        [--output-dir data/training/sft] \
        [--n-samples 3] \       # rollout passes per question (default 1)
        [--subset 200]          # limit rows for smoke testing

--n-samples N runs up to N passes over the dataset.  Pass k+1 skips any
question already solved in an earlier pass, so extra compute is spent only
on the remaining hard questions.  All distinct correct trajectories (even
multiple per question from different passes) are kept in sft_train.parquet,
giving the model diverse valid reasoning paths to learn from.

Output files (in --output-dir):
    sft_train.parquet       correct trajectories only  (messages + data_source columns)
    collected_<ts>.json     all trajectories including failures (for analysis / yield stats)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agent_engine.caching import CacheManager
from agent_engine.config import load_experiment_config
from agent_engine.core import AgenticOrchestrator, ToolRegistry
from agent_engine.datasets.evaluators.metrics import evaluate_answer
from agent_engine.models.base import ModelFamily, get_tool_call_format
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools import CodeGeneratorTool, WebSearchTool
from agent_engine.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_source_to_prompt_dataset(data_source: str) -> str:
    """Map training data_source to the PromptBuilder dataset name.

    Mirrors fine_tuning/rollout.py:_prompt_dataset_for_data_source so that
    collected trajectories use the same system prompts seen during RL training.
    """
    ds = data_source.lower()
    if ds == "deepmath" or ds.startswith("aime"):
        return "deepmath"
    return "gaia"


def _build_sft_messages(state) -> List[Dict[str, str]]:
    """Combine initial system+user messages with the full conversation turns.

    state.messages holds [system, user] (AgentFlow rewrites it each turn but
    always keeps these two at indices 0 and 1).
    state.output_messages holds the accumulated assistant + tool turns for
    tool-call and final-answer steps — but NOT the planning turn.

    The planning turn (Turn 0) output is stored separately in
    state.raw_query_analysis (full text including <think> tags).
    We prepend it as an assistant message so the SFT data includes the full
    planning → tool-call → answer trajectory that the model must learn.
    """
    initial = list(state.messages[:2])   # [system, user]
    turns: List[Dict[str, str]] = []

    # Turn 0: planning (only present in AgentFlow mode, not baseline)
    if state.raw_query_analysis:
        turns.append({"role": "assistant", "content": state.raw_query_analysis})

    turns.extend(state.output_messages)
    return initial + turns


def _setup_model_provider(model_cfg, api_keys: Dict[str, str], model_cache: Dict):
    """Instantiate a model provider, reusing cached instances for the same path."""
    # Include lora_adapter_path in the key so a LoRA instance and the base model
    # don't collide when both appear in the same experiment. Keying on
    # path_or_id alone silently hands the first-loaded provider to both roles.
    # Matches agent_engine.runner.providers.setup_model_provider.
    cache_key = f"{model_cfg.path_or_id}|lora:{getattr(model_cfg, 'lora_adapter_path', None) or ''}"
    if cache_key in model_cache:
        logger.info("Reusing cached model: %s (role: %s)", model_cfg.name, model_cfg.role)
        return model_cache[cache_key]

    backend = getattr(model_cfg, "backend", "vllm")
    if backend in ("openai",) or model_cfg.family == ModelFamily.GPT4:
        from agent_engine.models.api_provider import OpenAIProvider
        provider = OpenAIProvider(model_cfg, api_key=api_keys.get("openai"))
    elif backend == "anthropic" or model_cfg.family == ModelFamily.CLAUDE:
        from agent_engine.models.api_provider import AnthropicProvider
        provider = AnthropicProvider(model_cfg, api_key=api_keys.get("anthropic"))
    else:
        from agent_engine.models.vllm_provider import VLLMProvider
        provider = VLLMProvider(model_cfg)
        model_cache[cache_key] = provider

    return provider


def _setup_tools(config, cache_manager, api_keys, model_providers) -> ToolRegistry:
    """Build a ToolRegistry for web_search and code_generator in sub-agent mode."""
    registry = ToolRegistry()
    use_subagent_thinking = config.use_subagent_thinking()

    for tool_name in config.tools.enabled_tools:
        if tool_name == "web_search":
            provider_name = config.tools.web_tool_provider
            api_key = api_keys.get(provider_name)
            if not api_key:
                raise RuntimeError(
                    f"{provider_name.upper()}_API_KEY is required "
                    f"(web_tool_provider={provider_name!r})"
                )
            search_model = model_providers.get("web_search")
            registry.register(WebSearchTool(
                api_key=api_key,
                provider=provider_name,
                search_cache=cache_manager.search_cache,
                url_cache=cache_manager.url_cache,
                top_k=config.tools.top_k_results,
                max_doc_len=config.tools.max_doc_len,
                model_provider=search_model,
                fetch_urls=True,
                use_thinking=use_subagent_thinking,
                cache_manager=cache_manager,
                max_search_content_chars=config.tools.max_search_content_chars,
            ))
        elif tool_name == "code_generator":
            coding_model = model_providers.get("code_generator")
            registry.register(CodeGeneratorTool(
                timeout_seconds=60,
                temp_dir=str(config.cache_dir / "code_temp"),
                model_provider=coding_model,
                use_thinking=use_subagent_thinking,
                return_code=config.tools.return_code,
            ))

    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect SFT trajectories from teacher model.")
    parser.add_argument(
        "--config",
        default="experiments/configs/fine_tuning/collect_sft_qwen32b.yaml",
        help="Experiment config YAML (model + tool settings).",
    )
    parser.add_argument(
        "--parquet",
        default="data/training/train/combined_train.parquet",
        help="Training parquet to run (question, result, data_source columns).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training/sft",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help=(
            "Number of rollout passes per question (default 1). "
            "Pass k+1 skips questions already solved in earlier passes, "
            "so extra compute is spent only on remaining hard questions."
        ),
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=-1,
        help="Process only the first N rows (-1 = all). Useful for smoke testing.",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("=" * 70)
    logger.info("SFT data collection — %s", datetime.now())
    logger.info("=" * 70)
    logger.info("Config:    %s", args.config)
    logger.info("Parquet:   %s", args.parquet)
    logger.info("Output:    %s", args.output_dir)
    logger.info("n-samples: %d", args.n_samples)

    config = load_experiment_config(Path(args.config))
    set_seed(config.seed)

    # Read optional system_prompt_suffix directly from raw YAML (not in schema).
    import yaml as _yaml
    with open(args.config) as _f:
        _raw = _yaml.safe_load(_f)
    system_prompt_suffix: Optional[str] = _raw.get("system_prompt_suffix") or None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load training questions ──────────────────────────────────────────────
    df = pd.read_parquet(args.parquet)
    if args.subset > 0:
        df = df.iloc[: args.subset]
        logger.info("Subset mode: processing %d rows", len(df))
    logger.info("Loaded %d questions from %s", len(df), args.parquet)

    # ── GPU assignment (reuses run_experiment logic) ─────────────────────────
    try:
        from agent_engine.models.vllm_provider import resolve_gpu_assignments

        gpu_assignments = resolve_gpu_assignments(config)
        for role, model_cfg in config.models.items():
            path = model_cfg.path_or_id
            if path in gpu_assignments:
                util, gpu_ids = gpu_assignments[path]
                if model_cfg.gpu_memory_utilization is None:
                    model_cfg.gpu_memory_utilization = util
                if model_cfg.gpu_ids is None and gpu_ids is not None:
                    model_cfg.gpu_ids = gpu_ids
    except Exception as exc:
        logger.warning("GPU assignment resolution skipped (%s); using per-model defaults.", exc)

    # ── Models ───────────────────────────────────────────────────────────────
    api_keys = {
        "serper":    os.getenv("SERPER_API_KEY"),
        "tavily":    os.getenv("TAVILY_API_KEY"),
        "openai":    os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }
    model_cache: Dict[str, Any] = {}

    logger.info("Loading orchestrator: %s", config.get_model("orchestrator").name)
    orchestrator_model = _setup_model_provider(config.get_model("orchestrator"), api_keys, model_cache)

    model_providers: Dict[str, Any] = {}
    for tool_name in config.tools.enabled_tools:
        tool_cfg = config.get_model(tool_name)
        if tool_cfg is not None:
            logger.info("Loading %s model: %s", tool_name, tool_cfg.name)
            model_providers[tool_name] = _setup_model_provider(tool_cfg, api_keys, model_cache)
        else:
            logger.info("No models.%s in config — falling back to orchestrator model", tool_name)
            model_providers[tool_name] = orchestrator_model

    # ── Cache + tools ────────────────────────────────────────────────────────
    cache_manager = CacheManager(
        config.cache_dir,
        web_tool_provider=config.tools.web_tool_provider,
        dataset_name="sft_collection",   # separate cache namespace from benchmark runs
    )
    tools = _setup_tools(config, cache_manager, api_keys, model_providers)

    # ── Prompt builder ───────────────────────────────────────────────────────
    prompt_builder = PromptBuilder()
    tool_call_format = get_tool_call_format(config.get_model("orchestrator").family)

    # Build one system prompt per unique data_source in the parquet.
    # system_prompt_suffix (from YAML) is appended to every prompt when set —
    # useful for nudging the teacher toward tool use (e.g. code_generator for math).
    unique_sources = df["data_source"].unique().tolist()
    system_prompts: Dict[str, str] = {}
    for src in unique_sources:
        prompt_ds = _data_source_to_prompt_dataset(src)
        sp = prompt_builder.build_system_prompt(
            dataset_name=prompt_ds,
            tool_schemas=tools.get_all_schemas(),
            direct_tool_call=False,
            tool_call_format=tool_call_format,
        )
        if system_prompt_suffix:
            sp += system_prompt_suffix
        system_prompts[src] = sp
        logger.info("System prompt for data_source=%r → dataset_name=%r (%d chars)",
                    src, prompt_ds, len(system_prompts[src]))

    # ── Orchestrator ─────────────────────────────────────────────────────────
    orchestrator = AgenticOrchestrator(
        model_provider=orchestrator_model,
        tool_registry=tools,
        max_turns=config.max_turns,
        tool_limits={"web_search": config.tools.max_search_limit},
        use_thinking=config.use_orchestrator_thinking(),
        cache_manager=cache_manager,
        baseline=False,
    )

    # ── Run (multi-pass with skip-solved optimisation) ────────────────────────
    batch_size = config.batch_size if config.batch_size > 0 else len(df)
    correct_count = 0
    total_count = 0
    # question_ids already solved — skipped in subsequent passes to save compute.
    solved: set = set()
    # Lightweight per-source counters for the final summary (no full records in memory).
    by_source: Dict[str, Dict[str, int]] = {}

    # Attach a stable integer id to each row so we can track solved status
    # across passes independently of batch indexing.
    rows = [{"_qid": i, **r} for i, r in enumerate(df.to_dict("records"))]

    # Open the JSONL output file once and stream records as they arrive so that
    # progress is preserved even if the job is cancelled mid-run.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_path = output_dir / f"collected_{ts}.jsonl"
    jsonl_file = open(all_path, "w", encoding="utf-8")
    logger.info("Streaming results → %s", all_path)

    try:
        for pass_idx in range(args.n_samples):
            remaining = [r for r in rows if r["_qid"] not in solved]
            if not remaining:
                logger.info("All questions solved after pass %d — stopping early.", pass_idx)
                break

            logger.info(
                "=== Pass %d / %d  (%d questions remaining, %d already solved) ===",
                pass_idx + 1, args.n_samples, len(remaining), len(solved),
            )

            pass_correct = 0
            pass_total = 0

            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start : batch_start + batch_size]
                batch_end = batch_start + len(batch)
                logger.info(
                    "  Pass %d  batch %d–%d / %d",
                    pass_idx + 1, batch_start + 1, batch_end, len(remaining),
                )

                questions    = [r["question"]     for r in batch]
                question_ids = [r["_qid"]         for r in batch]
                sp_list      = [system_prompts[r["data_source"]] for r in batch]

                try:
                    states = orchestrator.run_batch(
                        questions=questions,
                        question_ids=question_ids,
                        system_prompts=sp_list,
                    )
                except Exception as exc:
                    logger.error(
                        "Pass %d  batch %d–%d failed: %s",
                        pass_idx + 1, batch_start + 1, batch_end, exc, exc_info=True,
                    )
                    for row in batch:
                        record = {
                            "question_id": row["_qid"],
                            "pass": pass_idx + 1,
                            "question": row["question"],
                            "data_source": row["data_source"],
                            "ground_truth": row["result"],
                            "prediction": None,
                            "correct": False,
                            "messages": [],
                            "error": str(exc),
                        }
                        jsonl_file.write(json.dumps(record, default=str) + "\n")
                        jsonl_file.flush()
                        total_count += 1
                        pass_total += 1
                        src = row["data_source"]
                        by_source.setdefault(src, {"questions": 0, "correct_trajectories": 0})
                        by_source[src]["questions"] += 1
                    continue

                for row, state in zip(batch, states):
                    prediction = state.answer or ""
                    eval_result = evaluate_answer(
                        prediction=prediction,
                        ground_truth=str(row["result"]),
                    )
                    correct = bool(eval_result.get("correct", False))
                    if correct:
                        correct_count += 1
                        pass_correct += 1
                        solved.add(row["_qid"])

                    messages = _build_sft_messages(state)
                    record = {
                        "question_id": row["_qid"],
                        "pass": pass_idx + 1,
                        "question": row["question"],
                        "data_source": row["data_source"],
                        "ground_truth": row["result"],
                        "prediction": prediction,
                        "correct": correct,
                        "messages": messages,
                        "turns": state.turn,
                        "tool_counts": state.tool_counts,
                    }
                    jsonl_file.write(json.dumps(record, default=str) + "\n")
                    jsonl_file.flush()
                    total_count += 1
                    pass_total += 1

                    src = row["data_source"]
                    by_source.setdefault(src, {"questions": 0, "correct_trajectories": 0})
                    by_source[src]["questions"] += 1
                    if correct:
                        by_source[src]["correct_trajectories"] += 1

                    logger.info(
                        "  [pass %d  qid %d]  data_source=%s  correct=%s  answer=%r",
                        pass_idx + 1, row["_qid"],
                        row["data_source"], correct, prediction[:80] if prediction else "",
                    )

                # Flush search / url cache after every batch
                cache_manager.save_caches()

            logger.info(
                "Pass %d complete — %d / %d correct this pass  (%d total solved)",
                pass_idx + 1, pass_correct, pass_total, len(solved),
            )

    finally:
        jsonl_file.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    cache_manager.save_caches()

    n_questions = len(rows)
    yield_pct = 100.0 * len(solved) / n_questions if n_questions else 0.0

    logger.info("=" * 70)
    logger.info(
        "Collection complete — %d / %d questions solved (%.1f%%)  |  "
        "%d correct trajectories total  |  %d passes",
        len(solved), n_questions, yield_pct, correct_count, args.n_samples,
    )
    for src, counts in sorted(by_source.items()):
        pct = 100.0 * counts["correct_trajectories"] / counts["questions"] if counts["questions"] else 0
        logger.info(
            "  %-20s  %d correct trajectories / %d attempts  (%.1f%%)",
            src, counts["correct_trajectories"], counts["questions"], pct,
        )
    logger.info("=" * 70)
    logger.info("JSONL output: %s  (%d records)", all_path, total_count)
    logger.info(
        "To build sft_train.parquet from this file run:\n"
        "  python scripts/build_sft_parquet.py %s --output-dir %s",
        all_path, output_dir,
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
