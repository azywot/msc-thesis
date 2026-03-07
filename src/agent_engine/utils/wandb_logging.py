"""Weights & Biases logging utilities.

This is intentionally lightweight:
- Import `wandb` only when needed (optional dependency).
- Avoid logging secrets (API keys/tokens).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional


def log_results_wandb(
    *,
    project: str,
    run_name: str,
    dataset_name: str,
    dataset_split: str,
    subset_num: Optional[int],
    model_name: str,
    mode: str,
    thinking_mode: str,
    direct_tool_call: bool,
    enable_search_tool: bool,
    enable_code_tool: bool,
    context_manager: bool,
    enable_text_inspector_tool: bool,
    enable_image_inspector_tool: bool,
    final_metrics: Optional[Dict[str, Any]],
    tool_stats: Optional[Dict[str, Any]],
    metrics_path: Optional[str] = None,
    config_summary: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
) -> None:
    """Log experiment results as a single row to a W&B table.

    Supports two ``tool_stats`` shapes emitted by different code paths:

    * **Nested** — ``tool_stats["overall"]["search_total"]`` (from
      ``run_experiment.py`` when saving per-level stats).
    * **Flat** — ``tool_stats["web_search"]`` (from the ``metrics["tool_usage"]``
      dict written directly to ``metrics.json``).

    Args:
        project: W&B project name.
        run_name: Display name for this run.
        dataset_name: Name of the evaluated dataset (e.g. ``"gaia"``).
        dataset_split: Dataset split used (e.g. ``"validation"``).
        subset_num: Number of examples evaluated (``None`` = full split).
        model_name: Orchestrator model identifier.
        mode: Experiment mode string (e.g. ``"subagent"``).
        thinking_mode: :class:`ThinkingMode` value as a string.
        direct_tool_call: Whether tools were called directly (no sub-agent LLM).
        enable_search_tool: Whether ``web_search`` was enabled.
        enable_code_tool: Whether ``code_generator`` was enabled.
        context_manager: Whether ``context_manager`` was enabled.
        enable_text_inspector_tool: Whether ``text_inspector`` was enabled.
        enable_image_inspector_tool: Whether ``image_inspector`` was enabled.
        final_metrics: Metrics dict with ``"overall"`` and ``"per_level"`` keys.
        tool_stats: Tool usage counts (see two-format note above).
        metrics_path: Local path to ``metrics.json`` (for display only).
        config_summary: Subset of :class:`ExperimentConfig` fields to store in
                        the W&B run config.
        config_path: Path to ``config.json`` to upload as a file artifact.
    """
    if not final_metrics:
        return

    try:
        import wandb  # type: ignore
    except Exception:
        return

    created_run = False
    if wandb.run is None:
        minimal_config: Dict[str, Any] = {}
        if isinstance(config_summary, dict):
            for key in ("seed", "max_turns", "batch_size"):
                v = config_summary.get(key)
                if v is not None and (not isinstance(v, str) or v.strip()):
                    minimal_config[key] = v

        wandb.init(project=project, name=run_name, config=minimal_config)
        created_run = True

    # Metrics (overall + per-level if present)
    overall = final_metrics.get("overall", {}) if isinstance(final_metrics, dict) else {}
    per_level = final_metrics.get("per_level", {}) if isinstance(final_metrics, dict) else {}

    accuracy = overall.get("accuracy")
    em = overall.get("em")
    f1 = overall.get("f1")

    def _lvl(metric_key: str, level: str):
        if isinstance(per_level, dict):
            return (per_level.get(level, {}) or {}).get(metric_key)
        return None

    l1_accuracy = _lvl("accuracy", "1")
    l2_accuracy = _lvl("accuracy", "2")
    l3_accuracy = _lvl("accuracy", "3")

    l1_em = _lvl("em", "1")
    l2_em = _lvl("em", "2")
    l3_em = _lvl("em", "3")

    # Optional: experiment description (for table readability)
    description: Optional[str] = None
    if isinstance(config_summary, dict):
        desc = config_summary.get("description")
        if isinstance(desc, str) and desc.strip():
            description = desc.strip()

    # Tool totals — support both formats:
    # (1) tool_stats["overall"] with search_total, code_total, etc.
    # (2) flat tool_stats with web_search, code_generator, etc. (from metrics["tool_usage"])
    _TOOL_ALIASES = {
        "search_total": ["search_total", "web_search"],
        "code_total": ["code_total", "code_generator"],
        "context_manager_total": ["context_manager_total", "context_manager"],
        "text_inspector_total": ["text_inspector_total", "text_inspector"],
        "image_inspector_total": ["image_inspector_total", "image_inspector"],
    }
    search_total = 0
    code_total = 0
    context_manager_total = 0
    text_inspector_total = 0
    image_inspector_total = 0
    try:
        overall_tools: Dict[str, Any] = {}
        if isinstance(tool_stats, dict) and isinstance(tool_stats.get("overall"), dict):
            overall_tools = tool_stats["overall"]
        elif isinstance(tool_stats, dict):
            overall_tools = tool_stats

        def _get(key: str, aliases: list) -> int:
            for alias in aliases:
                v = overall_tools.get(alias)
                if v is not None:
                    return int(v or 0)
            return 0

        if overall_tools:
            search_total = _get("search_total", _TOOL_ALIASES["search_total"])
            code_total = _get("code_total", _TOOL_ALIASES["code_total"])
            context_manager_total = _get("context_manager_total", _TOOL_ALIASES["context_manager_total"])
            text_inspector_total = _get("text_inspector_total", _TOOL_ALIASES["text_inspector_total"])
            image_inspector_total = _get("image_inspector_total", _TOOL_ALIASES["image_inspector_total"])
    except Exception:
        pass

    total_tool_calls = search_total + code_total + context_manager_total + text_inspector_total + image_inspector_total

    def _present(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
        return True

    log_data: Dict[str, Any] = {
        "dataset": str(dataset_name),
        "dataset_split": str(dataset_split),
        "subset_num": subset_num,
        "model_name": str(model_name),
        "mode": str(mode),
        "thinking_mode": str(thinking_mode),
        "direct_tool_call": bool(direct_tool_call),
        "enable_search_tool": bool(enable_search_tool),
        "enable_code_tool": bool(enable_code_tool),
        "context_manager": bool(context_manager),
        "enable_text_inspector_tool": bool(enable_text_inspector_tool),
        "enable_image_inspector_tool": bool(enable_image_inspector_tool),
        "description": description,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "L1_accuracy": float(l1_accuracy) if l1_accuracy is not None else None,
        "L2_accuracy": float(l2_accuracy) if l2_accuracy is not None else None,
        "L3_accuracy": float(l3_accuracy) if l3_accuracy is not None else None,
        "em": float(em) if em is not None else None,
        "f1": float(f1) if f1 is not None else None,
        "L1_em": float(l1_em) if l1_em is not None else None,
        "L2_em": float(l2_em) if l2_em is not None else None,
        "L3_em": float(l3_em) if l3_em is not None else None,
        "tool/search_total": search_total,
        "tool/code_total": code_total,
        "tool/context_manager_total": context_manager_total,
        "tool/text_inspector_total": text_inspector_total,
        "tool/image_inspector_total": image_inspector_total,
        "tool/total_tool_calls": total_tool_calls,
        "metrics_path": str(metrics_path) if metrics_path else None,
    }

    # Only log keys with non-empty values to avoid empty/duplicate columns in the table
    log_data = {k: v for k, v in log_data.items() if _present(v)}

    wandb.log(log_data, step=0)

    # Store the full config.json as a file artifact in W&B.
    if config_path and os.path.exists(config_path):
        try:
            wandb.save(config_path)
        except Exception:
            print(f"Failed to save config.json as a file artifact in W&B: {config_path}")
            pass

    if created_run:
        # HPC jobs can exit quickly; give the backend time to flush.
        time.sleep(2)
        try:
            wandb.finish()
        except Exception:
            pass

