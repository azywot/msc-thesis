"""Weights & Biases logging utilities.

This is intentionally lightweight:
- Import `wandb` only when needed (optional dependency).
- Avoid logging secrets (API keys/tokens).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def log_results_wandb(
    *,
    project: str,
    run_name: str,
    dataset_name: str,
    dataset_split: str,
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
) -> None:
    """Log a single W&B row with final metrics + tool usage."""
    if not final_metrics:
        return

    try:
        import wandb  # type: ignore
    except Exception:
        return

    created_run = False
    if wandb.run is None:
        safe_config: Dict[str, Any] = {}
        if isinstance(config_summary, dict):
            for k, v in config_summary.items():
                k_l = str(k).lower()
                if any(s in k_l for s in ("key", "token", "secret", "password")):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    safe_config[k] = v

        wandb.init(project=project, name=run_name, config=safe_config)
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

    # Tool totals
    search_total = 0
    code_total = 0
    context_manager_total = 0
    text_inspector_total = 0
    image_inspector_total = 0
    try:
        if isinstance(tool_stats, dict) and isinstance(tool_stats.get("overall"), dict):
            overall_tools = tool_stats["overall"]
            search_total = int(overall_tools.get("search_total", 0) or 0)
            code_total = int(overall_tools.get("code_total", 0) or 0)
            context_manager_total = int(overall_tools.get("context_manager_total", 0) or 0)
            text_inspector_total = int(overall_tools.get("text_inspector_total", 0) or 0)
            image_inspector_total = int(overall_tools.get("image_inspector_total", 0) or 0)
    except Exception:
        pass

    total_tool_calls = search_total + code_total + context_manager_total + text_inspector_total + image_inspector_total

    log_data: Dict[str, Any] = {
        "dataset": f"{dataset_name}_{dataset_split}",
        "dataset_split": str(dataset_split),
        "model_name": str(model_name),
        "mode": str(mode),
        "thinking_mode": str(thinking_mode),
        "direct_tool_call": bool(direct_tool_call),
        "enable_search_tool": bool(enable_search_tool),
        "enable_code_tool": bool(enable_code_tool),
        "context_manager": bool(context_manager),
        "enable_text_inspector_tool": bool(enable_text_inspector_tool),
        "enable_image_inspector_tool": bool(enable_image_inspector_tool),
        "accuracy": None if accuracy is None else float(accuracy),
        "L1_accuracy": None if l1_accuracy is None else float(l1_accuracy),
        "L2_accuracy": None if l2_accuracy is None else float(l2_accuracy),
        "L3_accuracy": None if l3_accuracy is None else float(l3_accuracy),
        "em": None if em is None else float(em),
        "f1": None if f1 is None else float(f1),
        "L1_em": None if l1_em is None else float(l1_em),
        "L2_em": None if l2_em is None else float(l2_em),
        "L3_em": None if l3_em is None else float(l3_em),
        "tool/search_total": search_total,
        "tool/code_total": code_total,
        "tool/context_manager_total": context_manager_total,
        "tool/text_inspector_total": text_inspector_total,
        "tool/image_inspector_total": image_inspector_total,
        "tool/total_tool_calls": total_tool_calls,
        "metrics_path": None if metrics_path is None else str(metrics_path),
    }

    wandb.log(log_data, step=0)

    if created_run:
        # HPC jobs can exit quickly; give the backend time to flush.
        time.sleep(2)
        try:
            wandb.finish()
        except Exception:
            pass

