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
    mind_map: bool,
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

    acc = overall.get("acc")
    gaia_score = overall.get("gaia_score")
    em = overall.get("em")

    def _lvl(metric_key: str, level: str):
        if isinstance(per_level, dict):
            return (per_level.get(level, {}) or {}).get(metric_key)
        return None

    l1_acc = _lvl("acc", "1")
    l2_acc = _lvl("acc", "2")
    l3_acc = _lvl("acc", "3")

    l1_gaia_score = _lvl("gaia_score", "1")
    l2_gaia_score = _lvl("gaia_score", "2")
    l3_gaia_score = _lvl("gaia_score", "3")

    l1_em = _lvl("em", "1")
    l2_em = _lvl("em", "2")
    l3_em = _lvl("em", "3")

    # Tool totals
    search_total = 0
    code_total = 0
    mind_map_total = 0
    text_inspector_total = 0
    image_inspector_total = 0
    try:
        if isinstance(tool_stats, dict) and isinstance(tool_stats.get("overall"), dict):
            overall_tools = tool_stats["overall"]
            search_total = int(overall_tools.get("search_total", 0) or 0)
            code_total = int(overall_tools.get("code_total", 0) or 0)
            mind_map_total = int(overall_tools.get("mind_map_total", 0) or 0)
            text_inspector_total = int(overall_tools.get("text_inspector_total", 0) or 0)
            image_inspector_total = int(overall_tools.get("image_inspector_total", 0) or 0)
    except Exception:
        pass

    total_tool_calls = search_total + code_total + mind_map_total + text_inspector_total + image_inspector_total

    log_data: Dict[str, Any] = {
        "dataset": f"{dataset_name}_{dataset_split}",
        "dataset_split": str(dataset_split),
        "model_name": str(model_name),
        "mode": str(mode),
        "thinking_mode": str(thinking_mode),
        "direct_tool_call": bool(direct_tool_call),
        "enable_search_tool": bool(enable_search_tool),
        "enable_code_tool": bool(enable_code_tool),
        "mind_map": bool(mind_map),
        "enable_text_inspector_tool": bool(enable_text_inspector_tool),
        "enable_image_inspector_tool": bool(enable_image_inspector_tool),
        "acc": None if acc is None else float(acc),
        "L1_acc": None if l1_acc is None else float(l1_acc),
        "L2_acc": None if l2_acc is None else float(l2_acc),
        "L3_acc": None if l3_acc is None else float(l3_acc),
        "gaia_score": None if gaia_score is None else float(gaia_score),
        "L1_gaia_score": None if l1_gaia_score is None else float(l1_gaia_score),
        "L2_gaia_score": None if l2_gaia_score is None else float(l2_gaia_score),
        "L3_gaia_score": None if l3_gaia_score is None else float(l3_gaia_score),
        "em": None if em is None else float(em),
        "L1_em": None if l1_em is None else float(l1_em),
        "L2_em": None if l2_em is None else float(l2_em),
        "L3_em": None if l3_em is None else float(l3_em),
        "tool/search_total": search_total,
        "tool/code_total": code_total,
        "tool/mind_map_total": mind_map_total,
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

