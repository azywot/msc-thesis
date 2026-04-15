"""Prompt utility helpers for model-specific instruction tweaks."""

from typing import Any

from ..models.base import ModelFamily


def should_append_step_by_step_instruction(model_or_provider: Any, use_thinking: bool) -> bool:
    """Return True when a DeepSeek thinking call should be explicitly primed.

    Accepts either a model provider (with ``.config.family``) or a model config
    (with ``.family``), so callers can share one predicate.
    """
    if not model_or_provider or not use_thinking:
        return False
    family = getattr(model_or_provider, "family", None)
    if family is None:
        family = getattr(getattr(model_or_provider, "config", None), "family", None)
    return family == ModelFamily.DEEPSEEK


def append_step_by_step_instruction(text: str) -> str:
    """Append ``Think step by step.`` to the end of text once."""
    stripped = text.rstrip()
    if stripped.endswith("Think step by step."):
        return stripped
    return f"{stripped}\n\nThink step by step."
