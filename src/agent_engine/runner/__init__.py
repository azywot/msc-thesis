"""Experiment runner: the wiring that turns a config into a completed run.

``scripts/`` holds thin CLI wrappers; the logic they call lives here so it can
be imported, tested, and reused.
"""

from agent_engine.runner.metrics import compute_metrics, level_key
from agent_engine.runner.providers import setup_model_provider

__all__ = ["compute_metrics", "level_key", "setup_model_provider"]
