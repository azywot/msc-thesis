"""Agent Engine - Clean Agentic Reasoning System.

A clean, extensible implementation of agentic reasoning with tool calling
for LLM research.

Main components:
- models: Model provider system (vLLM, OpenAI, Anthropic)
- core: Orchestrator and tool framework
- tools: Tool implementations (web search, code execution, etc.)
- datasets: Dataset loaders and evaluators
- config: YAML-based configuration system
- utils: Parsing, logging, and utilities
"""

__version__ = "0.1.0"

# Keep top-level imports lightweight.
#
# Rationale: some submodules (e.g. vLLM, torch) may not be available on login
# nodes or lightweight environments. Import those explicitly from their
# submodules when needed.

# Always-available: config + basic model types
from .config import ExperimentConfig, load_experiment_config, save_experiment_config
from .models.base import ModelConfig, ModelFamily

__all__ = [
    # Version
    "__version__",
    # Models
    "ModelConfig",
    "ModelFamily",
    # Config
    "ExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
]
