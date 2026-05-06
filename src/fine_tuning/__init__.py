"""Orchestrator fine-tuning pipeline for CoSMAS.

Heavy dependencies (agentflow, math_verify, vllm) are imported lazily so that
lightweight submodules (config, reward, data.prepare) can be used without
installing the full training stack.
"""

from .config import FinetuningConfig
from .reward import OrchestratorReward

# OrchestratorRollout requires agentflow — only import when available
try:
    from .rollout import OrchestratorRollout
    __all__ = ["FinetuningConfig", "OrchestratorReward", "OrchestratorRollout"]
except ImportError:
    __all__ = ["FinetuningConfig", "OrchestratorReward"]
