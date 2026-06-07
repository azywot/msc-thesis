"""Orchestrator fine-tuning pipeline for CoSMAS.

Heavy dependencies (agentflow, math_verify, vllm) are imported lazily so that
lightweight submodules (reward, data.prepare) can be used without installing
the full training stack.

Training is launched via ``scripts/launch_verl.py`` which reads
``experiments/configs/fine_tuning/config.yaml`` directly and forwards Hydra overrides
to verl — there is no Python config dataclass.
"""

from .reward import OrchestratorReward

# OrchestratorRollout requires agentflow (vendored) + vllm — only import when available
try:
    from .rollout import OrchestratorRollout
    __all__ = ["OrchestratorReward", "OrchestratorRollout"]
except ImportError:
    __all__ = ["OrchestratorReward"]
