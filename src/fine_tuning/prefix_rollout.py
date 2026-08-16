"""Prefix-RFT rollout: replay the first k teacher decisions, then go on-policy.

``k`` arrives per rollout in the task payload under ``prefix_k``; the driver owns
the schedule because it owns ``global_step``. ``k = 0`` reproduces
``OrchestratorRollout`` exactly, which is what every validation rollout and every
question without a demonstration gets.

The replay shims themselves live in ``fine_tuning.prefix_replay``, which imports
only ``agent_engine`` so it can be unit-tested without the training stack.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .prefix_replay import ReplayController, ReplayProvider, ReplayToolRegistry
from .rollout import OrchestratorRollout, _get_task_metadata


class PrefixOrchestratorRollout(OrchestratorRollout):
    """OrchestratorRollout that can start from a teacher-demonstration prefix."""

    def __init__(self, *args, demos_path=None, base_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.demos_path = demos_path or os.environ.get("PREFIX_DEMOS_PATH")
        self.base_model = base_model or os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
        self._store = None
        self._tokenizer = None
        self._active_controller: Optional[ReplayController] = None

    # ------------------------------------------------------------------ #
    # Lazily loaded resources                                              #
    # ------------------------------------------------------------------ #

    def _get_store(self):
        if self._store is None and self.demos_path:
            from verl_ext.prefix_rft.demos import DemoStore

            self._store = DemoStore.from_parquet(self.demos_path)
            n_questions, n_decisions = self._store.coverage()
            print(
                f"[PrefixOrchestratorRollout] demonstration store: "
                f"{n_questions} questions, {n_decisions} decisions"
            )
        return self._store

    def _get_tokenizer(self):
        if self._tokenizer is None and self.base_model:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer

    # ------------------------------------------------------------------ #
    # Seams from OrchestratorRollout                                       #
    # ------------------------------------------------------------------ #

    def _make_controller(self, task: Any) -> Optional[ReplayController]:
        k = int((task or {}).get("prefix_k", 0) or 0)
        if k <= 0:
            return None
        store = self._get_store()
        tokenizer = self._get_tokenizer()
        if store is None or tokenizer is None:
            return None
        question_text, _, _, _ = _get_task_metadata(task)
        steps = store.steps(question_text)
        if not steps:
            return None
        return ReplayController(steps, k, tokenizer)

    def _wrap_provider(self, provider, task):
        controller = self._make_controller(task)
        self._active_controller = controller
        if controller is None:
            return provider
        return ReplayProvider(provider, controller)

    def _wrap_tools(self, registry, task):
        controller = self._active_controller
        if controller is None:
            return registry
        return ReplayToolRegistry(registry, controller)
