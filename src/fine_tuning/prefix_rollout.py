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

from verl_ext.prefix_rft.dispatch import read_prefix_spec

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
        """Build a replay controller for this task, or None to run plain on-policy.

        The mode comes from the payload, so the driver and this worker cannot disagree
        about which experiment is running. The reading itself is in
        ``verl_ext.prefix_rft.dispatch.read_prefix_spec`` because this module is not
        importable in the env that has pytest.
        """
        mode, value = read_prefix_spec(task)
        if mode is None:
            # Only worth reporting when neither key is there, which means dispatch broke.
            print(
                "[PrefixOrchestratorRollout] neither prefix_k nor prefix_l in the task "
                f"payload — the daemon did not dispatch one. "
                f"keys={sorted((task or {}).keys())}"
            )
            return None
        if mode == "tokens":
            return self._make_token_controller(task, value)
        return self._make_step_controller(task, value)

    def _make_step_controller(self, task, k):
        """Step mode: replay k whole teacher decisions.

        Every ``return None`` here is a silent downgrade to GRPO, so each one says why.
        print(), not logging: INFO from this package does not reach the SLURM log, and a
        run where replay never happened is indistinguishable from a working one without
        these lines (job 25753032).
        """
        if k <= 0:
            return None
        store = self._get_store()
        tokenizer = self._get_tokenizer()
        if store is None or tokenizer is None:
            print(
                "[PrefixOrchestratorRollout] prefix_k=%s but no store/tokenizer; "
                "running on-policy" % k
            )
            return None
        question_text, _, _, _ = _get_task_metadata(task)
        steps = store.steps(question_text)
        if not steps:
            print(
                f"[PrefixOrchestratorRollout] prefix_k={k} but the store has no "
                f"demonstration for this question; running on-policy. "
                f"question={question_text[:60]!r}"
            )
            return None
        print(
            f"[PrefixOrchestratorRollout] replaying {k} of {len(steps)} teacher "
            f"decisions for {question_text[:50]!r}"
        )
        return ReplayController(steps, k, tokenizer)

    def _make_token_controller(self, task, l):
        """Token mode: replay a token fraction of the whole demonstration."""
        if l <= 0.0:
            return None
        store = self._get_store()
        tokenizer = self._get_tokenizer()
        if store is None or tokenizer is None:
            raise RuntimeError(
                f"prefix_l={l} was dispatched but this worker has no demonstration "
                "store or tokenizer; token mode cannot downgrade silently or the run "
                "is not the experiment it claims to be."
            )
        question_text, _, _, _ = _get_task_metadata(task)
        steps = store.steps(question_text)
        if not steps:
            # Coverage is partial by design (1358 of 1800), so this one is legitimate.
            print(
                f"[PrefixOrchestratorRollout] prefix_l={l:.3f} but the store has no "
                f"demonstration for this question; running on-policy. "
                f"question={question_text[:60]!r}"
            )
            return None
        ctrl = ReplayController.from_token_fraction(steps, l, tokenizer)
        print(
            f"[PrefixOrchestratorRollout] token prefix l={l:.3f}: {ctrl.k} of "
            f"{len(steps)} decisions replayed whole"
            + (
                f", then {ctrl.split_tokens} tokens of decision {ctrl.k + 1}"
                if ctrl.split_index is not None
                else " (no split)"
            )
        )
        return ctrl

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
