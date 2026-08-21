"""Prefix-RFT daemon: dispatch k per rollout, and mark replayed tokens.

Subclasses the vendored ``AgentModeDaemon``. Two vendored methods are overridden:

- ``_async_set_up`` (``daemon.py:264-315``) queues one task per rollout from a
  shared sample dict. We write a per-rollout copy carrying ``prefix_k`` so the
  first rollout of each question is the hybrid one. It is a verbatim COPY of the
  vendored method plus the five marked edits in ``daemon_edits.py``; keep it that
  way, because ``scripts/check_prefix_rft_trainer_sync.py`` regenerates it from the
  vendored source and compares. Cosmetic changes here (dropping a comment,
  reflowing a line) will fail that check even though they change no behaviour.
- ``get_train_data_batch`` (``daemon.py:668-844``) builds the training tensors.
  We call it, then attach ``prefix_mask`` and ``is_prefix_rollout_list``.

The decision logic itself lives in ``dispatch.py`` and ``masks.py``, which import
no verl, so it is unit-testable on CPU. This module is a thin wrapper and is
covered by an import check rather than by pytest.

Instances are normally produced by promoting an existing ``AgentModeDaemon`` in
place (see ``trainer.PrefixRFTTrainer._ensure_prefix_daemon``), because the
vendored ``fit()`` constructs the daemon itself. ``__init__`` therefore only has
to work for direct construction, and all added state is also set explicitly by
the promotion path.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import torch

from fine_tuning.agentflow import LLM, NamedResources
from fine_tuning.agentflow.verl.daemon import AgentModeDaemon

from .dispatch import prefix_spec_for
from .masks import build_prefix_mask

logger = logging.getLogger(__name__)


class PrefixRFTDaemon(AgentModeDaemon):
    """AgentModeDaemon that dispatches prefix lengths and reports a prefix mask."""

    def __init__(
        self,
        *args,
        schedule=None,
        demo_store=None,
        n_prefixed_rollouts=1,
        prefix_mode="steps",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedule = schedule
        self.demo_store = demo_store
        self.n_prefixed_rollouts = n_prefixed_rollouts
        self.prefix_mode = prefix_mode
        self._global_step = 0
        self.last_prefix_metrics = {}

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def _prefix_spec_for(self, sample, rollout_index, is_train):
        return prefix_spec_for(
            sample=sample,
            rollout_index=rollout_index,
            is_train=is_train,
            schedule=self.schedule,
            demo_store=self.demo_store,
            n_prefixed_rollouts=self.n_prefixed_rollouts,
            global_step=self._global_step,
            mode=self.prefix_mode,
        )

    async def _async_set_up(self, data, server_addresses, is_train=True):
        """Async helper to set up data and resources on the server."""
        self.clear_data_and_server()

        # Clear any orphaned rollouts from previous runs
        try:
            orphaned_rollouts = await self.server.retrieve_completed_rollouts()
            if orphaned_rollouts:
                logger.info(f"Cleared {len(orphaned_rollouts)} orphaned rollouts from previous runs")
        except Exception as e:
            logger.warning(f"Failed to clear orphaned rollouts: {e}")

        self.backend_llm_server_addresses = server_addresses
        self.is_train = is_train

        # 1. Update resources on the server for clients to use
        llm_resource = LLM(
            endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
            model=self.train_information.get("model", "default-model"),
            sampling_parameters={"temperature": self.train_information.get("temperature", 0.7)},
        )
        resources: NamedResources = {"main_llm": llm_resource}
        resources_id = await self.server.update_resources(resources)
        self._current_resources_id = resources_id

        # 2. Queue tasks for agents to process
        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        print(f"Queueing {num_samples} samples with {rollouts_per_sample} rollouts each for {'training' if is_train else 'validation'}")

        # PREFIX-RFT EDIT 1: collect the dispatched prefix lengths for metrics.
        ks = []

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id

            # For training, each sample is rolled out multiple times
            for j in range(rollouts_per_sample):
                # PREFIX-RFT EDIT 2: the vendored loop reuses one dict for every
                # rollout of a question. The prefix differs per rollout, so take
                # a copy and stamp it here. One key per mode: prefix_k in step
                # mode, prefix_l in token mode.
                sample = dict(original_sample)
                spec = self._prefix_spec_for(sample, j, is_train)
                sample.update(spec)
                ks.append(next(iter(spec.values())))

                task_metadata = {"data_id": data_id, "is_train": is_train}

                # Data ID is different from Rollout ID, as one data can have multiple rollouts.
                rollout_id = await self.server.queue_task(
                    sample=sample,  # PREFIX-RFT EDIT 3: the per-rollout copy
                    mode="train" if is_train else "val",
                    resources_id=resources_id,
                    metadata=task_metadata,
                )
                # Store original sample data to reconstruct batch information later
                # PREFIX-RFT EDIT 4: store the copy; get_train_data_batch needs
                # the per-rollout prefix to rebuild prefix_mask.
                self._task_id_to_original_sample[rollout_id] = sample
                self._total_tasks_queued += 1

        print(f"Total tasks queued: {self._total_tasks_queued}")

        # PREFIX-RFT EDIT 5: record what was actually dispatched. Without this the
        # run cannot be distinguished from plain GRPO by its logs. print(), not
        # logger.info(): INFO from this package does not reach the SLURM log, which
        # is why job 25753032 could not be diagnosed from its output. The vendored
        # daemon prints for the same reason.
        self.last_prefix_metrics = self._summarise_prefix_dispatch(ks)
        print(
            f"Prefix dispatch: {sum(1 for k in ks if k > 0)} of {len(ks)} rollouts "
            f"prefixed (is_train={is_train}); ks={ks}"
        )

    def _summarise_prefix_dispatch(self, values):
        """Metrics for what was dispatched this step.

        ``sample_l`` is called once more here to report the window. That draw already
        happened before token mode existed, so it is left as it is; adding a second one
        would change the curriculum the run actually sees.
        """
        prefixed = [v for v in values if v > 0]
        if self.schedule is not None:
            _, low, high = self.schedule.sample_l(global_step=self._global_step)
        else:
            low, high = 0.0, 0.0
        mean = float(np.mean(prefixed)) if prefixed else 0.0
        out = {
            "actor/n_prefixed_rollouts": len(prefixed),
            "actor/prefix_low": float(low),
            "actor/prefix_high": float(high),
        }
        # Different names on purpose: mean k and mean l are not comparable quantities
        # and must not land in the same W&B series.
        if self.prefix_mode == "tokens":
            out["actor/prefix_l"] = mean
        else:
            out["actor/prefix_steps"] = mean
        return out

    def get_train_data_batch(self, max_prompt_length, max_response_length, device):
        data_proto, metrics = super().get_train_data_batch(
            max_prompt_length, max_response_length, device
        )
        if data_proto is None:
            return data_proto, metrics

        mask_rows, is_prefix_rollout = self._rebuild_prefix_rows(max_response_length)
        # The last link in the chain: replay -> triplet metadata -> prefix_mask. Zero
        # here with a non-zero dispatch means the marking was lost in transit.
        print(
            f"Prefix mask: {sum(1 for r in is_prefix_rollout if r)} of "
            f"{len(is_prefix_rollout)} rollouts marked, "
            f"{sum(sum(row) for row in mask_rows)} prefix tokens"
        )
        n_rows = data_proto.batch["responses"].shape[0]
        if len(mask_rows) != n_rows:
            raise ValueError(
                f"prefix_mask has {len(mask_rows)} rows but the batch has {n_rows}; "
                "the base daemon's row filter and build_prefix_mask have diverged"
            )

        data_proto.batch["prefix_mask"] = torch.LongTensor(mask_rows).to(device)
        data_proto.non_tensor_batch["is_prefix_rollout_list"] = np.array(is_prefix_rollout)
        # actor/num_prefix_tokens and actor/off_ratio are logged by the trainer,
        # which has the response_mask to divide by; do not duplicate them here.
        metrics.update(self.last_prefix_metrics)
        return data_proto, metrics

    def _rebuild_prefix_rows(self, max_response_length):
        """Walk completed rollouts in the same order the base daemon does.

        The base method iterates ``self._completed_rollouts`` and applies the same
        two skips (orphaned rollout, no triplets); reproducing that order is what
        keeps the mask aligned with ``responses``. The row-count assertion in the
        caller is the guard if this ever drifts.
        """
        mask_rows, is_prefix_rollout = [], []
        for rollout_id, rollout in self._completed_rollouts.items():
            if rollout_id not in self._task_id_to_original_sample:
                continue
            if not rollout.triplets:
                continue
            traces = [
                {
                    "prompt_ids": t.prompt.get("token_ids", []),
                    "response_ids": t.response.get("token_ids", []),
                    "is_prefix": bool((t.metadata or {}).get("prefix", False)),
                    "prefix_len": int((t.metadata or {}).get("prefix_len", 0) or 0),
                }
                for t in rollout.triplets
            ]
            rows = build_prefix_mask(traces, max_response_length)
            mask_rows.extend(rows)
            rollout_is_prefixed = any(t["is_prefix"] for t in traces)
            is_prefix_rollout.extend([rollout_is_prefixed] * len(rows))
        return mask_rows, is_prefix_rollout
