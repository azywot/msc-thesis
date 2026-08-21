"""Prefix-RFT trainer.

Subclasses the vendored ``AgentFlowTrainer``. ``_train_step`` is a verbatim copy
of the vendored method with two marked edits, because it is one long method with
no smaller seam and ``VENDORED.md`` forbids editing the vendored file in place.

COPIED FROM: src/fine_tuning/agentflow/verl/trainer.py, AgentFlowTrainer._train_step,
vendored revision 8ed8f41 (see VENDORED.md). The two edits are marked
"PREFIX-RFT EDIT". ``scripts/check_prefix_rft_trainer_sync.py`` re-derives this copy
from the vendored file and fails if it has changed, so a re-vendor surfaces as a red
check rather than as silent divergence.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.utils.metric import reduce_metrics

from fine_tuning.agentflow.verl.trainer import AgentFlowTrainer, _timer

from .advantage import apply_prefix_advantage
from .daemon import PrefixRFTDaemon
from .demos import DemoStore
from .schedule import PrefixStepSchedule

logger = logging.getLogger(__name__)


class PrefixRFTTrainer(AgentFlowTrainer):
    """AgentFlowTrainer with prefix dispatch and prefix-aware advantages."""

    def _build_prefix_components(self):
        """Build the schedule and demonstration store from config.

        The schedule spans the actual run rather than the paper's fixed 500 steps
        (spec divergence 6): this run is roughly 112 steps, so a hardcoded 500
        would truncate the curriculum at 22% of its decay.
        """
        cfg = self.config.prefix_rft
        total_steps = int(getattr(self, "total_training_steps", 0) or 0) or 500
        schedule = PrefixStepSchedule(
            low_init=cfg.low_init,
            low_target=cfg.low_target,
            high=cfg.high,
            n_steps=total_steps,
            alpha=cfg.sampler_alpha,
            beta=cfg.sampler_beta,
            seed=cfg.seed,
        )
        store = DemoStore.from_parquet(cfg.demos_path)
        n_questions, n_decisions = store.coverage()
        logger.info(
            "Prefix-RFT: %d demonstrated questions, %d decisions, "
            "schedule %.2f -> %.2f over %d steps",
            n_questions,
            n_decisions,
            cfg.low_init,
            cfg.low_target,
            total_steps,
        )
        if n_questions == 0:
            raise ValueError(
                f"Demonstration store {cfg.demos_path} is empty; every rollout would "
                "run as plain GRPO and the run would silently not be Prefix-RFT."
            )
        return schedule, store

    def _ensure_prefix_daemon(self):
        """Promote the vendored daemon in place.

        AgentModeDaemon is constructed inside the vendored fit() (trainer.py:427).
        PrefixRFTDaemon adds only attributes, so reassigning __class__ and setting
        them explicitly is equivalent to having constructed it directly, and avoids
        copying fit() as well.
        """
        daemon = self.agent_mode_daemon
        if isinstance(daemon, PrefixRFTDaemon):
            return
        schedule, store = self._build_prefix_components()
        daemon.__class__ = PrefixRFTDaemon
        daemon.schedule = schedule
        daemon.demo_store = store
        daemon.n_prefixed_rollouts = int(self.config.prefix_rft.n_prefixed_rollouts)
        daemon.prefix_mode = str(self.config.prefix_rft.get("mode", "steps"))
        daemon._global_step = 0
        daemon.last_prefix_metrics = {}
        # print, not logger.info: INFO from this package does not reach the SLURM
        # log, so the promotion was invisible to job 25754573's Check B even though
        # it had happened.
        print(f"Promoted AgentModeDaemon to PrefixRFTDaemon (mode={daemon.prefix_mode})")

    def _apply_prefix_advantage(self, batch, metrics):
        """Rewrite the advantage on replayed tokens and log the prefix metrics."""
        if not self.config.get("prefix_rft", {}).get("enable", False):
            return
        if "prefix_mask" not in batch.batch.keys():
            return

        batch.batch["advantages"] = apply_prefix_advantage(
            advantages=batch.batch["advantages"],
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            prefix_mask=batch.batch["prefix_mask"],
            uid=batch.non_tensor_batch["uid"],
            rollout_id=batch.non_tensor_batch["rollout_id_list"],
            is_prefix_rollout=batch.non_tensor_batch["is_prefix_rollout_list"],
            num_rollouts_per_prefix=int(self.config.prefix_rft.n_prefixed_rollouts),
            singleton_baseline=str(self.config.prefix_rft.singleton_baseline),
        )

        prefix_mask = batch.batch["prefix_mask"]
        response_mask = batch.batch["response_mask"]
        n_prefix_tokens = int(prefix_mask.sum().item())
        n_response_tokens = int(response_mask.sum().item())
        metrics["actor/num_prefix_tokens"] = n_prefix_tokens
        metrics["actor/off_ratio"] = n_prefix_tokens / max(1, n_response_tokens)

        # How many prefixed turns were split mid-turn rather than replayed whole.
        # Derived from the mask so the driver needs nothing back from the worker;
        # it is near 0 in step mode and near 1 in token mode.
        per_row_prefix = prefix_mask.sum(dim=-1)
        per_row_response = response_mask.sum(dim=-1)
        n_prefixed_rows = int((per_row_prefix > 0).sum().item())
        n_split_rows = int(
            ((per_row_prefix > 0) & (per_row_prefix < per_row_response)).sum().item()
        )
        metrics["actor/prefix_split_fraction"] = n_split_rows / max(1, n_prefixed_rows)

        # The Figure 4 signature: reward-with-prefix should sit above the overall
        # training reward early on, and the gap should narrow as training proceeds.
        is_prefixed = np.asarray(batch.non_tensor_batch["is_prefix_rollout_list"], dtype=bool)
        scores = batch.batch["token_level_scores"].sum(dim=-1).float().cpu().numpy()
        if is_prefixed.any():
            metrics["actor/reward_with_prefix"] = float(scores[is_prefixed].mean())
        if (~is_prefixed).any():
            metrics["actor/reward_without_prefix"] = float(scores[~is_prefixed].mean())

    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        # PREFIX-RFT EDIT 1: the vendored fit() constructs a plain AgentModeDaemon
        # (trainer.py:427), so promote it here, and hand it the step the cosine
        # prefix schedule needs.
        if self.config.get('prefix_rft', {}).get('enable', False):
            self._ensure_prefix_daemon()
            self.agent_mode_daemon.set_global_step(self.global_steps)

        # data key check & no empty check
        print(f"Training data keys: {batch_dict.keys()}")
        for key, value in batch_dict.items():
            if isinstance(value, list):
                print(f"Training data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Training data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Training data {key} type: {type(value)}")

        # ensure no empty
        if not batch_dict or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in batch_dict.values()):
            raise ValueError("Training data is empty. Check your training dataset.")

        with _timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            # vLLM is awake at entry: fit() ran update_weights(0) at startup, and
            # every prior _train_step ends with update_weights, which wakes it.
            with _timer("gen", timing_raw):
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )

                if self.agent_mode_daemon._total_tasks_queued == 0:
                    raise ValueError("No training tasks were queued. Check data preparation.")

                self.agent_mode_daemon.run_until_all_finished()

                if len(self.agent_mode_daemon._completed_rollouts) == 0:
                    raise ValueError("No training tasks completed. Check server and agent execution.")

                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    device=gen_batch.batch["fake_ids"].device,
                )
                metrics.update(agent_metrics)
                if batch is None:
                    raise ValueError(
                        "All completed rollout traces had empty prompt+response IDs; "
                        "cannot run a training step. Check rollout workers and proxy token injection."
                    )
                self.agent_mode_daemon.clear_data_and_server()
                # Sleep vLLM so the actor/ref FSDP forward+backward gets the GPU.
                # update_weights() at the end of this step will wake it back up.
                self.checkpoint_manager.sleep_replicas()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                raise NotImplementedError("REMAX is not supported; use GRPO (adv_estimator=grpo) instead.")

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # PREFIX-RFT EDIT 2: overwrite the advantage on replayed teacher
            # tokens (paper section 3). Questions with no prefixed rollout are
            # left exactly as verl computed them.
            self._apply_prefix_advantage(batch, metrics)

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["agent_mode/n_dropped_sample_because_of_prompt"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_mini_batch"] = n_transition - n_remained_transition

            n_transition = len(batch)
            # make sure divisible by k_partitions for seqlen_balancing
            k_partitions = self.config.trainer.n_gpus_per_node  # equals n_gpus_per_node (typically num_workers or 8)
            n_remained_transition = n_transition // k_partitions * k_partitions
            if n_remained_transition != n_transition:
                batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_gpu_partitions"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Sync freshly-trained FSDP actor weights → vLLM and wake the rollout
            # engine so the next _train_step / _validate finds it ready. This is the
            # canonical HYBRID-mode wake path (verl ray_trainer.py:1532-1533).
            with _timer("update_weights", timing_raw):
                self.checkpoint_manager.update_weights(self.global_steps)

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics
