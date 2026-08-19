"""Hydra entrypoint for Prefix-RFT training.

Mirrors ``fine_tuning.agentflow.verl.entrypoint`` with four substitutions:
our hydra config, ``PrefixRFTWorker`` in place of verl's actor worker,
``PrefixRFTTrainer`` in place of ``AgentFlowTrainer``, and ``calculate_entropy``
forced on because the entropy clip needs it.

Launched as ``python -m verl_ext.prefix_rft`` by ``scripts/launch_verl.py`` when
``PREFIX_RFT`` is set.
"""

import os

import hydra
import ray
from omegaconf import open_dict


@hydra.main(
    config_path="pkg://verl_ext.prefix_rft/config",
    config_name="prefix_rft_trainer",
    version_base=None,
)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # VERL async rollout uses vLLM's v1 AsyncLLM; workers must not see VLLM_USE_V1 cleared/false
    # (raises ValueError in vllm.v1.engine.async_llm.AsyncLLM.from_vllm_config).
    _v1 = os.environ.get("VLLM_USE_V1", "").strip().lower()
    if _v1 not in ("1", "true", "yes", "on"):
        os.environ["VLLM_USE_V1"] = "1"

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_USE_V1": "1",
                },
            },
            num_cpus=config.ray_init.num_cpus,
            include_dashboard=False,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
    ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # The entropy clip ranks prefix tokens by current-policy entropy, so the
        # actor must compute it whatever else the config says.
        with open_dict(config):
            config.actor_rollout_ref.actor.calculate_entropy = True

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)

        if config.actor_rollout_ref.actor.strategy not in ["fsdp", "fsdp2"]:
            raise NotImplementedError(
                "Prefix-RFT supports the FSDP strategy only; the actor is installed by "
                "subclassing verl's FSDP worker."
            )
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        if config.actor_rollout_ref.rollout.mode != "async":
            raise NotImplementedError(
                "Prefix-RFT requires rollout.mode=async: the rollout workers reach the "
                "trainer through the AgentFlow daemon."
            )

        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        from .worker import PrefixRFTWorker

        actor_rollout_cls = PrefixRFTWorker
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Reference policy: the base worker, not ours. Only the actor is clipped.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        from fine_tuning.agentflow.verl.dataset import AgentDataset

        from .trainer import PrefixRFTTrainer

        train_dataset = AgentDataset(
            data_files=config.data.train_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        val_dataset = AgentDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config.data,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = PrefixRFTTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
