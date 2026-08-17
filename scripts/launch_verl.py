"""Launch the VERL training server.

Mirrors AgentFlow's train/train_agent.py: reads the training config, sets
environment variables, and spawns `python -m fine_tuning.agentflow.verl key=value ...`.

Usage:
    python scripts/launch_verl.py --config experiments/configs/fine_tuning/config.yaml
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch VERL training server.")
    parser.add_argument("--config", type=str, default="experiments/configs/fine_tuning/config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the launch command exactly as a real run would, then append Hydra's "
            "--cfg job so it resolves config_path, composes every override and exits. "
            "No Ray, no GPU, no weights. Catches the class of failure that otherwise "
            "only appears minutes into an allocation: an unresolvable config_path, a "
            "mistyped key, or a '+' prefix that Hydra's struct mode rejects."
        ),
    )
    args, unknown = parser.parse_known_args()

    # VERL workers forbid ROCR_VISIBLE_DEVICES alongside CUDA_VISIBLE_DEVICES (see
    # verl/single_controller/base/worker.py). Some HPC stacks export both even on NVIDIA nodes.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set environment variables from config.env
    for key, value in config.get("env", {}).items():
        os.environ[key] = str(value)
        print(f"  Exported {key}={value}")

    # Same as fine_tuning.agentflow.verl.entrypoint: VERL + vLLM v1 AsyncLLM require this.
    _v1 = os.environ.get("VLLM_USE_V1", "").strip().lower()
    if _v1 not in ("1", "true", "yes", "on"):
        os.environ["VLLM_USE_V1"] = "1"

    python_args = dict(config.get("python_args", {}))
    # Ray defaults num_cpus to the whole host under SLURM; that prestarts far more workers than the
    # job's CPU allocation and wedges worker registration. Prefer the scheduler's CPU count.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        python_args["ray_init.num_cpus"] = int(slurm_cpus)
        print(f"  ray_init.num_cpus={slurm_cpus} (from SLURM_CPUS_PER_TASK)")

    # USE_LORA was exported above via the env: loop; read it back now.
    # Default is "1" (LoRA on) — full-FT requires an explicit USE_LORA=false in config.yaml.
    use_lora = os.environ.get("USE_LORA", "1").strip().lower() in ("1", "true", "yes", "on")
    if use_lora:
        lora_cfg = config.get("lora", {}) or {}
        rank = int(lora_cfg.get("rank", 64))
        alpha = int(lora_cfg.get("alpha", 64))
        targets = str(lora_cfg.get("target_modules", "all-linear"))
        python_args["actor_rollout_ref.model.lora_rank"] = rank
        python_args["actor_rollout_ref.model.lora_alpha"] = alpha
        # No + prefix: target_modules is an existing VERL schema key (lora_target_modules is not).
        python_args["actor_rollout_ref.model.target_modules"] = targets
        # load_format=safetensors: vLLM must load base weights from disk on startup so the
        # FSDPVLLMShardingManager can push LoRA deltas on top (dummy_dtensor starts with zeros,
        # which breaks LoRA — base weights would be missing entirely).
        python_args["actor_rollout_ref.rollout.load_format"] = "safetensors"
        # layered_summon: sync FSDP→vLLM one layer at a time (lower peak GPU memory for LoRA).
        python_args["actor_rollout_ref.rollout.layered_summon"] = True
        # use_shm: pass weights via shared memory instead of torch RPC (faster, less contention).
        python_args["actor_rollout_ref.model.use_shm"] = True
        # LoRA trains ~1% of parameters; a 10× higher LR than full FT is standard practice.
        lora_lr = 1e-5
        python_args["actor_rollout_ref.actor.optim.lr"] = lora_lr
        # KV cache must be flushed between rollouts: vllm's add_lora swaps adapter weights but
        # cached prefixes were computed under the previous adapter → silent drift otherwise.
        # Verl's Qwen3-8B LoRA example leaves the default True; we override the full-FT False here.
        python_args["actor_rollout_ref.rollout.free_cache_engine"] = True
        # LoRA frees ~24 GB optimizer + ~4 GB ref shard per GPU → vLLM can take more KV cache.
        # Pushed from 0.6 to 0.70 alongside max_model_len=18432 (was 22528). Smoke 8B
        # observed only 27 GB peak torch alloc during training (94 GB H100), so growing
        # the vLLM KV pool fills the headroom and lifts rollout throughput. GPU 0 is
        # tightest (also hosts the sub-agent at util=0.12 ≈ 11 GB): 66 GB vLLM + 11 GB
        # sub-agent + 4 GB FSDP shard + 8 GB activations ≈ 89 GB / 94 GB (5 GB headroom).
        python_args["actor_rollout_ref.rollout.gpu_memory_utilization"] = 0.70
        # use_dynamic_bsz MUST remain False: VERL's dp_actor.update_policy calls
        # prepare_dynamic_batch without dp_group, so the cross-rank all_reduce that
        # syncs num_micro_batches is skipped.  When ranks have different total token
        # counts (e.g. one shard crosses the 45056 boundary and gets 2 micro-batches
        # while another stays at 1), FSDP AllGather on ranks 0/1 meets AllReduce on
        # ranks 2/3 at the same NCCL seqnum → 1-hour watchdog deadlock.  With
        # use_dynamic_bsz=False, every rank always does ppo_mini_batch_size //
        # ppo_micro_batch_size_per_gpu = 8 // 4 = 2 fixed micro-batches — safe.
        # (Bug 8 in CHANGELOG_ft_debugging.md; root fix is upstream dp_actor.py
        # passing dp_group=dist.group.WORLD to prepare_dynamic_batch.)
        python_args["actor_rollout_ref.actor.use_dynamic_bsz"] = False
        python_args["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] = 45056

        # Resume from a previously saved adapter (multi-stage training or warm restart).
        # Set lora.resume_adapter_path in the config to the saved lora_adapter/ directory;
        # leave unset (or null) for fresh training from the base model.
        resume_path = lora_cfg.get("resume_adapter_path") or None
        if resume_path:
            python_args["actor_rollout_ref.model.lora_adapter_path"] = str(resume_path)

        # lora.merge: False for vLLM (default, adapter deltas transferred natively).
        # Set to True only when using the SGLang rollout backend.
        lora_merge = bool(lora_cfg.get("merge", False))
        if lora_merge:
            python_args["actor_rollout_ref.model.lora.merge"] = True

        print(
            f"  LoRA enabled: rank={rank}, alpha={alpha}, targets={targets}, "
            f"lr={lora_lr} (overrides config), "
            f"load_format=safetensors, layered_summon=True, use_shm=True, "
            f"free_cache_engine=True, gpu_memory_utilization={python_args['actor_rollout_ref.rollout.gpu_memory_utilization']}, "
            f"use_dynamic_bsz=False (ppo_max_token_len_per_gpu=45056 kept as reference)"
            + (f", resume_adapter_path={resume_path}" if resume_path else "")
            + (", merge=True (SGLang)" if lora_merge else "")
        )
    else:
        print("  LoRA disabled: full-parameter training (USE_LORA=false)")

    # Save optimizer state only when explicitly requested (needed for resume; omit by default to save disk).
    # VERL controls checkpoint contents via actor_rollout_ref.actor.checkpoint.save_contents.
    #
    # LoRA branch: drop 'model' from save_contents. For LoRA, verl's FSDPCheckpointManager
    # writes the FULL PEFT-wrapped state dict (base + LoRA, ~17 GB for Qwen3-4B / ~33 GB for
    # Qwen3-8B) into model_world_size_*_rank_*.pt — base weights are identical to the HF cache
    # so this is pure disk waste. The LoRA adapter is saved separately and unconditionally in
    # fsdp_workers.py:1194-1223 (~250-500 MB), so dropping 'model' leaves a usable checkpoint.
    # Resume from a LoRA-only ckpt uses lora.resume_adapter_path → lora_adapter/ (warm restart;
    # mid-run auto-resume is not supported because the optimizer's tied-to-base FSDP state
    # cannot be reconstructed without model_world_size_*.pt).
    save_optimizer = os.environ.get("SAVE_OPTIMIZER", "false").strip().lower() in ("1", "true", "yes", "on")
    if use_lora:
        contents = ["optimizer", "extra"] # if save_optimizer else ["extra"]
    else:
        contents = ["model", "optimizer", "extra"] if save_optimizer else ["model"]
    contents_str = "[" + ",".join(f"'{c}'" for c in contents) + "]"
    python_args["actor_rollout_ref.actor.checkpoint.save_contents"] = contents_str
    python_args["actor_rollout_ref.actor.checkpoint.load_contents"] = contents_str
    print(f"  Save optimizer state: {save_optimizer} (save_contents={','.join(contents)})")

    # Build unique checkpoint dir: <base>/<experiment>/<DD-MM-YYYY_HH-MM>-<SLURM_JOB_ID>
    # USE_SCRATCH_CHECKPOINTS=true  → /scratch-shared/$USER/.../fine_tuning  (large quota, production).
    # USE_SCRATCH_CHECKPOINTS=false → experiments/results/fine_tuning        (GPFS home, smoke tests).
    # All three configs use LoRA; optimizer state is tiny (~10s MB). Smoke tests use false because
    # checkpoint volume is small enough for GPFS home. Production uses true for the rollout JSONs
    # (8 rollouts × 1800 questions × 2 epochs = 28 800 JSON files) which can fill GPFS home fast.
    experiment_name = os.environ.get("EXPERIMENT_NAME", "unknown")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_tag = os.environ.get("VERL_RUN_TAG") or f"{datetime.now().strftime('%d-%m-%Y_%H-%M')}-{job_id}"
    use_scratch = os.environ.get("USE_SCRATCH_CHECKPOINTS", "false").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if use_scratch:
        _user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
        # LoRA runs share a dedicated scratch root so adapters don't mix with full-FT shards.
        # Verl writes the adapter at <ckpt_dir>/global_step_<N>/actor/lora_adapter/.
        if use_lora:
            ckpt_base = f"/scratch-shared/{_user}/fine_tuning/lora_adapters"
        else:
            ckpt_base = f"/scratch-shared/{_user}/msc-thesis/fine_tuning"
    else:
        ckpt_base = "experiments/results/fine_tuning"
    ckpt_dir = f"{ckpt_base}/{experiment_name}/{run_tag}"
    python_args["trainer.default_local_dir"] = ckpt_dir
    print(f"  Checkpoint dir: {ckpt_dir}")

    # Inject --served-model-name via engine_kwargs.vllm so vLLM registers the HF id used by the
    # orchestrator. Skipped if the user has already pinned a value in the config.
    base_model = os.environ.get("BASE_MODEL", "").strip()
    if base_model:
        smn_key = "actor_rollout_ref.rollout.engine_kwargs.vllm.served_model_name"
        if smn_key not in python_args and f"+{smn_key}" not in python_args:
            python_args[f"+{smn_key}"] = base_model
            print(f"  served_model_name={base_model} (forwarded to vLLM HTTP server)")

    # Disable prefix caching: VERL's build_cli_args_from_config silently drops bool
    # False values, so enable_prefix_caching=False in the config is a NO-OP — vLLM V1
    # defaults to True.  Prefix caching + free_cache_engine + LoRA causes "CUDA error:
    # illegal memory access" — stale prefix-cache hash entries point to freed KV blocks
    # after the cache engine is rebuilt between training phases.
    # Workaround: inject "no_enable_prefix_caching=True" via engine_kwargs.  True bools
    # DO pass through build_cli_args → --no_enable_prefix_caching.  vLLM's
    # FlexibleArgumentParser normalises underscores to dashes →
    # --no-enable-prefix-caching, which argparse (BooleanOptionalAction) honours.
    # Six crashes: ft_23028433, ft_23031012, ft_23060623, ft_23071620, ft_23092138,
    # ft_23117863 — all had enable_prefix_caching=True in the vLLM init banner despite
    # the config setting False.
    npc_key = "+actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching"
    if npc_key not in python_args:
        python_args[npc_key] = "True"
        print("  no_enable_prefix_caching=True (disables prefix caching; workaround for build_cli_args bug)")

    # enforce_eager is NOT needed — the real crash cause was the early-completion
    # race condition in daemon.py (in-flight vLLM requests hit freed KV cache).
    # Removing it restores torch.compile + CUDA graphs for faster inference.

    # Keys not in VERL's structured Hydra schema must be prefixed with + (append, not override).
    # ray_init.num_cpus: custom key passed to our AgentFlowTrainer's ray.init() call.
    # trainer.val_every_epoch / save_every_epoch: AgentFlowTrainer-only extensions.
    for _key in ("ray_init.num_cpus", "trainer.val_every_epoch", "trainer.save_every_epoch"):
        if _key in python_args:
            python_args[f"+{_key}"] = python_args.pop(_key)

    # Prefix-RFT runs its own entrypoint, which substitutes the trainer, daemon, worker
    # and actor. Everything else about the launch is identical.
    prefix_rft = os.environ.get("PREFIX_RFT", "").strip().lower() in ("1", "true", "yes", "on")
    module = "verl_ext.prefix_rft" if prefix_rft else "fine_tuning.agentflow.verl"
    if prefix_rft:
        print(f"  Prefix-RFT enabled: launching {module}")

    # Build: python -u -m <module> key=value key=value ...  (-u: line-buffered logs under SLURM > redirect)
    command = [sys.executable, "-u", "-m", module]
    for key, value in python_args.items():
        if isinstance(value, list):
            # Hydra list syntax: key=[elem1,elem2]  (each element env-expanded)
            elems = ",".join(os.path.expandvars(str(v)) for v in value)
            command.append(f"{key}=[{elems}]")
        elif isinstance(value, str):
            command.append(f"{key}={os.path.expandvars(value)}")
        else:
            command.append(f"{key}={value}")
    command.extend(unknown)

    if args.dry_run:
        # --cfg job makes Hydra compose and print the config instead of calling main(),
        # so the whole override list is validated without starting anything.
        command.append("--cfg")
        command.append("job")
        print("Dry run — resolving the config without launching:")
    else:
        print("Launching VERL server:")
    print(" ".join(str(x) for x in command))
    print("-" * 60)

    if args.dry_run:
        result = subprocess.run(command, env=os.environ, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-4000:])
            print(result.stderr[-4000:])
            print(f"DRY RUN FAILED — the config does not resolve (exit {result.returncode}).")
            sys.exit(result.returncode)
        print(f"DRY RUN OK — {len(python_args)} overrides resolved against {module}.")
        return

    try:
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"VERL server exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
