"""Launch the VERL training server.

Mirrors AgentFlow's train/train_agent.py: reads the training config, sets
environment variables, and spawns `python -m fine_tuning.agentflow.verl key=value ...`.

Usage:
    python scripts/launch_verl.py --config experiments/configs/train/config.yaml
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch VERL training server.")
    parser.add_argument("--config", type=str, default="experiments/configs/train/config.yaml")
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
    use_lora = os.environ.get("USE_LORA", "false").strip().lower() in ("1", "true", "yes", "on")
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
        print(
            f"  LoRA enabled: rank={rank}, alpha={alpha}, targets={targets}, "
            f"lr={lora_lr} (overrides config), "
            f"load_format=safetensors, layered_summon=True, use_shm=True"
        )
    else:
        print("  LoRA disabled: full-parameter training (USE_LORA=false)")

    # Build unique checkpoint dir: <base>/<experiment>/<DD-MM-YYYY_HH-MM>-<SLURM_JOB_ID>
    # Full fine-tuning checkpoints (~47 GB/step) go to scratch-shared; LoRA/smoke stay local.
    experiment_name = os.environ.get("EXPERIMENT_NAME", "unknown")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_tag = os.environ.get("VERL_RUN_TAG") or f"{datetime.now().strftime('%d-%m-%Y_%H-%M')}-{job_id}"
    use_scratch = os.environ.get("USE_SCRATCH_CHECKPOINTS", "false").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if use_scratch:
        _user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
        ckpt_base = f"/scratch-shared/{_user}/msc-thesis/training"
    else:
        ckpt_base = "experiments/results/training"
    ckpt_dir = f"{ckpt_base}/{experiment_name}/{run_tag}"
    python_args["trainer.default_local_dir"] = ckpt_dir
    print(f"  Checkpoint dir: {ckpt_dir}")

    # Build: python -u -m fine_tuning.agentflow.verl key=value key=value ...  (-u: line-buffered logs under SLURM > redirect)
    command = [sys.executable, "-u", "-m", "fine_tuning.agentflow.verl"]
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

    print("Launching VERL server:")
    print(" ".join(str(x) for x in command))
    print("-" * 60)

    try:
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"VERL server exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
