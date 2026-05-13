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

    # vLLM V1 uses cuMemCreate/cuMemMap for weight tensors. PyTorch's default CUDA
    # caching allocator holds large free blocks without returning them to the driver,
    # so after each FSDP training step + checkpoint save the driver has no free physical
    # pages for vLLM's wake_up call. Expandable segments return emptied pages to the
    # driver immediately, preventing this OOM.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        python_args["actor_rollout_ref.model.lora_rank"] = int(lora_cfg.get("rank", 64))
        python_args["actor_rollout_ref.model.lora_alpha"] = int(lora_cfg.get("alpha", 16))
        python_args["+actor_rollout_ref.model.lora_target_modules"] = str(lora_cfg.get("target_modules", "all-linear"))
        print(
            f"  LoRA enabled: rank={lora_cfg.get('rank', 64)}, "
            f"alpha={lora_cfg.get('alpha', 16)}, "
            f"targets={lora_cfg.get('target_modules', 'all-linear')}"
        )
    else:
        print("  LoRA disabled: full-parameter training (USE_LORA=false)")

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
