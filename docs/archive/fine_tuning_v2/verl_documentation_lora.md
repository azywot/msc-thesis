> **HISTORICAL — not maintained.** Archived 2026-08-16 during the repository
> handover. Kept for the reasoning it records; paths, commands, and numbers in
> it may be stale. For current documentation see `docs/`.

# RL(HF) Algorithms with LoRA Support (FSDP Only)

**see: /home/xchen1/azywot/verl for the cloned repository**

*Last updated: 02/03/2026.*

We support LoRA (Low-Rank Adaptation) for reinforcement learning algorithms such as PPO, GRPO, and others.

LoRA is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into pre-trained weights (typically linear layers). This reduces memory footprint and compute cost, making it possible to fine-tune large models with limited hardware.

The benefits this brings include:

* Reinforcement learning with very large models (e.g. 70B+) with modest hardware (e.g. 8×80GB GPUs)
* Enable larger batch sizes due to reduced memory usage
* Simplify model transfer and deployment, as only LoRA adapters need to be saved
* Combine with techniques like SLoRA or CCoE to serve multiple LoRA adapters efficiently

This guide explains how to enable LoRA in RL training and configure related parameters.

---

# FSDP Backend Usage Guide

> **Note**
>
> This section applies to FSDP/FSDP2 backend only.

LoRA is available in `verl.trainer.ppo.ray_trainer.RayPPOTrainer`. Examples are provided via the `verl.trainer.main_ppo` entry point.

LoRA is supported via HuggingFace PEFT with FSDP/FSDP2 and both vLLM and SGLang rollout backends.

## Supported Strategies

```yaml
strategy=fsdp
```

or

```yaml
strategy=fsdp2
```

## Supported Rollout Backends

```yaml
rollout.name=vllm
```

or

```yaml
rollout.name=sglang
```

---

# Required Configurations for LoRA

## `actor_rollout_ref.model.lora_rank`

```yaml
actor_rollout_ref.model.lora_rank: int
```

Set to a reasonable value greater than 0 (e.g. `8`, `16`, `32`, `64`).

---

## `actor_rollout_ref.model.lora_alpha`

```yaml
actor_rollout_ref.model.lora_alpha: float
```

The alpha scaling term used in LoRA.

---

## `actor_rollout_ref.rollout.load_format`

```yaml
actor_rollout_ref.rollout.load_format: safetensors
```

Required. This enables vLLM to load the base model.

---

## `actor_rollout_ref.model.target_modules`

```yaml
actor_rollout_ref.model.target_modules: all-linear
```

Defines which modules LoRA should be applied to.

Typically set to:

```yaml
all-linear
```

---

# Optional Configurations for LoRA

## `actor_rollout_ref.model.lora_adapter_path`

```yaml
actor_rollout_ref.model.lora_adapter_path: <path>
```

Path to a pretrained LoRA adapter directory.

If provided, verl loads an existing adapter instead of creating a new one.

This enables multi-stage training from previously saved adapters.

The directory must contain:

* `adapter_model.safetensors`
* `adapter_config.json`

---

## `actor_rollout_ref.model.lora.merge`

```yaml
actor_rollout_ref.model.lora.merge: bool
```

Controls whether LoRA adapters are merged into the base model weights before transferring to the rollout engine (`vLLM` or `SGLang`).

### Behavior

* `True`

  * LoRA adapters are merged into base weights
  * Full merged weights are synchronized

* `False`

  * Only LoRA adapter deltas are transferred natively

### Important Note

For SGLang:

```yaml
merge=True
```

is currently required.

Native adapter loading (`merge=False`) for SGLang is planned.

---

# Recommended Options

## Use shared memory preload

```yaml
actor_rollout_ref.model.use_shm=True
```

Preloads the model into `/dev/shm` to improve model loading speed.

---

## Layered summon for large models

```yaml
actor_rollout_ref.rollout.layered_summon=True
```

Enables the actor model to gather FSDP shards layer-by-layer when synchronizing the LoRA adapter to vLLM.

This reduces GPU peak memory usage.

Recommended when:

* The model is very large (70B+)
* GPU memory is limited (<48GB)

---

# Best Practices and Notes

## Learning Rate

It is recommended to increase the learning rate by roughly one order of magnitude compared to full fine-tuning.

---

# LoRA Rank Recommendations

A very small `lora_rank` can lead to:

* Slower convergence
* Worse training performance

Recommendations from community experiments:

* For a `0.5B` model:

```yaml
lora_rank=32
```

produced convergence speed and final performance close to full fine-tuning.

* For a `32B` model:

```yaml
lora_rank=128
```

also achieved convergence speed and performance close to non-LoRA training.

General recommendation:

```yaml
lora_rank >= 32
```

---

# Reference Configuration

Reference configuration for RL training with the `Qwen2.5-72B` model using `8 × 80GB GPUs`:

```yaml
data.train_batch_size=64 \
actor_rollout_ref.model.use_shm=True \
actor_rollout_ref.model.lora_rank=32 \
actor_rollout_ref.model.lora_alpha=32 \
actor_rollout_ref.model.target_modules=all-linear \
actor_rollout_ref.actor.optim.lr=3e-5 \
actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.rollout.max_num_seqs=64 \
actor_rollout_ref.rollout.max_model_len=1536 \
actor_rollout_ref.rollout.max_num_batched_tokens=1536 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.rollout.layered_summon=True \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
```

---

# Example Scripts

## LoRA training from scratch

```text
examples/tuning/lora/run_qwen3_8b_fsdp.sh
```

---

## LoRA training from adapter path

```text
examples/tuning/lora/run_qwen3_8b_from_adapter_fsdp.sh
```

---

## LoRA training for VLMs

```text
examples/tuning/lora/run_qwen2_5_vl_7b_fsdp.sh
```

---

# Summary

For FSDP-based LoRA training in verl:

* Use `strategy=fsdp` or `strategy=fsdp2`
* Use PEFT-based LoRA integration
* Prefer `lora_rank >= 32`
* Use `layered_summon=True` for large models
* Use `safetensors` rollout loading
* Use `all-linear` as the default target module setup
* Increase learning rate relative to full fine-tuning

This setup enables efficient RL fine-tuning of very large language models while significantly reducing memory usage and training cost.
