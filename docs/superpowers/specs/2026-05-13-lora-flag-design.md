# LoRA Flag Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `lora: true/false` flag (default `false`, matching AgentFlow's full-parameter training) that conditionally enables PEFT LoRA adapters in the VERL fine-tuning pipeline.

**Architecture:** A `USE_LORA` key in each experiment config's `env:` section acts as the toggle. LoRA hyperparameters live in a dedicated `lora:` top-level section (not inside `python_args`). `launch_verl.py` is the single point that translates the flag into VERL Hydra overrides — when `USE_LORA=false`, no LoRA keys are passed to VERL (which defaults to `lora_rank=0` = full-parameter training). When `USE_LORA=true`, `launch_verl.py` injects `actor_rollout_ref.model.lora_rank`, `lora_alpha`, and `+actor_rollout_ref.model.lora_target_modules` from the `lora:` section.

**Tech Stack:** Python, PyYAML, VERL (Hydra CLI overrides), PEFT (when LoRA enabled).

---

## Files

- Modify: `experiments/configs/train/config.yaml`
- Modify: `experiments/configs/train/config_smoke.yaml`
- Modify: `scripts/launch_verl.py`
- Modify: `src/fine_tuning/config.py`

---

## AgentFlow reference defaults (full-parameter training, no LoRA)

From `/gpfs/home3/xchen1/azywot/AgentFlow/train/config.yaml`:
- No `lora_rank`, `lora_alpha`, `lora_target_modules` keys anywhere
- `actor_rollout_ref.ref.fsdp_config.param_offload: False` (not offloaded)
- `actor_rollout_ref.actor.fsdp_config.param_offload: False`
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload: False`

---

## Design details

### `config.yaml` and `config_smoke.yaml` changes

Remove these three keys from `python_args` (they currently exist unconditionally):
```yaml
actor_rollout_ref.model.lora_rank: 64
actor_rollout_ref.model.lora_alpha: 16
'+actor_rollout_ref.model.lora_target_modules': 'all-linear'
```

Add `USE_LORA: "false"` to `env:`.

Add a new top-level `lora:` section (used only when `USE_LORA=true`):
```yaml
lora:
  rank: 64
  alpha: 16
  target_modules: "all-linear"
```

### `launch_verl.py` changes

After reading `python_args` from the YAML, read `USE_LORA` from the environment (it was already exported via the `env:` loop):

```python
use_lora = os.environ.get("USE_LORA", "false").strip().lower() in ("1", "true", "yes", "on")
if use_lora:
    lora_cfg = config.get("lora", {})
    python_args["actor_rollout_ref.model.lora_rank"] = int(lora_cfg.get("rank", 64))
    python_args["actor_rollout_ref.model.lora_alpha"] = int(lora_cfg.get("alpha", 16))
    python_args["+actor_rollout_ref.model.lora_target_modules"] = lora_cfg.get("target_modules", "all-linear")
    print(f"  LoRA enabled: rank={lora_cfg.get('rank', 64)}, alpha={lora_cfg.get('alpha', 16)}, targets={lora_cfg.get('target_modules', 'all-linear')}")
else:
    print("  LoRA disabled: full-parameter training (USE_LORA=false)")
```

This block goes immediately after the `slurm_cpus` block and before building the `command` list.

### `FinetuningConfig` changes

Add `use_lora: bool = False` field. Update `from_yaml` to read the nested `lora:` section:

```python
use_lora: bool = False
lora_rank: int = 64
lora_alpha: int = 16
lora_target_modules: str = "all-linear"
```

In `from_yaml`, after loading flat keys, also flatten the `lora:` sub-dict:
```python
lora_section = data.get("lora", {})
if lora_section:
    if "rank" in lora_section:
        filtered["lora_rank"] = lora_section["rank"]
    if "alpha" in lora_section:
        filtered["lora_alpha"] = lora_section["alpha"]
    if "target_modules" in lora_section:
        filtered["lora_target_modules"] = lora_section["target_modules"]
env_section = data.get("env", {})
if "USE_LORA" in env_section:
    val = str(env_section["USE_LORA"]).strip().lower()
    filtered["use_lora"] = val in ("1", "true", "yes", "on")
```

---

## What does NOT change

- `peft_vllm_weight_sync_patch.py` — already idempotent; when `lora_rank=0`, VERL never calls `update_params` with `peft_config != None`, so the patch is a harmless no-op.
- `train_orchestrator.py` — rollout side is unaffected by LoRA.
- `verl/config.yaml` (the Hydra base) — no changes needed.
