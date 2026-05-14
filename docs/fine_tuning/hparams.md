# Hyperparameter Notes — CoSMAS RL Fine-Tuning

Decisions and rationale for key hyperparameters in the GRPO training pipeline.
Reference config: `experiments/configs/train/config.yaml`.

---

## Learning Rate

**Current value:** `1e-6`

1e-6 is appropriate for full fine-tuning of an 8B model with GRPO:

- Full FT (no LoRA) produces larger effective updates per step than LoRA, so a conservative LR is needed for stability.
- DeepSeek-R1-Zero and most VERL-based GRPO papers use 1e-6 at this scale.
- GRPO has high-variance policy gradient estimates (n=8 rollouts per prompt), making a conservative LR important for stable learning.

**On increasing to 1e-5:**

Risky without also adjusting `kl_loss_coef`. With `kl_loss_coef=0.001` (very weak), the model can drift far from the reference policy before KL pushes back. The clip ratios (0.2/0.3) help but don't fully compensate.

**Recommended escalation path if rewards plateau:**

1. Try `3e-6` first — often enough to escape a plateau without destabilizing.
2. Raise `kl_loss_coef` in proportion (e.g. `0.005`) when increasing LR.
3. Monitor KL divergence in W&B; if it spikes, reduce LR back.

Jumping straight from `1e-6` to `1e-5` with `kl_loss_coef=0.001` risks unstable or collapsed training.

---

## KL Loss

**Current value:** `kl_loss_coef=0.001`, `use_kl_loss=true`, `use_kl_in_reward=false`

The KL penalty is intentionally light. If LR is increased, scale `kl_loss_coef` up proportionally (e.g. `0.005` at `3e-6`, `0.01` at `1e-5`).
