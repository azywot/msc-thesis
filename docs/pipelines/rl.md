# RL pipeline — GRPO on the orchestrator

Reinforcement learning on the Qwen3-8B orchestrator with Flow GRPO. The
orchestrator is the only thing trained; sub-agents run on a **separate, frozen**
vLLM server, so the tool interface during training is identical to the one at
evaluation.

**The reference document is [`src/fine_tuning/README.md`](../../src/fine_tuning/README.md)**
— 765 lines covering architecture, GPU allocation, W&B metrics, checkpoint
layout, troubleshooting and design decisions. This page is orientation: what the
pipeline is for, where its pieces live, and how it hooks into the framework.
When the two disagree, the module README wins.

---

## Why RL, in this project

Failure analysis across 2,534 MAS failures identified **direct reasoning without
action** as the dominant failure mode: the orchestrator answers from parametric
knowledge instead of delegating to a sub-agent. GRPO on retrieval-intensive
(Search-R1: HotpotQA + NQ) and math-intensive (DeepMath) data is meant to create
pressure toward tool use — tool-less rollouts lose reward, tool-using rollouts
win.

That framing matters when reading results: the headline number to watch is not
only accuracy but **whether tool-use rates moved**.

## The essentials

| What | Detail |
|---|---|
| Trained | Qwen3-8B orchestrator only; sub-agents frozen at Qwen3-1.7B |
| Method | Flow GRPO — the final reward propagates to every turn (planning, tool calls, synthesis) |
| Reward | Binary: 1.0 correct / 0.0 otherwise, via the **same** `evaluate_answer()` the benchmarks use |
| Data | 1800 questions: 900 Search-R1 (85% HotpotQA / 15% NQ) + 900 DeepMath (difficulty ≥ 3) |
| Val / test | 50-row val (20 search / 10 math / 20 AIME) for checkpoint selection; 200-row test held out |
| LoRA | Rank 64, alpha 64, all-linear; ~250–500 MB adapter vs ~16 GB full FT |
| Rollouts | 8 per question in training (the GRPO group); 1 greedy in validation |
| Hardware | 4 × H100 NVL — GPU 0 hosts the frozen sub-agent plus verl, GPUs 1–3 verl only |
| Run time | 2 epochs, ~112 steps; ~20 min/step observed, ~40 h end to end |

> **The effective learning rate is `1e-5`**, set by `scripts/launch_verl.py`.
> `config.yaml` shows `1e-6`, which is the full-FT baseline and is *not* what a
> LoRA run uses. Read the launcher, not the config, when reporting
> hyperparameters.

## Running it

```bash
sbatch jobs/fine_tuning/004_smoke_8b.job     # smoke test first
sbatch jobs/fine_tuning/005_train.job        # the real run
```

Then merge and evaluate:

```bash
python scripts/merge_lora.py \
    --checkpoint <best_checkpoint> \
    --base-model Qwen/Qwen3-8B \
    --output-dir <path>
```

Only the **latest** and **best** adapter directories are kept; the rest are
deleted asynchronously after rotation. `best_checkpoint/` is a symlink updated
whenever validation reward improves.

> `merge_lora.py` is for the **RL** path only. It expects verl's `actor/` layout
> with a single consolidated shard and will refuse SFT checkpoints — see
> [sft.md](sft.md#checkpoint-handling).

---

## How it hooks into the framework

This is the only pipeline that runs the orchestrator *inside* a training loop,
and it does so without a parallel implementation:

| Piece | Role |
|---|---|
| `src/fine_tuning/rollout.py` — `OrchestratorRollout` | A verl `LitAgent` that builds a tool registry and `ModelConfig` per rollout, then runs the real `AgenticOrchestrator`. |
| `src/fine_tuning/rollout.py` — `_CapturingProvider` | Wraps the training engine's generation so the orchestrator's own loop drives it. |
| `src/fine_tuning/reward.py` — `OrchestratorReward` | Scores the finished trajectory with the same dataset evaluators the runner uses. |
| `src/fine_tuning/agentflow/` | Vendored upstream AgentFlow. **Do not restyle** — see its `VENDORED.md`. |
| `src/verl_ext/` | Local verl extensions: `folded_sft_dataset.py`, `checkpoint_utils.py`. |

The design rule this embodies: **the trajectory being trained on is produced by
exactly the code path used at evaluation.** Every shortcut — a simplified loop,
a proxy reward — reintroduces a train/inference gap. The SFT pipeline learned
that lesson the expensive way; see [sft.md](sft.md#the-format-rule--the-thing-to-get-right).

## Adding a different RL method

Reuse `OrchestratorRollout` rather than writing a new loop; swap the algorithm
in the verl config. See
[guides/add-an-adaptation-method.md](../guides/add-an-adaptation-method.md#level-3--online-adaptation-like-rl).
