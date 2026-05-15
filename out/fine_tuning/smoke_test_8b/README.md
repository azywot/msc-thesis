# Smoke Test — Qwen3-8B GRPO (Job 22756625)

Run on 2026-05-14, node `gcn114`, Snellius.  
Orchestrator: `Qwen/Qwen3-8B` (VERL/FSDP, 2 GPUs).  
Frozen sub-agent: `Qwen/Qwen3-1.7B` (single GPU, port 9998).  
Config: `experiments/configs/train/config_smoke8b.yaml`.

## Result: PASS

All 9 pre-flight checks passed. Training ran for **2 gradient steps** and saved a checkpoint. The job was then killed by the SLURM time limit (`SIGNAL Terminated`), which is expected — the smoke config is not meant to run to completion.

### Checkpoint

```
experiments/results/training/qwen3-8b-grpo-smoke/
  14-05-2026_18-06-22756625/
    global_step_2/
      actor/
        model_world_size_2_rank_0.pt
        model_world_size_2_rank_1.pt
```

### W&B run

Project `cosmas-rl-finetuning-smoke`, run `qwen3-8b-grpo-smoke` (`z7sl2ei7`).

## What to check

| Item | Where to look | Expected |
|---|---|---|
| Pre-flight checks | `smoke8b_22756625.log` lines 19–31 | "ALL 9 checks passed" |
| VERL started | `smoke8b_22756625_verl.log` | Ray + worker bootstrap messages |
| Rollouts processed | `smoke8b_22756625.log` | Tasks 1–28 received and posted |
| Training steps logged | `smoke8b_22756625_verl.log` tail | "Training finished at step 2." |
| Checkpoint on disk | path above | two `.pt` files, one per rank |

## Known non-issues in the logs

- **`NameError: name 'sqrt' is not defined`** — the orchestrator's generated code was buggy; the rollout completed with reward 0. Expected behaviour.
- **`BadRequestError: 400` (context 11078 > 8192)** — the 1.7B sub-agent hit its context limit on some web-search sub-tasks. Rollouts degraded gracefully to reward 0. Not a blocker.
- **`503 No backend LLM servers available` / `APIConnectionError`** — VERL server shut down after step 2 while a few in-flight rollouts were still running. Those rollouts posted `triplets=None`; training was already done.
- **`Connection reset by peer` / W&B `BrokenPipeError`** — teardown race during SLURM cancellation. Cosmetic.
- **Conda/module mixing warning** in `.err` — Snellius boilerplate, ignore.

## Files

| File | Contents |
|---|---|
| `smoke8b_22756625.log` | Main orchestration log: pre-flight, server startup, per-rollout trace |
| `smoke8b_22756625_verl.log` | VERL/Ray training log: gradient steps, W&B metrics |
| `smoke8b_22756625_subagent.log` | Frozen sub-agent vLLM server log |
| `smoke8b_22756625.err` | STDERR: warnings, tracebacks, SLURM termination notice |
