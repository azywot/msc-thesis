# Vendored code — do not restyle

Everything in this directory is a **copy of an external project**. It is not
CoSMAS code and it is not maintained here.

| | |
|---|---|
| Upstream | `https://github.com/shin-ee-chen/AgentFlow.git` |
| Vendored | 2026-06-01, in commit `8ed8f41` ("Feat/fine tuning (#20)") |
| Scope | 29 files — the subset the training pipeline actually uses |
| Local changes | Import rewrites only (below) |

---

## Why it is here

The training pipeline previously depended on a separate AgentFlow clone, wired
up by a `_agentflow_path.py` path hack plus a clone-and-`pip install` block in
every SLURM job script. Vendoring it means the whole pipeline installs with one
`pip install -e .` and a job script cannot silently run against a different
AgentFlow revision than the one on your laptop.

## What was changed

Deliberately as little as possible:

1. **Eight absolute imports rewritten** from `agentflow.*` to
   `fine_tuning.agentflow.*`, so the package resolves at its new location. This
   includes the Hydra config path in `verl/entrypoint.py`:

   ```python
   # upstream
   @hydra.main(config_path="pkg://agentflow/verl", config_name="config", ...)
   # here
   @hydra.main(config_path="pkg://fine_tuning.agentflow/verl", config_name="config", ...)
   ```

2. **`_agentflow_path.py` deleted** — the `sys.path` hack the vendored package
   makes unnecessary.

3. **`verl/peft_vllm_weight_sync_patch.py` removed** during the verl upgrade.
   It monkey-patched two FSDP→vLLM LoRA sync bugs in verl 0.5 (re-added
   `.base_layer.` keys, and a missing `llm_engine` on vLLM V1). verl 0.6.0
   deprecated `ShardingManager` entirely, so the patched class no longer exists
   and the workaround targeted dead code. Recoverable from git history if a
   comparable regression ever reappears.

No behaviour was changed. No formatting, naming, docstrings or style were
changed.

---

## Rules

**Do not restyle, reformat, or refactor these files.** Not to satisfy `black`,
not to fix a lint warning, not to match the surrounding project's conventions.
Every cosmetic edit widens the diff against upstream and makes the next
re-vendor a manual merge instead of a copy.

**Do not fix bugs here.** Fix them upstream, or work around them in
`src/fine_tuning/` where the workaround is visible as ours. A patch applied
inside vendored code is invisible to the next person who re-vendors, and it will
be silently reverted.

**To take a newer upstream version: re-vendor, do not patch.**

1. Clone upstream at the target revision.
2. Copy the same file subset over this directory.
3. Re-apply the import rewrites in change 1 above.
4. Run the RL smoke test — `sbatch jobs/fine_tuning/004_smoke_8b.job` — before
   anything long.
5. Update the table at the top of this file with the new revision and date.

**If you must diverge**, record it in the list above with the reason, so the
next re-vendor knows what to re-apply rather than discovering it as a mysterious
test failure.

---

## Related

- [`../README.md`](../README.md) — the RL pipeline this code serves
- [`../../verl_ext/`](../../verl_ext/) — our own verl extensions, which *are*
  CoSMAS code and *should* follow project style
- [`../../../docs/pipelines/rl.md`](../../../docs/pipelines/rl.md) — pipeline overview
