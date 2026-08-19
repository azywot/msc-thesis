# Add an adaptation method

"Adaptation" here means anything that improves the orchestrator without
changing the framework: better prompts, better weights, or something new. Three
methods already exist, and they hook in at three different levels. Placing a
fourth one starts with recognising which level it belongs to.

| Method | What it changes | Hook point |
|---|---|---|
| **GEPA** | the system prompt | `gepa_prompt_path` in the config bypasses `PromptBuilder` |
| **SFT** | the weights | a LoRA adapter or checkpoint loaded via `ModelConfig` |
| **RL (GRPO)** | the weights, from its own rollouts | `OrchestratorRollout` drives the real orchestrator inside verl |
| **Prefix-RFT** | the weights, from its own rollouts seeded by demonstrations | `PrefixOrchestratorRollout` replays teacher decisions inside verl |

The shared principle: **the orchestrator is never modified.** Every method
either changes what goes into it (prompt) or what it runs on (weights). If your
method requires editing `core/orchestrator.py`, the design is fighting you.

---

## Level 1 - prompt adaptation (like GEPA)

The cheapest hook. `PromptBuilder` normally assembles the system prompt from
templates; when a config sets

```yaml
gepa_prompt_path: experiments/results/gepa/gaia/<run>/best_candidate.json
```

the builder is bypassed entirely and `system_prompt` plus `planning_suffix` are
read from that file.

To add a prompt-level method, produce a JSON file with those two keys and point
a config at it. Nothing else changes - the same runner, the same orchestrator,
the same scorer, so the resulting numbers are directly comparable to a normal
run.

The optimisation side lives in `src/gepa_integration/`:

| File | Role |
|---|---|
| `adapter.py` | `AgentGEPAAdapter` - evaluates a candidate prompt by running real questions and returns scores plus reflective feedback. |
| `reflection.py` | Turns failures into the text the reflector model reads. |
| `seed.py` | The starting candidate. Imports `classify_failure` from the failure-mode analysis. |
| `data/` | Split construction. |

A different prompt optimiser reuses the same seam: implement its own adapter,
write a `best_candidate.json`, done. See [pipelines/gepa.md](../pipelines/gepa.md).

## Level 2 - weight adaptation, offline (like SFT)

If your method produces weights offline, there is no framework hook at all -
point a config at the result:

```yaml
models:
  orchestrator:
    path_or_id: Qwen/Qwen3-8B
    lora_adapter_path: /path/to/adapter    # a LoRA adapter
    max_lora_rank: 64                      # must match the training rank
```

or, for a fully fine-tuned model, set `path_or_id` straight to the checkpoint
directory and omit the adapter keys.

Two traps, both of which produce a run that completes and scores badly rather
than failing:

1. **`max_lora_rank` must match training.** vLLM's default is 16; ours is 64. A
   mismatch loads silently.
2. **The model cache is keyed on `(path_or_id, adapter)`**, not the path alone.
   That is deliberate - it is what lets a base model and its adapted version
   coexist in one run - and it means a config sharing `path_or_id` across roles
   only shares the instance if the adapters match too.

The far bigger trap is the **training data format**, and it is not a
configuration issue: SFT rows must match what inference actually builds. The
orchestrator's AgentFlow prompt is *memory-folded* - rebuilt each turn from
`query_analysis` and `action_history` - so training on native multi-turn
conversations teaches a format the model never sees at inference. That mismatch
made an early SFT run score *below* the base model. `src/verl_ext/folded_sft_dataset.py`
exists to produce in-format rows. See [pipelines/sft.md](../pipelines/sft.md).

(This is about the *row format*, distinct from *which data* fills the rows -
for changing the RL/GEPA source mix, see
[change-training-data.md](change-training-data.md).)

## Level 3 - online adaptation (like RL)

The deepest hook, and the only one that needs the orchestrator to run *inside*
the training loop. `src/fine_tuning/rollout.py` does this:

- `OrchestratorRollout` (a verl `LitAgent`) builds a tool registry and a
  `ModelConfig` per rollout, then runs the real `AgenticOrchestrator`.
- `_CapturingProvider` wraps the training engine's generation so the
  orchestrator's own loop drives it - the trajectory being trained on is
  therefore produced by exactly the code path used at evaluation.
- `reward.py`'s `OrchestratorReward` scores the finished trajectory with the
  same dataset evaluators the runner uses.

If you are adding an online method, copy this shape: **drive the real
orchestrator and score with the real evaluator.** Every shortcut - a simplified
loop, a proxy reward - reintroduces the train/inference gap that the format bug
above already cost this project once.

`src/fine_tuning/agentflow/` is vendored upstream code. Do not restyle it; see
`src/fine_tuning/agentflow/VENDORED.md`.

**Prefix-RFT is the worked example of extending Level 3 without touching vendored
code.** It reuses `OrchestratorRollout` and adds two identity seams (`_wrap_provider`,
`_wrap_tools`, both no-ops in the base class) rather than forking it, and every piece of
verl-facing logic lives in its own module under `src/verl_ext/prefix_rft/` so it can be
unit-tested without verl installed. See [pipelines/prefix-rft.md](../pipelines/prefix-rft.md).

---

## Choosing a level

Ask what the method changes:

- **Only the prompt** → level 1. No code in `agent_engine` changes at all.
- **Weights, trained offline** → level 2. Your work is a training script plus a
  data-format check; the framework only needs a config.
- **Weights, trained from the agent's own behaviour** → level 3. Reuse
  `OrchestratorRollout` rather than writing a new loop.

## Verify before scaling up

Whatever the level, prove the adapted artefact loads and changes behaviour on a
handful of questions before submitting a long job:

```bash
# level 1: does the prompt actually come from the file?
python -c "
from agent_engine.config.loader import load_experiment_config
c = load_experiment_config('<config>.yaml')
print(c.gepa_prompt_path)
"

# levels 2-3: a short run, small subset_num, then compare against the base config
python scripts/run_experiment.py --config <config>.yaml --output-dir /tmp/check
python scripts/analyze_results.py /tmp/check/raw_results.json --by-level --tools
```

An adapted run that scores *identically* to the base run usually means the
adaptation was not loaded - check the log for the adapter path before believing
a null result.
