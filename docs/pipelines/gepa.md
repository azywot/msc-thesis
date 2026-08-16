# GEPA pipeline - prompt adaptation

GEPA evolves the orchestrator's **system prompt** and **planning-turn suffix**
from the agent's own execution traces. No weights change: a Qwen3-32B reflector
reads full `<think>` traces, action histories and failure labels, and proposes
prompt rewrites.

Module documentation: `src/gepa_integration/README.md`.

---

## What is optimised

Two components per benchmark:

| Component | What it is |
|---|---|
| `system_prompt` | The full system prompt - preamble, few-shot example, final instructions. Tool schemas inside `<tools>…</tools>` are **protected and never modified**. |
| `planning_suffix` | The instruction block appended to the user query on turn 0, the planning turn. |

The agent configuration during optimisation is fixed and matches the
milestone-1 AgentFlow setup: Qwen3-8B orchestrator and sub-agents (same model,
shared vLLM instance), `thinking_mode: ORCHESTRATOR_ONLY`, sub-agent mode
(`direct_tool_call: false`).

## Training data

Sourced from open datasets that do **not** overlap the evaluation benchmarks,
so the held-out test sets stay clean:

- **GAIA preset** - 75% Search-R1 (85/15 HotpotQA/NQ) + 25% DeepMath (no
  difficulty filter).
- **MATH preset** - 75% DeepMath (difficulty ≥ 5) + 25% Search-R1.

Both are 300 examples: 150 `D_feedback` / 50 `D_pareto` / 100 test. Built by
`src/gepa_integration/data/prepare.py`. Multi-answer Search-R1 examples carry
`answer_aliases` so every valid answer string scores correctly.

Evaluation is on the 100 held-out test examples, never seen during optimisation.

---

## Running it

```bash
# 1. Download Search-R1 + DeepMath and build the GEPA data files
sbatch jobs/gepa/000_prep_gepa_data.job

# 2. Install the gepa package into the conda env
sbatch jobs/gepa/001_install_gepa_deps.job

# 3. CPU smoke test - imports, split integrity, evaluator
sbatch jobs/gepa/002_smoke_gepa.job

# 4. GPU smoke test - 1 GEPA step on 2 real examples (3xH100, ~1h)
sbatch jobs/gepa/003_smoke_gepa_gpu.job

# 5. Full optimisation, submitted independently (each ~24h, 3xH100)
sbatch jobs/gepa/006_run_gepa_gaia.job
sbatch jobs/gepa/007_run_gepa_math.job
```

Step by step locally, with a Qwen3-32B reflector already serving on port 8001:

```bash
python scripts/run_gepa.py --mode optimize --config experiments/configs/gepa/gaia.yaml
python scripts/run_gepa.py --mode evaluate --config experiments/configs/gepa/gaia.yaml
python scripts/run_gepa.py --mode diff     --config experiments/configs/gepa/gaia.yaml
```

`--mode diff` shows what the optimiser changed, which is the interesting output
scientifically - a prompt that improved for an unreadable reason is a weaker
result than one whose rewrite you can explain.

## Outputs

Under `experiments/results/gepa/<benchmark>/<TIMESTAMP>_<JOB_ID>/`:

| File | Contents |
|---|---|
| `best_candidate.json` | The optimised `{"system_prompt": ..., "planning_suffix": ...}` |
| `seed_candidate.json` | The starting candidate, for diffing |
| `gepa_results.json` | Held-out test evaluation, in `raw_results.json` format |
| `gepa_state.bin` | Full pickled optimisation state, written after each step - **auto-resumes** if `run_dir` already contains it |
| `generated_best_outputs_valset/` | Per-task best rollouts on the validation set (when `track_best_outputs: true`) |
| `optimize.stderr`, `evaluate.stderr` | Per-step stderr; replayed on failure |

The auto-resume is worth knowing about in both directions: a re-submitted job
continues rather than restarting, which is what you want after a timeout - and
is *not* what you want if you meant to start fresh. Use a new `run_dir` for a
clean run.

---

## Using an optimised prompt

No code changes. Point any inference config at the candidate file:

```yaml
gepa_prompt_path: experiments/results/gepa/gaia/<run>/best_candidate.json
```

This bypasses `PromptBuilder` entirely - `system_prompt` and `planning_suffix`
are read from the file. Because the builder is skipped, the dataset's normal
template is not consulted at all, so a GEPA config's results are comparable to a
normal run only in the sense that everything *else* is identical.

Ready-made configs: `experiments/configs/qwen3/gepa_inference/`.

## Extending

`AgentGEPAAdapter` (`src/gepa_integration/adapter.py`) is the seam: it evaluates
a candidate by running real questions and returns scores plus reflective
feedback. `reflection.py` turns failures into reflector-readable text and
`seed.py` builds the starting candidate (importing `classify_failure` from the
failure-mode analysis, so the seed knows about the taxonomy).

A different prompt optimiser only needs to produce a JSON file with the two keys
- see [guides/add-an-adaptation-method.md](../guides/add-an-adaptation-method.md).
