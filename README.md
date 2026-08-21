# CoSMAS - Collaborative Small-Agent System

MSc thesis research framework for evaluating **multi-agent collaboration with
small LLMs**.

CoSMAS asks one question: when a small model is given planning, structured
memory and tool sub-agents, does it actually do better - or does the machinery
just add cost? To answer that, it runs the same model, on the same benchmark,
with the same tools, in two modes:

- **AgentFlow** - a planning turn, structured memory rebuilt each turn, explicit
  sub-goals.
- **Baseline** - vanilla LLM-with-tools, a growing conversation, no planning.

One YAML key (`baseline: true`) switches between them. Everything else is held
constant, which is what makes the comparison worth anything.

**Benchmarks:** GAIA, HLE, GPQA, AIME, MATH500, AMC, MuSiQue, BigCodeBench.
**Models:** Qwen3 (0.6B–32B), DeepSeek-R1-Distill (7B, 32B), OLMo 3 (Think /
Instruct). **Backends:** vLLM (cluster), MLX (Apple Silicon), OpenAI / Anthropic.

![Framework diagram](data/static/FrameworkDiagram.jpg)

---

## Documentation

| I want to… | Read |
|---|---|
| Understand how it works | [docs/architecture.md](docs/architecture.md) |
| Know what a config key does | [docs/configuration.md](docs/configuration.md) |
| Run an evaluation | [docs/pipelines/evaluation.md](docs/pipelines/evaluation.md) |
| Optimise prompts (GEPA) | [docs/pipelines/gepa.md](docs/pipelines/gepa.md) |
| Fine-tune (SFT / RL / Prefix-RFT) | [docs/pipelines/sft.md](docs/pipelines/sft.md), [docs/pipelines/rl.md](docs/pipelines/rl.md), [docs/pipelines/prefix-rft.md](docs/pipelines/prefix-rft.md) |
| Add a benchmark | [docs/guides/add-a-benchmark.md](docs/guides/add-a-benchmark.md) |
| Add a tool or sub-agent | [docs/guides/add-a-tool-or-subagent.md](docs/guides/add-a-tool-or-subagent.md) |
| Add a model family | [docs/guides/add-a-model-family.md](docs/guides/add-a-model-family.md) |
| Add an adaptation method | [docs/guides/add-an-adaptation-method.md](docs/guides/add-an-adaptation-method.md) |
| Change the RL / GEPA training data | [docs/guides/change-training-data.md](docs/guides/change-training-data.md) |
| Contribute code | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Know what is broken | [docs/known-issues.md](docs/known-issues.md) |

`docs/archive/` holds superseded planning and status documents, each with a
HISTORICAL banner. They record reasoning worth keeping; their paths and numbers
are stale by definition.

---

## Install

**Cluster (conda + vLLM)** - one job builds the environment:

```bash
sbatch jobs/001_setup.job     # creates the `agent_engine` conda env, installs the project
squeue -u $USER
# log: out/setup/msc_thesis_env_setup_<job_id>.log
```

**Laptop (Apple Silicon, MLX):**

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[mlx]'
```

**Plain:**

```bash
pip install -e .          # core
pip install -e '.[vllm]'  # GPU backend
pip install -e '.[dev]'   # tests, black, isort
```

Then set API keys:

```bash
cp .env.example .env
```

`SERPER_API_KEY` **or** `TAVILY_API_KEY` is required (whichever matches
`web_tool_provider`; Serper is the default). `HF_TOKEN` is needed for gated
datasets. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `WANDB_API_KEY` are optional.

Datasets:

```bash
sbatch jobs/002_download_datasets.job          # all of them, on the cluster
python scripts/download_datasets.py --dataset gaia     # or one at a time
```

---

## Run one experiment

```bash
python scripts/run_experiment.py --config experiments/configs/qwen3/agentflow/gaia/qwen8B_subagent_tools_all.yaml
```

`--config` is required. Override the output location rather than overwriting a
previous run:

```bash
python scripts/run_experiment.py --config <config>.yaml --output-dir ./experiments/results/my_run
```

On a laptop, the `local/` configs use small models and need no cluster:

```bash
python scripts/run_experiment.py --config experiments/configs/local/qwen3_4b_gaia.yaml
```

A whole suite:

```bash
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline --local
```

Each run writes a timestamped directory containing `raw_results.json`,
`metrics.json`, `config.json` and `experiment.log`. A surviving
`raw_results.partial.json` means the run **did not finish**.

```bash
python scripts/analyze_results.py experiments/results/<run>/raw_results.json --by-level --tools
```

See [docs/pipelines/evaluation.md](docs/pipelines/evaluation.md) for what the
numbers mean and how to read a bad one.

---

## Repo map

```
src/
  agent_engine/        the framework
    config/            YAML schema + loader
    core/              orchestrator, execution state, tool base, batching
    models/            model families + vLLM / MLX / API providers
    tools/             web_search, code_generator, mind_map, text/image inspector, registry
    datasets/          loaders, DatasetSpec table, evaluators
    prompts/           templates/ + builder
    runner/            experiment loop, providers, tool wiring, metrics
    external/          serper, tavily, url_fetcher
    caching/           search + URL cache
    analysis/          failure-mode classifier and analyses over recorded runs
    utils/             tool-call parsing, logging, seeding
  fine_tuning/         RL (GRPO) + SFT; agentflow/ is vendored - see its VENDORED.md
  gepa_integration/    prompt optimisation
  verl_ext/            local verl extensions (folded SFT dataset, checkpoint utils)

scripts/               run_experiment, analyze_results, generate_configs,
                       download_datasets, export_prompts, GEPA + SFT + RL tooling,
                       plots/, tables/, failure_modes/ (CLI shims)
experiments/
  configs/             YAML configs by model family; most are generated
  scripts/             run_all_in_folder.sh
  results/             default output root
jobs/                  SLURM scripts - 001-007 setup, fine_tuning/, gepa/, grpo_inference/
tests/                 unit/ + characterization/ (behaviour-locking fixtures)
examples/              one runnable script per tool
docs/                  see the table above
```

---

## Concepts worth knowing before changing anything

**Everything is batched.** The unit of execution is a batch of questions, not a
question. Each turn is one `generate()` call across every unfinished question.
`batch_size: 1` turns this off, which is the setting for debugging.

**AgentFlow never shows the model its own past output.** The prompt is rebuilt
each turn from a query analysis plus an action history. Baseline grows a
conversation. They use *different prompt template files* - `*_dataset*.yaml` vs
`*_baseline*.yaml` - so changing one does not change the other.

**Most configs are generated.** Editing a generated YAML works until someone
runs `python scripts/generate_configs.py`, which silently reverts it. Change the
generator.

**A typo'd config key is silently ignored.** The schema does not forbid extra
keys, so a misspelling means the default is used and nothing warns you.

**The tool and dataset seams are registries, not if/elif chains.** Adding either
means adding a decorated factory or a spec row - never editing the orchestrator.

**Prefix-RFT has two prefix modes, and they are not interchangeable.** A teacher
demonstration is a sequence of decisions, so "a prefix of it" can be measured two
ways. `steps` (the default) replays a whole number of teacher *decisions*:
`k = clamp(floor(l * m), 0, m - 1)`. `tokens` is the paper's own measure: a token
fraction of the concatenated demonstration, which splits the decision that straddles
the budget so the model finishes a turn the teacher started. Three differences that
change how results compare:

- **Coverage differs unless you gate it.** `steps` cannot prefix the 273
  single-decision demonstrations at all (`k <= m-1 = 0`), so 1085 of 1358 are
  prefixable. `tokens` can split a single decision, so all 1358 are.
  `prefix_rft.min_demo_decisions: 2` makes `tokens` skip the same ones, which is what
  the shipped token config does so a steps-vs-tokens comparison is not confounded by
  which questions carry a prefix.
- **The ceiling differs.** `steps` is clamped to `m-1` decisions, so on the average
  3-decision demonstration it can never replay more than ~67% of the teacher, while
  `tokens` reaches 95%. `steps` is systematically less demonstration-heavy early on.
- **Only `tokens` sends a replayed turn to vLLM**, via `continue_final_message`. Whole
  replays are served locally and never leave the worker.

Pick with `--prefix-mode` on `scripts/launch_verl.py`, which is the driver. The rollout
workers infer the mode from the key the driver dispatches (`prefix_k` or `prefix_l`), so
there is no second setting to keep in sync. See
[docs/pipelines/prefix-rft.md](docs/pipelines/prefix-rft.md).

**One rollout in eight carries a prefix, and that is the paper's own setting.** A
prefixed rollout *replaces* an on-policy one rather than adding to it, so the rollout
budget is unchanged (paper A.2). This makes demonstration tokens roughly 5-10% of the
batch, which is the paper's own operating range and where it reports 51.8 against plain
RFT's 45.5. Run 012 measured 0.067. A single-digit `off_ratio` is therefore the target,
not a sign the prefix is too weak to matter. Separately, 1358 of the 1800 training
questions have a demonstration at all; the paper's Table 2 validates the method down to
1% coverage, so that is not the binding constraint either.

**The prefix schedule never becomes pure RL.** `l ~ U(low_t, 0.95)` with `low_t`
decaying 0.95 -> 0.05 moves training from demonstration-heavy to exploration-heavy, but
only the *lower* bound decays: mean `l` goes 0.95 -> 0.50 over the run, not to 0.05. One
rollout in eight is prefixed throughout, and it is trained with the RL loss under the
entropy filter, never a separate imitation loss.

### Prefix-RFT config keys

**Start by copying `experiments/configs/fine_tuning/config_prefix_rft.yaml`** (step
mode) or `config_prefix_rft_tokens.yaml` (token mode) rather than assembling one from
this table. Both are complete and every value cites the paper.

**Prerequisite:** the demonstration store must exist before any of this runs. Build it
with `sbatch jobs/fine_tuning/008_build_prefix_demos.job`.

Prefix-RFT is configured in **two blocks of the same file**, and this catches people
out. `env:` is read by both launcher scripts as environment variables; `python_args:`
becomes Hydra overrides and reaches the training driver only. Three settings live in
`env:`:

| `env:` key | Accepted values | Read by | Meaning |
|---|---|---|---|
| `PREFIX_RFT` | `"true"` / `"false"` | both launchers | The real master switch. Sends `launch_verl.py` to `verl_ext.prefix_rft` and `train_orchestrator.py` to `PrefixOrchestratorRollout`. Without it the `prefix_rft.*` keys below are inert and you get a plain GRPO run. |
| `PREFIX_DEMOS_PATH` | path to a parquet | rollout workers | Where the workers load the demonstration store. **Must match `prefix_rft.demos_path` below.** The workers ignore the Hydra key, so setting only that one leaves them on the default path and the driver dispatches prefixes for a store the workers never opened. `train_orchestrator.py` refuses to start if the two disagree. |
| `BASE_MODEL` | HF model id | rollout workers | Tokenizer used to build replayed and split turns. Must be the model being trained. |

Everything else is under `prefix_rft.` in `python_args:`, or passed as a Hydra override.

| Key | Accepted values | Default | Meaning |
|---|---|---|---|
| `enable` | `true` / `false` | `true` | Master switch. Also needs `PREFIX_RFT=true` in `env:` so both launchers take the Prefix-RFT path. |
| `mode` | `steps` / `tokens` | `steps` | How prefix length is measured. Anything else raises at dispatch. |
| `demos_path` | path to a parquet | `data/training/prefix_rft/prefix_demos.parquet` | The demonstration store, built by `scripts/build_prefix_demos.py`. |
| `min_demo_decisions` | int >= 1 | `1` | Fewest teacher decisions a question needs before it can carry a prefix. `steps` is structurally 2 already; set `tokens` to 2 for a controlled comparison against it, or leave 1 for the paper's full coverage. |
| `n_prefixed_rollouts` | int, `0` to `rollout.n` | `1` | Prefixed rollouts per prompt. Paper A.2 uses 1 of 8; a prefixed rollout replaces an on-policy one rather than adding to them. |
| `high` | float in `[0, 1]` | `0.95` | Upper bound of the `l` draw. Paper A.2. |
| `low_init` | float in `[0, 1]` | `0.95` | Lower bound at step 0. |
| `low_target` | float in `[0, 1]` | `0.05` | Lower bound at the end of the cosine decay. Must differ from `low_init`. |
| `sampler_alpha` / `sampler_beta` | float > 0 | `1.0` / `1.0` | Beta shape for the `l` draw; `1, 1` is the uniform draw A.2 specifies. |
| `entropy_keep_ratio` | float in `[0, 1]` | `0.2` | Share of prefix tokens keeping their advantage, highest-entropy first. Paper's top 20%. Also set `+actor_rollout_ref.prefix_entropy_keep_ratio` to the same value: the actor reads it from there. |
| `singleton_baseline` | `none` / `group` | `none` | Baseline for a single-rollout group. `none` reproduces the reference implementation. |
| `seed` | int | `42` | Seeds the schedule's sampler only. |

One more key sits outside the `prefix_rft.` block:
`+actor_rollout_ref.prefix_entropy_keep_ratio`, which must equal `entropy_keep_ratio`.
The actor reads it from there because verl turns `actor_rollout_ref.actor` into a
dataclass that rejects undeclared keys. The leading `+` is required: it tells Hydra to
append a key that is not in verl's schema, and without it the run fails at composition.

Flags on `scripts/launch_verl.py`: `--prefix-mode {steps,tokens}` overrides `mode`;
`--dry-run` composes every override and exits without touching a GPU, which is the
cheapest way to check a config you have edited.

---

## Tests

```bash
pytest -q                      # from the repo root
pytest tests/unit -q
pytest tests/characterization -q
```

`tests/characterization/` locks current behaviour against committed fixtures. If
one fails, a refactor changed behaviour - investigate before regenerating the
fixture. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Cluster notes (Snellius / SURF)

Working directory is `$HOME/azywot/msc-thesis/`.

| Job | Purpose |
|---|---|
| `jobs/001_setup.job` | Create the conda env, install the project |
| `jobs/002_download_datasets.job` | Download benchmark datasets |
| `jobs/003_run_examples.job` | Smoke-test a single example |
| `jobs/004_export_env.job` | Export conda env YAMLs |
| `jobs/005_export_prompts.job` | Export prompt templates + tool schemas |
| `jobs/006_create_configs.job` | Regenerate all experiment configs |
| `jobs/007_add_bigcodebench_libs.job` | Extra libraries for BigCodeBench |
| `jobs/fine_tuning/` | SFT and RL training - see the pipeline docs |
| `jobs/gepa/` | GEPA data prep, smoke tests, optimisation |
| `jobs/grpo_inference/` | Evaluation of base / SFT / GRPO checkpoints |

Overrides via `sbatch --export=ALL,...`: `ENV_NAME`, `PROJECT_DIR`, `DATA_DIR`.
