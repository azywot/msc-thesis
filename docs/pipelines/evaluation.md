# Evaluation pipeline

The default pipeline: run a config against a benchmark, get scored results.
Everything else in `pipelines/` produces an artefact that is ultimately measured
by this one.

---

## One run

```bash
python scripts/run_experiment.py --config experiments/configs/qwen3/agentflow/gaia.yaml
```

`--config` is required. `--output-dir` overrides the config's `output_dir`,
which is the safe way to avoid overwriting a previous run:

```bash
python scripts/run_experiment.py --config <config>.yaml --output-dir ./experiments/results/my_run
```

On a laptop (Apple Silicon, MLX backend), the `local/` configs use small models
and need no cluster:

```bash
python scripts/run_experiment.py --config experiments/configs/local/qwen3_4b_gaia.yaml
```

## A suite

```bash
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline --local
```

Without `--local` each config becomes a SLURM job; with it they run
sequentially in the current shell.

---

## What a run produces

Each run creates a timestamped subdirectory under `output_dir/`, e.g.
`all_validation_2026-02-22-22-25-02_<job_id>/`:

| File | Contents |
|---|---|
| `raw_results.json` | Per-example: question, prediction, ground truth, evaluation, tool calls, token usage, trace |
| `raw_results.partial.json` | Rolling checkpoint written during the run; **deleted on clean completion** |
| `metrics.json` | Aggregate accuracy, EM, F1, per-level breakdown, tool usage |
| `config.json` | The resolved config, so a result is reproducible without hunting for the YAML |
| `experiment.log` | Full run log |

> A surviving `raw_results.partial.json` means the run **did not finish**. It is
> the first thing to check when a result set looks short - the partial file is
> still valid JSON and still analysable, but it is not a complete run.

## Reading the results

```bash
python scripts/analyze_results.py experiments/results/<run>/raw_results.json --by-level --tools
```

`--by-level` gives the stratified breakdown (only meaningful for datasets whose
`DatasetSpec` sets `stratified=True`), `--tools` the per-tool call counts.

`metrics.json` structure:

```
overall:     accuracy, em, f1, num_correct ("N of M"), token_usage
tool_usage:  per-tool totals
per_level:   the same, per stratification bucket (stratified datasets only)
```

`accuracy` comes from each result row's `evaluation.accuracy`, which the
dataset's own `evaluate()` produced. Different benchmarks therefore mean
different things by "accuracy" - GAIA's scorer normalises numbers and lists,
GPQA compares option letters, BigCodeBench executes tests.

### Reading a low score

Before concluding the model is bad, check three things in `raw_results.json`:

1. **`metadata.max_turns_reached`** - the question ran out of turns and was
   force-answered. A cluster of these means `max_turns` is too low or the model
   is looping, not that its reasoning is wrong.
2. **Tool outputs that are error strings.** `url_fetcher` returns its errors *as
   page text* (`"Error fetching …"`), so a dead network looks like unhelpful
   search results rather than a crash.
3. **Empty `sub_goal` fields in `action_history`** (AgentFlow runs) - the model
   is not following the prompt format, which degrades the structured memory it
   sees on every later turn.

---

## The cache

Search results and fetched pages are cached under
`cache/<provider>/<dataset>/`:

```
cache/serper/gaia_validation/
    search_cache.json   query → results
    url_cache.json      url   → page text
    .cache.lock
```

This is what makes re-runs cheap and repeatable - a repeated run of the same
questions costs no API calls. Writes merge rather than overwrite (disk ∪ memory,
memory wins), so parallel SLURM workers sharing a cache directory do not clobber
each other.

Deleting a cache directory is safe; it only costs money and time.

## Failure-mode analysis

Beyond accuracy, runs can be classified into failure modes:

```bash
python -m agent_engine.analysis.failure_modes                       # scans the repo's runs
python -m agent_engine.analysis.failure_modes --output-dir ./out    # where to write the breakdown
```

It takes `--root` (the repo root, correctly defaulted) and `--output-dir`, not
a path to a single `raw_results.json` - it sweeps recorded runs and writes
`breakdown.json` / `breakdown.csv`.

`classify_failure` is **frozen** - the thesis's taxonomy counts come from it and
a characterization fixture replays it over recorded runs. Do not change its
body; add a new function if you need a different taxonomy.

The analyses over recorded runs live in `src/agent_engine/analysis/`
(`eval_runs/`, `fine_tuning/`), with thin CLI shims still at
`scripts/failure_modes/` for anything that referenced the old paths.

## Adding a benchmark

See [guides/add-a-benchmark.md](../guides/add-a-benchmark.md). Downloads:

```bash
python scripts/download_datasets.py --dataset gaia
python scripts/download_datasets.py --dataset bigcodebench --split v0.1.4 --subset 200
```
