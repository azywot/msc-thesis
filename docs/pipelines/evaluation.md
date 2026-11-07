# Evaluation pipeline

The default pipeline: run a config against a benchmark, get scored results.
Everything else in `pipelines/` produces an artefact that is ultimately measured
by this one.

---

## One run

```bash
python scripts/run_experiment.py --config experiments/configs/qwen3/agentflow/gaia/qwen8B_subagent_tools_all.yaml
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

## Text-only splits

Two splits drop every question that needs an image, so the text-only Qwen3
models are not penalised for what they cannot see.

| Split | Benchmark | Size | What is excluded |
|---|---|---|---|
| `all_validation_text` | GAIA | 155 | Questions with a `.jpg`, `.jpeg` or `.png` attachment |
| `test_text_subset_200` | HLE | 200 | Multimodal (image) questions |

HLE-text is built from the same seed-0 shuffle of full HLE as `test_subset_200`,
filtered to text-only questions and then truncated to 200. Its first 179
questions are therefore exactly the text-only questions of `test_subset_200`, in
order, so the two sets are nested. GAIA-text rows carry `source_index`, the line
index of the question in `all_validation.jsonl`. The helpers live in
`src/agent_engine/datasets/text_only.py`.

To rebuild the local files:

```bash
python scripts/download_datasets.py --dataset gaia --level all --split validation --text-only --local-only
python scripts/download_datasets.py --dataset hle --split test --subset 200 --seed 0 --text-only --local-only
```

The 24 Qwen3 text configs live in `experiments/configs/qwen3/baseline_text/` and
`experiments/configs/qwen3/agentflow_text/`. They come from the `baseline_text`
and `agentflow_text` suites of `scripts/generate_configs.py`.

## Sampling and seeds

A model entry can set `sampling_by_thinking` with a `thinking` and a
`non_thinking` block (each `temperature`, `top_p`, `top_k`). The block is picked
per call from the prompt's thinking state and replaces the plain
`temperature`/`top_p`/`top_k`. It is vLLM-only; other backends reject it. The text
configs use it so that repeated runs differ. Plain Qwen3 configs without it stay
greedy (temperature 0.0), so repeating them gives identical results.

`--seed N` overrides the config seed (and the seed of every model):

```bash
python scripts/run_experiment.py --config <config>.yaml --seed 3
```

A seeded run directory is named `{split}_seed{N}_{date}_{job}`. Every run (seeded
or not) writes a `config_fingerprint` to `config.json`: a hash of the config with
the seeds, output directory, run name, description, SLURM settings and W&B
settings left out. Runs of one configuration share a fingerprint. Each model entry
in `config.json` also records its `seed`. One sampling seed is used per attempt and
is shared by all requests in that attempt.

To launch several seeds:

```bash
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/baseline_text --dataset gaia --seeds 0,1,2,3,4
./jobs/scripts/launch_experiment.sh <config>.yaml --seed 2     # one config, one seed
python jobs/scripts/generate_job.py <config>.yaml --seed 2               # job script only
```

Each config and seed becomes its own SLURM job (or runs in turn with `--local`).

## pass@k

```bash
python scripts/pass_at_k.py RUN0 RUN1 RUN2 RUN3 RUN4 --k 1 3 5 --by-level --out pass_at_k.json
```

Each `RUN` is a run directory or a `raw_results.json` path. `--k` takes one or
more values (default 1 3 5). The estimator is the unbiased one: with n attempts
and c correct for a question, pass@k = 1 - C(n-c, k) / C(n, k). Questions with
fewer than k attempts are excluded and counted in the output. The script also
prints mean accuracy and its standard deviation over seeds (the sample standard
deviation, with n-1 in the denominator). `--k` values must be at least 1.

It refuses to combine runs unless all of these hold:

- at least two runs, each a different run directory, with distinct seeds (and
  distinct per-model seeds when `config.json` records them);
- the same `config_fingerprint` (a missing fingerprint also fails);
- the same set of question ids, with the same question text;
- no role with effective temperature 0 (greedy runs repeat themselves);
- no duplicate question id inside a run, and every result has `question_id` and `correct`.

`--allow-mismatch` turns only the fingerprint, question-set and question-text
problems into warnings. The run count, duplicate seeds or run directories, the
structural checks (including a result row with an `error` key) and the greedy
check cannot be waived. A warning is also printed when all runs have identical
predictions on every shared question, since sampling may then not be varying with
the seed.

Do not use a config with `subset_num > 0` for pass@k. Such a config picks a
random subset after `set_seed(config.seed)`, so each seed sees different
questions and pass@k will reject the runs. Use a full split (`subset_num: -1`, as
all the text configs do) or a small dedicated split file.

Runtime caveat: attempts share a common search cache; queries not yet cached are fetched independently by each attempt.

---

## What a run produces

Each run creates a timestamped subdirectory under `output_dir/`, e.g.
`all_validation_2026-02-22-22-25-02_<job_id>/` (`all_validation_seed0_<date>_<job_id>/` when run with `--seed 0`):

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
