# CLAUDE.md — CoSMAS (Collaborative Small-Agent System)

MSc thesis research framework for evaluating multi-agent collaboration with small LLMs.

## Project overview

CoSMAS is a configuration-driven multi-agent framework that compares two execution modes:
- **AgentFlow** (default): planning turn + structured memory loop + explicit sub-goals
- **Baseline**: vanilla LLM-with-tools, growing conversation, no planning

Benchmarks: GAIA, HLE, GPQA, AIME, MuSiQue, BigCodeBench (experiment configs exist); Models: Qwen3 (0.6B–32B). Backends: vLLM (cluster), MLX (Apple Silicon), OpenAI/Anthropic (API).

## Key directories

```
src/agent_engine/     # Main package
  config/             # YAML schema + loader (schema.py)
  core/               # Orchestrator + tool-calling loop
  models/             # vLLM / MLX / API providers
  tools/              # web_search, code_generator, mind_map, text/image inspector
  datasets/           # Loaders + evaluators + metrics
  prompts/            # Prompt templates + builder.py
  external/           # Serper, Tavily, URL fetching
  caching/            # Cache manager

scripts/
  run_experiment.py   # Main runner (requires --config)
  analyze_results.py  # Metrics + breakdowns
  download_datasets.py
  export_prompts.py
  plots/              # Plotting scripts (efficiency_plots, orchestrator_capabilities, etc.)
  tables/             # Table generation scripts

experiments/
  configs/            # YAML experiment configs (baseline/, 1_milestone_...AgentFlow/, local/, datasets/)
  configs/generate_configs.py  # Programmatic config generator
  results/            # Default output root

jobs/                 # SLURM job scripts for Snellius HPC
examples/             # Single-tool sanity check scripts
```

## Setup

**Local (Apple Silicon / MLX):**
```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[mlx]'
cp .env.example .env  # fill in SERPER_API_KEY, HF_TOKEN, etc.
```

**Cluster (conda / vLLM):**
```bash
pip install -e .           # core deps
pip install -e ".[vllm]"   # GPU backend
```

Required env vars (`.env`): `SERPER_API_KEY` or `TAVILY_API_KEY`, `HF_TOKEN` (for gated datasets), optionally `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `WANDB_API_KEY`.

## Running experiments

```bash
# Local MLX
python scripts/run_experiment.py --config experiments/configs/local/qwen3_4b_gaia.yaml

# Override output dir
python scripts/run_experiment.py --config <config.yaml> --output-dir ./experiments/results/my_run

# Batch (SLURM)
./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline
./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline --local  # sequential, no SLURM
```

## Config essentials

- `baseline: true` — skip planning turn, growing conversation (vanilla LLM baseline)
- `tools.direct_tool_call: true` — return raw tool output; `false` = sub-agent analyses it first
- `tools.return_code: true` — `code_generator` returns generated code instead of executing it (required for BigCodeBench)
- `thinking_mode` — `NO` / `ORCHESTRATOR_ONLY` / `SUBAGENTS_ONLY` / `ALL`
- `batch_size` — questions per batch (`-1` = all; `1` = no batching)
- `web_tool_provider` — `serper` (default, fetches full pages) or `tavily` (pre-cleaned content)

Regenerate all configs after changing `generate_configs.py`:
```bash
python experiments/configs/generate_configs.py
```

## Outputs (per run, under `output_dir/`)

`raw_results.json`, `metrics.json`, `config.json`, `experiment.log`, `raw_results.partial.json` (rolling checkpoint, deleted on clean finish).

Analyse:
```bash
python scripts/analyze_results.py experiments/results/<run>/raw_results.json --by-level --tools
```

## Code style

- Python 3.11+, `black` (line-length 100), `isort` (black profile)
- Tests: `pytest` — run from repo root
- No type annotations required (`disallow_untyped_defs = false`)

## Important notes

- If multiple model roles share the same `path_or_id`, the runner reuses the vLLM instance (no duplicate GPU memory)
- Prompt templates live in `src/agent_engine/prompts/`; AgentFlow uses `*_dataset*.yaml`, baseline uses `*_baseline*.yaml`
- Cache: `./cache/serper/<dataset>/` or `./cache/tavily/<dataset>/`
- HPC cluster is Snellius (SURF); working dir on cluster: `$HOME/azywot/msc-thesis/`
- **BigCodeBench**: split is a version string (e.g. `v0.1.4_subset_200`); the `test` field is a `unittest.TestCase` class run via `unittest.main`; `return_code: true` must be set so the tool returns code rather than executing it — `generate_configs.py` sets this automatically for all BigCodeBench tool-using configs; download via `python scripts/download_datasets.py --dataset bigcodebench --split v0.1.4 --subset 200`
- **BigCodeBench scorer** (`src/agent_engine/datasets/evaluators/bigcodebench_scorer.py`): assembles `prediction + test harness + unittest.main(...)` into a temp file and runs it in a subprocess; uses `re.search` to detect whether the prediction already contains the full function definition (with imports) and avoids double-prepending the `code_prompt` stub
