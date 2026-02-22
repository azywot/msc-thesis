# 🌌 Collaborative Small-Agent System (CoSMAS)

CoSMAS is a configuration-driven multi-agent research framework for investigating how small, collaborative language models can be composed to solve complex tasks efficiently. Its primary focus is the design and evaluation of collaboration mechanisms, enabling systematic comparison between single-model baselines and cooperative multi-agent configurations under controlled experimental conditions.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Installation](#installation)
3. [HPC / Cluster setup](#hpc--cluster-setup)
4. [Running experiments](#running-experiments)
5. [Examples](#examples)
6. [Configuration reference](#configuration-reference)
7. [Tools](#tools)
8. [Datasets](#datasets)
9. [Outputs](#outputs)

---

## Project structure

```
msc-thesis/
├── src/agent_engine/          # Main Python package
│   ├── config/                # YAML schema + loader
│   ├── core/                  # Orchestrator + tool-calling loop
│   ├── models/                # vLLM + API providers + locking/reuse
│   ├── tools/                 # web_search, code_generator, context_manager, inspectors
│   ├── datasets/              # loaders + evaluators + metrics
│   ├── prompts/               # prompt templates + builders
│   ├── external/              # Serper + URL fetching utilities
│   ├── caching/               # cache manager(s)
│   └── utils/                 # parsing/logging helpers
│
├── scripts/
│   ├── run_experiment.py      # Main runner (requires --config)
│   ├── analyze_results.py     # Metrics + breakdowns
│   ├── download_datasets.py   # Fetch/prepare datasets
│   └── export_prompts.py      # Dump prompt templates + tool schemas to JSON
│
├── experiments/
│   ├── configs/               # Experiment YAMLs (by dataset)
│   └── results/               # Default output root
│
├── jobs/                      # SLURM job scripts + HPC tooling
├── examples/                  # Small runnable single-tool examples
├── pyproject.toml
├── requirements.txt
└── environment.yml            # Conda env for HPC
```

**Common navigation:**

| Goal | Where to look |
|---|---|
| Run or configure an experiment | `scripts/run_experiment.py` + `experiments/configs/` |
| Change config schema / defaults | `src/agent_engine/config/schema.py` |
| Model providers, batching, GPU reuse | `src/agent_engine/models/` |
| Tool implementations | `src/agent_engine/tools/` |
| Dataset loaders + metrics | `src/agent_engine/datasets/` |
| SLURM job scripts | `jobs/` |
| Single-tool sanity checks | `examples/` |

---

## Installation

```bash
cd msc-thesis
pip install -e .
```

Optional dev extras:

```bash
pip install -e ".[dev]"
```

For HPC/Conda, follow the [cluster setup](#hpc--cluster-setup) section below.

---

## HPC / Cluster setup

Run these four steps once on Snellius (SURF) before launching any experiment. <br>
All commands assume `$HOME/thesis/msc-thesis/` as the working directory.

### 1. Build the conda environment

```bash
sbatch jobs/001_setup.job
squeue -u $USER    # monitor progress
```

Creates the `agent_engine` conda environment and installs the project.  
Log: `out/setup/msc_thesis_env_setup_<job_id>.log`

### 2. Set API keys

```bash
cp .env.example .env
nano .env
```

Minimum required:

```bash
SERPER_API_KEY=your_serper_key_here
```

Optional:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
```

### 3. Download datasets

```bash
sbatch jobs/002_download_datasets.job
```

Downloads all benchmark datasets to `/scratch-shared/$USER/data/`.  
Log: `out/datasets/download_datasets_<job_id>.log`

### 4. Verify the setup

```bash
sbatch jobs/003_test_simple.job
```

Runs a short single-example test using `experiments/configs/gaia/test_subagent.yaml`.  
Log: `out/test/example_subagent_<job_id>.log`

### Job files reference

| File | Purpose | Log |
|------|---------|-----|
| `jobs/001_setup.job` | Create conda env + install project | `out/setup/msc_thesis_env_setup_<job_id>.log` |
| `jobs/002_download_datasets.job` | Download benchmark datasets | `out/datasets/download_datasets_<job_id>.log` |
| `jobs/003_test_simple.job` | Smoke-test a single example | `out/test/example_subagent_<job_id>.log` |
| `jobs/004_export_env.job` | Export conda env YAMLs | `out/export_env/export_env_<job_id>.log` |
| `jobs/005_export_prompts.job` | Export prompt templates + tool schemas | `out/export_prompts/export_prompts_<job_id>.log` |

Optional overrides (via `sbatch --export=ALL,...`): `ENV_NAME`, `PROJECT_DIR`, `DATA_DIR`.

---

## Running experiments

### Locally

```bash
# Set required key
export SERPER_API_KEY="..."

# Run with a config file
python scripts/run_experiment.py --config experiments/configs/gaia/baseline.yaml

# Override output directory
python scripts/run_experiment.py --config experiments/configs/gaia/baseline.yaml \
    --output-dir ./experiments/results/my_run
```

### On SLURM

```bash
# Convenience wrapper: generates a job file from the config and submits it
./jobs/submit_job.sh experiment experiments/configs/gaia/baseline.yaml

# Or manually:
python jobs/scripts/generate_job.py experiments/configs/gaia/baseline.yaml
sbatch jobs/generated/gaia_qwen3_baseline.job
```


### Available experiment configs

```
experiments/configs/
├── gaia/
│   ├── baseline.yaml        # Full GAIA validation run
│   ├── test_direct.yaml     # Quick test — direct tool mode
│   └── test_subagent.yaml   # Quick test — sub-agent mode
└── template.yml             # Annotated template for new configs
```

To create a new config:

```bash
cp experiments/configs/gaia/baseline.yaml experiments/configs/gaia/my_run.yaml
nano experiments/configs/gaia/my_run.yaml
./jobs/submit_job.sh experiment experiments/configs/gaia/my_run.yaml
```

---

## Examples

The `examples/` directory contains one script per tool. Each script runs a single
question chosen to force the model to call the tool under test. They are the
recommended sanity check before launching a full experiment.

Run from the `msc-thesis/` root:

```bash
python examples/example_web_search.py        # web_search
python examples/example_code_generator.py    # code_generator
python examples/example_text_inspector.py    # text_inspector (reads fixtures/sample_document.txt)
python examples/example_image_inspector.py   # image_inspector (generates test PNG automatically)
python examples/example_context_manager.py   # web_search + context_manager (GraphRAG)
```

Prerequisites:

```bash
export SERPER_API_KEY="<your-key>"
export HF_HOME="/path/to/hf_cache"   # must contain Qwen/Qwen3-4B
```

Each script writes its output to `experiments/results/examples/<tool_name>/`:
- `result.json` — question, answer, turns used, tool call counts
- `trace.json` — full message + tool call history for debugging
- `example.log` — human-readable execution log

| Script | Tool tested |
|---|---|
| `example_web_search.py` | `web_search` |
| `example_code_generator.py` | `code_generator` |
| `example_text_inspector.py` | `text_inspector` |
| `example_image_inspector.py` | `image_inspector` |
| `example_context_manager.py` | `web_search` + `context_manager` |

---

## Configuration reference

Experiments are defined in YAML. A minimal example:

```yaml
name: "gaia_qwen3_baseline"
description: "GAIA validation — direct tool mode"

models:
  orchestrator:
    family: "qwen3"
    path_or_id: "Qwen/Qwen3-32B"
    role: "orchestrator"
    tensor_parallel_size: 2
    gpu_ids: [0, 1]

tools:
  enabled_tools: ["web_search", "code_generator"]
  direct_tool_call: true

dataset:
  name: "gaia"
  split: "all_validation"

max_turns: 15
thinking_mode: "ORCHESTRATOR_ONLY"
output_dir: "./experiments/results/gaia_baseline"
cache_dir: "./cache"
```

See `experiments/configs/template.yml` for a fully annotated version. Schema and defaults live in:
- `src/agent_engine/config/schema.py` — experiment / tools / dataset fields
- `src/agent_engine/models/base.py` — model generation defaults

### Key options

| Option | Values | Description |
|---|---|---|
| `tools.direct_tool_call` | `true` / `false` | Direct mode returns raw tool output to the planner; sub-agent mode uses a second LLM to analyse it first |
| `thinking_mode` | `NO` / `ORCHESTRATOR_ONLY` / `SUBAGENTS_ONLY` / `ALL` | Controls which roles emit extended reasoning (requires a thinking-capable model) |
| `batch_size` (config) | integer | Questions per batch (-1 = all; 1 = no batching) |

If multiple roles share the same `path_or_id`, the runner reuses the loaded vLLM instance and serialises access with per-model locks — no duplicate GPU memory.

---

## Tools

Enabled via `tools.enabled_tools` in the config:

| Tool | Description |
|---|---|
| `web_search` | Serper API search + URL fetching; optional LLM-based result analysis in sub-agent mode |
| `code_generator` | Execute Python in a subprocess; LLM generates the code in sub-agent mode |
| `context_manager` | Persistent per-question memory with optional GraphRAG indexing |
| `text_inspector` | Read and optionally analyse text files (PDF, DOCX, XLSX, CSV, …) |
| `image_inspector` | Vision-language analysis of images; requires a VLM in the config |

---

## Datasets

| Name | Key |
|---|---|
| GAIA | `gaia` |
| GPQA | `gpqa` |
| MATH500 | `math500` |
| AIME | `aime` |
| AMC | `amc` |
| Natural Questions | `nq` |
| TriviaQA | `triviaqa` |
| HotpotQA | `hotpotqa` |
| MuSiQue | `musique` |
| Bamboogle | `bamboogle` |
| 2WikiMultiHopQA | `2wiki` |

Download datasets before running:

```bash
python scripts/download_datasets.py --dataset gaia --split validation
```

---

## Outputs

Each run creates a timestamped subdirectory under `output_dir/`
(e.g. `all_validation_2026-02-22-22-25-02_<job_id>/`):

| File | Contents |
|---|---|
| `raw_results.json` | Per-example results: question, prediction, ground truth, metrics, tool calls |
| `raw_results.partial.json` | Rolling checkpoint written during the run; deleted on clean completion |
| `metrics.json` | Aggregate accuracy, EM, F1 |
| `config.json` | Serialised experiment config for reproducibility |
| `experiment.log` | Full runner log |

Analyse results:

```bash
python scripts/analyze_results.py experiments/results/<run_dir>/raw_results.json
python scripts/analyze_results.py experiments/results/<run_dir>/raw_results.json --by-level
python scripts/analyze_results.py experiments/results/<run_dir>/raw_results.json --tools
```

To log to W&B: set `use_wandb: true` + `wandb_project: <name>` in the YAML and provide `WANDB_API_KEY`.

---


### SLURM quick reference

```bash
scontrol show job <job_id>   # full job details
sacct -j <job_id>            # accounting info
scancel <job_id>             # cancel one job
scancel -u $USER             # cancel all your jobs
```
