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
7. [Baseline vs. AgentFlow](#baseline-vs-agentflow)
8. [Tools](#tools)
9. [Datasets](#datasets)
10. [Outputs](#outputs)

---

## Project structure

```
msc-thesis/
├── src/agent_engine/          # Main Python package
│   ├── config/                # YAML schema + loader
│   ├── core/                  # Orchestrator + tool-calling loop
│   ├── models/                # vLLM + MLX + API providers + locking/reuse
│   ├── tools/                 # web_search, code_generator, mind_map, inspectors
│   ├── datasets/              # loaders + evaluators + metrics
│   ├── prompts/               # prompt templates + builders
│   ├── external/              # Serper, Tavily + URL fetching utilities
│   ├── caching/               # cache manager(s)
│   └── utils/                 # parsing/logging helpers
│
├── scripts/
│   ├── run_experiment.py      # Main runner (requires --config)
│   ├── analyze_results.py     # Metrics + breakdowns (WIP)
│   ├── download_datasets.py   # Fetch/prepare datasets
│   └── export_prompts.py      # Dump prompt templates + tool schemas to JSON
│
├── experiments/
│   ├── configs/               # Experiment YAMLs organised by suite
│   │   ├── generate_configs.py    # Programmatic config generator (all suites)
│   │   ├── datasets/              # Hand-crafted per-dataset configs (test, dev)
│   │   │   └── gaia/  gpqa/  hle/  aime/  musique/
│   │   ├── baseline/              # Baseline suite (32B, no planning/structured memory)
│   │   ├── 1_milestone_no_img_no_mindmap_AgentFlow/  # AgentFlow suite (8B + 32B)
│   │   ├── local/             # MacBook/MLX configs (Qwen3-0.6B, 4B)
│   │   └── template.yml       # Annotated template for new configs
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
All commands assume `$HOME/azywot/msc-thesis/` as the working directory.

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

Web search (one required based on `web_tool_provider` in config, default is Serper):

```bash
SERPER_API_KEY=your_serper_key_here  # For web_tool_provider: "serper"
TAVILY_API_KEY=your_tavily_key_here  # For web_tool_provider: "tavily"
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
sbatch jobs/003_run_examples.job
```

Runs a short single-example test using `experiments/configs/datasets/gaia/test_subagent.yaml`.  
Log: `out/test/example_subagent_<job_id>.log`

### Job files reference

| File | Purpose | Log |
|------|---------|-----|
| `jobs/001_setup.job` | Create conda env + install project | `out/setup/msc_thesis_env_setup_<job_id>.log` |
| `jobs/002_download_datasets.job` | Download benchmark datasets | `out/datasets/download_datasets_<job_id>.log` |
| `jobs/003_run_examples.job` | Smoke-test a single example | `out/test/example_subagent_<job_id>.log` |
| `jobs/004_export_env.job` | Export conda env YAMLs | `out/export_env/export_env_<job_id>.log` |
| `jobs/005_export_prompts.job` | Export prompt templates + tool schemas | `out/export_prompts/export_prompts_<job_id>.log` |
| `jobs/006_create_configs.job` | Regenerate all experiment configs | `out/create_configs/create_configs_<job_id>.log` |

Optional overrides (via `sbatch --export=ALL,...`): `ENV_NAME`, `PROJECT_DIR`, `DATA_DIR`.

---

## Running experiments

### Locally (Apple Silicon — MLX)

Run small quantised models on a MacBook using the MLX backend. No GPU/CUDA required.

#### 1. Install MLX dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[mlx]'
```

#### 2. Set API keys in `.env`

```bash
cp .env.example .env
# Edit .env and fill in at minimum:
SERPER_API_KEY=...   # or TAVILY_API_KEY if using tavily
WANDB_API_KEY=...    # optional, if use_wandb: true
HF_TOKEN=...         # required for gated datasets (GAIA, GPQA, HLE)
```

#### 3. Download datasets

```bash
HF_HUB_DISABLE_XET_TRANSFER=1 python scripts/download_datasets.py \
    --dataset gaia --level all --split validation --output-dir ./data
```

#### 4. Run

```bash
# Qwen3-0.6B (fastest, least RAM)
python scripts/run_experiment.py --config experiments/configs/local/qwen3_0.6b_gaia.yaml

# Qwen3-4B (better quality)
python scripts/run_experiment.py --config experiments/configs/local/qwen3_4b_gaia.yaml
```

Pre-built local configs are in `experiments/configs/local/`. Key differences from cluster configs:

| Option | Local (MLX) | Cluster (vLLM) |
|---|---|---|
| `backend` | `mlx` | `vllm` (default) |
| `batch_size` | `1`–`5` (RAM-limited) | `-1` (all at once) |
| SLURM fields | ignored | used by job scripts |

> **Batching:** set `batch_size: N` (e.g. `3`) to process N questions in parallel on Apple Silicon's integrated GPU. Higher values use more RAM — start small and increase as needed.

---

### Locally (GPU — vLLM)

```bash
# Set required key (Serper or Tavily, depending on config)
export SERPER_API_KEY="..."  # If using web_tool_provider: "serper"
# OR
export TAVILY_API_KEY="..."  # If using web_tool_provider: "tavily"

# Run with a config file
python scripts/run_experiment.py --config experiments/configs/datasets/gaia/baseline.yaml

# Override output directory
python scripts/run_experiment.py --config experiments/configs/datasets/gaia/baseline.yaml \
    --output-dir ./experiments/results/my_run
```

### On SLURM

```bash
# Single config — generates a job file and submits it
./jobs/submit_job.sh experiment experiments/configs/datasets/gaia/baseline.yaml

# Or manually:
python jobs/scripts/generate_job.py experiments/configs/datasets/gaia/baseline.yaml
sbatch jobs/generated/gaia_qwen3_baseline.job
```

### Batch runs with `run_all_in_folder.sh`

Run all YAML configs found recursively under any folder — submits each as a separate SLURM job by default, or runs them sequentially with `--local`.

```bash
# Run example

# Run an entire suite
./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline
./experiments/scripts/run_all_in_folder.sh experiments/configs/1_milestone_no_img_no_mindmap_AgentFlow

# Run a single dataset within a suite
./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline/gaia

# Run locally (sequential, no SLURM)
./experiments/scripts/run_all_in_folder.sh experiments/configs/baseline/gaia --local
```

The script walks the folder recursively, so it works at any depth and picks up all `.yaml`/`.yml` files.


### Available experiment configs

Configs are organised into **suites** — self-contained families of experiments sharing the same naming scheme, output root, and model/dataset selection.

```
experiments/configs/
├── generate_configs.py                         # Unified config generator
├── datasets/                                   # Hand-crafted per-dataset configs
│   ├── gaia/   gpqa/   hle/   aime/   musique/ # test_direct, test_subagent, etc.
├── baseline/                                   # Baseline suite (vanilla LLM + tools)
│   ├── gaia/   hle/   gpqa/   aime/   musique/ # 4 configs × 5 datasets = 20 files
│   └── run_all.sh
├── 1_milestone_no_img_no_mindmap_AgentFlow/    # AgentFlow suite (full system)
│   ├── gaia/   hle/   gpqa/   aime/   musique/ # 12 configs × 5 datasets = 60 files
│   └── run_all.sh
├── local/                                      # MacBook/MLX configs
└── template.yml                                # Annotated template
```

#### Generating configs

```bash
# Regenerate all suites (baseline + agentflow)
python experiments/configs/generate_configs.py

# Regenerate a single suite
python experiments/configs/generate_configs.py --suite baseline
python experiments/configs/generate_configs.py --suite agentflow
```

Or via SLURM:

```bash
sbatch jobs/006_create_configs.job
```

#### Adding or customising a suite

All suite parameters live in the `SUITES` dict at the top of `generate_configs.py`:

```python
SUITES = {
    "my_suite": {
        "description_tag": "[My Suite]",
        "name_prefix":     "my_prefix",
        "output_dir_root": "./experiments/results/my_suite",
        "config_subdir":   "my_suite",       # → experiments/configs/my_suite/
        "baseline":        False,            # true = skip planning turn
        "variants":        VARIANTS_32B,     # or VARIANTS_ALL
        "num_gpus":        2,
        "wandb_project":   "my-wandb-project",
        "split_overrides": {},               # per-dataset split overrides
    },
}
```

Re-run the generator after any change to rebuild all YAML files.

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
python examples/example_mind_map.py   # web_search + mind_map (GraphRAG)
```

Prerequisites:

```bash
export SERPER_API_KEY="<your-key>"  # Or TAVILY_API_KEY, depending on config
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
| `example_mind_map.py` | `web_search` + `mind_map` |

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
| `baseline` | `true` / `false` | When `true`, skips the planning turn and uses a growing conversation instead of structured AgentFlow memory. Used for the vanilla LLM-with-tools comparison. Defaults to `false`. |

If multiple roles share the same `path_or_id`, the runner reuses the loaded vLLM instance and serialises access with per-model locks — no duplicate GPU memory.

---

## Baseline vs. AgentFlow

The framework supports two execution modes that are compared in the experiments.

### AgentFlow (default, `baseline: false`)

The full system, inspired by the original [AgentFlow](https://github.com/lupantech/AgentFlow) architecture.

**Turn 0 — Planning:** Before any tool use, the model receives the question and is asked to analyse it: identify objectives, list relevant tools, and sketch an approach. The output is stored as `query_analysis`.

**Turns 1–N — Structured memory loop:** Each subsequent turn reconstructs a fresh `[system, user]` prompt from structured state rather than appending to a growing conversation:

```
<original question>

**Query Analysis:**
<planning turn output>

**Previous Steps:**
Action Step 1:
  - Tool: web_search
  - Sub-goal: <explicit intermediate goal the model stated>
  - Command: <tool call JSON>
  - Result: <tool output>
...
```

The model is instructed to emit a `<sub_goal>` tag before every `<tool_call>`, making each intermediate intent explicit and trackable.

**System prompt:** loaded from `*_dataset*.yaml` templates (e.g. `gaia.yaml`, `gpqa.yaml`), which include `reasoning` and `sub_goal` scaffolding in the few-shot examples.

---

### Baseline (`baseline: true`)

A vanilla LLM-with-tools agent — the same model and tools, but without structured memory or planning.

**No planning turn:** `query_analysis` is never generated.

**Growing conversation:** The raw `state.messages` list grows each turn exactly as in a standard chat API interaction — assistant message appended, then tool response appended. The model sees the full message history, not a reconstructed structured prompt.

**No sub-goal bookkeeping:** `action_history` is not populated; the "Plan so far" and "Previous Steps" sections never appear.

**System prompt:** loaded from `*_baseline*.yaml` templates (e.g. `gaia_baseline.yaml`), which omit the `reasoning` and `sub_goal` lines from few-shot examples while keeping the same answer-format instructions and tool usage examples.

---

### What the comparison isolates

| Component | Baseline | Our framework |
|---|---|---|
| Persona framing | "reasoning assistant" (same) | "reasoning assistant" (same) |
| Planning turn (query analysis) | ✗ | ✓ |
| Memory construction per turn | growing raw conversation | reconstructed structured memory |
| Explicit sub-goal chain | ✗ | ✓ |
| Few-shot reasoning scaffolding | tool call + result only | reasoning + sub_goal + tool call + result |
| Extra LLM call per question | 0 | 1 (planning turn) |

Both conditions use the same model weights, tools, answer format (`\boxed{}`), and `max_turns` budget. The only variables are the structural components listed above — they are the system being evaluated, not confounders.

---

Enabled via `tools.enabled_tools` in the config:

| Tool | Description |
|---|---|
| `web_search` | Serper (default) or Tavily API search; provider set via `web_tool_provider` config. **Serper**: fetches and caches full page content. **Tavily**: uses pre-cleaned, structured content directly (no URL fetching). Optional LLM-based result analysis in sub-agent mode. |
| `code_generator` | Execute Python in a subprocess; LLM generates the code in sub-agent mode |
| `mind_map` | Persistent per-question memory with optional GraphRAG indexing |
| `text_inspector` | Read and optionally analyse text files (PDF, DOCX, XLSX, CSV, …) |
| `image_inspector` | Vision-language analysis of images; requires a VLM in the config |

### Web Search Providers

The `web_search` tool supports two providers via `web_tool_provider` config:

- **Serper** (default): Traditional search API that returns URLs. The tool fetches and caches full page content.
  - Cache structure: `./cache/serper/<dataset_name>/search_cache.json` and `./cache/serper/<dataset_name>/url_cache.json`

- **Tavily**: AI-native search engine designed for LLMs. Returns pre-cleaned, structured content with no URL fetching needed.
  - Cache structure: `./cache/tavily/<dataset_name>/search_cache.json` (no URL cache needed)
  - Faster and more efficient for AI agents, though cannot be used for comparison between direct tool calling vs. agentic setup due to proprietary LLM-based results formatting.

---

## Datasets

**Benchmarks used in experiments:**

| Name | Key | Split used | Type |
|---|---|---|---|
| GAIA | `gaia` | `all_validation` | General QA (multi-step, tool-requiring) |
| HLE (Humanity's Last Exam) | `hle` | `test_subset_200` | Expert-level single QA |
| GPQA | `gpqa` | `diamond` | Graduate-level multiple-choice science |
| AIME | `aime` | `train` | Competition mathematics |
| MuSiQue | `musique` | `validation_subset_200` | Multi-hop reasoning |

**Additional datasets — loaders available, not yet in experiment configs:**

| Name | Key |
|---|---|
| MATH500 | `math500` |
| AMC | `amc` |
| Natural Questions | `nq` |
| TriviaQA | `triviaqa` |
| HotpotQA | `hotpotqa` |
| Bamboogle | `bamboogle` |
| 2WikiMultiHopQA | `2wiki` |

**Download datasets before running:**

```bash
python scripts/download_datasets.py --dataset gaia --split validation
```

**Prompt templates:** GAIA, HLE, and MuSiQue share the `gaia.yaml` system prompt template; GPQA uses `gpqa.yaml`; AIME uses `math.yaml`. In baseline mode the corresponding `*_baseline.yaml` templates are loaded instead. The mapping lives in `src/agent_engine/prompts/builder.py` (`PromptBuilder.build_system_prompt`).

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
