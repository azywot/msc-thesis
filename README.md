# Agent Engine (`msc-thesis`)

Configuration-driven agentic reasoning system with multi-turn tool calling, built for reproducible LLM experiments on HPC/SLURM.

## Key features
- **YAML experiment configs**: one canonical schema (`src/agent_engine/config/schema.py`)
- **Multiple model backends**: local vLLM and API providers (OpenAI/Anthropic)
- **Tool calling**: web search, code execution, mind-map memory, file inspectors
- **Batching & concurrency**: multi-question batching + concurrent API calls + batched URL fetching
- **Model instance reuse**: share the same local model across roles with thread-safe locks

## Project structure

```
msc-thesis/
├── README.md                      # This file
├── pyproject.toml                 # Package + tooling config
├── requirements.txt               # Python deps (pip)
├── environment.yml                # Conda env (HPC-friendly)
│
├── src/
│   └── agent_engine/              # Main Python package
│       ├── config/                # YAML schema + loader
│       ├── core/                  # Orchestrator + tool-calling loop
│       ├── models/                # vLLM + API providers + locking/reuse
│       ├── tools/                 # web_search, code_generator, context_manager, inspectors
│       ├── datasets/              # loaders + evaluators + metrics
│       ├── prompts/               # prompt templates + builders
│       ├── external/              # Serper + URL fetching utilities
│       ├── caching/               # cache manager(s)
│       └── utils/                 # parsing/logging helpers
│
├── scripts/                       # User-facing entrypoints
│   ├── run_experiment.py          # Main runner (requires --config)
│   ├── analyze_results.py         # Metrics + breakdowns
│   ├── download_datasets.py       # Fetch/prepare datasets
│   └── export_prompts.py          # Prompt export utilities
│
├── experiments/
│   ├── configs/                   # Experiment YAMLs (by dataset)
│   └── results/                   # Default output root (per config output_dir)
│
├── jobs/                          # SLURM/HPC workflow
│   ├── QUICKSTART.md              # HPC quickstart
│   ├── submit_job.sh              # Convenience submit wrapper
│   ├── scripts/                   # Job generation helpers
│   ├── templates/                 # SLURM templates
│   └── generated/                 # Generated .job files
│
├── docs/                          # Deeper documentation
├── examples/                      # Small runnable examples
└── tests/                         # Unit tests
```

### Where to start (common tasks)
- **Run an experiment**: `scripts/run_experiment.py` + `experiments/configs/**`
- **Change config fields / defaults**: `src/agent_engine/config/schema.py`
- **Model providers / batching / reuse**: `src/agent_engine/models/`
- **Tools**: `src/agent_engine/tools/`
- **Datasets + evaluation**: `src/agent_engine/datasets/`
- **SLURM**: `jobs/QUICKSTART.md`, `jobs/submit_job.sh`, `jobs/scripts/`

## Installation

```bash
cd msc-thesis
pip install -e .
```

Optional (dev tools):

```bash
pip install -e ".[dev]"
```

HPC/Conda setup is documented in `jobs/QUICKSTART.md`.

## Quick start

### 1) Set API keys

```bash
# Required for `web_search`
export SERPER_API_KEY="..."

# Optional (API models / W&B)
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export WANDB_API_KEY="..."
```

### 2) Run an experiment

```bash
python scripts/run_experiment.py --config experiments/configs/gaia/baseline.yaml
```

Increase batching (faster on vLLM; default is 8):

```bash
python scripts/run_experiment.py --config experiments/configs/gaia/baseline.yaml --batch-size 16
```

Override the output directory (optional):

```bash
python scripts/run_experiment.py --config experiments/configs/gaia/baseline.yaml --output-dir ./experiments/results/my_run
```

### 3) Analyze results

```bash
python scripts/analyze_results.py ./experiments/results/gaia_baseline/results.json --by-level --tools
```

## Configuration

Experiments are defined in YAML and loaded via `src/agent_engine/config/loader.py`.

```yaml
# experiments/configs/gaia/baseline.yaml
name: "gaia_qwen3_baseline"
description: "GAIA validation (direct tool mode)"

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

Defaults live in:
- `src/agent_engine/config/schema.py` (experiment/tools/dataset defaults)
- `src/agent_engine/models/base.py` (model generation defaults)

## Modes

- **Direct vs sub-agent tools**: `tools.direct_tool_call: true/false`
- **Thinking**: `thinking_mode: NO | ORCHESTRATOR_ONLY | SUBAGENTS_ONLY | ALL`
- **Batching**: `--batch-size N` (N=1 disables batching)
- **Model instance reuse** (local models): if multiple roles share the same `path_or_id`, the runner reuses the loaded model instance and serializes access with per-model locks.

## Tools

Enabled via `tools.enabled_tools`:
- `web_search` (`Serper` + optional URL fetching / sub-agent analysis)
- `code_generator` (generate + run Python)
- `context_manager` (persistent memory)
- `text_inspector` (read/analyze text files)
- `image_inspector` (vision analysis; requires a vision-capable model)

## Datasets

Supported dataset names include:
- **GAIA**: `gaia`
- **GPQA**: `gpqa`
- **Math**: `math500`, `aime`, `amc`
- **QA**: `nq`, `triviaqa`, `hotpotqa`, `musique`, `bamboogle`, `2wiki`

## Outputs

`output_dir/` contains:
- `results.json` (main per-example results)
- `results_partial.json` (periodic checkpoint saves)
- `experiment.log` (runner logs)
- legacy compatibility: `<split>.<month>.<day>,<hour>:<min>.json` and `.metrics.json`
- if run via SLURM: `<experiment_name>_<job_id>.log`

## Running on SLURM

```bash
./jobs/submit_job.sh experiment experiments/configs/gaia/baseline.yaml
```

Or generate a job file from a config:

```bash
python jobs/scripts/generate_job.py experiments/configs/gaia/baseline.yaml
sbatch jobs/generated/<experiment_name>.job
```

## Documentation
- `jobs/QUICKSTART.md` (HPC/SLURM workflow)
- `experiments/configs/gaia/README.md` (GAIA config variants)



# Examples — agent_engine (sub-agent mode)

Each script tests a **single tool** in sub-agent mode using the shared config at
`experiments/configs/gaia/test_subagent.yaml`.  They are designed to force the
model to actually call the tool under test rather than answering from its weights.

All scripts should be run from the `msc-thesis/` root directory.

---

## Prerequisites

```bash
# Required for web_search
export SERPER_API_KEY="<your-key>"

# Required for all examples (models loaded via vLLM)
# Make sure HF_HOME points to a cache with Qwen/Qwen3-4B
```

---

## Files

| File | Tool tested | Question type |
|---|---|---|
| `example_web_search.py` | `web_search` | Recent real-world fact the model cannot know from training data |
| `example_code_generator.py` | `code_generator` | Multi-part maths problem requiring actual code execution |
| `example_text_inspector.py` | `text_inspector` | Five specific questions about a local document (`fixtures/sample_document.txt`) |
| `example_image_inspector.py` | `image_inspector` | Visual questions about a bar-chart PNG (generated on the fly) |
| `example_context_manager.py` | `web_search` + `context_manager` | Multi-step task: search for facts, then use context_manager to store/query (verifies GraphRAG indexing) |

Support files:

| File | Purpose |
|---|---|
| `_common.py` | Shared helpers (model init, tool wiring, orchestrator factory, prompt builder) — mirrors `scripts/run_experiment.py` |
| `simple_example.py` | Original minimal example (web_search + code_generator, direct mode) |
| `fixtures/sample_document.txt` | Synthetic annual report used by the text-inspector example |
| `fixtures/make_test_image.py` | Standalone helper to regenerate the bar-chart PNG manually |

---

## Running the examples

```bash
# Web search
python examples/example_web_search.py

# Code generation
python examples/example_code_generator.py

# Text inspection (reads fixtures/sample_document.txt)
python examples/example_text_inspector.py

# Image inspection (generates fixtures/test_chart.png automatically)
python examples/example_image_inspector.py

# Context manager + web search (GraphRAG indexing verification)
python examples/example_context_manager.py
```

Each script saves its output under `experiments/results/examples/<tool_name>/`:
- `result.json` — question, answer, turns used, tool call counts
- `trace.json` — full execution state (all messages, tool_calls) for debugging
- `example.log` — execution log including a readable trace (every message and tool call with arguments). For the code_generator example, the log also includes the generated code and the temp file path where it was written before execution.

---

## How it relates to the full experiment runner

The examples use `orchestrator.run()` (single question) instead of
`orchestrator.run_batch()`. Everything else — model caching, tool constructor
arguments, `thinking_mode`, `CacheManager`, `PromptBuilder` — is identical to
`scripts/run_experiment.py`. For a single question `run()` and `run_batch(batch_size=1)`
produce equivalent results.

The config field `tools.enabled_tools` in the YAML is **overridden** in each
example so that only the one tool under test is registered, even though the YAML
lists all five tools.
