# 🌌 Collaborative Small-Agent System (CoSMAS)

CoSMAS is a configuration-driven multi-agent research framework for investigating how small, collaborative language models can be composed to solve complex tasks. It supports systematic comparison between single-model baselines and cooperative multi-agent configurations under controlled experimental conditions.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Installation](#installation)
3. [HPC / Cluster setup](#hpc--cluster-setup)
4. [Running experiments](#running-experiments)
5. [Examples](#examples)
6. [Configuration reference](#configuration-reference)
7. [Model families](#model-families)
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
│   ├── analyze_results.py     # Metrics + breakdowns
│   ├── download_datasets.py   # Fetch/prepare datasets
│   └── export_prompts.py      # Dump prompt templates + tool schemas to JSON
│
├── experiments/
│   ├── configs/               # Experiment YAMLs (by model/dataset)
│   │   └── local/             # MacBook/MLX configs (Qwen3-0.6B, 4B)
│   └── results/               # Default output root
│
├── jobs/                      # SLURM job scripts + HPC tooling
├── examples/                  # Small runnable single-tool examples
├── pyproject.toml
├── requirements.txt
└── environment.yml            # Conda env for HPC
```

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

Run these steps once on Snellius (SURF) before launching any experiment.  
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

```bash
SERPER_API_KEY=...    # web_tool_provider: "serper" (default)
TAVILY_API_KEY=...    # web_tool_provider: "tavily"

# Optional
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...          # required for gated datasets (GAIA, GPQA, HLE)
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
WANDB_API_KEY=...    # optional
HF_TOKEN=...         # required for gated datasets (GAIA, GPQA, HLE)
```

#### 3. Download datasets

```bash
HF_HUB_DISABLE_XET_TRANSFER=1 python scripts/download_datasets.py \
    --dataset gaia --level all --split validation --output-dir ./data
```

#### 4. Run

```bash
python scripts/run_experiment.py --config experiments/configs/local/qwen3_0.6b_gaia.yaml
python scripts/run_experiment.py --config experiments/configs/local/qwen3_4b_gaia.yaml
```

Pre-built local configs are in `experiments/configs/local/`. Key differences from cluster configs:

| Option | Local (MLX) | Cluster (vLLM) |
|---|---|---|
| `backend` | `mlx` | `vllm` (default) |
| `batch_size` | `1`–`5` (RAM-limited) | `-1` (all at once) |
| SLURM fields | ignored | used by job scripts |

> **Batching:** set `batch_size: N` to process N questions in parallel. Higher values use more RAM — start small and increase as needed.

---

### Locally (GPU — vLLM)

```bash
export SERPER_API_KEY="..."

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
├── deepseek/                # DeepSeek-R1-Distill and R1-0528 configs
├── phi4/                    # Phi-4-mini-instruct and Phi-4-mini-reasoning configs
├── local/                   # MLX configs for Apple Silicon
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

The `examples/` directory contains one script per tool. Each runs a single question designed to exercise that tool — recommended sanity check before a full experiment.

```bash
python examples/example_web_search.py        # web_search
python examples/example_code_generator.py    # code_generator
python examples/example_text_inspector.py    # text_inspector
python examples/example_image_inspector.py   # image_inspector
python examples/example_mind_map.py          # web_search + mind_map (GraphRAG)
```

Prerequisites:

```bash
export SERPER_API_KEY="<your-key>"
export HF_HOME="/path/to/hf_cache"   # must contain Qwen/Qwen3-4B
```

Each script writes output to `experiments/results/examples/<tool_name>/`:

| File | Contents |
|---|---|
| `result.json` | question, answer, turns used, tool call counts |
| `trace.json` | full message + tool call history |
| `example.log` | human-readable execution log |

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
| `tools.direct_tool_call` | `true` / `false` | Direct mode returns raw tool output to the planner; sub-agent mode routes it through a second LLM first |
| `thinking_mode` | `NO` / `ORCHESTRATOR_ONLY` / `SUBAGENTS_ONLY` / `ALL` | Controls which roles emit extended reasoning (requires a thinking-capable model) |
| `batch_size` | integer | Questions per batch (-1 = all at once; 1 = sequential) |

If multiple roles share the same `path_or_id`, the runner reuses the loaded vLLM instance and serialises access with per-model locks — no duplicate GPU memory.

---

## Model families

The `ModelFamily` enum (`src/agent_engine/models/base.py`) controls generation defaults, thinking-mode handling, tool-call format, and vLLM chat-template rendering. Set it via the `family` key in any model block of a config.

### Supported families

| Family key | Example models | Thinking | Backend |
|---|---|---|---|
| `qwen3` | Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B} | Auto | vLLM, MLX |
| `qwen2.5` | Qwen/Qwen2.5-{7B,14B,32B}-Instruct | No | vLLM, MLX |
| `qwq` | Qwen/QwQ-32B | Auto | vLLM |
| `deepseek_r1` | deepseek-ai/DeepSeek-R1-Distill-Qwen-{7B,14B,32B} | Auto | vLLM |
| `deepseek_r1_0528` | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | Auto | vLLM |
| `phi4` | microsoft/Phi-4-mini-{instruct,reasoning} | Opt-in | vLLM |
| `llama3` | meta-llama/Llama-3.x-… | No | vLLM |
| `mistral` | mistralai/Mistral-{7B,8B}-Instruct-… | No | vLLM |
| `gpt4` | gpt-4o, gpt-4o-mini, … | No | API |
| `claude` | claude-3-5-sonnet-…, claude-3-7-sonnet-… | No | API |

---

### DeepSeek family

Two sub-families are supported. Both are reasoning models with `supports_thinking: true` set automatically. They share the same native tool-call token structure but differ in prompt handling.

#### `deepseek_r1` — DeepSeek-R1-Distill-Qwen-{7B, 14B, 32B}

January 2025 release, Qwen2.5 backbone.

Generation defaults: `temperature: 0.6`, `top_p: 0.95`, `max_model_len: 32768`.

**Prompt handling:**
- **System prompt:** per the [DeepSeek-R1 usage recommendations](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B#usage-recommendations), all instructions should be in the user turn. `VLLMProvider._merge_system_into_user` merges any leading `system` message into the first `user` message automatically before the chat template is applied.
- **Thinking:** `<think>\n` is appended to the rendered prompt to enforce reasoning (`<think>\n</think>\n\n` to suppress it), matching the template's generation prompt `<｜Assistant｜><think>\n`.

**Multi-seed evaluation:** the usage recommendations advise averaging results across multiple seeds when benchmarking:

```yaml
models:
  orchestrator:
    seed: 42   # repeat with seed: 0, 1, 43, … and aggregate metrics.json files
```

#### `deepseek_r1_0528` — DeepSeek-R1-0528-Qwen3-8B

May 2025 release, Qwen3 architecture with DeepSeek-R1-0528 tokenizer.

Generation defaults: same as `deepseek_r1`.

**Prompt handling (updated vs R1-Distill):**
- **System prompt:** fully supported — passed through unchanged to the chat template. No merging into the user turn.
- **Thinking:** no manual `<think>` injection needed. The model reasons autonomously. Per the [updated usage recommendations](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B), this model can be run in the same manner as Qwen3-8B.

Available configs: `experiments/configs/deepseek/`

---

### Phi-4 family

**`phi4`** — Microsoft Phi-4-mini-instruct and Phi-4-mini-reasoning (3.8 B parameters, 128 K context, February 2025).

| Model | HuggingFace ID | Thinking |
|---|---|---|
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | No |
| Phi-4-mini-reasoning | `microsoft/Phi-4-mini-reasoning` | Opt-in |

Unlike DeepSeek and Qwen3, thinking capability is **not auto-detected** for Phi-4. For `Phi-4-mini-reasoning` set `supports_thinking: true` explicitly:

```yaml
models:
  orchestrator:
    family: "phi4"
    path_or_id: "microsoft/Phi-4-mini-reasoning"
    supports_thinking: true   # required — not inferred from family

thinking_mode: "ORCHESTRATOR_ONLY"
```

Generation defaults: `temperature: 0.0`, `top_p: 0.8`, `top_k: 20`, `repetition_penalty: 1.1`.

Available configs: `experiments/configs/phi4/`

---

### Tool calling formats

The prompt builder and parser are both family-aware. Each family uses the model's **native chat-template tokens** to delimit tool calls, matching the training distribution exactly.

| Family | Tool call format | Tool result format |
|---|---|---|
| `qwen3`, `qwen2.5`, `qwq` | `<tool_call>\n…\n</tool_call>` | `<tool_response>…</tool_response>` |
| `deepseek_r1`, `deepseek_r1_0528` | `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}`<br>` ```json`<br>`{arguments}`<br>` ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>` | `<｜tool▁output▁begin｜>…<｜tool▁output▁end｜>` |
| `phi4` | `<\|tool_call\|>\n…\n<\|/tool_call\|>` | `<tool_response>…</tool_response>` |

**DeepSeek format notes:**
- Uses the exact native special tokens from both chat templates ([R1-Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)).
- The JSON body contains **arguments only** — the function name comes from the `<｜tool▁sep｜>` header, not from the JSON. This matches the native template exactly.
- Stop sequence: `<｜tool▁call▁end｜>` (with `<｜tool▁calls▁end｜>` as fallback).

**JSON format by family:**

For Qwen/Phi-4 families, the JSON body is a wrapper:
```json
{"name": "function_name", "arguments": {"param": "value"}}
```

For DeepSeek families, the JSON body contains arguments only (name is in the header):
```json
{"param": "value"}
```

**Validation sources:**
- Qwen3 `<tool_call>` / `</tool_call>` and `<tool_response>` / `</tool_response>`: [`Qwen/Qwen3-8B` — tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json)
- DeepSeek-R1-Distill native token structure: [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- DeepSeek-R1-0528 native token structure (identical): [`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- Phi-4-mini `<|tool_call|>` (ID 200025) / `<|/tool_call|>` (ID 200026): [`microsoft/Phi-4-mini-instruct` — tokenizer_config.json](https://huggingface.co/microsoft/Phi-4-mini-instruct/raw/main/tokenizer_config.json)

**Parser priority** (`src/agent_engine/utils/parsing.py` — `parse_tool_call`):

1. Native DeepSeek tokens (`<｜tool▁call▁begin｜>…<｜tool▁call▁end｜>`) — name from header, args from JSON body
2. Qwen XML tags (`<tool_call>…</tool_call>`)
3. Phi-4 pipe tags (`<|tool_call|>…<|/tool_call|>`)
4. Markdown JSON code blocks (` ```json … ``` `) — fallback
5. Bare JSON `{"name": …, "arguments": …}` — last resort

---

## Tools

Enabled via `tools.enabled_tools` in the config:

| Tool | Description |
|---|---|
| `web_search` | Serper or Tavily API search; provider set via `web_tool_provider`. **Serper** fetches and caches full page content. **Tavily** returns pre-cleaned, structured content directly. |
| `code_generator` | Execute Python in a subprocess; LLM generates code in sub-agent mode |
| `mind_map` | Persistent per-question memory with optional GraphRAG indexing |
| `text_inspector` | Read and analyse text files (PDF, DOCX, XLSX, CSV, …) |
| `image_inspector` | Vision-language analysis of images; requires a VLM in the config |

### Web search cache structure

| Provider | Search cache | URL cache |
|---|---|---|
| Serper | `./cache/serper/<dataset>/search_cache.json` | `./cache/serper/<dataset>/url_cache.json` |
| Tavily | `./cache/tavily/<dataset>/search_cache.json` | — (not needed) |

---

## Datasets

### Fully supported benchmarks

| Name | Key |
|---|---|
| GAIA | `gaia` |
| HLE (Humanity's Last Exam) | `hle` |
| GPQA | `gpqa` |

### Partially wired (downloadable, extendable)

| Name | Key |
|---|---|
| MATH500 | `math500` |
| AIME | `aime` |
| AMC | `amc` |
| Natural Questions | `nq` |
| TriviaQA | `triviaqa` |
| HotpotQA | `hotpotqa` |
| MuSiQue | `musique` |
| Bamboogle | `bamboogle` |
| 2WikiMultiHopQA | `2wiki` |

Download before running:

```bash
python scripts/download_datasets.py --dataset gaia --split validation
```

**Prompt template note:** GAIA, HLE, and other single-question QA datasets share the same system prompt template (`gaia.yaml`). Dataset names like `hle` are transparently mapped in `PromptBuilder.build_system_prompt`.

---

## Outputs

Each run creates a timestamped subdirectory under `output_dir/`  
(e.g. `all_validation_2026-02-22-22-25-02_<job_id>/`):

| File | Contents |
|---|---|
| `raw_results.json` | Per-example results: question, prediction, ground truth, metrics, tool calls |
| `raw_results.partial.json` | Rolling checkpoint written during the run; deleted on clean completion |
| `metrics.json` | Aggregate accuracy, EM, F1 |
| `config.json` | Serialised experiment config |
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
