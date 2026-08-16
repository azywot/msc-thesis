# Configuration reference

Every experiment is one YAML file. The loader
(`src/agent_engine/config/loader.py`) parses it into the Pydantic models in
`src/agent_engine/config/schema.py`, and **the schema is the authority** - if
this page and `schema.py` disagree, the schema is right and this page is stale.

> **A typo'd key is silently ignored.** The models do not set
> `extra="forbid"`, so `bacthed_size: 8` or `baseline_mode: true` loads
> without complaint and the default is used instead. Nothing warns you; the
> run just quietly measures something other than what you meant. When a config
> change appears to have no effect, check the spelling against the tables below
> before checking anything else.

---

## A minimal config

```yaml
name: qwen3_8b_gaia
description: Qwen3-8B on GAIA validation, AgentFlow mode

models:
  orchestrator:
    name: Qwen3-8B
    family: qwen3
    path_or_id: Qwen/Qwen3-8B
    role: orchestrator

dataset:
  name: gaia
  split: validation

tools:
  enabled_tools: [web_search, code_generator]
  direct_tool_call: true

max_turns: 15
batch_size: -1
thinking_mode: NO
baseline: false
output_dir: ./experiments/results/qwen3_8b_gaia
```

Only `name` and, in practice, `models.orchestrator` are load-bearing; everything
else has a default. `dataset` is optional at the schema level because the
orchestrator is also driven directly by the fine-tuning rollout code.

---

## Top level (`ExperimentConfig`)

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | str | *required* | Short identifier; used in output directory names. |
| `description` | str | `""` | Free text. |
| `models` | dict[str, ModelConfig] | `{}` | Role → model. See [Models](#models-modelconfig). |
| `tools` | ToolsConfig | defaults | See [Tools](#tools-toolsconfig). |
| `dataset` | DatasetConfig \| null | `null` | See [Dataset](#dataset-datasetconfig). |
| `max_turns` | int | `15` | Reasoning turns per question before an answer is forced. |
| `batch_size` | int | `-1` | Questions per batched generation call. `-1` = whole dataset in one batch; `1` = no batching. |
| `seed` | int | `0` | Propagated into every `ModelConfig` that does not set its own. |
| `thinking_mode` | enum | `NO` | `NO`, `ORCHESTRATOR_ONLY`, `SUBAGENTS_ONLY`, `ALL`. |
| `baseline` | bool | `false` | `true` = vanilla LLM-with-tools. See [Baseline vs AgentFlow](architecture.md#baseline-vs-agentflow). |
| `output_dir` | path | `./experiments/results` | Run outputs are written here. |
| `use_wandb` | bool | `false` | |
| `wandb_project` | str \| null | `null` | Required when `use_wandb` is true. |
| `cache_dir` | path | `./cache` | Root of the search/URL cache. |
| `slurm` | SlurmConfig | defaults | See [SLURM](#slurm-slurmconfig). |
| `gepa_prompt_path` | path \| null | `null` | See [GEPA prompts](#gepa-prompt-path). |

### `batch_size`

Batching is what makes a 200-question run tractable: all questions advance one
turn together, so each turn is a single large `generate()` call instead of 200
small ones. The cost is memory - `-1` on a large dataset with long contexts can
exceed VRAM. Start at `-1`, drop to a fixed size if the run OOMs.

`batch_size: 1` is not just "slower"; it is a genuinely different execution
path (no cross-question batching at all), which makes it the right setting when
debugging a single question.

### `thinking_mode`

Which components emit `<think>` output. `ORCHESTRATOR_ONLY` and
`SUBAGENTS_ONLY` exist so the two can be varied independently - several thesis
ablations turn on exactly that split.

How thinking is switched on differs by family and is handled for you: Qwen3 and
QwQ take an `enable_thinking` kwarg, DeepSeek is forced via a `<think>` prefix,
and OLMo 3 Think always thinks. See
[add-a-model-family](guides/add-a-model-family.md).

---

## Models (`ModelConfig`)

`models` maps a **role** to a model. `orchestrator` is the agent itself. Other
roles are named after tools and are used only when `tools.direct_tool_call` is
`false`, where each tool gets its own sub-agent LLM:

```yaml
models:
  orchestrator:
    name: Qwen3-8B
    family: qwen3
    path_or_id: Qwen/Qwen3-8B
    role: orchestrator
  web_search:
    name: Qwen3-1.7B
    family: qwen3
    path_or_id: Qwen/Qwen3-1.7B
    role: web_search
```

> If two roles share the same `path_or_id` **and** the same LoRA adapter, the
> runner reuses one loaded instance instead of paying for the weights twice.
> The adapter is part of the cache key - a base model and the same model with an
> adapter are two different instances.

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | str | *required* | Label for logs and W&B. |
| `family` | enum | *required* | `qwen3`, `qwen2.5`, `qwq`, `llama3`, `mistral`, `deepseek`, `olmo-think`, `olmo-instruct`, `gpt4`, `claude`. |
| `path_or_id` | str | *required* | HF model ID or absolute path to a checkpoint. |
| `role` | str | *required* | Should match the key it sits under. |
| `backend` | str | `vllm` | `vllm`, `mlx`, `openai`, `anthropic`. |
| `max_model_len` | int | `32768` | Prompt + generation, in tokens. |
| `max_tokens` | int | `8192` | New tokens per call. |
| `temperature` | float | `0.0` | `0.0` = greedy, the default for reproducibility. |
| `top_p` | float | `0.8` | |
| `top_k` | int | `20` | |
| `repetition_penalty` | float | `1.1` | |
| `supports_thinking` | bool \| null | `null` | Derived from `family` when unset. Setting it by hand overrides that. |
| `tensor_parallel_size` | int \| null | `null` | GPUs to shard across; auto-detected when unset. |
| `gpu_memory_utilization` | float \| null | `null` | Fraction of VRAM for this model; resolved at load time when unset. |
| `gpu_ids` | list[int] \| null | `null` | Specific GPU indices. |
| `seed` | int \| null | `null` | Falls back to the top-level `seed`. |
| `lora_adapter_path` | str \| null | `null` | PEFT adapter directory. |
| `max_lora_rank` | int | `64` | **Must match the training rank.** vLLM's own default is 16; ours is 64. |

Family defaults are not all identical - OLMo 3 ships sampling parameters from
its model card (`temperature=0.6`, `top_p=0.95`, `top_k=-1`,
`repetition_penalty=1.0`) rather than the table above. Check the configs under
`experiments/configs/olmo3/` before copying numbers from a Qwen config.

---

## Tools (`ToolsConfig`)

```yaml
tools:
  enabled_tools: [web_search, code_generator]
  direct_tool_call: true
  return_code: false
  web_tool_provider: serper
```

| Key | Type | Default | Notes |
|---|---|---|---|
| `enabled_tools` | list[str] | `[web_search, code_generator]` | `web_search`, `code_generator`, `mind_map`, `text_inspector`, `image_inspector`. |
| `direct_tool_call` | bool | `true` | See below. |
| `return_code` | bool | `false` | `code_generator` returns the generated code instead of executing it. |
| `web_tool_provider` | str | `serper` | `serper` (fetches full pages) or `tavily` (pre-cleaned content). |
| `max_search_limit` | int | `10` | `web_search` calls per question. |
| `top_k_results` | int | `5` | Search results per query. |
| `max_doc_len` | int | `3000` | Characters per fetched document snippet. |
| `max_search_content_chars` | int | `14000` | Total characters of formatted results passed to a sub-agent before truncation. |

### `direct_tool_call`

The single most consequential tool setting.

- `true` - the orchestrator gets the tool's raw output. No sub-agent, no extra
  model, no extra GPU.
- `false` - each tool runs its own sub-agent LLM that reads the raw output and
  returns an analysis. Requires a `models` entry per tool, and turns those tools
  into *batched* tools: their work is deferred and flushed as one grouped
  generation call per turn. See
  [architecture.md](architecture.md#deferred-tools-and-batching).

### `return_code`

Required for BigCodeBench, where the harness executes the prediction itself and
a tool that already ran the code would produce the wrong artefact.
`scripts/generate_configs.py` sets this automatically for BigCodeBench
tool-using configs - do not rely on remembering it by hand.

---

## Dataset (`DatasetConfig`)

```yaml
dataset:
  name: gaia
  split: validation
  data_dir: ./data
  subset_num: -1
```

| Key | Type | Default | Notes |
|---|---|---|---|
| `name` | str | *required* | Must be a key in `DATASET_SPECS`. See [add-a-benchmark](guides/add-a-benchmark.md). |
| `split` | str | *required* | `validation`, `test`, … For BigCodeBench this is a version string, e.g. `v0.1.4_subset_200`. |
| `data_dir` | path | `./data` | Where `scripts/download_datasets.py` put things. |
| `subset_num` | int | `-1` | Number of examples; `-1` = the whole split. |

`name` is looked up **case-sensitively** in `DATASET_SPECS`
(`src/agent_engine/datasets/spec.py`). `gaia` resolves; `GAIA` falls through to
the unregistered path, which changes both prompt-template selection and whether
sampling is stratified. Use the exact registered spelling.

---

## SLURM (`SlurmConfig`)

Used by the job generator, not by the runner itself.

| Key | Type | Default |
|---|---|---|
| `partition` | str | `gpu_h100` |
| `num_gpus` | int \| null | `1` |
| `ntasks` | int | `1` |
| `cpus_per_task` | int | `8` |
| `time` | str | `04:00:00` |
| `conda_env` | str | `agent_engine` |

---

## GEPA prompt path

```yaml
gepa_prompt_path: experiments/results/gepa/gaia/<run>/best_candidate.json
```

When set, `PromptBuilder` is bypassed entirely: `system_prompt` and
`planning_suffix` are read from the file instead of being assembled from
templates. This is how a GEPA-optimised prompt is evaluated - see
[pipelines/gepa.md](pipelines/gepa.md). Because it short-circuits the builder,
a config with this key set ignores the dataset's normal template.

---

## Where configs live

```
experiments/configs/
  qwen3/       agentflow/  baseline/  orchestrator_capacity/
               subagent_orchestrator_ablation/  structured_memory_ablation/
               gepa_inference/  lora_inference/  sft_inference/
  deepseek/    agentflow/  baseline/          DeepSeek-R1-Distill (7B, 32B)
  olmo3/       think/  instruct/
  local/       three small-model configs for a laptop (MLX)
  gepa/        GEPA optimisation runs + splits/
  fine_tuning/ RL / SFT training configs
  datasets_examples/   one worked example per benchmark
  template.yml
```

Most of these are **generated** by `scripts/generate_configs.py`, which owns
eleven suites:

```
agentflow  baseline  lora_inference  sft_inference  orch_capacity
structured_memory_ablation  subagent_orchestrator_ablation
olmo-think-agentflow  olmo-think-baseline
olmo-instruct-agentflow  olmo-instruct-baseline
```

Editing a generated file by hand works right up until someone regenerates the
suite, at which point the edit is silently reverted. Change the generator
instead:

```bash
python scripts/generate_configs.py                      # rewrite every suite
python scripts/generate_configs.py --output-root /tmp/preview   # dry run
```

`--output-root` writes elsewhere, which is the safe way to see what a generator
change would produce before overwriting the committed tree.

The configs under `gepa/`, `fine_tuning/`, `local/` and `datasets_examples/` are
hand-written and the generator does not touch them.

### Adding or customising a suite

A suite is one entry in the `SUITES` dict near the top of
`scripts/generate_configs.py`. It is the cross product of a set of **datasets**
and a set of **variants** (model/tool/thinking combinations), written out as one
YAML per pair.

Seven keys are required - every existing suite sets all of them:

```python
SUITES = {
    "my_suite": {
        "description_tag": "[My Suite]",           # prepended to each config's description
        "name_prefix":     "my_prefix",            # prepended to each config's name
        "output_dir_root": "./experiments/results/my_suite",
        "config_subdir":   "qwen3/my_suite",       # -> experiments/configs/qwen3/my_suite/
        "baseline":        False,                  # True = skip the planning turn
        "wandb_project":   "benchmarks",
        "split_overrides": {},                     # per-dataset split overrides
        # plus a source of variants, below
    },
}
```

Variants normally come from one of two keys:

| Key | Use |
|---|---|
| `variants` | One list for every dataset. Pick one of the sixteen `VARIANTS_*` lists defined above `SUITES`, or write your own. |
| `variants_by_dataset` | `{dataset: [...]}` when one benchmark needs a different set; falls back to `variants` for datasets not listed. |

The exception is a suite whose `variant_type` derives its own combinations.
`subagent_orchestrator_ablation` sets neither key: its `subagent_orch_ablation`
path builds a leave-one-out config per tool from each dataset's own tool list,
so the variants are computed rather than declared. If you write a new
`variant_type`, it owns that decision too.

The rest are optional, and each defaults to the behaviour you would expect if
you omitted it:

| Key | Default | Effect |
|---|---|---|
| `num_gpus` | - | GPU count for the generated SLURM job. |
| `force_num_gpus` | `False` | Use the suite's `num_gpus` even when the variant's model declares its own. |
| `datasets` | all seven | Restrict to a subset of `gaia`, `hle`, `gpqa`, `aime`, `math500`, `musique`, `bigcodebench`. |
| `variant_type` | `"standard"` | Selects a different config-building path: `orch_capacity`, `lora_inference`, or `subagent_orch_ablation`. |
| `no_thinking_mode` | `False` | Force `thinking_mode: NO` across the suite. |
| `lora_adapter_path` | - | Required when `variant_type` is `lora_inference`. |
| `adapter_label`, `adapter_desc` | `"LoRA"`, `"LoRA-adapted"` | Naming for adapter-based suites. |
| `train_job` | `"005_train.job"` | The training job referenced in generated descriptions. |

Then regenerate - preview first:

```bash
python scripts/generate_configs.py --output-root /tmp/preview
diff -r /tmp/preview/qwen3/my_suite experiments/configs/qwen3/my_suite
python scripts/generate_configs.py
```

> **The generator deletes before it writes.** `generate_suite` clears
> `*/*.yaml` and `*/*.yml` under the suite directory first, so a hand-written
> file living in a generated suite's folder does not survive. Keep hand-written
> configs in `local/`, `gepa/` or `fine_tuning/`, which the generator never
> touches.
>
> **Regeneration reverts hand-edits to generated files**, silently. It is
> currently a no-op against the committed tree - `tests/unit/test_wiring_invariants.py`
> asserts exactly that - so if you need a different value, put it in the
> generator, not in the output.
