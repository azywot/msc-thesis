# Fine-Tuning Design Notes

Technical notes and decisions for the GRPO fine-tuning pipeline.
For the full step-by-step guide see `docs/fine_tuning_README.md`.
For failure-mode motivation see `docs/failure_modes_fine_tuning_alignment.md`.

---

## Validation and Test Datasets

**AgentFlow's default:** AIME 2024.
**This project:** three non-overlapping splits of Search-R1 (NQ/HotpotQA) + DeepMath, carved in order — test first, then val, then train.

AIME is one of the five evaluation benchmarks reported in the thesis. Using it for checkpoint selection introduces selection bias — the best checkpoint would be chosen on questions you later report results on, inflating the AIME numbers. The held-out DeepMath val split is in-distribution with the math training data, never seen during training, and never part of the final benchmark runs. The test split is held out entirely and used only for final reporting after the best checkpoint is selected via val.

Files written to `data/training/`:
```
train/combined_train.parquet    1800 rows  (900 search + 900 math, shuffled)

val/val_search.parquet           100 rows  NQ + HotpotQA
val/val_deepmath.parquet         100 rows  DeepMath-103K (difficulty >= 5)
val/val_combined.parquet         200 rows  both merged — offline analysis only

test/test_search.parquet         100 rows  NQ + HotpotQA
test/test_deepmath.parquet       100 rows  DeepMath-103K (difficulty >= 5)
test/test_combined.parquet       200 rows  both merged — final reporting only
```

Source proportions (85% HotpotQA / 15% NQ within Search-R1; 50/50 search/math overall) are identical across all three splits.

VERL's `data.val_files` points at `val_search.parquet` and `val_deepmath.parquet` (two separate files) so W&B reports `val_0/reward_mean` and `val_1/reward_mean` separately. A combined file would hide per-domain divergence. The test split is never used by VERL during training.

---

## Token Budget and Truncation Risk

`data.max_response_length` is set to **4096** (doubled from AgentFlow's 2048 default, which targets single-step Planner rollouts).

Qwen3-8B with `THINKING_MODE: ORCHESTRATOR_ONLY` generates a `<think>...</think>` block before every action. On training questions:
- NQ / HotpotQA: thinking traces ~300–800 tokens (factual, short)
- DeepMath easy–medium: ~500–1000 tokens
- DeepMath hard: **500–1500 tokens**

Combined with tool calls and synthesis, most rollouts land at 800–2500 tokens total. The 4096 budget covers the typical case.

**Risk on hard DeepMath:** If the model thinks extensively before dispatching code, total tokens can approach or exceed 4096. When `data.truncation: 'truncate'` fires, the final answer token is lost → reward = 0. Spurious zeros push the model away from long thinking traces, which is the opposite of what you want.

**Detection:** watch `val_1/reward_mean` (DeepMath) in W&B epoch 1. If it stays near zero while `val_0/reward_mean` (Search-R1) and `actor/reward_mean` are rising, truncation is biting.
**Fix:** set `data.max_response_length: 8192` and relaunch.

---

## Answer Format: `\boxed{}` not `<answer>` tags

AgentFlow's rollout code appends this suffix to every user question:

```python
output_fmt = (
    " When ready, output the final answer enclosed in "
    "<answer> and </answer> tags."
)
```

This is **excluded** from `msc-thesis`. The thesis inference stack uses `\boxed{ANSWER}` as the final-answer format (parsed by `extract_answer()` in `parsing.py`). Injecting `<answer>` tags during training would create a distribution mismatch: the reward function calls `extract_answer()` which looks for `\boxed{}`, so a model trained to emit `<answer>` tags would get reward = 0 on every rollout that only emits those tags.

`_build_rollout_question()` in `rollout.py` returns the raw question text unmodified.

---

## Sub-Agent Design and Train/Eval Distribution Shift

Sub-agent token generations (web search analyser, code generator) are **excluded from the GRPO objective**. VERL treats them as environment interactions, not as the model being trained.

Sub-agents run on a **separate, frozen vLLM server** (`SUBAGENT_ENDPOINT`, default port 9998) that is started before training and never updated. The default sub-agent model is `Qwen/Qwen3-1.7B`. This means:

- The orchestrator is trained against a stable, consistent tool interface at every rollout.
- The sub-agent weights never co-evolve with the orchestrator during training.
- At eval time, sub-agents also run the same fixed base model through the standard `VLLMProvider`, so there is **no train/eval distribution shift for sub-agents**.

`_build_tool_registry(subagent_endpoint, subagent_model)` raises `EnvironmentError` if `SUBAGENT_ENDPOINT` is not set — this enforces the invariant that a frozen server is always explicitly provided before training starts. The job scripts (009, 010) launch the frozen server first, then the VERL server.

**Why a smaller sub-agent model (1.7B)?**
The 1.7B model is significantly cheaper to load alongside the 8B orchestrator on the same node, leaving more GPU memory for the VERL actor/rollout. Sub-agent tasks (retrieve-and-summarise, write-and-execute) are narrow and well-specified by the orchestrator, so the quality difference from using a 1.7B model instead of 8B is small in practice.

---

## Max Turns During Training vs Evaluation

| Setting | Training (`OrchestratorRollout`) | Evaluation (`run_experiment.py`) |
|---|---|---|
| `max_turns` | **5** (default in `OrchestratorRollout.__init__`) | **15** (from config YAML) |
| `max_tokens` | 2048 (sub-agent) | 8192 (sub-agent, from config) |
| `temperature` (orchestrator) | 0.7 (train), 0.0 (val) | 0.0 (greedy, from config) |
| `temperature` (sub-agents) | 0.0 (greedy, always) | 0.0 (greedy, from config) |
| Planning turn | enabled (`baseline=False`) | enabled (unless `baseline: true`) |

Fewer turns during training keeps rollouts shorter and reduces the chance of token budget overflow. At eval time, 15 turns allow more tool-use rounds. The distribution shift is deliberate: training questions (NQ, HotpotQA, DeepMath) typically resolve in 1–3 tool calls; the orchestrator will not be harmed by having more turns available at eval.

---

## Reward Function

`OrchestratorReward` in `reward.py` returns a binary reward: **1.0** if correct, **0.0** otherwise. It calls `evaluate_answer(prediction, ground_truth)` from `metrics.py`.

`ground_truth` is `task["result"]`, which is the **first** golden answer (see `normalise_search_r1_row` — `result` holds the first element of the `golden_answers` list). The remaining golden answers are stored in `extra_info.golden_answers` but are not used by the reward function.

For NQ and HotpotQA, questions often have 2–4 valid answers. A prediction matching a non-first alias will be evaluated against only the first alias. `evaluate_answer` uses containment matching (`contains_match`) so it is lenient — if the ground-truth string appears anywhere in the prediction it scores 1.0 — but it does not iterate over all aliases.

**Implication:** the training reward is slightly more restrictive than the evaluation metric (which uses `evaluate_musique` with all aliases after the golden-answers fix). On questions where the model predicts a valid non-first answer, the reward will be 0 at training time and 1 at evaluation time. This is a minor bias toward the first alias during training. The effect is small because NQ/HotpotQA golden-answer lists are short and the containment matcher catches most paraphrases.

---

## Data Preparation Notes

### Search-R1 streaming shuffle

`_download_search_r1()` uses HuggingFace *streaming* mode with `ds.shuffle(seed=seed, buffer_size=...)`. Streaming shuffle is buffer-based (reservoir sampling) and **not bit-for-bit reproducible** across runs, even with the same seed, because shard download order can differ between runs. The same `seed` will produce a similar but not identical subset.

`_download_deepmath()` uses non-streaming `ds.shuffle(seed=seed)`, which IS fully reproducible. ✓

**Practical impact:** negligible — the training set is large enough that row ordering variation does not affect the learned policy. Just note that `--seed 42` does not guarantee an identical parquet file for Search-R1.

### Training mix sizes

All three splits share the same source proportions (85% HotpotQA / 15% NQ within Search-R1; 50/50 search/math overall). Rows are carved in order: test first, then val, then train — guaranteeing no cross-split contamination.

| Use case | `--n-search` | `--n-math` | val per domain | test per domain |
|---|---|---|---|---|
| Default (GRPO) | 900 | 900 | 100 | 100 |
| Smoke test | 50 | 50 | 10 | 10 |

Total rows: 1800 train + 200 val + 200 test = 2200 drawn from each dataset.

### `data_source` routing in rollout prompts

`_prompt_dataset_for_data_source()` maps:
- `"deepmath"` → `"deepmath"` prompt template (math system prompt)
- anything else (including `"nq"`, `"hotpotqa"`) → `"gaia"` prompt template

This is correct: NQ and HotpotQA are retrieval QA tasks that use the same system prompt as GAIA at inference time.

---

## Shared Tool Registry Across Episodes

`OrchestratorRollout._get_or_build_tools()` lazily creates a single `ToolRegistry` on the first episode and reuses it for all subsequent episodes in the same worker process. The URL cache and search cache therefore accumulate across rollouts during a training run. This is intentional — cache hits speed up later rollouts that ask the same question (GRPO generates `rollout_n=8` rollouts per question, so cache hits are common within the same group). Disk usage for `rollout_dir` JSON logs grows at ~1–5 KB per rollout; on a 5-epoch run with 128 training questions/epoch and 8 rollouts each, this is ~5–25 MB.

---

## Reproducibility

Seed propagation is wired through the full stack:
- `ExperimentConfig.seed` → propagated to `ModelConfig.seed` for all models at load time (see `config/loader.py`)
- `ModelConfig.seed` passed to vLLM `LLM(seed=...)` and `SamplingParams(seed=...)` ✓
- `ModelConfig.seed` passed as `seed=` to OpenAI API calls ✓
- Training rollouts deliberately do NOT fix a per-episode seed — diversity is required for GRPO

MLX backend: no seed control in `mlx_lm.generate`/`batch_generate` — non-deterministic with temperature > 0. Not applicable to training (vLLM only on cluster).

Setting `PYTHONHASHSEED` at runtime (in `set_seed()`) affects child processes only; it cannot retroactively change hash randomisation in the running Python process.

---

## Two Conda Environments

| Env | Purpose | vLLM |
|---|---|---|
| `cosmas-train` | Training (VERL, rollout workers, AgentFlow) | 0.9.2 |
| `agent_engine` | Inference and evaluation | 0.12.0 |

The version split is a hard constraint. VERL 0.5.0 requires vLLM ~0.9.x; the inference stack pins 0.12.0. Never mix them: running `run_experiment.py` in `cosmas-train` or `train_orchestrator.py` in `agent_engine` will fail on import.
