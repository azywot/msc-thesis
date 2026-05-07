# Changelog

All notable changes to CoSMAS (Collaborative Small-Agent System) are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — feat/fine-tuning

### Added
- **RL fine-tuning pipeline** for the orchestrator (`src/fine_tuning/`)
  - `FinetuningConfig` dataclass (LoRA, GRPO, training hyperparams)
  - `OrchestratorReward` — binary reward via `evaluate_answer()` from `metrics.py`
  - `OrchestratorRollout(LitAgent)` — wraps `AgenticOrchestrator` as a VERL rollout worker
  - `data/prepare.py` — downloads Search-R1 + DeepMath-103K, carves out held-out DeepMath val split, converts to VERL parquet schema
- `scripts/launch_verl.py` — starts VERL training server (mirrors AgentFlow `train_agent.py`)
- `scripts/train_orchestrator.py` — starts rollout workers with `NullTracer` (no AgentOps required)
- `train/config.yaml` — VERL + AgentFlow config for Qwen3-8B GRPO with LoRA rank-64, 4×A100
- `jobs/train_orchestrator.sh` — SLURM job script for Snellius
- `jobs/environment_train.yml` — conda env pinned to AgentFlow stack (verl==0.5.0, vllm==0.9.2)
- `OpenAIProvider`: optional `base_url` parameter for vLLM-compatible API endpoints
- Design spec and implementation plan in `docs/superpowers/`
- `docs/failure_modes_fine_tuning_alignment.md` — analysis linking thesis failure modes to fine-tuning design
- `pyproject.toml`: `[training]` optional extras group

### Changed
- `train/config.yaml`: validation set switched from `aime24.parquet` to `deepmath_val.parquet` — AIME is an evaluation benchmark and must not be used for checkpoint selection (selection bias)
- `train/config.yaml`: `ENABLE_TOOLS` now includes both `web_search` and `code_generator` (was `web_search` only)
- `train/config.yaml`: `data.max_response_length` increased from 2048 to 4096 — msc-thesis runs a full multi-turn orchestrator loop per rollout vs. AgentFlow's single planning step, requiring a larger response budget
- `train/config.yaml`: `THINKING_MODE: ORCHESTRATOR_ONLY` — training matches the evaluation condition; `OrchestratorRollout` and `train_orchestrator.py` wired to forward this to `AgenticOrchestrator(use_thinking=...)`
- `data/prepare.py`: both training domains now get held-out val splits carved out before the training subsample — `val_search.parquet` (200 Search-R1), `val_deepmath.parquet` (200 DeepMath), `val_combined.parquet` (merged, for offline analysis); added `--n-val-search` CLI arg; AIME download removed
- `train/config.yaml`, `train/config_smoke.yaml`: `data.val_files` is now a two-element list — VERL logs `val_0/reward_mean` (search) and `val_1/reward_mean` (math) separately in W&B
- `scripts/launch_verl.py`: list values in `python_args` are now converted to Hydra list syntax (`key=[elem1,elem2]`) so multi-file `data.val_files` reaches VERL correctly
- `rollout.py`: `data_source` added to every saved rollout JSON record for offline per-domain analysis

---

## [0.5.0] — 2026-04-26

### Added
- Failure analysis documentation for experiments
- Repository version alignment investigation (#19)

---

## [0.4.0] — 2026-04-22

### Added
- Shared subagent memory — context from previous steps passed to sub-agents
- `[Attachment]` marker fix for file-inspector tools

### Changed
- Full alignment with `multi-agent-tools` experiment baselines
- Updated thesis visualizations (plots, tables)

### Fixed
- Test suite after sub-agent context changes

---

## [0.3.0] — 2026-04-20

### Added
- **DeepSeek model family** (`ModelFamily.DEEPSEEK`)
  - JSON_SINGLE tool-call format (`{"tool_call": {...}}`)
  - Force tool-call prefix injection on turn 1 (prevents hallucinated reasoning-only turns)
  - `<tool_response>` stop token to prevent fabricated tool responses
  - System message merged into first user turn (no system-role slot)
  - DS-7B and DS-32B experiment configs
- **OLMo 3 model families** (`OLMO_THINK`, `OLMO_INSTRUCT`)
  - Pythonic tool-call format (`<function_calls>`)
  - `role: tool` → `role: environment` rewrite for OLMo Think's chat template
  - `functions=""` injection to suppress "no functions" suffix
  - Sampling defaults matching HF model cards (T=0.6, top_p=0.95, max_tokens=32768)
  - OLMo experiment configs (think + instruct variants)
- `_sanitize_tool_arguments` — drops unexpected kwargs for strict tool signatures
- MATH500 dataset support (subset of 200)

### Changed
- `_force_tool_call` disabled in baseline mode (preserves pure-baseline comparison)
- Stop tokens keyed by `ToolCallFormat`

---

## [0.2.0] — 2026-04-15

### Added
- **BigCodeBench** benchmark support
  - `CodeGeneratorTool` with `return_code: true` mode (returns code instead of executing)
  - `bigcodebench_scorer.py` — assembles prediction + test harness and runs via `unittest`
  - Auto-set `return_code: true` in `generate_configs.py` for BigCodeBench
- **Orchestrator capacity ablation** experiments (`experiments/configs/qwen3/orchestrator_capacity/`)
- **Subagent–orchestrator ablation** experiments
- **Structured memory ablation** experiments
- Main results table and Figure 3 generation scripts (`scripts/tables/`, `scripts/plots/`)
- Efficiency plots (token usage, timing breakdowns)
- LaTeX table export scripts

### Changed
- Reorganized `experiments/configs/` by model family (`qwen3/`, `deepseek/`, `olmo3/`)
- `generate_configs.py` moved to `scripts/`

---

## [0.1.0] — 2026-03-11

### Added
- **AgentFlow alignment** — orchestrator loop matches AgentFlow's planner structure
  - Planning turn (Turn 0): query analysis before any tool calls
  - Structured memory prompt: `_build_memory_prompt()` rebuilds context each turn
  - `<sub_goal>` tag extraction and storage in `action_history`
  - Action history formatted as `Action Step N` blocks
- **MuSiQue** multi-hop QA dataset + evaluator (with answer aliases)
- **AIME** math competition dataset + experiment configs
- **Tavily** web search provider option (`web_tool_provider: tavily`)
- **Reasoning context** injection for code generator sub-agent (`attachment_context`)
- Context manager renamed to **mind map** (`mind_map` tool)
  - GraphRAG-backed knowledge indexing (`graph_rag.py`)
  - Pre-tool reasoning indexed before `web_search`, `code_generator`, `mind_map` calls
- Baseline mode (`baseline: true`) — skips planning turn, uses growing conversation
- `thinking_mode` config flag (`NO` / `ORCHESTRATOR_ONLY` / `SUBAGENTS_ONLY` / `ALL`)
- HLE (Humanity's Last Exam) dataset support
- `resolve_gpu_assignments` for multi-GPU tensor parallelism
- `batch_size` config — amortizes LLM calls across questions in one turn
- SLURM job templates (`jobs/`)
- `export_prompts.py` script
- Rolling checkpoint (`raw_results.partial.json`) for crash recovery

### Changed
- `planner` → `orchestrator` naming throughout codebase
- System prompt saved to `config.json` output
- Improved W&B logging (mind map stats, token usage)

### Fixed
- Mind map caching and W&B logging
- GPQA formatting (no "Choices:" header)
- Reproducibility: fixed random seed propagation

---

## [0.0.1] — 2026-02-17

### Added
- Initial CoSMAS framework
- `AgenticOrchestrator` — multi-turn reasoning loop with tool calling (Qwen3 JSON format)
- Tools: `WebSearchTool` (Serper), `CodeGeneratorTool`, `TextInspectorTool`, `ImageInspectorTool`
- Datasets: GAIA, GPQA (initial support)
- Model providers: `VLLMProvider`, `OpenAIProvider`, `AnthropicProvider`, `MLXProvider`
- YAML-based experiment config system (`experiments/configs/`)
- Prompt templates (`src/agent_engine/prompts/templates/`)
- `run_experiment.py` main runner
- `analyze_results.py` metrics script
- `download_datasets.py` helper
- Fine-tuning placeholder stubs (`src/fine_tuning/`)
- W&B logging integration
- Unit test suite
