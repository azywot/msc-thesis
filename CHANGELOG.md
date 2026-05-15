# Changelog

All notable changes to CoSMAS (Collaborative Small-Agent System) are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — feat/gepa-integration

### Added
- **GEPA prompt optimisation** (`src/gepa_integration/`) — system adaptation chapter implementation
  - `seed.py` — `build_seed_candidate()` renders the two-component seed (`system_prompt` + `planning_suffix`) from YAML templates; `build_splits()` generates failure-stratified train / random val / random test splits from any existing `raw_results.json`
  - `adapter.py` — `AgentGEPAAdapter` implementing the GEPA `GEPAAdapter` protocol: `evaluate()` runs the orchestrator under `ORCHESTRATOR_ONLY` thinking and returns per-example scores; `make_reflective_dataset()` serialises execution traces into GEPA's reflective dataset format for both `system_prompt` and `planning_suffix` components
  - Reflective records include the orchestrator's raw `<think>` blocks (via `raw_query_analysis` and `output_messages`) so the Qwen3-32B reflector can diagnose reasoning failures, not just answer failures
- **`scripts/run_gepa.py`** — four-mode CLI for the full GEPA pipeline:
  - `--mode splits` — generate and save train/val/test split JSON files; train set is failure-stratified (65% failures proportional across all six failure modes, 35% successes) using `classify_failure()` from `scripts/failure_modes/analyze_failure_modes.py`
  - `--mode optimize` — run GEPA optimisation loop (GAIA: 80 train / 45 val; GPQA: 100 train / 48 val); saves `best_candidate.json` and `seed_candidate.json` to `run_dir`
  - `--mode evaluate` — evaluate best candidate on held-out test set (GAIA: 40q, GPQA: 50q); outputs `gepa_results.json` in `raw_results.json` format for use with existing `analyze_results.py`
  - `--mode diff` — print unified diff of `system_prompt` and `planning_suffix` between seed and best candidate
- **GEPA experiment configs** (`experiments/configs/gepa/`)
  - `gaia.yaml` — GAIA optimisation: Qwen3-8B agent + sub-agents, Qwen3-32B reflector (port 8001), 150 rollouts, sub-agent mode (`direct_tool_call: false`) matching the milestone baseline
  - `gpqa.yaml` — GPQA Diamond: same setup, multiple-choice routing via `example.metadata["choices"]`
  - `splits/gaia_splits.json` — pre-generated splits: 80 train / 45 val / 40 test (seed=1, failure-stratified)
  - `splits/gpqa_splits.json` — pre-generated splits: 100 train / 48 val / 50 test (seed=1, failure-stratified)
- **`jobs/gepa/` — SLURM job sequence** for the full GEPA pipeline:
  - `000_prep_gepa_data.job` — generates failure-stratified splits for GAIA and GPQA via `run_gepa.py --mode splits`; safe to re-run (deterministic, seed=1)
  - `001_install_gepa_deps.job` — installs `gepa==0.0.22` (pinned in `requirements.txt`) into the `agent_engine` conda env; smoke-tests all imports
  - `002_smoke_gepa.job` — CPU-only pre-flight checks: imports, seed candidate structure, source `raw_results.json` presence, splits integrity (sizes + no-overlap), dataset loading, `evaluate_answer` spot-checks
  - `003_smoke_gepa_gpu.job` — end-to-end GPU smoke test (3×H100 NVL): Qwen3-8B agent on GPU 0 (tp=1), Qwen3-32B reflector on GPUs 1–2 (tp=2); runs 1 GEPA step on 2 GAIA train examples then evaluates on 2 held-out test examples; asserts `gepa_results.json` schema
  - `004_run_gepa.job` — full optimisation run (4×H100 NVL, 12h): GAIA then GPQA; supports `REGEN_SPLITS=1`, `SKIP_GAIA=1`, `SKIP_GPQA=1` overrides
- **`scripts/smoke_gepa.py`** — standalone pre-flight smoke test (no GPU); run locally or via `002_smoke_gepa.job`
- **`experiments/configs/gepa/smoke_test.yaml`** — minimal GEPA config for `003_smoke_gepa_gpu.job`: 2 train / 2 val / 2 test examples, 1 GEPA step (budget=2, minibatch=2), Qwen3-32B reflector, `max_turns=3`
- **`experiments/configs/gepa/splits/smoke_splits.json`** — pre-generated splits for smoke test (6 GAIA question IDs)
- **`tests/gepa_integration/`** — 32 unit tests covering `ExecutionState.raw_query_analysis`, orchestrator `planning_suffix` param + constants, `build_seed_candidate`, `build_splits` (size, no-overlap, failure ratio, JSON output), `_extract_thinking`, and all `AgentGEPAAdapter` methods

### Fixed
- `scripts/run_gepa.py` — `_build_tool_registry` used non-existent `direct_mode=` constructor argument on all three tools; replaced with the correct `model_provider=` pattern (direct mode = `model_provider=None`, sub-agent mode = pass the shared `VLLMProvider`)
- `scripts/run_gepa.py` — all configs changed to `direct_tool_call: false`; `_build_tool_registry` now accepts `model_provider` and wires it into tools when in sub-agent mode, with `use_thinking` derived from `thinking_mode`; model provider is created before the tool registry in both `run_optimize` and `run_evaluate` so it can be passed in
- `scripts/run_gepa.py` — `build_seed_candidate` now reads `max_search_limit` from the YAML config instead of silently using the default
- `scripts/run_gepa.py` — `run_evaluate` now passes `tool_limits` from the config to `AgenticOrchestrator`
- `jobs/gepa/003_smoke_gepa_gpu.job`, `jobs/gepa/004_run_gepa.job` — removed `--enable-thinking` from `vllm serve` (not a valid server flag; thinking is a per-request sampling parameter)

### Changed
- `src/agent_engine/core/state.py` — `ExecutionState` gains `raw_query_analysis: Optional[str] = None`; stores the full planning-turn output including `<think>` blocks before stripping
- `src/agent_engine/core/orchestrator.py` — planning-turn suffix strings extracted as module-level constants (`_DEFAULT_PLANNING_SUFFIX_NO_TOOLS`, `_DEFAULT_PLANNING_SUFFIX_TOOLS`); `AgenticOrchestrator.__init__` gains optional `planning_suffix` parameter (default `None` = use constants as before); `_run_planning_turn` stores raw text in `state.raw_query_analysis` before stripping thinking tags

---

## [Unreleased] — feat/fine-tuning

### Added
- **Qwen3-8B smoke-test config and job** (`experiments/configs/train/config_smoke8b.yaml`, `jobs/012_smoke_8b.job`)
  - Mirrors `config_smoke.yaml` (4B, 1 GPU) but targets Qwen3-8B with `N_GPUS=2` — actor FSDP-sharded to ~8 GB/GPU, vLLM TP=1 data-parallel (each GPU holds a full 8B copy during rollout)
  - Total: 3 H100 NVL GPUs — GPU 0 exclusive to frozen Qwen3-1.7B sub-agent (`util=0.40`); GPUs 1–2 for VERL
  - `param_offload=true` / `optimizer_offload=true` / `free_cache_engine=true` retained for the same reason as the 4B smoke: vLLM + actor shard on GPU during rollout risks CUDA memory fragmentation
  - `USE_SCRATCH_CHECKPOINTS: true` — 8B checkpoints (even from smoke runs) written to `/scratch-shared/$USER/msc-thesis/training/qwen3-8b-grpo-smoke/`; job script prints the path at startup and re-uses it in the checkpoint verification step
  - `max_model_len: 6144` = `max_prompt_length (4096) + max_response_length (2048)` — identical to 4B smoke to keep the test fast

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
- `AgentFlow/agentflow/verl/entrypoint.py` calls `peft_vllm_weight_sync_patch.apply_patch()` in the main process, the Ray `worker_process_setup_hook`, and the `TaskRunner` actor so the fix is active in every process that runs `FSDPVLLMShardingManager`
- `train/config.yaml`: validation set switched from `aime24.parquet` to `deepmath_val.parquet` — AIME is an evaluation benchmark and must not be used for checkpoint selection (selection bias)
- `train/config.yaml`: `ENABLE_TOOLS` now includes both `web_search` and `code_generator` (was `web_search` only)
- `train/config.yaml`: `data.max_response_length` increased from 2048 to 4096 — msc-thesis runs a full multi-turn orchestrator loop per rollout vs. AgentFlow's single planning step, requiring a larger response budget
- `train/config.yaml`: `THINKING_MODE: NO` — thinking disabled for training (reverted from `ORCHESTRATOR_ONLY` in *Adjust configs* commit after smoke runs; reduces rollout latency and avoids KV pressure from long thinking traces; can be re-enabled if evaluation shows benefit); `OrchestratorRollout` and `train_orchestrator.py` wired to forward this to `AgenticOrchestrator(use_thinking=...)`
- `rollout.py`: sub-agents now run during training via a shared VERL endpoint, matching AgentFlow's `vllm-local-<BASE_MODEL>` pattern — sub-agent tokens are environment context (not GRPO trajectory); `direct_tool_call=False` to match evaluation interface; `CodeGeneratorTool` registered with sub-agent provider (was missing entirely)
- Two conda environments (`cosmas-train` vLLM 0.9.2 / `agent_engine` vLLM 0.12.0) intentionally kept separate — consolidation investigated but blocked by a three-way VERL 0.5.0 / vLLM / AgentFlow version constraint; `docs/fine_tuning_README.md` documents the rationale
- `data/prepare.py`: both training domains now get held-out val splits carved out before the training subsample — `val_search.parquet` (200 Search-R1), `val_deepmath.parquet` (200 DeepMath), `val_combined.parquet` (merged, for offline analysis); added `--n-val-search` CLI arg; AIME download removed
- **Dataset curation for GRPO** (`data/prepare.py`)
  - Three non-overlapping splits: **1800 train / 200 val / 200 test** (test → val → train carve order guarantees no contamination)
  - `--search-source {hotpotqa,nq,both}` (default `both`) — controls which Search-R1 sources are included
  - `--hotpot-ratio` (default `0.85`) — HotpotQA fraction within Search-R1; same ratio applied identically to train, val, and test so source proportions are stratified
  - `--deepmath-min-difficulty` (default `5`) — filters DeepMath-103K to difficulty ≥ threshold (range 1–9); hard problems produce cleaner GRPO signal
  - `--n-search` / `--n-math` defaults lowered to **900** (from 10 000); `--n-val-*` defaults to **100** (from 200); new `--n-test-search` / `--n-test-math` args (default **100**)
  - `build_test_files()` — writes `test/test_search.parquet`, `test/test_deepmath.parquet`, `test/test_combined.parquet`
  - `_search_source_quotas(n, source, hotpot_ratio)` — pure helper; unit-tested
  - `_passes_difficulty_filter(raw, min_difficulty)` — pure helper; fail-open for missing field; unit-tested
  - `extra_info.difficulty` stored on DeepMath rows (int, coerced from string if needed; absent when field not in Hub row)
  - `jobs/008_prepare_fine_tuning_data.job` updated to `900/900` train + `100/100` val + `100/100` test with all new flags explicit
- `train/config.yaml`, `train/config_smoke.yaml`: `data.val_files` is now a two-element list — VERL logs `val_0/reward_mean` (search) and `val_1/reward_mean` (math) separately in W&B
- `scripts/launch_verl.py`: list values in `python_args` are now converted to Hydra list syntax (`key=[elem1,elem2]`) so multi-file `data.val_files` reaches VERL correctly
- `rollout.py`: `data_source` added to every saved rollout JSON record for offline per-domain analysis
- **SLURM fine-tuning jobs — GPU split and memory** (`jobs/009_test_small_ft_example.job`, `jobs/010_ft_orchestrator.job`)
  - **009 (smoke)** requests **2 GPUs**: frozen sub-agent on GPU 0; VERL on GPU 1 only (`N_GPUS: 1`). Smoke uses **Qwen/Qwen3-4B** so FSDP + colocated rollout vLLM fit one **40GB** card (Qwen3-8B OOMs loading the second vLLM weights copy). Full **010** stays **Qwen3-8B** with **4 GPUs** / `N_GPUS: 3`.
  - **010 (full run)** requests **4 GPUs** (typical Snellius cap): sub-agent on GPU 0; VERL on GPUs 1–3. Matches `config.yaml` `N_GPUS: 3` and `ROLLOUT_TP_SIZE: 1` (TP=2 would require an even training GPU count).
  - Sub-agent **`VLLM_USE_V1=0`**, **`--gpu-memory-utilization 0.40`**, **`--max-model-len 8192`** on both jobs — reduces V1 `torch.compile` + overly small util leaving no KV budget (`smoke_*_subagent.log` “available KV cache memory” errors).
  - After VERL start, **poll `http://127.0.0.1:9999/task`** with `curl` (10 min smoke / 15 min full) before rollout workers; clear failure if VERL dies or timeout; hints to check `*_verl.log` and `/scratch-local/${USER}.${SLURM_JOB_ID}/ray/`.
- **`experiments/configs/train/config_smoke.yaml`**: **2-GPU** smoke (`N_GPUS: 1`); **`BASE_MODEL: Qwen/Qwen3-4B`**, **`EXPERIMENT_NAME: qwen3-4b-grpo-smoke`**; shorter seq caps + rollout **`max_model_len`** / **`gpu_memory_utilization`**; **FSDP CPU offload off** (4B fits; 8B did not). Manual-launch: `CUDA_VISIBLE_DEVICES=1` for VERL after sub-agent on GPU 0.
- **`experiments/configs/train/config.yaml`**: **`N_GPUS: 3`**, **`ROLLOUT_TP_SIZE: 1`**; rollout **`gpu_memory_utilization: 0.30`** (was `0.4`) so vLLM’s `free_mem ≥ util × totalVRAM` check passes on **40GB A100** when rollout shares GPUs with FSDP; **`actor_rollout_ref.ref.fsdp_config.param_offload: true`** to shrink ref-model GPU footprint for colocated vLLM. Header documents **4-GPU** Slurm split and sub-agent `vllm` flags.
- **`scripts/launch_verl.py`**: If **`SLURM_CPUS_PER_TASK`** is set, overrides **`ray_init.num_cpus`** for Hydra so Ray does not see the whole node under Slurm; drops **`ROCR_VISIBLE_DEVICES`** when **`CUDA_VISIBLE_DEVICES`** is set (VERL worker compatibility).

### Changed
- **Checkpoint and rollout paths are now unique and co-located** (`scripts/launch_verl.py`, `scripts/train_orchestrator.py`, `jobs/009`, `jobs/010`, `experiments/configs/train/config*.yaml`)
  - Both VERL checkpoints and rollout JSONs land under the same run directory `<base>/<experiment>/<DD-MM-YYYY_HH-MM-JOBID>/`, eliminating the previous mismatch where each script computed its own timestamp independently.
  - Job scripts export `VERL_RUN_TAG` before starting any process; `launch_verl.py` and `train_orchestrator.py` read it first and fall back to self-computing only for manual (non-SLURM) runs.
  - `launch_verl.py` now sets `trainer.default_local_dir` explicitly via Hydra override, replacing VERL's default `checkpoints/<PROJECT_NAME>/<EXPERIMENT_NAME>/` path.
  - Full fine-tuning checkpoints (~47 GB/step) go to `/scratch-shared/$USER/msc-thesis/training/` (`USE_SCRATCH_CHECKPOINTS: true` in `config.yaml`); smoke and LoRA runs stay under `experiments/results/training/` (`USE_SCRATCH_CHECKPOINTS: false` in `config_smoke.yaml`).
  - Checkpoint verification in `jobs/009` and `jobs/010` corrected to match VERL's actual output layout (`global_step_<N>/actor/model_world_size_*_rank_*.pt`) — previous scripts checked `checkpoint_step_*/actor/lora_weights.pt` which VERL never writes.

### Changed
- **Switch full training to H100 and use all 4 GPUs** (`jobs/010_ft_orchestrator.job`, `experiments/configs/train/config.yaml`, `experiments/configs/train/config_smoke.yaml`)
  - `jobs/010`: partition changed from `gpu_a100` (40 GB) to `gpu_h100` (~94 GB NVL). Sub-agent now starts at `--gpu-memory-utilization 0.08` (~7.5 GB) instead of 0.40 so GPU 0 can be shared with VERL. `CUDA_VISIBLE_DEVICES` changed from `1,2,3` to `0,1,2,3`; VERL now uses all 4 H100 GPUs.
  - `config.yaml`: `N_GPUS: 3 → 4`; `gpu_memory_utilization: 0.30 → 0.45` (42.3 GB vLLM budget per GPU — 3.5× more KV cache than before); `actor_rollout_ref.ref.fsdp_config.param_offload: true → false` (H100 94 GB has ample headroom, eliminates PCIe transfer on every ref forward). Memory budget: 81.3 GB per VERL GPU (GPUs 1–3); 88.8 GB on GPU 0 (shared with sub-agent), 5.2 GB activation headroom.
  - `config_smoke.yaml`: `gpu_memory_utilization: 0.28 → 0.55` (leverages H100 94 GB on GPU 1; non-vLLM = 18 GB with both offloads, total = 70.2 GB < 94 GB); `max_num_batched_tokens: 4096 → 6144` (was smaller than `max_model_len: 6144` — corrected alignment). Smoke keeps N_GPUS=1 and Qwen3-4B (fast pipeline validation, not throughput benchmark).

### Added
- **`scripts/merge_lora.py`** — post-training LoRA merger: loads a VERL FSDP actor checkpoint, detects whether it is a LoRA or full-parameter run, and saves a merged HuggingFace model. LoRA path: normalises VERL's HF-style keys to PEFT's `base_model.model.*` namespace, calls `PeftModel.load_state_dict(strict=False)` then `merge_and_unload()`. Full-param path: loads state dict directly and saves with `safe_serialization=True`. Tokenizer is copied from the checkpoint's `actor/huggingface/` subdir.

### Added
- **Checkpoint rotation** (`src/fine_tuning/agentflow/verl/trainer.py`)
  - `_rotate_checkpoints(is_best, epoch, val_reward)` — maintains `latest_checkpoint/` and `best_checkpoint/` symlinks pointing into VERL's `global_step_N/` dirs after every epoch
  - `best_checkpoint_info.json` written alongside the checkpoint dir whenever a new best is recorded: `{"epoch": N, "step": N, "val_reward": 0.xxxx}`
  - Old step dirs that are no longer referenced by either symlink are deleted in a background thread (checkpoint dirs are 8-32 GB; synchronous deletion would stall the training loop)
  - `_save_checkpoint()` override disables VERL's built-in `max_actor_ckpt_to_keep` rotation (wrapped in `open_dict` to handle OmegaConf struct configs safely)
- **`total_training_steps` fix for `train_max_samples`** (`src/fine_tuning/agentflow/verl/trainer.py`)
  - `_create_dataloader()` override recomputes `self.total_training_steps` after truncating the dataset to `train_max_samples`, so `is_last_step` and the tqdm progress bar are based on the truncated dataloader rather than the full dataset the base class measured (was showing `2/25` instead of `2/2` in smoke runs)
- **`ModelFamily` detection helper** (`src/fine_tuning/rollout.py`)
  - `_model_family_from_id(model_id)` — derives `ModelFamily` from a HuggingFace model ID string; used by `_make_model_config()` for both orchestrator and sub-agent providers so non-Qwen models (DeepSeek, OLMo) get the correct prompt/tool-call format

### Changed
- **Epoch-boundary validate-then-save ordering** (`src/fine_tuning/agentflow/verl/trainer.py`)
  - When `val_every_epoch=true` and `save_every_epoch=true`, validation now always runs before the save so the `val/reward` score is available to drive `is_best` checkpoint selection; previously these were independent code paths that could diverge
  - `best_val_reward` tracked across epochs; `_rotate_checkpoints` receives `is_best` flag based on cumulative best
  - In-loop validation/save (`test_freq` / `save_freq`) now correctly guarded by `not val_every_epoch` / `not save_every_epoch`; no double-validation on the final epoch
  - `done` flag + `break` replaces `return` on `is_last_step` so the post-epoch block (validate → save → rotate) always fires, including for the final step
  - Natural-epoch-exhaustion path (when `train_max_samples` truncates below `total_training_steps`) now has a matching cleanup block (`progress_bar.close()`, `del logger`)
  - `agentops` import in `reward.py` moved inside `reward()` with `ImportError` fallback (was a top-level import that failed when agentops is not installed)
- **`actor_rollout_ref.rollout.free_cache_engine: false`** added to `experiments/configs/train/config.yaml` to make the intent explicit (prevents vLLM from freeing the cache engine between rollout and update phases on multi-GPU runs where memory is sufficient)

### Changed
- **Flow GRPO: propagate final reward to all turns** (`src/fine_tuning/rollout.py`)
  - Previously, only the last triplet in a multi-turn rollout received `reward=reward_value`; all intermediate turns (planning, tool-call steps) had `reward=None`, so they contributed no gradient signal.
  - Now every triplet in the trajectory receives the same final sparse reward, matching the AgentFlow Flow GRPO design (`daemon.py:656-695`). GRPO advantage normalisation within each question group is unchanged — only the per-turn reward assignment changes.
  - **Why it matters:** the CoSMAS orchestrator runs a planning turn + one or more tool-call turns + a synthesis turn. Training with reward only on the synthesis step ignores whether the model correctly decided to call a tool, which tool to call, and how to formulate the query — all learnable behaviours. Flow GRPO exposes those decisions to the gradient.

### Fixed
- **LoRA Hydra key mismatch** (`scripts/launch_verl.py`): `+actor_rollout_ref.model.lora_target_modules` used a `+` prefix which adds a new orphaned key (`lora_target_modules`) not present in VERL's schema; VERL reads `actor_rollout_ref.model.target_modules` (no prefix, no `lora_` prefix). Changed to `actor_rollout_ref.model.target_modules` without `+`.
- **LoRA vLLM startup failure** (`scripts/launch_verl.py`): `actor_rollout_ref.rollout.load_format` was left at the default `dummy_dtensor` when LoRA was enabled. vLLM's `dummy_dtensor` starts with zero weights, so FSDP→vLLM base-weight sync was missing entirely — LoRA deltas were pushed on top of zeros. Added `actor_rollout_ref.rollout.load_format=safetensors` when `USE_LORA=true` so vLLM loads the base weights from disk on startup.
- **LoRA weight-sync performance** (`scripts/launch_verl.py`): Added `actor_rollout_ref.rollout.layered_summon=True` (layer-by-layer FSDP→vLLM sync reduces peak GPU memory during sync) and `actor_rollout_ref.model.use_shm=True` (shared memory for weight transfer) when `USE_LORA=true`, matching the VERL LoRA reference configuration.
- **LoRA debug placeholder** (`src/fine_tuning/agentflow/verl/peft_vllm_weight_sync_patch.py`): `TensorLoRARequest(lora_path="simon_lora_path", ...)` had a leftover debug string as the adapter identifier. Changed to `"cosmas_lora_adapter"`.
- **`config.yaml` vLLM KV cache exhaustion** (`experiments/configs/train/config.yaml`): Without `max_model_len` and `max_num_batched_tokens`, vLLM defaulted to Qwen3-8B's native ~40960-token context window, leaving only ~2 usable KV slots at `gpu_memory_utilization=0.30` on 40 GB A100. Added both set to 22528 (18432 max_prompt + 4096 max_response), matching the pattern already in `config_smoke.yaml`.
- **`jobs/009` log directory** (`jobs/009_test_small_ft_example.job`): `mkdir -p out/fine_tuning` did not create the `smoke_test/` subdirectory that `#SBATCH --output` and vLLM log redirects write to; changed to `mkdir -p out/fine_tuning/smoke_test`.
- **Ray dashboard crash on HPC nodes** (`AgentFlow/agentflow/verl/entrypoint.py`): `MetricsHead` (Prometheus) timed out on Snellius nodes where Prometheus is unavailable, cascading into the raylet failing to register with GCS and Ray refusing to start. Added `include_dashboard=False` to `ray.init()`.
- **VERL→vLLM LoRA weight-key mismatch** (`AgentFlow/agentflow/verl/peft_vllm_weight_sync_patch.py`): `FSDPVLLMShardingManager.update_params` called `replace_lora_wrapper` which re-added `.base_layer.` to keys that `__collect_lora_params` had already stripped, causing `KeyError: '*.qkv_proj.base_layer.weight'` in vLLM's Qwen3 `load_weights` during the first FSDP→vLLM weight sync. Patch bypasses `replace_lora_wrapper` on the first-sync path (`base_sync_done=False`) so vLLM receives standard HuggingFace parameter names; `base_sync_done` is still set to `True` afterwards.
- **Ray + Slurm**: Training jobs no longer wedge on worker registration when Ray auto-detects all host CPUs — `ray_init.num_cpus` aligned with **`#SBATCH --cpus-per-task=16`** in YAML and enforced from **`SLURM_CPUS_PER_TASK`** at launch.
- **Fine-tuning GPU memory (Snellius 40GB A100)**: Avoided CUDA OOM and vLLM **`init_device` / KV-cache** startup failures by **not** sharing GPU 0 between the frozen sub-agent and VERL (FSDP + rollout), by **conservative** rollout `gpu_memory_utilization`, **ref-only FSDP `param_offload`** in `config.yaml`, and sub-agent **`VLLM_USE_V1=0`** with **`--gpu-memory-utilization 0.40`** so `max_model_len` 8192 has enough KV budget.
- **`data/prepare.py` Search-R1 schema** — `PeterJinGo/nq_hotpotqa_train` exposes answers as `golden_answers` and the source as `data_source`; normalization now reads those Hub fields, keeps the first answer as `result` / `extra_info.groundtruth`, and preserves the full answer-alias list in `extra_info.golden_answers`. The Search-R1 downloader now skips empty normalized rows and raises only if it cannot fill the requested val/train counts.
- **`data/prepare.py` DeepMath schema** — `zwhe99/DeepMath-103K` exposes `question` and `final_answer`; normalization previously used non-existent `problem` / `answer`, producing empty DeepMath rows in parquets. Rows now prefer `question` + `final_answer`, with legacy fallbacks for `problem` / `answer` / `instruction`. `_download_deepmath` scans the shuffled split and skips the rare Hub row with an empty normalized question or answer instead of failing the whole job; it raises only if the dataset runs out before filling the requested val/train counts (schema mismatch or pervasive corruption). Unit tests cover Hub-shaped and legacy rows.

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
