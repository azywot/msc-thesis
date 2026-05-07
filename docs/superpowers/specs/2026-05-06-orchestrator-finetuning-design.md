# Design Spec: RL Fine-Tuning Pipeline for the Orchestrator

**Date:** 2026-05-06
**Status:** Approved
**Scope:** `msc-thesis` — `src/fine_tuning/` + supporting scripts and configs

---

## 1. Objective

Fine-tune the orchestrator (Qwen3-8B) in the CoSMAS multi-agent system using reinforcement learning, following the AgentFlow training methodology. Only the orchestrator is trained; sub-agents (web search analyser, code generator) remain frozen.

The pipeline replicates AgentFlow's GRPO-based RL loop, using:
- [`shin-ee-chen/AgentFlow`](https://github.com/shin-ee-chen/AgentFlow) as a library (installed from the local clone)
- VERL as the RL training backend
- LoRA for parameter-efficient fine-tuning of Qwen3-8B
- Search-R1 + DeepMath-103K as training data (mixed 50/50)
- The existing `metrics.py` evaluation logic as the reward signal

After training, the LoRA adapter is merged into the base model and used directly in existing msc-thesis inference configs — no changes to the inference code.

---

## 2. Architecture

```
Training time
─────────────────────────────────────────────────────────────────────────

              ┌──────────────────────────────────────────────┐
              │  agentflow.verl  (VERL backend, Ray cluster) │
              │                                              │
              │  AgentFlowTrainer(RayPPOTrainer)             │
              │  ├── GRPO advantage estimator                │
              │  ├── FSDP actor: Qwen3-8B + LoRA rank-64     │
              │  ├── Reference policy: Qwen3-8B (frozen)     │
              │  └── AgentModeDaemon  :9999                  │
              └─────────────────┬────────────────────────────┘
                                │  HTTP  (tasks ↓ / triplets ↑)
              ┌─────────────────▼────────────────────────────┐
              │  scripts/train_orchestrator.py               │
              │                                              │
              │  agentflow.Trainer(tracer=NullTracer())       │
              │  └── OrchestratorRollout(LitAgent)           │
              │      ├── AgenticOrchestrator (planning loop) │
              │      ├── tools: web_search, code_generator   │
              │      └── OrchestratorReward (metrics.py)     │
              └──────────────────────────────────────────────┘

Inference (unchanged)
─────────────────────────────────────────────────────────────────────────

  util/model_merger.py  →  merged HF model  →  path_or_id in YAML
                                                ↓
                                          VLLMProvider (no changes)
```

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| RL backend | VERL via `agentflow.verl` | Replicates AgentFlow exactly; handles Ray cluster, FSDP, GRPO |
| Algorithm | GRPO (n=8 rollouts per question) | Same as AgentFlow; no value network needed |
| Model weights | LoRA rank-64, all-linear layers | ~130 MB checkpoints vs ~16 GB full FT; sufficient compute efficiency for thesis |
| Training data | Search-R1 (NQ + HotpotQA) + DeepMath-103K | Same sources as AgentFlow's `combined_train.parquet` |
| Validation | DeepMath val split (200 held-out questions) | AIME is an evaluation benchmark — using it for checkpoint selection would introduce selection bias in reported results; held-out DeepMath is in-distribution, never evaluated on |
| Reward | Binary exact-match via `metrics.py` | Directly comparable to evaluation benchmark numbers |
| Inference integration | Merge LoRA → HF model → `path_or_id` in YAML | Zero changes to `VLLMProvider` or any inference code |
| Environments | Two separate conda envs (training / inference) | Avoids vLLM version conflict (training: 0.9.2, inference: 0.12.0) |

---

## 3. File and Folder Structure

```
msc-thesis/
├── src/fine_tuning/                    # Implement all stubs
│   ├── __init__.py
│   ├── config.py                       # FinetuningConfig dataclass
│   ├── rollout.py                      # OrchestratorRollout(LitAgent)
│   ├── reward.py                       # OrchestratorReward
│   └── data/
│       ├── __init__.py
│       └── prepare.py                  # Download + convert → parquet
│
├── scripts/
│   ├── train_orchestrator.py           # Entry point: Trainer.fit() (rollout workers)
│   └── launch_verl.py                  # Entry point: starts VERL server (mirrors train_agent.py)
│
├── train/
│   └── config.yaml                     # Training config (4 GPU, LoRA)
│
├── jobs/
│   ├── train_orchestrator.sh           # SLURM job script (Snellius)
│   └── environment_train.yml           # Conda env for training
│
└── data/
    └── training/
        ├── train/
        │   └── combined_train.parquet  # Search-R1 + DeepMath (prepared)
        └── val/
            └── deepmath_val.parquet    # 200 held-out DeepMath questions (never in training set)
            # NOTE: do NOT use aime24.parquet here — AIME is an evaluation benchmark
```

### Mapping from AgentFlow

| AgentFlow source | msc-thesis equivalent |
|---|---|
| `train/rollout.py` → `AgentFlowRollout` | `src/fine_tuning/rollout.py` → `OrchestratorRollout` |
| `train/utils.py` → `compute_score()` | `src/fine_tuning/reward.py` → `OrchestratorReward` |
| `train/config.yaml` | `train/config.yaml` |
| `train/train_agent.py` | `scripts/launch_verl.py` (VERL server) + `scripts/train_orchestrator.py` (rollout workers) |
| `agentflow.verl.*` | imported directly — no copy |
| `util/model_merger.py` | imported directly — no copy |

---

## 4. Component Specifications

### 4.1 `src/fine_tuning/config.py` — `FinetuningConfig`

Dataclass holding all fine-tuning hyperparameters and paths. Loaded from the training YAML at startup.

Fields:
- `base_model: str` — HuggingFace model ID (e.g. `Qwen/Qwen3-8B`)
- `lora_rank: int` — LoRA rank (default: 64)
- `lora_alpha: int` — LoRA alpha (default: 16)
- `lora_target_modules: str` — target layers (default: `all-linear`)
- `train_data: str` — path to training parquet
- `val_data: str` — path to validation parquet
- `output_dir: str` — checkpoint output directory
- `n_gpus: int`, `rollout_tp_size: int`
- `train_temperature: float`, `test_temperature: float`
- `seed: int`
- `wandb_project: str`, `wandb_run_name: str`

---

### 4.2 `src/fine_tuning/rollout.py` — `OrchestratorRollout(LitAgent)`

Wraps `AgenticOrchestrator` as an AgentFlow `LitAgent`. Implements `training_rollout_async` and `validation_rollout_async`.

Responsibilities:
1. Receive a task from the `AgentModeDaemon` (contains `question`, `result`, `extra_info`)
2. Construct an `ApiProvider` (from `src/agent_engine/models/api_provider.py`) pointed at the VERL vLLM server endpoint obtained from `resources.get("main_llm").endpoint` — this is how the rollout workers call the model being trained, matching AgentFlow's `get_agent(..., openai_base_url=vllm_base_url)` pattern
3. Build a `ToolRegistry` (web_search, code_generator) using frozen sub-agent configs from the training YAML
4. Instantiate `AgenticOrchestrator(model_provider=api_provider, tool_registry=...)`
5. Run the full orchestrator loop: planning turn + tool loop + answer extraction
6. Call `OrchestratorReward` to compute the binary reward
7. Package as a rollout result (question, response text, reward) and return to daemon

Closely mirrors `AgentFlowRollout` + `Rollout` in `train/rollout.py`, but constructs `AgenticOrchestrator` instead of `construct_solver()`.

---

### 4.3 `src/fine_tuning/reward.py` — `OrchestratorReward`

Routes reward computation to `metrics.py` based on `data_source`:

```
data_source in {"nq", "hotpotqa"}   →  mode="qa"   (containment-based accuracy)
data_source in {"math", "deepmath"} →  mode="gen"  (exact/near-exact match)
```

The `deepmath_val.parquet` validation split uses `data_source="deepmath"`, so the existing `mode="gen"` branch handles it without any additional routing.

Returns `1.0` (correct) or `0.0` (incorrect). Binary reward, matching AgentFlow's `eval()` function.

---

### 4.4 `src/fine_tuning/data/prepare.py` — data preparation

CLI script: `python src/fine_tuning/data/prepare.py [options]`

Arguments: `--n-search INT`, `--n-math INT`, `--output-dir PATH`, `--seed INT`

Steps:
1. Download `PeterJinGo/SearchR1-nq_hotpotqa_train` from HuggingFace
2. Download `zwhe99/DeepMath-103K` from HuggingFace
3. Normalise both to the VERL schema:
   ```python
   {"data_source": str, "question": str, "result": str,
    "extra_info": {"idx": int, "groundtruth": str}}
   ```
4. Subsample DeepMath: carve out 200 questions first (fixed seed) as the validation split **before** subsampling the training portion; these 200 questions must never appear in `combined_train.parquet`
5. Subsample remainder (default: 10k search + 10k math for fast runs; 50k + 50k for full)
6. Shuffle combined set with fixed seed
7. Write `data/training/train/combined_train.parquet`
8. Write the 200 held-out DeepMath questions to `data/training/val/deepmath_val.parquet`

**Why not AIME24?** AIME is one of the five evaluation benchmarks reported in the thesis. Using it for checkpoint selection would mean the best checkpoint is chosen based on the same questions you later report results on — selection bias. The held-out DeepMath slice is in-distribution with the math training data, never evaluated on, and gives a stable per-epoch learning curve.

---

### 4.5 `scripts/train_orchestrator.py` — entry point

Two entry point scripts, both reading the same `train/config.yaml`.

**`scripts/launch_verl.py`** — starts the VERL training server. Mirrors `train/train_agent.py` from AgentFlow exactly:
1. Parse `--config` YAML argument
2. Set environment variables from `config.env`
3. Build `python -m agentflow.verl key=value key=value ...` command from `config.python_args`
4. Launch via `subprocess.run()`

**`scripts/train_orchestrator.py`** — starts the rollout workers. Runs after VERL is up:
1. Parse `--config` YAML argument
2. Set environment variables from `config.env`
3. Copy config YAML to `output_dir/config.yaml` (reproducibility)
4. Log git commit hash + Snellius `SLURM_JOB_ID` to W&B run config
5. Instantiate `NullTracer` (avoids AgentOps dependency):
   ```python
   class NullTracer(BaseTracer):
       def init(self): pass
       def teardown(self): pass
       def init_worker(self, worker_id): pass
       def teardown_worker(self, worker_id): pass
   ```
6. Start `Trainer(n_workers=N_WORKERS, tracer=NullTracer())`
7. Instantiate `OrchestratorRollout` from config
8. Call `trainer.fit(agent, f"http://localhost:{port}/")`

---

### 4.6 `train/config.yaml`

```yaml
env:
  BASE_MODEL: 'Qwen/Qwen3-8B'
  N_GPUS: 4
  ROLLOUT_TP_SIZE: 2
  EXPERIMENT_NAME: 'qwen3-8b-grpo-search-math'
  PROJECT_NAME: 'cosmas-rl-finetuning'
  BASE_DATA_DIR: 'data/training'
  ENABLE_TOOLS: ["web_search", "code_generator"]
  TOOL_STEPS: 5
  TRAIN_TEMPERATURE: 0.7
  TEST_TEMPERATURE: 0.0
  N_WORKERS: 1

python_args:
  agentflow.port: 9999
  algorithm.adv_estimator: 'grpo'
  data.train_files: '${BASE_DATA_DIR}/train/combined_train.parquet'
  data.val_files:   '${BASE_DATA_DIR}/val/deepmath_val.parquet'
  data.train_batch_size: 32
  actor_rollout_ref.rollout.n: 8
  actor_rollout_ref.actor.ppo_mini_batch_size: 8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: 4
  actor_rollout_ref.rollout.multi_turn.format: 'hermes'
  actor_rollout_ref.model.path: '${BASE_MODEL}'
  actor_rollout_ref.model.lora_rank: 64
  actor_rollout_ref.model.lora_alpha: 16
  actor_rollout_ref.model.lora_target_modules: 'all-linear'
  actor_rollout_ref.model.enable_gradient_checkpointing: true
  actor_rollout_ref.actor.optim.lr: 1e-6
  actor_rollout_ref.actor.use_kl_loss: true
  actor_rollout_ref.actor.kl_loss_coef: 0.001
  actor_rollout_ref.actor.clip_ratio_low: 0.2
  actor_rollout_ref.actor.clip_ratio_high: 0.3
  actor_rollout_ref.rollout.gpu_memory_utilization: 0.6
  actor_rollout_ref.rollout.tensor_model_parallel_size: '${ROLLOUT_TP_SIZE}'
  trainer.n_gpus_per_node: '${N_GPUS}'
  trainer.logger: ['console', 'wandb']
  trainer.project_name: '${PROJECT_NAME}'
  trainer.experiment_name: '${EXPERIMENT_NAME}'
  trainer.save_freq: 2
  trainer.test_freq: 2
  trainer.total_epochs: 5
  trainer.val_before_train: true
  data.max_prompt_length: 18432
  data.max_response_length: 2048
  data.truncation: 'truncate'
  data.train_max_samples: 128
  algorithm.use_kl_in_reward: false
  trainer.critic_warmup: 0
  actor_rollout_ref.rollout.name: 'vllm'
  actor_rollout_ref.actor.fsdp_config.param_offload: false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload: false
  actor_rollout_ref.ref.fsdp_config.param_offload: false
  actor_rollout_ref.actor.use_remove_padding: true
```

---

### 4.7 `jobs/train_orchestrator.sh` — SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=cosmas-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/%j_train.log

module load 2023
module load CUDA/12.1.1
conda activate cosmas-train

cd $HOME/azywot/msc-thesis

# Start VERL server in background (parses config, constructs Hydra args, calls python -m agentflow.verl)
python scripts/launch_verl.py --config train/config.yaml &
VERL_PID=$!

# Wait for Ray + vLLM to be ready
sleep 60

# Start rollout workers (connects to VERL daemon at port 9999)
python scripts/train_orchestrator.py --config train/config.yaml

wait $VERL_PID
```

---

### 4.8 `jobs/environment_train.yml` — training conda env

```yaml
name: cosmas-train
channels:
  - defaults
dependencies:
  - python=3.11
  - pip:
    - torch==2.7.0
    - torchvision==0.22.0
    - torchaudio==2.7.0
    - transformers==4.53.3
    - flash-attn==2.8.1
    - vllm==0.9.2
    - verl==0.5.0
    - agentflow @ file://$HOME/azywot/AgentFlow/agentflow
    - -e .
    - datasets>=3.0.0
    - pyarrow>=15.0.0
    - wandb>=0.16.0
    - filelock
    - sympy
    - omegaconf
    - codetiming
```

---

## 5. Inference Integration

After training completes, merge the LoRA adapter into the base model:

```bash
conda activate cosmas-train
python $HOME/azywot/AgentFlow/util/model_merger.py \
  --base_model Qwen/Qwen3-8B \
  --lora_path experiments/results/training/<run>/checkpoint_best/actor/lora_weights.pt \
  --output_dir experiments/results/training/<run>/merged_model/
```

Then update any existing experiment YAML:

```yaml
# experiments/configs/qwen3/agentflow/qwen3_8b_gaia_ft.yaml
model:
  path_or_id: /path/to/experiments/results/training/<run>/merged_model/
  # everything else unchanged
```

No changes to `VLLMProvider`, `AgenticOrchestrator`, or evaluation scripts.

---

## 6. Logging & Reproducibility

**W&B** is configured via `trainer.logger: ['console', 'wandb']` in the config YAML. VERL's `Tracking` class handles logging automatically.

Logged per step:
- `actor/reward_mean`, `actor/reward_std`
- `actor/kl_divergence`, `actor/entropy_loss`, `actor/pg_loss`
- `val/reward_mean` (every `test_freq` steps)
- `agent_mode/n_dropped_sample_*`
- `timing/*`

Reproducibility:
- Random seed set in config, passed to orchestrator via `src/agent_engine/utils/seed.py`
- Config YAML copied to `output_dir/config.yaml` at run start
- Git commit hash + `SLURM_JOB_ID` logged to W&B run config by entrypoint

---

## 7. Checkpointing

VERL's `_save_checkpoint()` saves every `save_freq` steps. With LoRA, only adapter weights are saved (~130 MB per checkpoint vs ~16 GB full FT).

Layout:
```
experiments/results/training/<run_name>/
├── config.yaml
├── checkpoint_step_2/actor/lora_weights.pt
├── checkpoint_step_4/actor/lora_weights.pt
└── checkpoint_best/ → symlink to best val checkpoint
```

---

## 8. Potential Pitfalls

| Risk | Mitigation |
|---|---|
| VERL API changes between versions | Pin `verl==0.5.0` in `environment_train.yml`; do not upgrade without testing |
| AgentFlow Trainer requires AgentOps | `NullTracer` in `scripts/train_orchestrator.py` removes this dependency entirely |
| Search-R1 / DeepMath column name mismatch | `data/prepare.py` normalises all columns to VERL schema before writing parquet |
| Two vLLM versions (0.9.2 training / 0.12.0 inference) | Separate conda envs; rollout workers use `vllm==0.9.2` via training env |
| OOM on 4× A100 during rollout | Tune `gpu_memory_utilization` (0.55–0.65); enable gradient checkpointing (already on) |
| Long rollouts overflowing `max_prompt_length` | `data.truncation: 'truncate'` ensures silent truncation rather than crash |

---

## 9. Out of Scope

- Fine-tuning sub-agents (web search analyser, code generator) — frozen throughout
- SFT warm-up stage — going directly to GRPO
- Reward model — binary exact-match only
- Multi-node training — single node, 4 GPUs
