# GAIA Experiment Configurations

Ready-to-use YAML configs for running GAIA validation with different execution modes, thinking strategies, and model setups.

## 🎯 Key Concept: Mode Independence

Two orthogonal settings can be mixed freely:

1. **Tool Execution Mode** (`tools.direct_tool_call`)
   - `true` = Direct execution (tools return raw results)
   - `false` = Sub-agent mode (LLM analyzes tool results)

2. **Thinking Mode** (`thinking_mode`)
   - `NO` = No thinking mode
   - `PLANNER_ONLY` = Only orchestrator uses thinking
   - `SUBAGENTS_ONLY` = Only tool sub-agents use thinking
   - `ALL` = Both use thinking

## Available configurations

### `baseline.yaml` (recommended)
- **setup**: 1 model (planner), direct tools, `thinking_mode: PLANNER_ONLY`
- **resources (typical)**: 2× H100 (TP=2), ~8–10h for full validation

Run:

```bash
./jobs/submit_job.sh experiment experiments/configs/gaia/baseline.yaml
```

### `direct_mode.yaml`
Same as `baseline.yaml` (kept as an explicit “direct mode” example).

### `multimodel_subagent.yaml`
- **setup**: 3 models (planner + tool sub-agents), sub-agent tools, `thinking_mode: ALL`
- **resources (typical)**: 4× H100, ~12–16h

### `subagent_planner_thinking.yaml` (independence demo)
- **setup**: sub-agent tools, but `thinking_mode: PLANNER_ONLY`
- **meaning**: sub-agents run tools, but only the planner uses thinking

### `subagent_no_thinking.yaml`
- **setup**: sub-agent tools, `thinking_mode: NO`
- **goal**: fastest sub-agent variant (no thinking overhead)

## Configuration matrix (quick compare)

| Config | Models | direct_tool_call | thinking_mode | GPUs (typical) | Use case |
|--------|--------|------------------|---------------|----------------|----------|
| `baseline.yaml` | 1 (32B) | `true` | `PLANNER_ONLY` | 2 | Production baseline |
| `direct_mode.yaml` | 1 (32B) | `true` | `PLANNER_ONLY` | 2 | Same as baseline |
| `multimodel_subagent.yaml` | 3 (32B+14B+14B) | `false` | `ALL` | 4 | Max capability |
| `subagent_planner_thinking.yaml` | 3 (32B+14B+14B) | `false` | `PLANNER_ONLY` | 4 | Independence demo |
| `subagent_no_thinking.yaml` | 3 (32B+14B+14B) | `false` | `NO` | 4 | Fast sub-agents |

## 📊 Configuration Matrix

| Config | Models | Instance Reuse | direct_tool_call | thinking_mode | GPUs | Runtime | Best For |
|--------|--------|----------------|------------------|---------------|------|---------|----------|
| `baseline.yaml` | 1 (32B) | N/A | ✅ true | PLANNER_ONLY | 2 | ~8-10h | **Production** |
| `subagent_shared_model.yaml` 🆕 | 1 (32B shared) | ✅ Yes | ❌ false | PLANNER_ONLY | 2 | ~10-12h | **Sub-agents, low memory** |
| `multimodel_subagent.yaml` | 3 separate | ❌ No | ❌ false | ALL | 4 | ~12-16h | **Max capability** |
| `subagent_planner_thinking.yaml` | 3 separate | ❌ No | ❌ false | PLANNER_ONLY | 4 | ~11-14h | **Ablations** |
| `subagent_no_thinking.yaml` | 3 separate | ❌ No | ❌ false | NO | 4 | ~10-13h | **Fast sub-agents** |

## 🛠️ Creating Custom Configs

### Minimal Template (Direct Mode)

```yaml
name: "my_gaia_experiment"
models:
  planner:
    family: "qwen3"
    path_or_id: "Qwen/Qwen3-32B"
    role: "planner"
    gpu_ids: [0, 1]

tools:
  direct_tool_call: true
dataset:
  name: "gaia"
  split: "all_validation"
output_dir: "./experiments/results/my_experiment"
```

*Unspecified parameters use defaults from `config/schema.py` and `models/base.py`*

### Shared Model Template (Sub-Agent Mode)

```yaml
name: "my_subagent_experiment"
models:
  planner:
    path_or_id: "Qwen/Qwen3-32B"
    gpu_ids: [0, 1]
  search:
    path_or_id: "Qwen/Qwen3-32B"  # Reuses planner!
  code:
    path_or_id: "Qwen/Qwen3-32B"  # Reuses planner!

tools:
  direct_tool_call: false  # Enable sub-agents
dataset:
  name: "gaia"
  split: "all_validation"
output_dir: "./experiments/results/my_experiment"
```

## 🚀 Running Experiments

### Submit to SLURM

```bash
# Using convenience wrapper
./jobs/submit_job.sh experiment experiments/configs/gaia/baseline.yaml

# Or manually generate + submit
python jobs/scripts/generate_job.py experiments/configs/gaia/baseline.yaml
sbatch jobs/generated/gaia_qwen3_baseline.job
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live log (from output_dir)
tail -f experiments/results/gaia_baseline/gaia_qwen3_baseline_<job_id>.log

# Check runner logs
tail -f experiments/results/gaia_baseline/experiment.log
```

### Local Testing (Small Subset)

```bash
# For a quick local smoke test, duplicate a config and set:
#   dataset.subset_num: 5
#
# Then run:
python scripts/run_experiment.py --config experiments/configs/gaia/baseline.yaml --output-dir ./test_run
```

## 📊 Results and Analysis

### Generated Artifacts

Every run produces in `output_dir/`:

```
experiments/results/gaia_baseline/
├── results.json                    # Main results (full traces)
├── results_partial.json            # Checkpoint saves
├── all_validation.02.17,14:30.json # Legacy format (compatibility)
├── all_validation.02.17,14:30.metrics.json
├── experiment.log                  # Runner logs
└── gaia_qwen3_baseline_12345.log   # SLURM logs (if submitted)
```

### Analyze Results

**Basic analysis:**
```bash
python scripts/analyze_results.py \
    experiments/results/gaia_baseline/results.json
```

**Detailed breakdown:**
```bash
# By difficulty level
python scripts/analyze_results.py \
    experiments/results/gaia_baseline/results.json \
    --by-level

# Tool usage analysis
python scripts/analyze_results.py \
    experiments/results/gaia_baseline/results.json \
    --tools

# Both
python scripts/analyze_results.py \
    experiments/results/gaia_baseline/results.json \
    --by-level --tools
```

### Expected Performance

| Model | Mode | Thinking | GAIA Validation Accuracy |
|-------|------|----------|-------------------------|
| Qwen3-32B | Direct | Planner Only | ~40-45% |
| Qwen3-32B | Sub-agent (shared) | Planner Only | ~42-47% |
| Qwen3-32B + 14B tools | Sub-agent | ALL | ~45-50% |

*Note: Actual results vary by seed, tool usage, and configuration details.*

## 📈 W&B Logging (Optional)

Enable experiment tracking with Weights & Biases:

**1. Set in YAML:**
```yaml
use_wandb: true
wandb_project: "my-gaia-experiments"
```

**2. Set API key:**
```bash
export WANDB_API_KEY="your_key_here"
```

**Logged Metrics:**
- Overall accuracy, F1, exact match
- Per-level performance (Level 1, 2, 3)
- Tool usage statistics
- Safe config summary (no API keys)

## 🆕 Advanced Features

### Model Instance Reuse

When using sub-agent mode with the same model:
```yaml
models:
  planner:
    path_or_id: "Qwen/Qwen3-32B"
    gpu_ids: [0, 1]
  search:
    path_or_id: "Qwen/Qwen3-32B"  # Auto-reuses planner instance!
  code:
    path_or_id: "Qwen/Qwen3-32B"  # Auto-reuses planner instance!
```

**Benefits:**
- ✅ 3× GPU memory savings
- ✅ Thread-safe sharing
- ✅ No configuration changes needed

See `docs/MODEL_INSTANCE_REUSE.md` for details.

### Batch Processing

Control parallelization:
```bash
# Process 16 questions per turn (faster on vLLM)
python scripts/run_experiment.py --config config.yaml --batch-size 16

# Process 1 at a time (debugging)
python scripts/run_experiment.py --config config.yaml --batch-size 1
```

## 📚 Further Reading

- **`../../../jobs/QUICKSTART.md`** - HPC setup and SLURM workflow
- **`../../../README.md`** - Main documentation

---

**Last Updated**: 2026-02-17  
**Status**: ✅ All configs tested and production-ready
