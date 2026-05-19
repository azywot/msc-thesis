# VERL upgrade — 0.5.0 → 0.7.1

*Date: 2026-05-19. Files touched: `jobs/environment_train.yml`, `pyproject.toml`.*

## Why upgrade

The training stack was pinned to `verl==0.5.0` (matching AgentFlow's `scripts/setup_stable_gpu.sh`). Two problems with that pin:

1. **vLLM V1 incompatibility for LoRA.** verl 0.5.0's `FSDPVLLMShardingManager.update_params` calls `self.inference_engine.llm_engine.add_lora(lora_request)`, but vLLM V1 dropped the `llm_engine` attribute on `WorkerWrapperBase`. We carried an in-repo monkey-patch (`src/fine_tuning/agentflow/verl/peft_vllm_weight_sync_patch.py`) to work around it — see CHANGELOG "Bug 2 fixed" entry and the patch's docstring for the full mechanics.
2. **No actor/ref LoRA sharing.** Each LoRA RLVR step held a separate base-model copy for actor and ref, blowing GPU memory on anything past ~8B.

verl 0.7.1's release notes call out *"LoRA training enhancement with megatron-bridge: actor/ref share, LoRA adapter only refit"* and native vLLM V1 compatibility — exactly the two pain points above.

## What changed

### `jobs/environment_train.yml`

| Package | Before | After | Why |
|---|---|---|---|
| `verl` | 0.5.0 | **0.7.1** | The upgrade itself |
| `vllm` | 0.9.2 | **0.17.0** | verl 0.7.1's release notes target `vllm==0.17.0` (the v0.6.0 release deprecated `ShardingManager`, so older vllms aren't on the supported integration path) |
| `torch` | 2.7.0 | **2.10.0** | Forced by `vllm/requirements/cuda.txt` at tag `v0.17.0`: `torch==2.10.0` |
| `torchvision` | 0.22.0 | **0.25.0** | Compatibility table at github.com/pytorch/vision: torch 2.10 ↔ torchvision 0.25 |
| `torchaudio` | 2.7.0 | **2.10.0** | PyTorch convention: torchaudio version tracks torch |
| `transformers` | 4.53.3 | **4.56.0** | `vllm/requirements/common.txt` at `v0.17.0`: `transformers >= 4.56.0, < 5` |
| `python` | 3.11 | **3.12** | flash-attn 2.8.1's GitHub release assets ship `cu12*torch2.10*cp312`/`cp313` wheels but no `cp311` build; staying on 3.11 would force a 30-90min source build under `--no-build-isolation`. `pyproject.toml`'s `requires-python = ">=3.11"` is satisfied; vllm 0.17, transformers 4.56, and verl 0.7.1 all support 3.12 |

### `pyproject.toml`

`[project.optional-dependencies].training` had `verl==0.5.0` — bumped to `0.7.1` so `pip install -e ".[training]"` doesn't fight the conda env. The other training deps (filelock, omegaconf, codetiming, agentops, …) stay as-is.

### Files **deleted** alongside the bump

- **`src/fine_tuning/agentflow/verl/peft_vllm_weight_sync_patch.py`** — the monkey-patch that worked around verl 0.5's two FSDP→vLLM LoRA sync bugs. verl 0.6.0 deprecated `ShardingManager` entirely, so `FSDPVLLMShardingManager` (the class the patch monkey-patched) no longer exists in 0.7.1 — the workaround targets dead code. Recoverable from git history if a regression surfaces.
- **`src/fine_tuning/agentflow/verl/entrypoint.py`** — removed `from . import peft_vllm_weight_sync_patch` and the three `apply_patch()` call sites (main `run_ppo`, Ray runtime env's `worker_process_setup_hook`, and the `TaskRunner.run` body). The `VLLM_USE_V1=1` enforcement stays — it's not patch-related.

### Files **not** changed

- **`flash-attn==2.8.1`** is installed out-of-band (see `jobs/008_prepare_fine_tuning_data.job`). It is *not* pinned in the env file. The 2.8.1 wheel was built against torch 2.7; whether it imports cleanly under torch 2.10 needs verification on first env build. If broken, bump to whatever flash-attn release lists torch 2.10 wheels.

## Risks introduced by this bump

| Risk | Mitigation |
|---|---|
| verl 0.5 → 0.7 has two breaking-release minor bumps. AgentFlow's vendored integration (`src/fine_tuning/agentflow/verl/{entrypoint,trainer,dataset}.py`) was written against 0.5 APIs | Run `jobs/009_test_small_ft_example.job` end-to-end before any production fine-tune. Expect breakage and triage |
| `transformers` 4.53 → 4.56 may shift chat-template defaults for Qwen3 / DeepSeek / OLMo families. Tool-call rendering in `src/agent_engine/models/vllm_provider.py` depends on those templates | Re-run the smoke tests under `tests/agent_engine/models/` and the OLMo/DeepSeek family parity tests after env rebuild |
| `torch` 2.7 → 2.10 may surface compiled-op / determinism changes. flash-attn 2.8.1 may not have torch-2.10 wheels | Linked above — flash-attn upgrade likely needed |
| The four old "fine-tuning pipeline bugs" recorded in `MEMORY.md` were all fixed *under verl 0.5.0*. Some may resurface or be moot under 0.7.1 | Bugs 1, 3, and 4 are fixes to our own code (rollout return values, OpenAI/Anthropic provider signatures, `_save_rollout` assertion), so they remain version-independent. Bug 2 was the `peft_vllm_weight_sync_patch.py` workaround we just deleted — if a regression surfaces, the file is recoverable from git history |

## How to validate

1. Rebuild env: `conda env remove -n cosmas-train && conda env create -f jobs/environment_train.yml`
2. Reinstall flash-attn: `pip install flash-attn==2.8.1 --no-build-isolation` — if this fails on torch 2.10, find a compatible flash-attn release
3. Reinstall project: `pip install -e ".[training]"`
4. Run the smoke job: `sbatch jobs/009_test_small_ft_example.job`
5. Watch for:
   - Successful first FSDP→vLLM weight sync (no `KeyError: '*.qkv_proj.base_layer.weight'`)
   - Successful second sync (no `AttributeError: 'WorkerWrapperBase' object has no attribute 'llm_engine'`)
   - Non-`None` rewards in `rollout_data/`
6. If green, schedule the patch deletion as a follow-up CHANGELOG entry.
