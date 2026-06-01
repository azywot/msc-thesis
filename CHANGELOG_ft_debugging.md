# Fine-Tuning Debugging Changelog

Chronological record of bugs found and fixes applied to the VERL+AgentFlow RL
fine-tuning pipeline (Qwen3-8B, GRPO, LoRA rank 64, 4×A100-95 GB on Snellius).

---

## Bug 1 — Missing return values in rollout (smoke_22698716)

**Symptom:** All rollout rewards were `None`.

**Root cause:** `OrchestratorRollout._run_episode` did not `return reward_value`,
and `training_rollout_async` / `validation_rollout_async` did not return the
coroutine result.

**Fix:** Added the missing `return` statements in `src/fine_tuning/rollout.py`.

---

## Bug 2 — vLLM V1 `llm_engine` attribute missing during weight sync (smoke_22698716)

**Symptom:** `AttributeError: 'AsyncLLM' object has no attribute 'llm_engine'`
on the second weight sync (when `base_sync_done=True`).

**Root cause:** `FSDPVLLMShardingManager.update_params` assumed
`inference_engine.llm_engine.add_lora()`, but vLLM V1's `AsyncLLM` has no
`llm_engine` attribute.

**Fix:** In `AgentFlow/agentflow/verl/peft_vllm_weight_sync_patch.py`, fall back
to `inference_engine.add_lora()` when `llm_engine` is absent.

---

## Bug 3 — Missing `force_tool_call` parameter on API providers (smoke_22712786)

**Symptom:** `TypeError: apply_chat_template() got an unexpected keyword argument
'force_tool_call'` on every rollout → all answers `None`.

**Root cause:** `OpenAIProvider.apply_chat_template()` and
`AnthropicProvider.apply_chat_template()` were missing the `force_tool_call: bool
= False` parameter that vLLM/MLX providers have.

**Fix:** Added the parameter to both signatures in
`src/agent_engine/models/api_provider.py` (ignored by API providers).

---

## Bug 4 — Stale rollout-data assertion crash (smoke_22712786)

**Symptom:** `AssertionError` in `_save_rollout` after reward was computed but
before it could be returned → `final_reward=None`.

**Root cause:** `assert existing < self.rollout_n` fired because (a) old rollout
data from previous SLURM jobs accumulated in the same directory, (b) multiple
validation passes per job hit the same idx subdirectories.

**Fix:** Removed the assertion and `FileLock` block from `_save_rollout` in
`src/fine_tuning/rollout.py`.

---

## Bug 5 — VERL silently drops `enable_prefix_caching=False` (ft_23028433 onward)

**Symptom:** vLLM init banner always shows `enable_prefix_caching=True` despite
the VERL config setting it to `False`.

**Root cause:** `build_cli_args_from_config` in
`verl/workers/rollout/vllm_rollout/utils.py` handles bool values as:
```python
if isinstance(v, bool):
    if v:
        cli_args.append(f"--{k}")
    # False → silently skipped!
```
Since `--enable-prefix-caching` uses `BooleanOptionalAction` (default True in
vLLM V1), skipping `False` means the default (True) wins.

**Fix (final, in `scripts/launch_verl.py`):** Inject
`no_enable_prefix_caching=True` via `engine_kwargs.vllm`. Because it is a bool
`True`, `build_cli_args_from_config` emits `--no_enable_prefix_caching`. vLLM's
`FlexibleArgumentParser` normalises underscores → dashes, producing
`--no-enable-prefix-caching`, which `BooleanOptionalAction` honours.

**Earlier attempted fixes that did NOT work:**
- Setting `enable_prefix_caching: false` in `config.yaml` — silent no-op (this bug).
- Monkey-patch in `async_server.py` — dead code (see Bug 6).
- `compilation_config={"cudagraph_mode": "PIECEWISE"}` — mitigated a related
  flash_attn crash but did not address the prefix-caching root cause.
- `compilation_config={"cudagraph_mode": "NONE"}` — same: didn't help.
- `enforce_eager=True` — disabled all compilation; crash persisted because prefix
  caching was still on.

---

## Bug 6 — `PatchedvLLMServer` is dead code in verl 0.7.1 (ft_23060623)

**Symptom:** The monkey-patch for Bug 5 (placed in `PatchedvLLMServer` in
`src/fine_tuning/agentflow/verl/async_server.py`) never executed.

**Root cause:** verl 0.7.1 hardcodes
`server_class = ray.remote(vLLMHttpServer)` in `vLLMReplica.__init__` and
ignores the `custom_async_server` config field. Additionally, `async_server.py`
imported `AsyncvLLMServer` (nonexistent in verl 0.7.1), so the module would fail
to import.

**Fix:**
- Changed import to `vLLMHttpServer`.
- Removed the dead monkey-patch code.
- Added a NOTE comment that `PatchedvLLMServer` is dead code in verl 0.7.1.
- Moved the actual fix to `scripts/launch_verl.py` (see Bug 5).

---

## Bug 7 — Early completion race: KV cache freed while vLLM requests in-flight (THE ROOT CAUSE)

**Symptom:** `CUDA error: illegal memory access` at
`gpu_model_runner.py:1733` (`sampled_token_ids.tolist()`). Crashes persisted
through EVERY compilation/caching setting change: cudagraph_mode PIECEWISE →
NONE, enforce_eager=True, enable_prefix_caching=False. Always on exactly ONE of
four vLLM servers, always when rollout completion < 100%.

**Root cause:** Race condition between early completion and `sleep_replicas()`.

1. The daemon's polling loop hits the early-completion threshold (≥90% done, 2 min
   no progress) and `break`s out — but 6-7 orchestrator workers are still
   mid-conversation, making vLLM HTTP calls.
2. `generate_sequences()` returns to the VERL trainer.
3. The trainer IMMEDIATELY calls `self.checkpoint_manager.sleep_replicas()` (line
   1322 of `ray_trainer.py`), which frees the KV cache on ALL vLLM servers.
4. An in-flight worker's vLLM call hits freed KV cache memory → CUDA crash.

**Evidence:**
- ALL seven crashes had early completion (< 100%): 97.3%, 97.7%, 98.0%, etc.
- ALL successful steps had 100% completion — zero crashes at 100%.
- The CUDA error timestamp always falls AFTER the "Finished" message but while
  HTTP 200s are still being returned (the in-flight requests are active).

**Fix:** Added a drain phase in `daemon.py` `_async_run_until_finished()`:
after the polling loop exits, wait up to 90 s for `_processing_tasks` to empty
(i.e., all in-flight vLLM calls complete) before returning control to VERL.

**Why previous "fixes" didn't help:** They were all red herrings. The crash had
nothing to do with cudagraphs, torch.compile, or prefix caching — it was purely
a concurrency bug where GPU memory was freed under running kernels.

---

## Bug 8 — NCCL collective mismatch: different micro-batch counts per rank (ft_23153423)

**Symptom:** `ProcessGroupNCCL watchdog timeout` after exactly 3600 s (the
`nccl_timeout` value) during `update_actor` at step 28. Ranks 0/1 were blocked
on `_ALLGATHER_BASE (NumelIn=311165952)`, ranks 2/3 on `ALLREDUCE (NumelIn=1)`,
both at NCCL seqnum 398964 — the same sequence counter, different operations.

**Root cause:** VERL's `DataParallelPPOActor.update_policy` calls
`prepare_dynamic_batch(mini_batch, max_token_len=45056)` **without** a
`dp_group` argument.  Inside `rearrange_micro_batches`, the cross-rank
`all_reduce(MAX)` that synchronises `num_micro_batches` across ranks is gated on
`dp_group is not None` — so it silently does nothing.

When the step-28 mini-batch happened to have an uneven token-length distribution
across GPUs (ranks 0/1 shards crossed the 45056-token boundary, giving
`ceildiv(total, 45056) = 2` micro-batches; ranks 2/3 stayed below, giving 1),
ranks 0/1 executed a second FSDP forward pass (AllGather for layer weights)
while ranks 2/3 had already finished backward and entered `_optimizer_step` →
`clip_grad_norm_` (AllReduce of the scalar gradient norm, NumelIn=1).  The two
groups submitted different NCCL ops at the same seqnum → 1-hour deadlock.

The bug is stochastic: it only fires when `_balance_batch`'s Karmarkar-Karp
partition leaves at least one GPU shard with a token total just above 45056.
Steps 21–27 were all fine; step 28's specific data distribution triggered it.

**Fix (in `scripts/launch_verl.py`):** Set `actor_rollout_ref.actor.use_dynamic_bsz=False`.
With fixed batching, every rank always creates exactly
`ppo_mini_batch_size // ppo_micro_batch_size_per_gpu = 8 // 4 = 2` micro-batches —
deterministic, all ranks identical, no collective mismatch possible.

**Upstream root fix (not applied here):** Pass `dp_group=torch.distributed.group.WORLD`
to `prepare_dynamic_batch` in `verl/workers/actor/dp_actor.py` line 560.  The
existing synchronisation code in `rearrange_micro_batches` then fires and aligns
`num_micro_batches` across all ranks via `all_reduce(MAX)`.

**Why `use_dynamic_bsz=False` is safe here:**  Average response length ≈ 250 tokens,
prompt ≈ 1100 tokens.  Each fixed micro-batch of 4 samples × ~1350 tokens ≈ 5400
tokens — well within GPU memory.  Even worst-case (4 × max_prompt 7892 +
max_response 2048) ≈ 40 k tokens is under the 95 GB A100 envelope.

---

## Crash timeline

| Job | Step | Completion | Error site | Real cause |
|-----|------|-----------|------------|------------|
| ft_23028433 | 2 | < 100% | flash_attn.py:330 | Early completion race (Bug 7) |
| ft_23031012 | 21 | < 100% | flash_attn.py:330 | Early completion race (Bug 7) |
| ft_23060623 | 30 | < 100% | flash_attn.py:330 | Early completion race (Bug 7) |
| ft_23071620 | 25 | 97.3% | gpu_model_runner.py:1733 | Early completion race (Bug 7) |
| ft_23092138 | 21 | 98.0% (val) | gpu_model_runner.py:1733 | Early completion race (Bug 7) |
| ft_23117863 | 21 | 98.0% (val) | gpu_model_runner.py:1733 | Early completion race (Bug 7) |
| ft_23118789 | 25 | 97.7% | gpu_model_runner.py:1733 | Early completion race (Bug 7) |
| ft_23153423 | 28 | 100% rollout | ProcessGroupNCCL.cpp:632 | Dynamic bsz collective mismatch (Bug 8) |

---

## Files modified

| File | Changes |
|------|---------|
| `src/fine_tuning/agentflow/verl/daemon.py` | **THE FIX**: drain in-flight vLLM requests (up to 90 s) after early completion before returning to VERL |
| `scripts/launch_verl.py` | Injects `no_enable_prefix_caching=True` (Bug 5 workaround); `enforce_eager` removed; `use_dynamic_bsz=False` (Bug 8 fix) |
| `src/fine_tuning/agentflow/verl/async_server.py` | Fixed dead import (`AsyncvLLMServer` → `vLLMHttpServer`), removed dead monkey-patch, added dead-code NOTE |
| `experiments/configs/fine_tuning/config.yaml` | `save_freq`/`test_freq` 10→5; updated comments |
