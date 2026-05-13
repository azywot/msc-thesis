"""Fix VERL's FSDP→vLLM weight-key mismatch and vLLM V1 llm_engine incompatibility.

Fix 1 — key mismatch (base_sync_done=False)
--------------------------------------------
In ``FSDPVLLMShardingManager.__enter__`` the first weight sync (``base_sync_done=False``)
calls ``__collect_lora_params``, which correctly strips ``.base_layer.`` from PEFT parameter
names so the keys are in standard HuggingFace format (e.g. ``q_proj.weight``).

``update_params`` then calls ``replace_lora_wrapper``, which *re-adds* ``.base_layer.`` for any
module that is a LoRA target (e.g. ``q_proj.weight`` → ``q_proj.base_layer.weight``).

vLLM's ``load_weights`` for Qwen2/Qwen3 remaps ``q_proj``→``qkv_proj`` via
``stacked_params_mapping``, producing ``qkv_proj.base_layer.weight``, but ``params_dict``
only contains ``qkv_proj.weight`` → ``KeyError``.

In the first-sync path (``not base_sync_done``), skip ``replace_lora_wrapper`` by calling
the original ``update_params`` with ``peft_config=None``.  ``base_sync_done`` is still set
to ``True`` after ``model.load_weights`` completes, so subsequent syncs use the
``TensorLoRARequest`` path as intended.

Fix 2 — vLLM V1 missing llm_engine (base_sync_done=True)
----------------------------------------------------------
``vLLMAsyncRollout`` (mode=async, the AgentFlow default) uses ``WorkerWrapperBase`` as
``inference_engine``.  VERL's ``update_params`` second-sync path calls
``self.inference_engine.llm_engine.add_lora(lora_request)``, but ``WorkerWrapperBase``
has no ``llm_engine`` attribute in vLLM V1 → ``AttributeError``.

When ``inference_engine`` has no ``llm_engine``, call ``inference_engine.add_lora()``
directly.  ``WorkerWrapperBase.__getattr__`` routes this to ``worker.add_lora`` →
``model_runner.add_lora`` → ``lora_manager.add_adapter`` → ``_load_adapter`` (already
patched by ``VLLMHijack.hijack()`` inside ``FSDPVLLMShardingManager.__init__`` to accept
``TensorLoRARequest``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_cosmas_patched"


def apply_patch() -> None:
    """Apply monkey-patches.  Safe to call multiple times (idempotent)."""
    _patch_fsdp_vllm_sharding_manager()


def _patch_fsdp_vllm_sharding_manager() -> None:
    try:
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
    except ImportError:
        return  # VERL not available in this process — nothing to patch

    if getattr(FSDPVLLMShardingManager.update_params, _PATCH_ATTR, False):
        return  # already patched in this process

    _orig = FSDPVLLMShardingManager.update_params

    def _patched_update_params(self, updated_params, peft_config=None):
        if peft_config is not None and not self.base_sync_done:
            # Fix 1: replace_lora_wrapper would re-add .base_layer. that __collect_lora_params
            # just stripped — pass peft_config=None to skip it; base_sync_done is set as usual.
            logger.info(
                "[cosmas] First FSDP→vLLM sync: bypassing replace_lora_wrapper "
                "(peft key-mismatch fix, base_sync_done=False path)"
            )
            return _orig(self, updated_params, peft_config=None)

        if peft_config is not None and self.base_sync_done:
            # Fix 2: vLLMAsyncRollout uses WorkerWrapperBase as inference_engine, which has
            # no .llm_engine in vLLM V1. Call add_lora() directly; WorkerWrapperBase.__getattr__
            # routes it to worker.add_lora → model_runner.add_lora → lora_manager.add_adapter
            # → _load_adapter (already patched by VLLMHijack to accept TensorLoRARequest).
            if not hasattr(self.inference_engine, "llm_engine"):
                import time
                from dataclasses import asdict

                from verl.utils.vllm_utils import TensorLoRARequest

                lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                lora_request = TensorLoRARequest(
                    lora_name=f"{lora_int_id}",
                    lora_int_id=lora_int_id,
                    lora_path="simon_lora_path",
                    peft_config=asdict(peft_config),
                    lora_tensors=updated_params,
                )
                self.inference_engine.add_lora(lora_request)
                logger.info(
                    "[cosmas] V1/SPMD LoRA sync: add_lora via inference_engine.add_lora "
                    f"(no llm_engine, {len(updated_params)} params)"
                )
                return

        return _orig(self, updated_params, peft_config=peft_config)

    setattr(_patched_update_params, _PATCH_ATTR, True)
    FSDPVLLMShardingManager.update_params = _patched_update_params
    logger.info("[cosmas] FSDPVLLMShardingManager.update_params patched (peft key-mismatch fix)")
