"""vLLM model provider for local inference with tensor parallelism."""

import os
from typing import Any, Dict, List, Optional, Union

from vllm import LLM, SamplingParams

from .base import BaseModelProvider, GenerationResult, ModelConfig, ModelFamily
from .llm_shared import get_llm_lock
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VLLMProvider(BaseModelProvider):
    """vLLM provider for local models with tensor parallelism."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Cache dir: TRANSFORMERS_CACHE → HF_HUB_CACHE → HF_HOME/hub
        hf_hub_cache = os.environ.get(
            "TRANSFORMERS_CACHE",
            os.environ.get(
                "HF_HUB_CACHE",
                os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
            ),
        )
        tensor_parallel_size = self._resolve_tensor_parallel_size(config)
        logger.info(
            "Model %s: tensor_parallel_size=%d (source: %s)",
            config.name,
            tensor_parallel_size,
            "config" if config.tensor_parallel_size is not None
            else ("gpu_ids" if config.gpu_ids else "auto-detected"),
        )

        # Pin to config.gpu_ids then restore env so other models see full device set
        prev_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if config.gpu_ids:
                visible_list = (
                    [x.strip() for x in prev_cuda_visible.split(",") if x.strip()]
                    if prev_cuda_visible
                    else None
                )
                if visible_list:
                    mapped = [visible_list[i] for i in config.gpu_ids]
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mapped)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))

            self.llm = LLM(
                model=config.path_or_id,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=config.max_model_len,
                gpu_memory_utilization=config.gpu_memory_utilization,
                seed=config.seed,
                download_dir=hf_hub_cache,
                enforce_eager=True,
            )
        finally:
            if prev_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda_visible

        self.tokenizer = self.llm.get_tokenizer()
        self._lock = get_llm_lock(config.path_or_id)

    @staticmethod
    def _resolve_tensor_parallel_size(config) -> int:
        """Determine tensor_parallel_size from config or environment.

        Resolution order:
        1. Explicit value in config (non-None)
        2. len(config.gpu_ids) when gpu_ids are pinned
        3. All visible CUDA devices (torch.cuda.device_count()), min 1
        """
        if config.tensor_parallel_size is not None:
            return config.tensor_parallel_size
        if config.gpu_ids:
            return len(config.gpu_ids)
        try:
            import torch
            n = torch.cuda.device_count()
            # Use all GPUs only when utilization is high; else single-GPU
            if n > 1 and config.gpu_memory_utilization >= 0.95:
                return n
            return 1
        except Exception:
            return 1

    def generate(self, prompts: Union[List[str], List[Dict[str, Any]]]) -> List[GenerationResult]:
        """Generate completions using vLLM. Supports text prompts and multimodal (image) prompts.

        For multimodal prompts, pass a list of dicts: [{"prompt": str, "multi_modal_data": {"image": PIL.Image}}].
        vLLM's native multimodal API is used; tokenizer.encode is skipped for image prompts.
        """
        # Detect multimodal inputs (VLM image analysis)
        is_multimodal = (
            prompts
            and isinstance(prompts[0], dict)
            and "prompt" in prompts[0]
            and "multi_modal_data" in prompts[0]
        )

        if is_multimodal:
            return self._generate_multimodal(prompts)

        return self._generate_text(prompts)

    def _generate_multimodal(
        self, prompts: List[Dict[str, Any]]
    ) -> List[GenerationResult]:
        """Generate for VLM multimodal (image + text) inputs."""
        safe_max_tokens = min(self.config.max_tokens, 2048)
        sampling_params = SamplingParams(
            max_tokens=safe_max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
            seed=self.config.seed,
        )

        with self._lock:
            outputs = self.llm.generate(
                prompts,
                sampling_params=sampling_params,
            )

        results = []
        for output in outputs:
            results.append(
                GenerationResult(
                    text=output.outputs[0].text,
                    finish_reason=output.outputs[0].finish_reason,
                    usage={
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(output.outputs[0].token_ids),
                        "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                    },
                    metadata={"model": self.config.name, "role": self.config.role},
                )
            )
        return results

    def _generate_text(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate for text-only prompts."""
        try:
            if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'model_config'):
                max_model_len = self.llm.llm_engine.model_config.max_model_len
            elif hasattr(self.llm, 'model_config'):
                max_model_len = self.llm.model_config.max_model_len
            else:
                max_model_len = self.config.max_model_len or 32768
                logger.warning(
                    "Could not determine max_model_len from vLLM engine; using %s",
                    max_model_len,
                )
        except Exception as e:
            max_model_len = self.config.max_model_len or 32768
            logger.exception(
                "Error getting max_model_len from vLLM engine; using %s",
                max_model_len,
            )

        sampling_params_list = []
        valid_prompts = []
        for idx, prompt in enumerate(prompts):
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            available_tokens = max_model_len - prompt_tokens
            safe_max_tokens = min(self.config.max_tokens, max(512, available_tokens))

            if available_tokens < 100:
                logger.warning(
                    "Prompt %s is too long (%s tokens). Truncating to fit.",
                    idx + 1,
                    prompt_tokens,
                )
                target_length = max_model_len - 1024
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                truncated_tokens = tokens[-target_length:]
                prompt = self.tokenizer.decode(truncated_tokens)
                safe_max_tokens = 1024
                logger.info(
                    "Truncated prompt %s to %s tokens.",
                    idx + 1,
                    len(truncated_tokens),
                )

            valid_prompts.append(prompt)
            # Pause generation after a tool call
            stop_kwargs: Dict[str, Any] = {}
            if self.config.role == "orchestrator":
                stop_kwargs = {"stop": ["</tool_call>"], "include_stop_str_in_output": True}
            params = SamplingParams(
                max_tokens=safe_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                seed=self.config.seed,
                **stop_kwargs,
            )
            sampling_params_list.append(params)

        with self._lock:  # shared instance may be used from multiple threads
            outputs = self.llm.generate(valid_prompts, sampling_params=sampling_params_list)

        results = []
        for output in outputs:
            result = GenerationResult(
                text=output.outputs[0].text,
                finish_reason=output.outputs[0].finish_reason,
                usage={
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                    "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                },
                metadata={
                    "model": self.config.name,
                    "role": self.config.role,
                }
            )
            results.append(result)

        return results

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False
    ) -> str:
        """Apply model-specific chat template."""
        if use_thinking and self.config.family == ModelFamily.QWEN3:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking=True,
            )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def cleanup(self):
        """Release GPU memory."""
        if hasattr(self, 'llm'):
            del self.llm
            self.llm = None
