"""vLLM model provider for local model inference.

This module provides a concrete implementation of BaseModelProvider using vLLM
for efficient local model inference with tensor parallelism support.
"""

import os
from typing import Dict, List

from vllm import LLM, SamplingParams

from .base import BaseModelProvider, GenerationResult, ModelConfig, ModelFamily
from .llm_shared import get_llm_lock
from ..utils.logging import get_logger

logger = get_logger(__name__)


class VLLMProvider(BaseModelProvider):
    """vLLM provider for local models with tensor parallelism."""

    def __init__(self, config: ModelConfig):
        """Initialize vLLM provider.

        Args:
            config: ModelConfig with model path and generation settings
        """
        super().__init__(config)

        # Set CUDA devices if specified
        if config.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))

        # Initialize vLLM engine
        self.llm = LLM(
            model=config.path_or_id,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            trust_remote_code=True,
        )

        # Get tokenizer
        self.tokenizer = self.llm.get_tokenizer()
        
        # Get lock for thread-safe generation on shared instances
        self._lock = get_llm_lock(config.path_or_id)

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate completions using vLLM.

        Args:
            prompts: List of formatted prompt strings

        Returns:
            List of GenerationResult objects
        """
        # Get max_model_len from vLLM engine
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

        # Prepare per-prompt sampling params to handle context length limits
        sampling_params_list = []
        valid_prompts = []
        
        for idx, prompt in enumerate(prompts):
            # Calculate prompt length
            prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            
            # Calculate available tokens
            available_tokens = max_model_len - prompt_tokens
            
            # Determine safe max_tokens (reserve at least 512 tokens for generation if possible)
            safe_max_tokens = min(self.config.max_tokens, max(512, available_tokens))
            
            if available_tokens < 100:  # Extremely tight context
                logger.warning(
                    "Prompt %s is too long (%s tokens). Truncating to fit.",
                    idx + 1,
                    prompt_tokens,
                )
                # Truncate prompt to allow at least 1024 generation tokens
                target_length = max_model_len - 1024
                # Keep the last target_length tokens (preserving end is better for chat)
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
            
            params = SamplingParams(
                max_tokens=safe_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                seed=self.config.seed,
            )
            sampling_params_list.append(params)

        # Generate with per-prompt sampling params (thread-safe via lock)
        with self._lock:
            outputs = self.llm.generate(valid_prompts, sampling_params=sampling_params_list)

        # Convert to GenerationResult
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
        """Apply model-specific chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            use_thinking: Whether to enable thinking mode (for Qwen3)

        Returns:
            Formatted prompt string
        """
        # Handle thinking mode for Qwen3 models
        if use_thinking and self.config.family == ModelFamily.QWEN3:
            # Qwen3 thinking mode: add thinking parameter
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                thinking=True  # Enable thinking mode
            )
        else:
            # Standard chat template
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def cleanup(self):
        """Release GPU memory."""
        if hasattr(self, 'llm'):
            # vLLM doesn't have explicit cleanup, but we can delete the object
            # to help with garbage collection
            del self.llm
            self.llm = None
