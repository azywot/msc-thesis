"""API-based model providers (OpenAI, Anthropic).

This module provides implementations for cloud-based model APIs.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from openai import OpenAI
from anthropic import Anthropic

from .base import BaseModelProvider, GenerationResult, ModelConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider for GPT models."""

    def __init__(self, config: ModelConfig, api_key: str = None):
        """Initialize OpenAI provider.

        Args:
            config: ModelConfig with model ID and generation settings
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__(config)
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate completions using OpenAI API with concurrent requests.

        Args:
            prompts: List of formatted prompt strings

        Returns:
            List of GenerationResult objects
        """
        if len(prompts) == 1:
            # Single prompt: no concurrency overhead
            return self._generate_single(prompts[0])
        
        # Multiple prompts: use ThreadPoolExecutor for concurrent API calls
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as executor:
            future_to_idx = {
                executor.submit(self._generate_single, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()[0]
                except Exception as e:
                    logger.exception("OpenAI generation failed for prompt %s", idx)
                    results[idx] = GenerationResult(
                        text=f"Error: {e}",
                        finish_reason="error",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        metadata={"model": self.config.path_or_id, "role": self.config.role}
                    )
        
        return results
    
    def _generate_single(self, prompt: str) -> List[GenerationResult]:
        """Generate a single completion (helper for batching).

        Args:
            prompt: Single prompt string

        Returns:
            List with single GenerationResult
        """
        response = self.client.chat.completions.create(
            model=self.config.path_or_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed,
        )

        result = GenerationResult(
            text=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            metadata={
                "model": response.model,
                "role": self.config.role,
            }
        )
        return [result]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False
    ) -> str:
        """Apply chat template (OpenAI handles this internally).

        For OpenAI, we return the last message content as the template
        is applied by the API itself.

        Args:
            messages: List of message dicts
            use_thinking: Ignored for OpenAI

        Returns:
            Last message content
        """
        # For OpenAI, we typically send the full conversation
        # For compatibility, return the last user message
        if messages:
            return messages[-1].get("content", "")
        return ""

    def cleanup(self):
        """Cleanup (no-op for API providers)."""
        pass


class AnthropicProvider(BaseModelProvider):
    """Anthropic API provider for Claude models."""

    def __init__(self, config: ModelConfig, api_key: str = None):
        """Initialize Anthropic provider.

        Args:
            config: ModelConfig with model ID and generation settings
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        super().__init__(config)
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        """Generate completions using Anthropic API with concurrent requests.

        Args:
            prompts: List of formatted prompt strings

        Returns:
            List of GenerationResult objects
        """
        if len(prompts) == 1:
            # Single prompt: no concurrency overhead
            return self._generate_single(prompts[0])
        
        # Multiple prompts: use ThreadPoolExecutor for concurrent API calls
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(8, len(prompts))) as executor:
            future_to_idx = {
                executor.submit(self._generate_single, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()[0]
                except Exception as e:
                    logger.exception("Anthropic generation failed for prompt %s", idx)
                    results[idx] = GenerationResult(
                        text=f"Error: {e}",
                        finish_reason="error",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        metadata={"model": self.config.path_or_id, "role": self.config.role}
                    )
        
        return results
    
    def _generate_single(self, prompt: str) -> List[GenerationResult]:
        """Generate a single completion (helper for batching).

        Args:
            prompt: Single prompt string

        Returns:
            List with single GenerationResult
        """
        response = self.client.messages.create(
            model=self.config.path_or_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )

        result = GenerationResult(
            text=response.content[0].text,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            metadata={
                "model": response.model,
                "role": self.config.role,
            }
        )
        return [result]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False
    ) -> str:
        """Apply chat template (Anthropic handles this internally).

        Args:
            messages: List of message dicts
            use_thinking: Ignored for Anthropic

        Returns:
            Last message content
        """
        if messages:
            return messages[-1].get("content", "")
        return ""

    def cleanup(self):
        """Cleanup (no-op for API providers)."""
        pass
