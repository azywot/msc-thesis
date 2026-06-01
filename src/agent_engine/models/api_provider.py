"""API-based model providers (OpenAI, Anthropic).

This module provides implementations for cloud-based model APIs.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from openai import OpenAI
from anthropic import Anthropic

from .base import BaseModelProvider, GenerationResult, ModelConfig, _ENABLE_THINKING_KWARG_FAMILIES
from ..utils.logging import get_logger, format_messages_as_chat

logger = get_logger(__name__)


class OpenAIProvider(BaseModelProvider):
    """OpenAI API provider for GPT models."""

    def __init__(self, config: ModelConfig, api_key: str = None, base_url: str = None):
        """Initialize OpenAI provider.

        Args:
            config: ModelConfig with model ID and generation settings
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional base URL for OpenAI-compatible APIs (e.g. vLLM server).
                      When set, overrides the default openai.com endpoint.
        """
        super().__init__(config)
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
        )

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
            prompt: Single prompt string or JSON-encoded messages list

        Returns:
            List with single GenerationResult
        """
        raw_messages = None
        use_thinking = False
        try:
            payload = json.loads(prompt)
            if isinstance(payload, dict) and "messages" in payload:
                raw_messages = payload["messages"]
                use_thinking = bool(payload.get("use_thinking", False))
            elif isinstance(payload, list):
                raw_messages = payload
        except (json.JSONDecodeError, TypeError):
            pass

        if raw_messages is not None:
            # Map "tool" -> "user" (OpenAI native tool role is for function-calling)
            messages = [
                {**m, "role": "user"} if m.get("role") == "tool" else m
                for m in raw_messages
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        logger.debug("OpenAI request:\n%s", format_messages_as_chat(messages))

        extra_body = {}
        if self.config.family in _ENABLE_THINKING_KWARG_FAMILIES:
            extra_body["chat_template_kwargs"] = {"enable_thinking": use_thinking}

        response = self.client.chat.completions.create(
            model=self.config.path_or_id,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed,
            extra_body=extra_body or None,
        )

        # Capture token IDs when the backend includes them (e.g. VERL vLLM proxy).
        # openai-python exposes extra JSON fields via __getattr__ → model_extra.
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        raw_response_token_ids = getattr(response, "response_token_ids", None)
        response_token_ids = raw_response_token_ids[0] if raw_response_token_ids else None

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
            },
            messages=raw_messages,
            prompt_token_ids=list(prompt_token_ids) if prompt_token_ids is not None else None,
            response_token_ids=list(response_token_ids) if response_token_ids is not None else None,
        )
        return [result]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False,
        force_tool_call: bool = False,
    ) -> str:
        """Serialize the full conversation for the OpenAI-compatible API.

        Returns:
            JSON-encoded payload with messages and use_thinking flag
            (deserialized in _generate_single).
        """
        return json.dumps(
            {"messages": messages, "use_thinking": use_thinking},
            ensure_ascii=False,
        )

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
            prompt: Single prompt string or JSON-encoded messages list

        Returns:
            List with single GenerationResult
        """
        raw_messages = None
        system = None
        try:
            payload = json.loads(prompt)
            if isinstance(payload, dict) and "messages" in payload:
                raw_messages = payload["messages"]
            elif isinstance(payload, list):
                raw_messages = payload
        except (json.JSONDecodeError, TypeError):
            pass

        if raw_messages is not None:
            # Anthropic requires: system as top-level param, no "tool" role.
            messages = []
            for m in raw_messages:
                role = m.get("role")
                if role == "system":
                    system = m.get("content", "")
                elif role == "tool":
                    messages.append({"role": "user", "content": m.get("content", "")})
                else:
                    messages.append(m)
        else:
            messages = [{"role": "user", "content": prompt}]

        log_messages = ([{"role": "system", "content": system}] if system else []) + messages
        logger.debug("Anthropic request:\n%s", format_messages_as_chat(log_messages))

        kwargs = dict(
            model=self.config.path_or_id,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

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
            },
            messages=raw_messages,
        )
        return [result]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False,
        force_tool_call: bool = False,
    ) -> str:
        """Serialize the full conversation for the Anthropic API.

        Args:
            messages: List of message dicts
            use_thinking: Ignored for Anthropic
            force_tool_call: Ignored for Anthropic

        Returns:
            JSON-encoded messages list (deserialized in _generate_single)
        """
        return json.dumps(messages, ensure_ascii=False)

    def cleanup(self):
        """Cleanup (no-op for API providers)."""
        pass
