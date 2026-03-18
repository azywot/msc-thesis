"""MLX model provider for local inference on Apple Silicon Macs."""

import json
from typing import Any, Dict, List, Optional, Union

from .base import BaseModelProvider, GenerationResult, ModelConfig, _DEEPSEEK_FAMILIES
from ..utils.logging import get_logger, format_messages_as_chat

logger = get_logger(__name__)


class MLXProvider(BaseModelProvider):
    """MLX provider for local models on Apple Silicon using mlx-lm.

    Supports both single-prompt generation (mlx_lm.generate) and batched
    generation (mlx_lm.batch_generate). Use batch_size > 1 in the experiment
    config to enable batching — all prompts in the batch are processed in
    parallel on the Apple Silicon GPU.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "mlx-lm is required for the MLX backend. "
                "Install it with: uv pip install -e '.[mlx]'"
            )

        logger.info("Loading MLX model: %s", config.path_or_id)
        self.model, self.tokenizer = load(config.path_or_id)
        logger.info("MLX model loaded: %s", config.name)

    def generate(self, prompts: Union[List[str], List[Dict[str, Any]]]) -> List[GenerationResult]:
        """Generate completions for a batch of prompts.

        Uses mlx_lm.batch_generate for >1 prompts (parallel GPU execution)
        and mlx_lm.generate for a single prompt.
        """
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        sampler = make_sampler(
            temp=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        rep_penalty = self.config.repetition_penalty
        logits_processors = make_logits_processors(
            repetition_penalty=rep_penalty if rep_penalty != 1.0 else None,
        )

        # Decode all prompts from JSON payloads and render via chat template
        rendered_list: List[str] = []
        msgs_list: List[Optional[List[Dict[str, Any]]]] = []
        for idx, prompt in enumerate(prompts):
            msgs: Optional[List[Dict[str, Any]]] = None
            use_thinking = False
            if isinstance(prompt, str):
                try:
                    payload = json.loads(prompt)
                    if isinstance(payload, dict) and "messages" in payload:
                        msgs = payload["messages"]
                        use_thinking = payload.get("use_thinking", False)
                    elif isinstance(payload, list):
                        msgs = payload
                except (json.JSONDecodeError, TypeError):
                    pass

            if msgs is not None:
                rendered = self._render_messages(msgs, use_thinking)
                logger.debug(
                    "MLX request (prompt %d/%d):\n%s",
                    idx + 1, len(prompts), format_messages_as_chat(msgs),
                )
            else:
                rendered = prompt if isinstance(prompt, str) else str(prompt)

            rendered_list.append(rendered)
            msgs_list.append(msgs)

        responses = self._generate_responses(
            rendered_list, sampler, logits_processors,
        )

        results = []
        for rendered, msgs, response in zip(rendered_list, msgs_list, responses):
            response = self._truncate_tool_call(response)
            prompt_tokens = len(self.tokenizer.encode(rendered))
            completion_tokens = len(self.tokenizer.encode(response))
            results.append(GenerationResult(
                text=response,
                finish_reason="stop",
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                metadata={"model": self.config.name, "role": self.config.role},
                messages=msgs,
            ))

        return results

    def _generate_responses(
        self,
        rendered_list: List[str],
        sampler,
        logits_processors,
    ) -> List[str]:
        """Dispatch to batch_generate (>1 prompts) or generate (1 prompt)."""
        if len(rendered_list) == 1:
            from mlx_lm import generate as mlx_generate
            response = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=rendered_list[0],
                max_tokens=self.config.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False,
            )
            return [response]

        # Batch path: tokenize first (batch_generate needs List[List[int]])
        from mlx_lm import batch_generate
        token_ids = [self.tokenizer.encode(r) for r in rendered_list]
        batch_result = batch_generate(
            self.model,
            self.tokenizer,
            prompts=token_ids,
            max_tokens=self.config.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        # batch_generate returns a BatchResponse; .responses is a list of strings
        return list(batch_result.responses)

    def _truncate_tool_call(self, response: str) -> str:
        """Truncate output at </tool_call> for the orchestrator role.

        mlx_lm has no string-level stop support, so we post-process.
        This matches vLLM's stop=["</tool_call>"] behaviour.
        """
        if self.config.role != "orchestrator" or "<tool_call>" not in response:
            return response
        end_tag = "</tool_call>"
        tag_pos = response.find(end_tag)
        if tag_pos != -1:
            return response[:tag_pos + len(end_tag)]
        # Model opened <tool_call> but didn't close it
        return response + end_tag

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False,
    ) -> str:
        """Serialize messages as JSON payload (same contract as VLLMProvider)."""
        return json.dumps(
            {"messages": messages, "use_thinking": use_thinking},
            ensure_ascii=False,
        )

    def cleanup(self):
        """Release model from memory."""
        if hasattr(self, "model"):
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None

    def _render_messages(self, msgs: List[Dict[str, Any]], use_thinking: bool) -> str:
        """Apply tokenizer chat template, matching VLLMProvider behaviour.

        DeepSeek templates don't accept ``enable_thinking``; thinking is
        controlled by manually closing the ``<think>`` block.
        """
        if self.config.family in _DEEPSEEK_FAMILIES:
            rendered = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            if not (use_thinking and self.config.supports_thinking):
                if rendered.endswith("<think>\n"):
                    rendered += "</think>\n\n"
                else:
                    rendered += "<think>\n</think>\n\n"
            return rendered

        thinking_flag = use_thinking and self.config.supports_thinking
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking_flag,
        )
