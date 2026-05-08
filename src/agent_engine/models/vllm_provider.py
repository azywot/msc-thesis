"""vLLM model provider for local inference with tensor parallelism."""

import json
import os
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

from vllm import LLM, SamplingParams

from .base import (
    BaseModelProvider, GenerationResult, ModelConfig, ModelFamily, ToolCallFormat, get_tool_call_format,
    _ENABLE_THINKING_KWARG_FAMILIES, _NO_SYSTEM_PROMPT_FAMILIES, _THINK_PREFIX_FAMILIES,
    _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES, _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES,
    merge_system_into_user, rewrite_tool_role_to_environment, suppress_no_functions_suffix,
)

# Stop token to inject after a tool call, keyed by tool-call format.
# None means no stop token (full output is generated; parse_tool_call handles extraction).
# JSON_SINGLE has no natural closing tag, so we stop on ``<tool_response>`` — the
# marker the model is prone to hallucinate after its own tool call. The real
# tool result is appended by the orchestrator, never produced by the model, so
# this is safe while preventing fabricated tool responses from leaking into
# subsequent turns (critical for DeepSeek baseline mode).
_TOOL_CALL_STOP_TOKEN: Dict[ToolCallFormat, Optional[str]] = {
    ToolCallFormat.JSON: "</tool_call>",
    ToolCallFormat.PYTHONIC: "</function_calls>",
    ToolCallFormat.JSON_SINGLE: "<tool_response>",
}

from .llm_shared import get_llm_lock
from ..utils.logging import get_logger, format_messages_as_chat

logger = get_logger(__name__)

# Families that are served via an API and do not occupy local GPU memory.
_API_FAMILIES = frozenset({ModelFamily.GPT4, ModelFamily.CLAUDE})


class VLLMProvider(BaseModelProvider):
    """vLLM provider for local models with tensor parallelism."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Cache dir: HF_HOME/hub
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hf_hub_cache = os.path.join(hf_home, "hub")
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
                gpu_memory_utilization=config.gpu_memory_utilization if config.gpu_memory_utilization is not None else 0.95,
                seed=config.seed if config.seed is not None else 0,
                download_dir=hf_hub_cache,
                enforce_eager=True,
                trust_remote_code=True,
                enable_prefix_caching=False,
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
            n = torch.cuda.device_count()
            # Use all GPUs only when utilization is high (or still unresolved); else single-GPU
            if n > 1 and (config.gpu_memory_utilization is None or config.gpu_memory_utilization >= 0.95):
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
        # apply_chat_template returns JSON; render each prompt string here.
        rendered_prompts = []
        decoded_messages_list: List[Optional[List[Dict[str, Any]]]] = []
        for p in prompts:
            raw_prompt = p.get("prompt", "")
            msgs = None
            try:
                payload = json.loads(raw_prompt)
                if isinstance(payload, dict) and "messages" in payload:
                    msgs = payload["messages"]
                    use_thinking = payload.get("use_thinking", False)
                    force_tool_call = payload.get("force_tool_call", False)
                    raw_prompt = self._render_messages(msgs, use_thinking, force_tool_call)
            except (json.JSONDecodeError, TypeError):
                pass
            rendered_prompts.append({**p, "prompt": raw_prompt})
            decoded_messages_list.append(msgs)

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
                rendered_prompts,
                sampling_params=sampling_params,
            )

        return [
            self._make_result(output, decoded_messages_list[i])
            for i, output in enumerate(outputs)
        ]

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
        except Exception:
            max_model_len = self.config.max_model_len or 32768
            logger.exception(
                "Error getting max_model_len from vLLM engine; using %s",
                max_model_len,
            )

        sampling_params_list = []
        valid_prompts = []
        decoded_messages: List[Optional[List[Dict[str, Any]]]] = []

        for idx, prompt in enumerate(prompts):
            # Decode JSON payload from apply_chat_template; fall back to plain string.
            msgs: Optional[List[Dict[str, Any]]] = None
            use_thinking = False
            force_tool_call = False
            try:
                payload = json.loads(prompt)
                if isinstance(payload, dict) and "messages" in payload:
                    msgs = payload["messages"]
                    use_thinking = payload.get("use_thinking", False)
                    force_tool_call = payload.get("force_tool_call", False)
                elif isinstance(payload, list):
                    msgs = payload
            except (json.JSONDecodeError, TypeError):
                pass

            decoded_messages.append(msgs)

            # Apply the tokenizer chat template to get the rendered string.
            if msgs is not None:
                rendered = self._render_messages(msgs, use_thinking, force_tool_call)
                logger.debug(
                    "vLLM request (prompt %d/%d):\n%s",
                    idx + 1, len(prompts), format_messages_as_chat(msgs),
                )
            else:
                rendered = prompt  # plain string passed directly (backward compat)

            prompt_tokens = len(self.tokenizer.encode(rendered, add_special_tokens=False))
            available_tokens = max_model_len - prompt_tokens
            safe_max_tokens = min(self.config.max_tokens, max(512, available_tokens))

            if available_tokens < 100:
                logger.warning(
                    "Prompt %s is too long (%s tokens). Truncating to fit.",
                    idx + 1,
                    prompt_tokens,
                )
                target_length = max_model_len - 1024
                tokens = self.tokenizer.encode(rendered, add_special_tokens=False)
                truncated_tokens = tokens[-target_length:]
                rendered = self.tokenizer.decode(truncated_tokens)
                safe_max_tokens = 1024
                logger.info(
                    "Truncated prompt %s to %s tokens.",
                    idx + 1,
                    len(truncated_tokens),
                )

            valid_prompts.append(rendered)
            # Pause generation after a tool call; token depends on the family's format.
            # For JSON_SINGLE (DeepSeek) we stop on ``<tool_response>`` to prevent the
            # model from hallucinating a fake tool response after its own tool call.
            stop_kwargs: Dict[str, Any] = {}
            if self.config.role == "orchestrator":
                fmt = get_tool_call_format(self.config.family)
                stop_token = _TOOL_CALL_STOP_TOKEN.get(fmt)
                if stop_token:
                    stop_kwargs = {"stop": [stop_token], "include_stop_str_in_output": True}
            params = SamplingParams(
                max_tokens=safe_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                seed=self.config.seed if self.config.seed is not None else 0,
                **stop_kwargs,
            )
            sampling_params_list.append(params)

        with self._lock:  # shared instance may be used from multiple threads
            outputs = self.llm.generate(valid_prompts, sampling_params=sampling_params_list)

        return [
            self._make_result(output, decoded_messages[i])
            for i, output in enumerate(outputs)
        ]

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        use_thinking: bool = False,
        force_tool_call: bool = False,
    ) -> str:
        """Serialize messages for generate().

        Returns a JSON-encoded payload so generate() has access to the raw
        messages list (for logging and result attachment). The tokenizer
        template is applied inside _generate_text.
        """
        return json.dumps(
            {"messages": messages, "use_thinking": use_thinking, "force_tool_call": force_tool_call},
            ensure_ascii=False,
        )

    def cleanup(self):
        """Release GPU memory."""
        if hasattr(self, 'llm'):
            del self.llm
            self.llm = None

    def _render_messages(
        self, msgs: List[Dict[str, Any]], use_thinking: bool, force_tool_call: bool = False
    ) -> str:
        """Apply the tokenizer chat template to a messages list.

        For Qwen3/QwQ, passes ``enable_thinking`` so the template can suppress
        the reasoning block when thinking is disabled.  For DeepSeek R1 (and
        similar families), the system message is merged into the first user turn
        and one of three suffixes is appended after the generation-prompt token:

        * ``force_tool_call=True``:  a brief closed ``<think>`` block followed by
          ``<sub_goal>`` to prime the model into emitting a tool call rather than
          reasoning to a direct answer.  Works regardless of *use_thinking*.
        * ``use_thinking=True`` (no force): ``<think>\\n`` to open a free reasoning
          block.
        * otherwise: ``<think>\\n\\n</think>\\n`` to suppress reasoning.
        """
        if self.config.family in _NO_SYSTEM_PROMPT_FAMILIES:
            msgs = merge_system_into_user(msgs)

        # OLMo 3 Think: rename role=tool → environment (template drops ``tool``).
        if self.config.family in _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES:
            msgs = rewrite_tool_role_to_environment(msgs)

        # OLMo 3 Think: inject functions="" to kill the "no functions" suffix.
        if self.config.family in _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES:
            msgs = suppress_no_functions_suffix(msgs)

        if self.config.family in _ENABLE_THINKING_KWARG_FAMILIES:
            rendered = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=(use_thinking and self.config.supports_thinking),
            )
        else:
            rendered = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

        if self.config.family in _THINK_PREFIX_FAMILIES:
            if force_tool_call:
                # Closed think block + open <sub_goal> tag for prefix-completion.
                rendered += "<think>\nI need to call a tool to answer this question.\n</think>\n<sub_goal>"
            elif use_thinking and self.config.supports_thinking:
                rendered += "<think>\n"
            else:
                rendered += "<think>\n\n</think>\n"

        return rendered

    def _make_result(self, output: Any, messages: Optional[List[Dict[str, Any]]]) -> GenerationResult:
        """Build a GenerationResult from a single vLLM output object."""
        return GenerationResult(
            text=output.outputs[0].text,
            finish_reason=output.outputs[0].finish_reason,
            usage={
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
            },
            metadata={"model": self.config.name, "role": self.config.role},
            messages=messages,
        )


def resolve_gpu_assignments(config) -> Dict[str, Tuple[float, Optional[List[int]]]]:
    """Compute (gpu_memory_utilization, gpu_ids) for each distinct local model path.

    - 1 distinct local model                  → util=0.95, no pinning (all GPUs visible)
    - N distinct models, total GPUs needed ≤ available
                                              → pin each model to its own GPU slice;
                                                large models (14B/32B/72B) get 2 GPUs,
                                                others get 1 GPU
    - N distinct models, total GPUs needed > available
                                              → util = 0.9 / N (shared, no pinning)
    """
    try:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    except Exception:
        num_gpus = 1

    def _is_local(cfg) -> bool:
        return (
            cfg is not None
            and cfg.family not in _API_FAMILIES
            and getattr(cfg, "backend", "vllm") == "vllm"
        )

    def _is_large(path: str) -> bool:
        return any(s in path.lower() for s in ("14b", "32b", "72b"))

    orch_cfg = config.get_model("orchestrator")

    # Collect distinct local model paths preserving load order (orchestrator first).
    seen: set = set()
    ordered_paths: List[str] = []

    def _add(path: str) -> None:
        if path not in seen:
            seen.add(path)
            ordered_paths.append(path)

    if _is_local(orch_cfg):
        _add(orch_cfg.path_or_id)

    if not config.tools.direct_tool_call:
        for tool_name in config.tools.enabled_tools:
            model_cfg = config.get_model(tool_name)
            if model_cfg is None:
                pass  # falls back to orchestrator instance (same path already in set)
            elif _is_local(model_cfg):
                _add(model_cfg.path_or_id)

    n = len(ordered_paths)
    main_path = orch_cfg.path_or_id if _is_local(orch_cfg) else None

    assignment: Dict[str, Tuple[float, Optional[List[int]]]] = {}

    if n == 0:
        pass  # all API models, nothing to assign
    elif n == 1:
        # Single distinct model: give it all visible GPUs.
        assignment[ordered_paths[0]] = (0.95, None)
        logger.info("GPU assignment (single model): util=0.95")
    else:
        # Multiple distinct models: pin each to its own GPU slice.
        # Large models (14B+) require 2 GPUs for tensor parallelism; others need 1.
        pairs = ([main_path] + [p for p in ordered_paths if p != main_path]) if main_path else ordered_paths
        gpus_per_model = {p: (2 if _is_large(p) else 1) for p in pairs}
        total_needed = sum(gpus_per_model.values())

        if total_needed <= num_gpus:
            idx = 0
            for path in pairs:
                g = gpus_per_model[path]
                assignment[path] = (0.9, list(range(idx, idx + g)))
                idx += g
            logger.info("GPU assignment (multi-model, pinned): %s",
                        {p: v[1] for p, v in assignment.items()})
        else:
            # Not enough GPUs to pin individually; share evenly.
            util = round(0.9 / n, 4)
            for path in ordered_paths:
                assignment[path] = (util, None)
            logger.info("GPU assignment (%d models sharing %d GPU(s)): util=%.4f", n, num_gpus, util)

    return assignment
