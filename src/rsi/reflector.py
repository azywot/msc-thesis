"""Reflector clients (in-process and OpenAI-compatible) and proposal parsing."""

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel

from agent_engine.utils.parsing import strip_thinking_tags


class ReflectorSettings(BaseModel):
    backend: Literal["in_process", "openai"] = "in_process"
    model: str
    base_url: Optional[str] = None
    max_model_len: int = 32768
    max_tokens: int = 16384
    input_budget_tokens: int = 14000
    per_call_chars: int = 1500
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True
    timeout_s: int = 900


def _error_item(seed, exc):
    return {"text": "", "finish_reason": f"error: {exc}", "seed": seed}


class InProcessReflector:
    """Samples from the executor's own vLLM engine with reflector-specific SamplingParams.

    Never mutates provider.config. Reuses the provider's lock and LoRA request.
    """

    def __init__(
        self,
        provider,
        max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        enable_thinking=True,
    ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking

    def sample(self, messages, seeds):
        if not seeds:
            return []
        try:
            from vllm import SamplingParams

            # provider.apply_chat_template() returns a JSON payload that only
            # VLLMProvider.generate() understands; llm.generate needs the rendered string.
            prompt = self.provider._render_messages(messages, self.enable_thinking)
            params = [
                SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    seed=s,
                )
                for s in seeds
            ]
            with self.provider._lock:
                outputs = self.provider.llm.generate(
                    [prompt] * len(seeds),
                    sampling_params=params,
                    lora_request=getattr(self.provider, "lora_request", None),
                )
            return [
                {
                    "text": o.outputs[0].text,
                    "finish_reason": o.outputs[0].finish_reason,
                    "seed": s,
                }
                for o, s in zip(outputs, seeds)
            ]
        except Exception as exc:
            return [_error_item(s, exc) for s in seeds]


class ReflectorClient:
    """Reflector served by a separate OpenAI-compatible vLLM server (32B control)."""

    def __init__(
        self,
        base_url,
        model,
        max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        enable_thinking=True,
        timeout_s=900,
    ):
        import openai

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s)

    def sample(self, messages, seeds):
        out = []
        for s in seeds:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=s,
                    extra_body={
                        "top_k": self.top_k,
                        "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                    },
                )
                choice = resp.choices[0]
                out.append(
                    {
                        "text": choice.message.content or "",
                        "finish_reason": choice.finish_reason,
                        "seed": s,
                    }
                )
            except Exception as exc:
                out.append(_error_item(s, exc))
        return out


def build_reflector(settings, executor_provider):
    if settings.backend == "in_process":
        return InProcessReflector(
            executor_provider,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            enable_thinking=settings.enable_thinking,
        )
    if not settings.base_url:
        raise ValueError("reflector backend 'openai' requires base_url")
    return ReflectorClient(
        settings.base_url,
        settings.model,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        enable_thinking=settings.enable_thinking,
        timeout_s=settings.timeout_s,
    )


@dataclass
class Proposal:
    unit: str
    replacement: object
    rationale: str
    expected_effect: str


class ProposalParseError(ValueError):
    pass


_FENCE = re.compile(r"```json\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_proposal(text):
    body = strip_thinking_tags(text or "")
    if not body:
        raise ProposalParseError("empty reflector output")
    fences = _FENCE.findall(body)
    if fences:
        blob = fences[-1].strip()
    else:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ProposalParseError("no JSON object found in reflector output")
        blob = body[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ProposalParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProposalParseError("proposal JSON is not an object")
    for key in ("unit", "replacement"):
        if key not in data:
            raise ProposalParseError(f"missing required field '{key}'")
    if not isinstance(data["unit"], str) or not data["unit"]:
        raise ProposalParseError("'unit' must be a non-empty string")
    return Proposal(
        unit=data["unit"],
        replacement=data["replacement"],
        rationale=str(data.get("rationale", "")),
        expected_effect=str(data.get("expected_effect", "")),
    )
