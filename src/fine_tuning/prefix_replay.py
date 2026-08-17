"""Replay shims for Prefix-RFT: serve teacher decisions in place of generation.

The orchestrator is not modified. Two shims sit between it and its resources:

- ``ReplayController`` holds the teacher's decisions and a counter. While the
  counter is below ``k`` it answers generation requests with the teacher's stored
  response and tool requests with the teacher's stored result.
- ``ReplayToolRegistry`` consults the controller before delegating.

Because the orchestrator still builds every prompt itself, a replayed turn is
conditioned on exactly the prompt inference would have built. Token IDs are
produced with the same two calls the vendored proxy uses
(``fine_tuning/agentflow/verl/daemon.py:216-225``): ``apply_chat_template`` with
``add_generation_prompt=True`` and no ``enable_thinking`` kwarg, and ``encode``
with ``add_special_tokens=False`` and no appended EOS. Replayed and generated
turns are therefore tokenised identically.

This module deliberately imports only ``agent_engine``, never
``fine_tuning.rollout``, which pulls in agentflow and so needs agentops. Keeping
the shims free of that dependency is what lets them be unit-tested on CPU.
"""

from __future__ import annotations

import json
from typing import Optional

from agent_engine.core.tool import ToolResult
from agent_engine.models.base import GenerationResult


class ReplayController:
    """Serves the first ``k`` teacher decisions in place of live generation."""

    def __init__(self, steps: list[dict], k: int, tokenizer):
        self.steps = list(steps)
        self.k = max(0, min(int(k), len(self.steps)))
        self.tokenizer = tokenizer
        self._served = 0
        # Armed by a replayed tool-call decision, consumed by the next tool lookup.
        self._pending_tool_result: Optional[str] = None

    @property
    def exhausted(self) -> bool:
        return self._served >= self.k

    @property
    def replayed_turns(self) -> int:
        return self._served

    def next_response(self, prompt_payload: str) -> Optional[dict]:
        """Return the next teacher decision with proxy-identical token IDs."""
        if self.exhausted:
            return None
        step = self.steps[self._served]
        self._served += 1
        self._pending_tool_result = step["tool_result"] if step["tool_name"] else None

        messages = self._decode_messages(prompt_payload)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        text = str(step["response"])
        response_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return {
            "text": text,
            "prompt_token_ids": list(prompt_ids),
            "response_token_ids": list(response_ids),
        }

    def next_tool_result(self) -> Optional[str]:
        """Consume the stored result armed by the last replayed decision.

        Single-use on purpose. If it returned the same result on every call, the
        first live tool call after replay ended would be served a stale teacher
        result instead of executing, and the trajectory would silently diverge
        from anything the policy actually did.
        """
        out = self._pending_tool_result
        self._pending_tool_result = None
        return out

    @staticmethod
    def _decode_messages(prompt_payload: str) -> list[dict]:
        """Mirror OpenAIProvider._generate_one's decoding (api_provider.py:82-100).

        The tool -> user remap happens before the request leaves the provider, so
        the proxy tokenises the remapped list and we must too.
        """
        raw = None
        try:
            payload = json.loads(prompt_payload)
            if isinstance(payload, dict) and "messages" in payload:
                raw = payload["messages"]
            elif isinstance(payload, list):
                raw = payload
        except (json.JSONDecodeError, TypeError):
            raw = None
        if raw is None:
            raw = [{"role": "user", "content": prompt_payload}]
        return [{**m, "role": "user"} if m.get("role") == "tool" else m for m in raw]


class _ReplayedTool:
    """Returns a stored tool result.

    Accepts ``**kwargs`` so the orchestrator's argument sanitiser
    (``_sanitize_tool_arguments``) treats the signature as permissive and passes
    the model's arguments through untouched.
    """

    def __init__(self, name: str, output: str):
        self.name = name
        self._output = output

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=self._output, metadata={"replayed": True})


class ReplayToolRegistry:
    """Wraps a ToolRegistry, serving stored results while the controller is replaying."""

    def __init__(self, registry, controller: ReplayController):
        self._registry = registry
        self._controller = controller

    def get(self, name: str):
        # next_tool_result() returns None once the last served decision was not a
        # tool call, and the controller stops serving past k, so no extra guard
        # is needed here.
        pending = self._controller.next_tool_result()
        if pending is not None:
            return _ReplayedTool(name, pending)
        return self._registry.get(name)

    def __getattr__(self, item):
        return getattr(self._registry, item)


class ReplayProvider:
    """Serves replayed turns, then delegates to the capturing provider."""

    def __init__(self, capturing, controller: ReplayController):
        object.__setattr__(self, "_capturing", capturing)
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_capturing"), name)

    def generate(self, prompts: list) -> list:
        controller = object.__getattribute__(self, "_controller")
        capturing = object.__getattribute__(self, "_capturing")
        results = []
        for prompt in prompts:
            replayed = controller.next_response(prompt)
            if replayed is None:
                results.extend(capturing.generate([prompt]))
                continue
            # Record the replayed turn where the capturing provider records its
            # own, so the rollout builds one triplet per decision in order.
            capturing.captured_turns.append(
                {
                    "prompt_ids": replayed["prompt_token_ids"],
                    "response_ids": replayed["response_token_ids"],
                    "response_text": replayed["text"],
                    "is_prefix": True,
                }
            )
            # Proof that replay reached the generation path, not merely that a
            # controller was constructed. _make_controller's print says only the
            # latter, which is why job 25753400 looked like replay had happened.
            print(
                f"[ReplayProvider] served replayed turn "
                f"{controller.replayed_turns}/{controller.k} "
                f"({len(replayed['response_token_ids'])} response tokens)"
            )
            n_prompt = len(replayed["prompt_token_ids"])
            n_response = len(replayed["response_token_ids"])
            results.append(
                GenerationResult(
                    text=replayed["text"],
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": n_prompt,
                        "completion_tokens": n_response,
                        "total_tokens": n_prompt + n_response,
                    },
                    metadata={"replayed": True},
                    prompt_token_ids=replayed["prompt_token_ids"],
                    response_token_ids=replayed["response_token_ids"],
                )
            )
        return results
