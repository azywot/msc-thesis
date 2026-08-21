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
from verl_ext.prefix_rft.budget import split_for_budget


class ReplayController:
    """Serves the first ``k`` teacher decisions in place of live generation."""

    def __init__(self, steps: list[dict], k: int, tokenizer):
        self.steps = list(steps)
        self.k = max(0, min(int(k), len(self.steps)))
        self.tokenizer = tokenizer
        self._served = 0
        # Armed by a replayed tool-call decision, consumed by the next tool lookup.
        self._pending_tool_result: Optional[str] = None
        # Token mode only: the decision that straddles the budget, and how much of
        # it the teacher supplies. None in step mode, where turns are never split.
        self.split_index: Optional[int] = None
        self.split_tokens = 0
        self._split_served = False

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

    @classmethod
    def from_token_fraction(cls, steps: list[dict], l: float, tokenizer) -> "ReplayController":
        """Build a controller whose prefix is a token fraction of the whole demonstration.

        The paper measures the prefix in tokens (A.2). Whole decisions are replayed
        while they fit the budget, and the decision that straddles it is split, so the
        model finishes a turn the teacher started.
        """
        lengths = [
            len(tokenizer.encode(str(s["response"]), add_special_tokens=False)) for s in steps
        ]
        n_full, r = split_for_budget(lengths, l)
        ctrl = cls(steps, k=n_full, tokenizer=tokenizer)
        if r > 0:
            # split_for_budget guarantees n_full < len(steps) whenever r > 0.
            ctrl.split_index = n_full
            ctrl.split_tokens = r
        return ctrl

    def next_partial(self, prompt_payload: str) -> Optional[dict]:
        """Return the prefill for the split decision, or None.

        Offered once, and only after every whole decision has been served. Unlike
        ``next_response`` this deliberately does not arm ``_pending_tool_result``:
        the model writes its own tool call after the prefill, so the teacher's stored
        result does not apply to it.
        """
        if self.split_index is None or self._split_served:
            return None
        if self._served != self.split_index:
            return None
        self._split_served = True

        step = self.steps[self.split_index]
        ids = self.tokenizer.encode(str(step["response"]), add_special_tokens=False)
        prefix_ids, prefix_text = self._safe_prefix(ids[: self.split_tokens])
        if not prefix_ids:
            return None

        messages = self._decode_messages(prompt_payload)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return {
            "messages": messages,
            "prefix_text": prefix_text,
            "prefix_ids": list(prefix_ids),
            "prompt_token_ids": list(prompt_ids),
        }

    def _safe_prefix(self, ids) -> tuple[list, str]:
        """Longest head of ``ids`` whose decoded text re-encodes to itself.

        The prefill travels to vLLM as text, so a boundary inside a mergeable token
        pair comes back as different ids and shifts every prefix position in the mask.
        Nothing downstream would notice: the run would train and report success. So
        the boundary is backed off a token at a time until the round trip holds.
        """
        candidate = list(ids)
        while candidate:
            text = self.tokenizer.decode(candidate)
            if list(self.tokenizer.encode(text, add_special_tokens=False)) == candidate:
                return candidate, text
            candidate = candidate[:-1]
        return [], ""

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

    # Special methods must be declared explicitly. Python looks dunders up on the
    # TYPE, not the instance, so __getattr__ never sees them: len(registry) on a
    # proxy that only defines __getattr__ raises
    #     TypeError: object of type 'ReplayToolRegistry' has no len()
    # The orchestrator calls len(self.tools) while building the planning prompt
    # (orchestrator.py:145 and :578), so every prefixed episode died there, was
    # caught by the rollout's except clause, and completed as an empty on-policy
    # trajectory - dispatch and replay both looked healthy and nothing reached the
    # loss (job 25754114). Mirror ToolRegistry's dunders (tool.py:183-189).

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name) -> bool:
        return name in self._registry


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
                partial = controller.next_partial(prompt)
                if partial is not None:
                    results.extend(self._generate_from_prefill(capturing, controller, partial))
                else:
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
                    "prefix_len": len(replayed["response_token_ids"]),
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

    def _generate_from_prefill(self, capturing, controller, partial) -> list:
        """Have the model finish a teacher turn, then record it as one whole turn.

        vLLM formats the final assistant message open-ended under
        ``continue_final_message`` and returns only the continuation. The daemon's proxy
        will inject token IDs for the request it saw, whose prompt contains the partial
        assistant message; those are wrong for training, so the captured turn is
        overwritten here with the prompt the turn would have had and a response that is
        teacher tokens followed by generated ones. This is what keeps the vendored daemon
        out of the change.
        """
        payload = json.dumps(
            {
                "messages": partial["messages"]
                + [{"role": "assistant", "content": partial["prefix_text"]}],
                "use_thinking": False,
                "continue_final_message": True,
            }
        )
        before = len(capturing.captured_turns)
        results = capturing.generate([payload])
        if len(capturing.captured_turns) != before + 1:
            raise RuntimeError(
                "ReplayProvider expected exactly one captured turn for a split prefill, "
                f"got {len(capturing.captured_turns) - before}. The turn cannot be "
                "corrected, so prefix_mask would mark tokens that are not the teacher's."
            )

        continuation = results[0].text or ""
        prefix_ids = list(partial["prefix_ids"])
        cont_ids = list(controller.tokenizer.encode(continuation, add_special_tokens=False))

        turn = capturing.captured_turns[-1]
        turn["prompt_ids"] = list(partial["prompt_token_ids"])
        turn["response_ids"] = prefix_ids + cont_ids
        turn["response_text"] = partial["prefix_text"] + continuation
        turn["is_prefix"] = True
        turn["prefix_len"] = len(prefix_ids)

        # Same reason as the whole-replay print: a run where the split never happened
        # is otherwise indistinguishable from one where it did.
        print(
            f"[ReplayProvider] served split turn: {len(prefix_ids)} teacher tokens + "
            f"{len(cont_ids)} generated"
        )

        # Return the whole turn, not just the continuation, so the orchestrator parses
        # the complete decision.
        results[0] = GenerationResult(
            text=turn["response_text"],
            finish_reason=results[0].finish_reason,
            usage=results[0].usage,
            metadata={"replayed": "partial"},
            prompt_token_ids=turn["prompt_ids"],
            response_token_ids=turn["response_ids"],
        )
        return results
