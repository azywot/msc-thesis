"""A model provider that replays a fixed script of outputs.

Returns queued outputs in order, one per prompt in each ``generate()`` batch,
and records what it was asked so a trace fixture can assert on prompt
construction and batching as well as on control flow.

Why not reuse ``tests/unit/test_smoke.py:_MockProvider``: that one returns the
*same* text for every prompt.  A trace fixture needs a different output per turn
(a ``web_search`` call, then a ``code_generator`` call, then a final answer), so
it needs a queue.
"""

from typing import Dict, List, Optional

from agent_engine.models.base import (
    BaseModelProvider,
    GenerationResult,
    ModelConfig,
    ModelFamily,
)

_DEFAULT_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class ScriptedProvider(BaseModelProvider):
    """Pops one queued output per prompt, in order.

    Attributes:
        calls: One entry per ``generate()`` invocation, holding that call's
            prompt list.  Batch boundaries are preserved here -- which prompts
            the orchestrator groups into a single call *is* the batching
            behaviour under test, and a flat list would erase it.
        prompts_seen: Every prompt, flattened, for assertions that don't care
            about grouping.
    """

    def __init__(
        self,
        outputs: List[str],
        usage: Optional[Dict[str, int]] = None,
        name: str = "scripted",
    ):
        config = ModelConfig(
            name=name,
            family=ModelFamily.QWEN3,
            path_or_id=name,
            role="orchestrator",
            seed=7,
        )
        super().__init__(config)
        self._queue = list(outputs)
        self._consumed = 0
        self._usage = dict(usage) if usage is not None else dict(_DEFAULT_USAGE)
        self.calls: List[List[str]] = []
        self.prompts_seen: List[str] = []

    @property
    def remaining(self) -> int:
        """Unconsumed scripted outputs.  A trace test should end at 0."""
        return len(self._queue)

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        self.calls.append(list(prompts))
        self.prompts_seen.extend(prompts)
        results = []
        for _ in prompts:
            if not self._queue:
                raise AssertionError(
                    f"ScriptedProvider ran out of outputs after {self._consumed} "
                    f"(call #{len(self.calls)} asked for {len(prompts)}). "
                    "The orchestrator made more generate() calls than the script expects."
                )
            self._consumed += 1
            results.append(
                GenerationResult(
                    text=self._queue.pop(0),
                    finish_reason="stop",
                    usage=dict(self._usage),
                )
            )
        return results

    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "".join(parts) + f"|thinking={use_thinking}|force={force_tool_call}"

    def cleanup(self):
        pass
