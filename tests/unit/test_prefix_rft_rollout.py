"""Tests for Prefix-RFT replay in the rollout worker.

The shims under test live in ``fine_tuning.prefix_replay``, which deliberately
imports only ``agent_engine``. ``fine_tuning.rollout`` pulls in agentflow, which
needs agentops, absent from the agent_engine env; keeping the shims separate is
what lets them be tested on CPU.
"""

import json

import pytest

from fine_tuning.prefix_replay import ReplayController, ReplayToolRegistry


class _FakeTokenizer:
    """Stands in for the HF tokenizer, mirroring the proxy's two calls."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [len(m["content"]) for m in messages]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def _steps():
    return [
        {"response": "plan", "tool_name": None, "tool_result": None},
        {"response": "call", "tool_name": "web_search", "tool_result": "hits"},
        {"response": "final", "tool_name": None, "tool_result": None},
    ]


def _payload(messages):
    return json.dumps({"messages": messages, "use_thinking": False})


def test_controller_serves_k_responses_then_stops():
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    first = ctrl.next_response(_payload([{"role": "user", "content": "ab"}]))
    second = ctrl.next_response(_payload([{"role": "user", "content": "abc"}]))
    assert first["text"] == "plan"
    assert second["text"] == "call"
    assert ctrl.exhausted is True
    assert ctrl.next_response(_payload([{"role": "user", "content": "x"}])) is None
    assert ctrl.replayed_turns == 2


def test_controller_tokenises_exactly_as_the_proxy_does():
    ctrl = ReplayController(_steps(), k=1, tokenizer=_FakeTokenizer())
    out = ctrl.next_response(_payload([{"role": "user", "content": "ab"}]))
    assert out["prompt_token_ids"] == [2]
    assert out["response_token_ids"] == [ord(c) for c in "plan"]


def test_controller_maps_tool_role_to_user_before_templating():
    ctrl = ReplayController(_steps(), k=1, tokenizer=_FakeTokenizer())
    messages = [{"role": "tool", "content": "xyz"}]
    out = ctrl.next_response(_payload(messages))
    assert out["prompt_token_ids"] == [3]


def test_k_zero_replays_nothing():
    ctrl = ReplayController(_steps(), k=0, tokenizer=_FakeTokenizer())
    assert ctrl.exhausted is True
    assert ctrl.next_response(_payload([{"role": "user", "content": "a"}])) is None


def test_k_is_clamped_to_the_number_of_steps():
    ctrl = ReplayController(_steps(), k=99, tokenizer=_FakeTokenizer())
    assert ctrl.k == 3


def test_tool_results_are_served_in_order_for_replayed_steps():
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    assert ctrl.next_tool_result() is None
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    assert ctrl.next_tool_result() == "hits"


def test_replay_registry_delegates_once_exhausted():
    from agent_engine.core.tool import ToolResult

    class _RealTool:
        name = "web_search"

        def execute(self, **kwargs):
            return ToolResult(success=True, output="live", metadata={})

    class _RealRegistry:
        def get(self, name):
            return _RealTool()

        def list_tools(self):
            return ["web_search"]

        def get_all_schemas(self):
            return []

    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    registry = ReplayToolRegistry(_RealRegistry(), ctrl)

    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    replayed = registry.get("web_search").execute(query="q")
    assert replayed.output == "hits"
    assert replayed.metadata["replayed"] is True

    assert registry.get("web_search").execute(query="q").output == "live"


def test_replay_registry_delegates_attributes():
    class _RealRegistry:
        def get(self, name):
            return None

        def list_tools(self):
            return ["web_search", "code_generator"]

    ctrl = ReplayController(_steps(), k=0, tokenizer=_FakeTokenizer())
    registry = ReplayToolRegistry(_RealRegistry(), ctrl)
    assert registry.list_tools() == ["web_search", "code_generator"]


def test_provider_emits_generation_results_and_marks_captured_turns():
    from fine_tuning.prefix_replay import ReplayProvider

    class _Capturing:
        def __init__(self):
            self.captured_turns = []

        def generate(self, prompts):
            from agent_engine.models.base import GenerationResult

            self.captured_turns.append(
                {"prompt_ids": [1], "response_ids": [2], "response_text": "live"}
            )
            return [GenerationResult(text="live", finish_reason="stop")]

    capturing = _Capturing()
    ctrl = ReplayController(_steps(), k=1, tokenizer=_FakeTokenizer())
    provider = ReplayProvider(capturing, ctrl)

    first = provider.generate([_payload([{"role": "user", "content": "ab"}])])
    assert first[0].text == "plan"
    assert capturing.captured_turns[0]["is_prefix"] is True

    second = provider.generate([_payload([{"role": "user", "content": "ab"}])])
    assert second[0].text == "live"
    assert capturing.captured_turns[1].get("is_prefix", False) is False


def test_a_live_tool_call_after_replay_is_not_served_a_stale_result():
    """The failure this guards: replay ends, the policy makes its own tool call,
    and the registry hands back the teacher's last result instead of executing."""
    from agent_engine.core.tool import ToolResult

    class _RealTool:
        def execute(self, **kwargs):
            return ToolResult(success=True, output="live", metadata={})

    class _RealRegistry:
        def get(self, name):
            return _RealTool()

    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    registry = ReplayToolRegistry(_RealRegistry(), ctrl)

    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))  # plan
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))  # tool call
    assert registry.get("web_search").execute(query="q").output == "hits"

    # Replay is exhausted; every later lookup must reach the real tool.
    assert ctrl.next_response(_payload([{"role": "user", "content": "a"}])) is None
    assert registry.get("web_search").execute(query="q").output == "live"
    assert registry.get("web_search").execute(query="q").output == "live"


def test_replay_registry_supports_the_special_methods_the_orchestrator_uses():
    """Python looks dunders up on the TYPE, so __getattr__ never sees them.

    The orchestrator calls len(self.tools) while building the planning prompt. A proxy
    that only defines __getattr__ raises TypeError there, the rollout's except clause
    swallows it, and the episode completes as an empty on-policy trajectory — dispatch
    and replay both look healthy while nothing reaches the loss (job 25754114).
    """
    from agent_engine.core.tool import ToolRegistry

    real = ToolRegistry()
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    registry = ReplayToolRegistry(real, ctrl)

    # Every special method the real registry defines must survive the proxy.
    assert len(registry) == len(real)
    assert ("web_search" in registry) == ("web_search" in real)


def test_the_proxies_expose_every_dunder_their_target_defines():
    """Generic guard: adding a dunder to ToolRegistry must not silently bypass the proxy."""
    from agent_engine.core.tool import ToolRegistry

    # Every class gets __dict__/__doc__/__module__/__weakref__ for free; only
    # deliberately defined special methods are at risk from the proxy.
    automatic = {"__dict__", "__doc__", "__module__", "__weakref__", "__init__"}
    special = {
        name
        for name in vars(ToolRegistry)
        if name.startswith("__") and name.endswith("__") and name not in automatic
    }
    assert special, "ToolRegistry defines no special methods; this test would be vacuous"
    missing = {n for n in special if n not in vars(ReplayToolRegistry)}
    assert not missing, (
        f"ReplayToolRegistry does not define {sorted(missing)}; __getattr__ does not "
        "cover special methods, so these would fail on the proxy."
    )
