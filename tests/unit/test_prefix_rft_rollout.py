"""Tests for Prefix-RFT replay in the rollout worker.

The shims under test live in ``fine_tuning.prefix_replay``, which deliberately
imports only ``agent_engine``. ``fine_tuning.rollout`` pulls in agentflow, which
needs agentops, absent from the agent_engine env; keeping the shims separate is
what lets them be tested on CPU.
"""

import json

import pytest

from fine_tuning.prefix_replay import (
    ReplayController,
    ReplayProvider,
    ReplayToolRegistry,
)


class _FakeTokenizer:
    """Stands in for the HF tokenizer, mirroring the proxy's two calls."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [len(m["content"]) for m in messages]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


class _RoundTripTokenizer(_FakeTokenizer):
    """_FakeTokenizer plus a decode that round-trips, since ord/chr is a bijection."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _LossyTokenizer(_RoundTripTokenizer):
    """A prefix ending on "n" does not survive the round trip.

    Stands in for a real tokenizer splitting a mergeable pair: the decoded text
    re-encodes to more ids than it came from, which would shift every position in the
    mask. Only ``decode`` lies, so the decision token lengths ``from_token_fraction``
    measures are unaffected and the budget arithmetic stays readable.
    """

    def decode(self, ids):
        text = super().decode(ids)
        if text.endswith("n"):
            return text + "?"
        return text


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


# --------------------------------------------------------------------------- #
# token mode: a prefix measured in tokens, which can split a decision           #
# --------------------------------------------------------------------------- #


def test_token_fraction_replays_whole_decisions_then_splits_one():
    # responses are "plan" (4), "call" (4), "final" (5); total 13.
    # l = 0.8 -> budget 10: two whole decisions (8) then 2 tokens of "final".
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    assert ctrl.k == 2
    assert ctrl.split_index == 2
    assert ctrl.split_tokens == 2


def test_the_split_turn_is_served_after_the_whole_ones():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    assert ctrl.next_response(payload)["text"] == "plan"
    assert ctrl.next_response(payload)["text"] == "call"
    assert ctrl.next_response(payload) is None
    partial = ctrl.next_partial(payload)
    assert partial["prefix_text"] == "fi"
    assert partial["prefix_ids"] == [ord("f"), ord("i")]
    assert partial["prompt_token_ids"] == [2]


def test_the_split_is_served_only_once():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    ctrl.next_response(payload)
    ctrl.next_response(payload)
    assert ctrl.next_partial(payload) is not None
    assert ctrl.next_partial(payload) is None


def test_no_partial_is_offered_before_the_whole_decisions_are_done():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    assert ctrl.next_partial(_payload([{"role": "user", "content": "ab"}])) is None


def test_a_boundary_landing_on_a_decision_edge_offers_no_partial():
    # budget 8 is exactly "plan" + "call".
    ctrl = ReplayController.from_token_fraction(
        _steps(), l=8 / 13, tokenizer=_RoundTripTokenizer()
    )
    payload = _payload([{"role": "user", "content": "ab"}])
    assert ctrl.split_index is None
    ctrl.next_response(payload)
    ctrl.next_response(payload)
    assert ctrl.next_partial(payload) is None


def test_a_split_decision_does_not_arm_the_teacher_tool_result():
    """The model writes its own tool call after the prefill, so the teacher's stored
    result must not be served for it."""
    steps = [
        {"response": "plan", "tool_name": None, "tool_result": None},
        {"response": "call", "tool_name": "web_search", "tool_result": "hits"},
    ]
    # total 8, l = 0.75 -> budget 6: "plan" whole, then 2 tokens of "call".
    ctrl = ReplayController.from_token_fraction(steps, l=0.75, tokenizer=_RoundTripTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    ctrl.next_response(payload)
    assert ctrl.next_tool_result() is None
    ctrl.next_partial(payload)
    assert ctrl.next_tool_result() is None


def test_a_fully_replayed_decision_still_arms_its_tool_result():
    steps = [
        {"response": "call", "tool_name": "web_search", "tool_result": "hits"},
        {"response": "final", "tool_name": None, "tool_result": None},
    ]
    # total 9, l = 0.7 -> budget 6: "call" whole, then 2 tokens of "final".
    ctrl = ReplayController.from_token_fraction(steps, l=0.7, tokenizer=_RoundTripTokenizer())
    ctrl.next_response(_payload([{"role": "user", "content": "ab"}]))
    assert ctrl.next_tool_result() == "hits"


def test_the_boundary_backs_off_until_the_text_round_trips():
    """A prefix whose text does not re-encode to itself is shortened, not sent.

    l = 0.85 over 13 tokens gives a budget of 11: "plan" and "call" whole, then 3
    tokens of "final". "fin" does not round-trip, so the boundary drops to "fi".
    """
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.85, tokenizer=_LossyTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    ctrl.next_response(payload)
    ctrl.next_response(payload)
    assert ctrl.split_tokens == 3
    partial = ctrl.next_partial(payload)
    assert partial["prefix_ids"] == [ord("f"), ord("i")]
    assert partial["prefix_text"] == "fi"


def test_a_single_decision_demonstration_can_be_split():
    steps = [{"response": "final", "tool_name": None, "tool_result": None}]
    ctrl = ReplayController.from_token_fraction(steps, l=0.5, tokenizer=_RoundTripTokenizer())
    assert ctrl.k == 0
    assert ctrl.split_index == 0
    partial = ctrl.next_partial(_payload([{"role": "user", "content": "ab"}]))
    assert partial["prefix_text"] == "fi"


def test_step_mode_controllers_have_no_split():
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    assert ctrl.split_index is None
    assert ctrl.split_tokens == 0
    assert ctrl.next_partial(_payload([{"role": "user", "content": "ab"}])) is None


# --------------------------------------------------------------------------- #
# serving a split turn                                                          #
# --------------------------------------------------------------------------- #


class _FakeCapturing:
    """Stands in for _CapturingProvider: records one turn per generated result."""

    def __init__(self, reply="nal"):
        self.captured_turns = []
        self.calls = []
        self._reply = reply

    def generate(self, prompts):
        from agent_engine.models.base import GenerationResult

        self.calls.append(prompts[0])
        self.captured_turns.append(
            {"prompt_ids": [0], "response_ids": [1], "response_text": self._reply}
        )
        return [
            GenerationResult(
                text=self._reply,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                metadata={},
                prompt_token_ids=[0],
                response_token_ids=[1],
            )
        ]


def test_provider_stitches_a_split_turn_back_into_one_turn():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing(reply="nal")
    provider = ReplayProvider(capturing, ctrl)
    payload = _payload([{"role": "user", "content": "ab"}])

    provider.generate([payload])  # "plan"
    provider.generate([payload])  # "call"
    out = provider.generate([payload])  # split: "fi" + "nal"

    turn = capturing.captured_turns[-1]
    assert turn["response_text"] == "final"
    assert turn["response_ids"] == [ord(c) for c in "final"]
    assert turn["prefix_len"] == 2
    assert turn["is_prefix"] is True
    assert turn["prompt_ids"] == [2]
    assert out[0].text == "final"


def test_the_split_request_asks_vllm_to_continue_the_assistant_message():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing()
    provider = ReplayProvider(capturing, ctrl)
    payload = _payload([{"role": "user", "content": "ab"}])

    provider.generate([payload])
    provider.generate([payload])
    provider.generate([payload])

    sent = json.loads(capturing.calls[-1])
    assert sent["continue_final_message"] is True
    assert sent["messages"][-1] == {"role": "assistant", "content": "fi"}


def test_whole_replays_record_their_full_length_as_the_prefix():
    ctrl = ReplayController(_steps(), k=1, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing()
    provider = ReplayProvider(capturing, ctrl)
    provider.generate([_payload([{"role": "user", "content": "ab"}])])
    turn = capturing.captured_turns[-1]
    assert turn["is_prefix"] is True
    assert turn["prefix_len"] == len("plan")
    assert capturing.calls == []  # never reached vLLM


def test_after_the_split_the_provider_delegates_normally():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing()
    provider = ReplayProvider(capturing, ctrl)
    payload = _payload([{"role": "user", "content": "ab"}])
    for _ in range(4):
        provider.generate([payload])
    last = capturing.captured_turns[-1]
    assert "prefix_len" not in last
    assert json.loads(capturing.calls[-1]).get("continue_final_message") is None
