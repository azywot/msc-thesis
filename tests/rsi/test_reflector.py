import sys
import types
from types import SimpleNamespace

import pytest

from rsi.reflector import (
    InProcessReflector,
    ProposalParseError,
    ReflectorClient,
    ReflectorSettings,
    build_reflector,
    parse_proposal,
)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params=None, lora_request=None):
        self.calls.append((prompts, sampling_params))
        self.lora = lora_request
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"t{p.seed}", finish_reason="stop")])
            for p in sampling_params
        ]


class FakeVLLMProvider:
    def __init__(self):
        import threading

        self.llm = FakeLLM()
        self._lock = threading.Lock()
        self.lora_request = "LORA"
        self.config = SimpleNamespace(temperature=0.7, role="orchestrator")
        self.rendered_with_thinking = []

    def _render_messages(self, msgs, use_thinking, force_tool_call=False):
        self.rendered_with_thinking.append(use_thinking)
        return "RENDERED"


def install_fake_openai(monkeypatch, reply="x", finish="stop", fail_seed=None):
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            if kw["seed"] == fail_seed:
                raise RuntimeError("boom")
            msg = SimpleNamespace(content=reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    return calls


def test_parse_fenced_json_after_thinking():
    t = '<think>long</think>Here:\n```json\n{"unit": "web_search", "replacement": "{query} {formatted_results}", "rationale": "r", "expected_effect": "e"}\n```'
    p = parse_proposal(t)
    assert p.unit == "web_search" and p.replacement == "{query} {formatted_results}"


def test_parse_bare_object():
    t = 'ok {"unit": "orchestrator", "replacement": {"turn1n_prompt": "x"}}'
    assert parse_proposal(t).unit == "orchestrator"


def test_parse_uses_last_fence():
    t = '```json\n{"unit": "a", "replacement": "1"}\n```\n```json\n{"unit": "b", "replacement": "2"}\n```'
    assert parse_proposal(t).unit == "b"


@pytest.mark.parametrize(
    "bad",
    ["no json here", '{"unit": "orchestrator"}', '{"unit": "orchestrator", "replacement": ', ""],
)
def test_parse_errors(bad):
    with pytest.raises(ProposalParseError):
        parse_proposal(bad)


def test_in_process_reflector_uses_own_sampling_on_the_shared_engine():
    prov = FakeVLLMProvider()
    out = InProcessReflector(prov, max_tokens=99).sample(
        [{"role": "user", "content": "u"}], [5, 6, 7]
    )
    ((prompts, params),) = prov.llm.calls
    assert len(prompts) == 3 and [p.seed for p in params] == [5, 6, 7]
    assert all(p.temperature == 0.6 and p.max_tokens == 99 and not p.stop for p in params)
    assert prov.config.temperature == 0.7 and prov.rendered_with_thinking == [True]
    assert prov.llm.lora == "LORA"
    assert [o["seed"] for o in out] == [5, 6, 7]
    assert out[0] == {"text": "t5", "finish_reason": "stop", "seed": 5}


def test_in_process_errors_become_items():
    prov = FakeVLLMProvider()
    prov.llm.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom"))
    out = InProcessReflector(prov, max_tokens=9).sample([{"role": "user", "content": "u"}], [1, 2])
    assert [o["finish_reason"] for o in out] == ["error: oom"] * 2
    assert [o["seed"] for o in out] == [1, 2] and all(o["text"] == "" for o in out)


def test_sample_passes_seed_and_thinking(monkeypatch):
    calls = install_fake_openai(monkeypatch, reply="x")
    out = ReflectorClient("http://h/v1", "m", max_tokens=10).sample(
        [{"role": "user", "content": "u"}], [5, 6]
    )
    assert [c["seed"] for c in calls] == [5, 6]
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert [o["seed"] for o in out] == [5, 6]


def test_client_exception_becomes_error_item(monkeypatch):
    install_fake_openai(monkeypatch, fail_seed=6)
    out = ReflectorClient("http://h/v1", "m", max_tokens=10).sample([], [5, 6])
    assert out[0]["finish_reason"] == "stop"
    assert out[1] == {"text": "", "finish_reason": "error: boom", "seed": 6}


def test_build_reflector(monkeypatch):
    install_fake_openai(monkeypatch)
    prov = FakeVLLMProvider()
    assert isinstance(build_reflector(ReflectorSettings(model="m"), prov), InProcessReflector)
    s = ReflectorSettings(model="m", backend="openai", base_url="http://h/v1")
    assert isinstance(build_reflector(s, prov), ReflectorClient)
    with pytest.raises(ValueError):
        build_reflector(ReflectorSettings(model="m", backend="openai"), prov)


def test_parse_json_fence_with_inner_code_fence():
    import json

    template = "Task: {task}\n```python\nprint('hi')\n```\nContext: {context}"
    obj = {"unit": "code_generator", "rationale": "r", "expected_effect": "e"}
    obj["replacement"] = template
    t = "<think>x</think>Proposal:\n```json\n" + json.dumps(obj, indent=2) + "\n```\n"
    p = parse_proposal(t)
    assert p.unit == "code_generator" and p.replacement == template


def test_parse_inner_fence_falls_back_when_later_fence_is_broken():
    import json

    good = json.dumps({"unit": "web_search", "replacement": "a ```python x``` b"})
    t = "```json\n" + good + '\n```\nand a broken one:\n```json\n{"unit": \n```'
    assert parse_proposal(t).unit == "web_search"


def test_settings_reject_budget_over_model_len():
    with pytest.raises(ValueError, match="max_model_len"):
        ReflectorSettings(model="m", input_budget_tokens=20_000, max_tokens=16_384)
    ok = ReflectorSettings(model="m", input_budget_tokens=16_384, max_tokens=16_384)
    assert ok.input_budget_tokens + ok.max_tokens == ok.max_model_len
