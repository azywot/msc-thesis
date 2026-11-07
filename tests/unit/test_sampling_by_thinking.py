"""Per-thinking-state sampling: ModelConfig.sampling_for and the vLLM provider."""

import json
import threading
from types import SimpleNamespace

import pytest

from agent_engine.models.base import ModelConfig

QWEN = dict(name="q", family="qwen3", path_or_id="Qwen/Qwen3-8B", role="orchestrator")
CARD = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "non_thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}


def test_sampling_for_without_override_uses_model_fields():
    # Pinning a sampling field opts out of the Qwen3 model-card default.
    m = ModelConfig(**QWEN, temperature=0.0)
    assert m.sampling_for(True) == {"temperature": 0.0, "top_p": 0.8, "top_k": 20}
    assert m.sampling_for(False) == m.sampling_for(True)


def test_sampling_for_picks_block_by_thinking():
    m = ModelConfig(**QWEN, sampling_by_thinking=CARD)
    assert m.sampling_for(True) == CARD["thinking"]
    assert m.sampling_for(False) == CARD["non_thinking"]


def test_override_rejected_on_non_vllm_backend():
    with pytest.raises(ValueError):
        ModelConfig(**QWEN, backend="mlx", sampling_by_thinking=CARD)


@pytest.mark.parametrize("family", ["gpt4", "claude"])
def test_override_rejected_on_api_family_with_default_backend(family):
    cfg = dict(QWEN, family=family, path_or_id="some-api-model")
    with pytest.raises(ValueError, match="sampling_by_thinking"):
        ModelConfig(**cfg, sampling_by_thinking=CARD)


def test_config_json_includes_model_seed():
    d = _cfg_dict(ModelConfig(**QWEN, seed=5))["models"]["orchestrator"]
    assert d["seed"] == 5


def test_family_defaults_still_apply_with_override():
    m = ModelConfig(
        name="d", family="deepseek", path_or_id="x", role="orchestrator", sampling_by_thinking=CARD
    )
    assert m.max_tokens == 32768 and m.temperature == 0.6


class _RecordingLLM:
    def __init__(self):
        self.params = None

    def generate(self, prompts, sampling_params=None, lora_request=None):
        self.params = sampling_params
        return [object() for _ in prompts]


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [0] * 10


def _provider(config):
    from agent_engine.models.vllm_provider import VLLMProvider

    p = object.__new__(VLLMProvider)
    p.config = config
    p.tokenizer = _Tok()
    p.llm = _RecordingLLM()
    p._lock = threading.Lock()
    p.lora_request = None
    p._render_messages = lambda msgs, use_thinking, force_tool_call=False: "rendered"
    p._make_result = lambda output, msgs: "r"
    return p


def _payload(use_thinking):
    return json.dumps(
        {"messages": [{"role": "user", "content": "hi"}], "use_thinking": use_thinking}
    )


def test_vllm_text_path_builds_params_per_prompt():
    cfg = ModelConfig(**QWEN, sampling_by_thinking=CARD, seed=7)
    p = _provider(cfg)
    p._generate_text([_payload(True), _payload(False)])
    a, b = p.llm.params
    assert (a.temperature, a.top_p, a.top_k) == (0.6, 0.95, 20)
    assert (b.temperature, b.top_p, b.top_k) == (0.7, 0.8, 20)
    assert a.seed == 7 and b.seed == 7


def test_vllm_text_path_unchanged_without_override():
    # temperature > 0 so vLLM does not normalise top_p/top_k for greedy decoding
    cfg = ModelConfig(**QWEN, temperature=0.5)
    p = _provider(cfg)
    p._generate_text([_payload(True)])
    (a,) = p.llm.params
    assert (a.temperature, a.top_p, a.top_k) == (0.5, 0.8, 20)
    assert a.repetition_penalty == cfg.repetition_penalty


def test_vllm_multimodal_path_builds_params_per_prompt():
    cfg = ModelConfig(**QWEN, sampling_by_thinking=CARD, seed=7)
    p = _provider(cfg)
    mm = lambda t: {"prompt": _payload(t), "multi_modal_data": {"image": None}}
    p._generate_multimodal([mm(True), mm(False)])
    a, b = p.llm.params
    assert a.temperature == 0.6 and b.temperature == 0.7
    assert a.seed == 7 and b.seed == 7


def _cfg_dict(orchestrator):
    from agent_engine.runner.experiment import _config_to_dict

    cfg = SimpleNamespace(
        name="n",
        description="",
        seed=1,
        thinking_mode=SimpleNamespace(value="no"),
        dataset=SimpleNamespace(name="d", split="s"),
        tools=SimpleNamespace(enabled_tools=[], direct_tool_call=False),
        has_model=lambda role: role == "orchestrator",
        get_model=lambda role: orchestrator if role == "orchestrator" else None,
    )
    return _config_to_dict(cfg)


def test_config_json_includes_sampling_by_thinking():
    with_card = ModelConfig(**QWEN, sampling_by_thinking=CARD)
    without = ModelConfig(**QWEN, temperature=0.0)
    d1 = _cfg_dict(with_card)["models"]["orchestrator"]
    d2 = _cfg_dict(without)["models"]["orchestrator"]
    assert d1["sampling_by_thinking"] == CARD
    assert d2["sampling_by_thinking"] is None
    assert {k: v for k, v in d1.items() if k != "sampling_by_thinking"}.keys() == d2.keys() - {
        "sampling_by_thinking"
    }
