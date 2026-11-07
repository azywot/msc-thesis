"""Qwen3 on vLLM defaults to the model-card sampling settings instead of greedy decoding."""

import importlib.util
from pathlib import Path

import pytest

from agent_engine.models.base import QWEN3_MODEL_CARD_SAMPLING, ModelConfig

QWEN = dict(name="q", family="qwen3", path_or_id="Qwen/Qwen3-8B", role="orchestrator")
CARD_THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
CARD_NON_THINKING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def test_card_constant_has_model_card_values():
    assert QWEN3_MODEL_CARD_SAMPLING == {
        "thinking": CARD_THINKING,
        "non_thinking": CARD_NON_THINKING,
    }


def test_qwen3_vllm_default_samples_with_model_card():
    m = ModelConfig(**QWEN)
    assert m.sampling_for(True) == CARD_THINKING
    assert m.sampling_for(False) == CARD_NON_THINKING


def test_qwen3_explicit_backend_vllm_gets_default_too():
    m = ModelConfig(**QWEN, backend="vllm")
    assert m.sampling_for(True) == CARD_THINKING


@pytest.mark.parametrize(
    "pinned", [{"temperature": 0.0}, {"top_p": 0.9}, {"top_k": 50}, {"temperature": 0.3}]
)
def test_explicit_sampling_field_keeps_config_values(pinned):
    m = ModelConfig(**QWEN, **pinned)
    assert m.sampling_by_thinking is None
    expected = {"temperature": 0.0, "top_p": 0.8, "top_k": 20, **pinned}
    assert m.sampling_for(True) == expected == m.sampling_for(False)


def test_explicit_sampling_by_thinking_wins_over_default():
    custom = {
        "thinking": {"temperature": 0.9, "top_p": 0.9, "top_k": 40},
        "non_thinking": {"temperature": 0.1, "top_p": 0.5, "top_k": 10},
    }
    m = ModelConfig(**QWEN, sampling_by_thinking=custom)
    assert m.sampling_for(True) == custom["thinking"]
    assert m.sampling_for(False) == custom["non_thinking"]


@pytest.mark.parametrize("backend", ["mlx", "openai", "anthropic"])
def test_qwen3_non_vllm_backend_keeps_old_default_and_does_not_raise(backend):
    m = ModelConfig(**QWEN, backend=backend)
    assert m.sampling_by_thinking is None
    assert m.temperature == 0.0


@pytest.mark.parametrize("family", ["qwq", "qwen2.5", "llama3"])
def test_other_families_unchanged(family):
    m = ModelConfig(**{**QWEN, "family": family})
    assert m.sampling_by_thinking is None
    assert m.sampling_for(True) == {"temperature": 0.0, "top_p": 0.8, "top_k": 20}


@pytest.mark.parametrize("family", ["deepseek", "olmo-think", "olmo-instruct"])
def test_deepseek_and_olmo_family_defaults_unchanged(family):
    m = ModelConfig(**{**QWEN, "family": family})
    assert m.sampling_by_thinking is None
    assert m.temperature == 0.6 and m.top_p == 0.95


def test_default_is_not_shared_between_instances():
    a = ModelConfig(**QWEN)
    b = ModelConfig(**QWEN)
    assert a.sampling_by_thinking is not b.sampling_by_thinking


def test_generator_constant_matches_model_default():
    path = Path(__file__).resolve().parents[2] / "scripts" / "generate_configs.py"
    spec = importlib.util.spec_from_file_location("generate_configs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.QWEN3_MODEL_CARD_SAMPLING == QWEN3_MODEL_CARD_SAMPLING
