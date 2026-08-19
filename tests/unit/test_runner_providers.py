"""Model-provider instance caching in agent_engine.runner.providers.

The cache exists so several roles (orchestrator, web_search, code_generator)
sharing one model path load the weights once.  The subtlety is the key: two
roles can share a ``path_or_id`` and still need *different* providers when one
of them carries a LoRA adapter.  Keying on the path alone hands the
first-loaded provider to both, so a role silently runs with the wrong weights.
"""

import pytest

from agent_engine.models.base import ModelConfig, ModelFamily
from agent_engine.runner.providers import setup_model_provider


class _FakeVLLM:
    """Stands in for VLLMProvider so no weights are loaded."""

    instances = 0

    def __init__(self, config):
        self.config = config
        type(self).instances += 1


@pytest.fixture
def fake_vllm(monkeypatch):
    import agent_engine.models.vllm_provider as vp

    _FakeVLLM.instances = 0
    monkeypatch.setattr(vp, "VLLMProvider", _FakeVLLM)
    return _FakeVLLM


def _config(role, path="Qwen/Qwen3-8B", lora=None):
    return ModelConfig(
        name=f"cfg-{role}",
        family=ModelFamily.QWEN3,
        path_or_id=path,
        role=role,
        lora_adapter_path=lora,
    )


def test_same_model_is_loaded_once_across_roles(fake_vllm):
    cache = {}
    a = setup_model_provider(_config("orchestrator"), {}, cache)
    b = setup_model_provider(_config("web_search"), {}, cache)

    assert a is b, "two roles on the same path should share one provider"
    assert fake_vllm.instances == 1


def test_lora_and_base_do_not_share_a_cache_entry(fake_vllm):
    """The regression this module exists for."""
    cache = {}
    base = setup_model_provider(_config("web_search"), {}, cache)
    lora = setup_model_provider(
        _config("orchestrator", lora="/adapters/run-42/lora_adapter"), {}, cache
    )

    assert base is not lora, "LoRA model must not reuse the base model's provider"
    assert fake_vllm.instances == 2
    assert len(cache) == 2


def test_distinct_lora_adapters_do_not_share_a_cache_entry(fake_vllm):
    cache = {}
    first = setup_model_provider(_config("orchestrator", lora="/adapters/step_20"), {}, cache)
    second = setup_model_provider(_config("orchestrator", lora="/adapters/step_40"), {}, cache)

    assert first is not second
    assert fake_vllm.instances == 2


def test_no_cache_still_returns_a_provider(fake_vllm):
    """``model_cache=None`` is the uncached path and must not raise."""
    provider = setup_model_provider(_config("orchestrator"), {}, None)

    assert isinstance(provider, _FakeVLLM)
    assert fake_vllm.instances == 1
