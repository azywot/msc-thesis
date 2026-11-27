from types import SimpleNamespace

import pytest

from agent_engine.models import vllm_provider
from agent_engine.models.base import ModelFamily
from agent_engine.models.vllm_provider import model_instance_key, resolve_gpu_assignments


def _cfg(path="Qwen/Qwen3-8B", lora=None):
    return SimpleNamespace(
        family=ModelFamily.QWEN3, backend="vllm", path_or_id=path, lora_adapter_path=lora
    )


def _config(orch, subs):
    models = {"orchestrator": orch, **subs}
    return SimpleNamespace(
        get_model=lambda role: models.get(role),
        tools=SimpleNamespace(direct_tool_call=False, enabled_tools=list(subs)),
    )


def _gpus(monkeypatch, n):
    monkeypatch.setattr(vllm_provider.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vllm_provider.torch.cuda, "device_count", lambda: n)


def test_instance_key_format():
    assert model_instance_key(_cfg()) == "Qwen/Qwen3-8B|lora:"
    assert model_instance_key(_cfg(lora="/a/b")) == "Qwen/Qwen3-8B|lora:/a/b"


def test_lora_and_base_same_path_get_separate_gpus(monkeypatch):
    _gpus(monkeypatch, 2)
    orch = _cfg(lora="/a/adapter")
    sub = _cfg()
    config = _config(orch, {"web_search": sub, "code_generator": _cfg()})
    out = resolve_gpu_assignments(config)
    assert out[model_instance_key(orch)] == (0.9, [0])
    assert out[model_instance_key(sub)] == (0.9, [1])
    assert len(out) == 2


def test_single_model_unchanged(monkeypatch):
    _gpus(monkeypatch, 2)
    orch = _cfg()
    config = _config(orch, {"web_search": _cfg()})
    out = resolve_gpu_assignments(config)
    assert out == {model_instance_key(orch): (0.95, None)}


def test_lora_and_base_share_when_gpus_short(monkeypatch):
    _gpus(monkeypatch, 1)
    orch = _cfg(lora="/a/adapter")
    sub = _cfg()
    out = resolve_gpu_assignments(_config(orch, {"web_search": sub}))
    assert out[model_instance_key(orch)] == (0.45, None)
    assert out[model_instance_key(sub)] == (0.45, None)
