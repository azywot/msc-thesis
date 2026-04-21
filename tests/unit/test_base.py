"""Unit tests for agent_engine.models.base."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from agent_engine.models.base import (
    ModelFamily,
    ModelConfig,
    ToolCallFormat,
    _ENABLE_THINKING_KWARG_FAMILIES,
    _NO_SYSTEM_PROMPT_FAMILIES,
    _THINK_PREFIX_FAMILIES,
    _THINKING_FAMILIES,
    get_tool_call_format,
    merge_system_into_user,
)


# ---------------------------------------------------------------------------
# ToolCallFormat enum
# ---------------------------------------------------------------------------

class TestToolCallFormat:
    def test_json_exists(self):
        assert ToolCallFormat.JSON.value == "json"

    def test_pythonic_exists(self):
        assert ToolCallFormat.PYTHONIC.value == "pythonic"

    def test_json_single_exists(self):
        assert ToolCallFormat.JSON_SINGLE.value == "json_single"


# ---------------------------------------------------------------------------
# get_tool_call_format
# ---------------------------------------------------------------------------

class TestGetToolCallFormat:
    def test_deepseek_is_json_single(self):
        assert get_tool_call_format(ModelFamily.DEEPSEEK) == ToolCallFormat.JSON_SINGLE

    def test_qwen3_defaults_to_json(self):
        assert get_tool_call_format(ModelFamily.QWEN3) == ToolCallFormat.JSON

    def test_qwq_defaults_to_json(self):
        assert get_tool_call_format(ModelFamily.QWQ) == ToolCallFormat.JSON

    def test_olmo_think_is_pythonic(self):
        assert get_tool_call_format(ModelFamily.OLMO_THINK) == ToolCallFormat.PYTHONIC

    def test_olmo_instruct_is_pythonic(self):
        assert get_tool_call_format(ModelFamily.OLMO_INSTRUCT) == ToolCallFormat.PYTHONIC

    def test_llama3_defaults_to_json(self):
        assert get_tool_call_format(ModelFamily.LLAMA3) == ToolCallFormat.JSON

    def test_mistral_defaults_to_json(self):
        assert get_tool_call_format(ModelFamily.MISTRAL) == ToolCallFormat.JSON


# ---------------------------------------------------------------------------
# Family frozensets
# ---------------------------------------------------------------------------

class TestFamilyFrozensets:
    def test_deepseek_in_thinking_families(self):
        assert ModelFamily.DEEPSEEK in _THINKING_FAMILIES

    def test_deepseek_in_no_system_prompt(self):
        assert ModelFamily.DEEPSEEK in _NO_SYSTEM_PROMPT_FAMILIES

    def test_deepseek_in_think_prefix(self):
        assert ModelFamily.DEEPSEEK in _THINK_PREFIX_FAMILIES

    def test_qwen3_in_enable_thinking_kwarg(self):
        assert ModelFamily.QWEN3 in _ENABLE_THINKING_KWARG_FAMILIES

    def test_qwq_in_enable_thinking_kwarg(self):
        assert ModelFamily.QWQ in _ENABLE_THINKING_KWARG_FAMILIES

    def test_deepseek_not_in_enable_thinking_kwarg(self):
        assert ModelFamily.DEEPSEEK not in _ENABLE_THINKING_KWARG_FAMILIES

    def test_olmo_not_in_no_system_prompt(self):
        assert ModelFamily.OLMO_THINK not in _NO_SYSTEM_PROMPT_FAMILIES


# ---------------------------------------------------------------------------
# ModelConfig family defaults
# ---------------------------------------------------------------------------

class TestModelConfigDefaults:
    def _make(self, family, **kwargs):
        return ModelConfig(
            name="test",
            family=family,
            path_or_id="test/model",
            role="orchestrator",
            **kwargs,
        )

    def test_deepseek_temperature_default(self):
        cfg = self._make(ModelFamily.DEEPSEEK)
        assert cfg.temperature == 0.6

    def test_deepseek_top_p_default(self):
        cfg = self._make(ModelFamily.DEEPSEEK)
        assert cfg.top_p == 0.95

    def test_deepseek_max_tokens_default(self):
        cfg = self._make(ModelFamily.DEEPSEEK)
        assert cfg.max_tokens == 32768

    def test_qwen3_temperature_default(self):
        cfg = self._make(ModelFamily.QWEN3)
        assert cfg.temperature == 0.0  # global default

    def test_deepseek_supports_thinking_derived(self):
        cfg = self._make(ModelFamily.DEEPSEEK)
        assert cfg.supports_thinking is True

    def test_qwen3_supports_thinking_derived(self):
        cfg = self._make(ModelFamily.QWEN3)
        assert cfg.supports_thinking is True

    def test_llama3_does_not_support_thinking(self):
        cfg = self._make(ModelFamily.LLAMA3)
        assert cfg.supports_thinking is False

    def test_supports_thinking_explicit_override(self):
        cfg = self._make(ModelFamily.LLAMA3, supports_thinking=True)
        assert cfg.supports_thinking is True

    def test_family_coerced_from_string(self):
        cfg = ModelConfig(
            name="test", family="deepseek", path_or_id="x", role="orchestrator"
        )
        assert cfg.family == ModelFamily.DEEPSEEK


# ---------------------------------------------------------------------------
# merge_system_into_user
# ---------------------------------------------------------------------------

class TestMergeSystemIntoUser:
    def test_merges_system_into_first_user(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = merge_system_into_user(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "You are helpful.\n\nHello!"

    def test_no_system_message_unchanged(self):
        msgs = [{"role": "user", "content": "Hello!"}]
        result = merge_system_into_user(msgs)
        assert result == msgs

    def test_empty_list_unchanged(self):
        assert merge_system_into_user([]) == []

    def test_system_without_user_unchanged(self):
        msgs = [{"role": "system", "content": "Only system"}]
        result = merge_system_into_user(msgs)
        assert result == []  # system removed but no user to merge into

    def test_preserves_subsequent_messages(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second user"},
        ]
        result = merge_system_into_user(msgs)
        assert len(result) == 3
        assert result[0]["content"] == "sys\n\nfirst user"
        assert result[1]["role"] == "assistant"
        assert result[2]["content"] == "second user"
