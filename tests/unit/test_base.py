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
    _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES,
    _THINK_PREFIX_FAMILIES,
    _THINKING_FAMILIES,
    _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES,
    get_tool_call_format,
    merge_system_into_user,
    rewrite_tool_role_to_environment,
    suppress_no_functions_suffix,
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

    def test_olmo_think_in_tool_role_as_environment(self):
        assert ModelFamily.OLMO_THINK in _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES

    def test_olmo_instruct_in_tool_role_as_environment(self):
        # No-op for Instruct (template already aliases), but included for uniformity.
        assert ModelFamily.OLMO_INSTRUCT in _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES

    def test_olmo_think_in_suppress_no_functions(self):
        assert ModelFamily.OLMO_THINK in _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES

    def test_olmo_instruct_not_in_suppress_no_functions(self):
        # Instruct template does NOT add the suffix to user-supplied system
        # messages, so it must not be in this set.
        assert ModelFamily.OLMO_INSTRUCT not in _SUPPRESS_NO_FUNCTIONS_SUFFIX_FAMILIES

    def test_deepseek_not_in_tool_role_as_environment(self):
        # DeepSeek uses its own merged-system-into-user path; not an OLMo quirk.
        assert ModelFamily.DEEPSEEK not in _TOOL_ROLE_AS_ENVIRONMENT_FAMILIES


# ---------------------------------------------------------------------------
# OLMo 3 family defaults
# ---------------------------------------------------------------------------

class TestOlmoDefaults:
    def _make(self, family):
        return ModelConfig(
            name="test", family=family, path_or_id="allenai/Olmo-3-7B-X", role="orchestrator"
        )

    def test_olmo_think_temperature(self):
        assert self._make(ModelFamily.OLMO_THINK).temperature == 0.6

    def test_olmo_think_top_p(self):
        assert self._make(ModelFamily.OLMO_THINK).top_p == 0.95

    def test_olmo_think_max_tokens(self):
        assert self._make(ModelFamily.OLMO_THINK).max_tokens == 32768

    def test_olmo_think_top_k_disabled(self):
        # HF card recipe does not set top_k; -1 means "disabled" in vLLM.
        assert self._make(ModelFamily.OLMO_THINK).top_k == -1

    def test_olmo_think_repetition_penalty_noop(self):
        # HF card recipe does not set repetition_penalty; 1.0 is the no-op value.
        assert self._make(ModelFamily.OLMO_THINK).repetition_penalty == 1.0

    def test_olmo_instruct_top_k_disabled(self):
        assert self._make(ModelFamily.OLMO_INSTRUCT).top_k == -1

    def test_olmo_instruct_repetition_penalty_noop(self):
        assert self._make(ModelFamily.OLMO_INSTRUCT).repetition_penalty == 1.0

    def test_olmo_think_supports_thinking(self):
        assert self._make(ModelFamily.OLMO_THINK).supports_thinking is True

    def test_olmo_instruct_does_not_support_thinking(self):
        assert self._make(ModelFamily.OLMO_INSTRUCT).supports_thinking is False


# ---------------------------------------------------------------------------
# rewrite_tool_role_to_environment (OLMo 3 Think)
# ---------------------------------------------------------------------------

class TestRewriteToolRoleToEnvironment:
    def test_renames_single_tool_message(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "<function_calls>\nf()\n</function_calls>"},
            {"role": "tool", "tool_name": "f", "content": "result"},
        ]
        result = rewrite_tool_role_to_environment(msgs)
        assert result[-1]["role"] == "environment"
        assert result[-1]["content"] == "result"

    def test_preserves_tool_name_metadata(self):
        msgs = [{"role": "tool", "tool_name": "web_search", "content": "snippets"}]
        assert rewrite_tool_role_to_environment(msgs)[0]["tool_name"] == "web_search"

    def test_does_not_mutate_input(self):
        msgs = [{"role": "tool", "content": "x"}]
        rewrite_tool_role_to_environment(msgs)
        assert msgs[0]["role"] == "tool"

    def test_no_tool_messages_returns_same_list(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert rewrite_tool_role_to_environment(msgs) is msgs

    def test_rewrites_multiple_tool_turns(self):
        msgs = [
            {"role": "tool", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "tool", "content": "c"},
        ]
        result = rewrite_tool_role_to_environment(msgs)
        assert [m["role"] for m in result] == ["environment", "assistant", "environment"]


# ---------------------------------------------------------------------------
# suppress_no_functions_suffix (OLMo 3 Think)
# ---------------------------------------------------------------------------

class TestSuppressNoFunctionsSuffix:
    def test_injects_empty_functions_on_system(self):
        msgs = [
            {"role": "system", "content": "You may call tools X, Y, Z..."},
            {"role": "user", "content": "q"},
        ]
        result = suppress_no_functions_suffix(msgs)
        assert result[0]["functions"] == ""
        assert result[0]["content"] == msgs[0]["content"]

    def test_preserves_existing_functions_key(self):
        msgs = [{"role": "system", "content": "x", "functions": "existing"}]
        result = suppress_no_functions_suffix(msgs)
        assert result[0]["functions"] == "existing"

    def test_no_system_message_unchanged(self):
        msgs = [{"role": "user", "content": "q"}]
        assert suppress_no_functions_suffix(msgs) is msgs

    def test_empty_list_unchanged(self):
        assert suppress_no_functions_suffix([]) == []

    def test_does_not_mutate_input(self):
        msgs = [{"role": "system", "content": "x"}]
        suppress_no_functions_suffix(msgs)
        assert "functions" not in msgs[0]


class TestMergeSystemIntoUserPreservesOrder:
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
