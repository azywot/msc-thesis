import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unittest.mock import patch, MagicMock
from agent_engine.models.api_provider import OpenAIProvider
from agent_engine.models.base import ModelConfig, ModelFamily


def _config():
    return ModelConfig(
        name="test",
        family=ModelFamily.QWEN3,
        path_or_id="Qwen/Qwen3-8B",
        role="orchestrator",
    )


def test_default_base_url_is_none():
    with patch("agent_engine.models.api_provider.OpenAI") as mock_openai:
        OpenAIProvider(_config(), api_key="k")
        _, kwargs = mock_openai.call_args
        assert kwargs.get("base_url") is None


def test_custom_base_url_is_forwarded():
    with patch("agent_engine.models.api_provider.OpenAI") as mock_openai:
        OpenAIProvider(_config(), api_key="k", base_url="http://localhost:9000/v1")
        _, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == "http://localhost:9000/v1"
