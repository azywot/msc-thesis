import pytest

from agent_engine.tools.code_generator import CodeGeneratorTool
from agent_engine.tools.web_search import WebSearchTool


class FakeProvider:
    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return ""

    def generate(self, prompts):
        return []


@pytest.fixture
def SCHEMAS():
    return [
        WebSearchTool(
            api_key="k", search_cache={}, url_cache={}, model_provider=FakeProvider()
        ).get_schema(),
        CodeGeneratorTool(model_provider=FakeProvider()).get_schema(),
    ]
