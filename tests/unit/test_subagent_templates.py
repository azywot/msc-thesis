import json
from types import SimpleNamespace

import pytest

from agent_engine.prompts.templating import find_placeholders, render_placeholders
from agent_engine.tools.code_generator import CODE_GENERATOR_TASK_TEMPLATE, CodeGeneratorTool
from agent_engine.tools.web_search import WEB_SEARCH_ANALYSIS_TEMPLATE, WebSearchTool


class CaptureProvider:
    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return json.dumps(messages)

    def generate(self, prompts):
        return [SimpleNamespace(text="out", usage={"total_tokens": 1}) for _ in prompts]


def captured_messages(prompt):
    return json.loads(prompt)


@pytest.fixture
def web():
    return WebSearchTool(
        api_key="key", search_cache={}, url_cache={}, model_provider=CaptureProvider()
    )


@pytest.fixture
def code():
    return CodeGeneratorTool(model_provider=CaptureProvider())


@pytest.fixture
def state():
    return SimpleNamespace(question_id=0, attachments=None, metadata={})


def test_render_placeholders_single_pass():
    assert render_placeholders("{a} {b} {a}", {"a": "{b}", "b": "x"}) == "{b} x {b}"
    assert render_placeholders("keep {unknown} and {", {"a": "1"}) == "keep {unknown} and {"


def test_web_seed_template_reproduces_todays_prompt(web):
    msgs = captured_messages(web.build_analysis_prompt("q1", "PAGES"))
    text = msgs[0]["content"]
    assert text.startswith(
        "**Task Instruction:**\n\nYou are tasked with reading and analyzing web pages"
    )
    assert "- **Current Search Query:**\nq1\n\n- **Searched Web Pages:**\nPAGES\n" in text
    assert text.endswith('based on the current search query "q1".\n')
    assert find_placeholders(WEB_SEARCH_ANALYSIS_TEMPLATE) == [
        "query",
        "formatted_results",
        "query",
    ]


def test_web_override_template_is_used(web):
    web.set_analysis_template("Extract facts for {query} from {formatted_results}.")
    prompt = web.build_analysis_prompt("q", "P")
    assert captured_messages(prompt)[0]["content"] == "Extract facts for q from P."


def test_analysis_cache_is_per_template(web):
    web.analysis_cache["q"] = "seed analysis"
    web.set_analysis_template("T {query} {formatted_results}")
    assert "q" not in web.analysis_cache
    web.set_analysis_template(WEB_SEARCH_ANALYSIS_TEMPLATE)
    assert web.analysis_cache["q"] == "seed analysis"


def test_code_seed_template_reproduces_todays_prompt(code):
    text = captured_messages(code.build_task_prompt("add 1+1"))[0]["content"]
    assert text.startswith("You are a code generator. Generate ONLY executable Python code")
    assert "\n\nProblem: add 1+1\n\nRequirements:\n- Output ONLY the Python code\n" in text
    assert text.endswith("Python code:")
    with_ctx = captured_messages(code.build_task_prompt("t", context="C"))[0]["content"]
    assert "\n\nContext:\n\nC\n\nProblem: t\n\n" in with_ctx
    assert find_placeholders(CODE_GENERATOR_TASK_TEMPLATE) == ["context_block", "task"]


def test_code_override_template_is_used(code):
    code.set_task_template("{context_block}DO {task}")
    text = captured_messages(code.build_task_prompt("t", context="C"))[0]["content"]
    assert text == "Context:\n\nC\n\nDO t"


def test_code_prepare_payload_keeps_task_and_context(code, state):
    job = code.prepare(state, {"name": "code_generator", "arguments": {"task": "t"}}, {"task": "t"})
    assert job.payload["task"] == "t" and "context" in job.payload
