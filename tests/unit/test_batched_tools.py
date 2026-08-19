"""The real tools' BatchedTool implementations.

The orchestrator trace fixture (B3) drives *fake* tools, so it gates the
orchestrator's control flow but says nothing about the prepare/batch/finalize
logic that moved out of the orchestrator and into ``WebSearchTool`` and
``CodeGeneratorTool``.  Mutating either tool leaves B3 green.  These tests cover
that gap.

Every expectation here is the behaviour of the orchestrator methods this logic
replaced: ``_schedule_web_job``, ``_run_web_analysis_batch``,
``_schedule_code_job`` and ``_run_code_generation_batch``.
"""

from types import SimpleNamespace

import pytest

from agent_engine.core.batching import BatchJob, BatchedTool, flush_batches
from agent_engine.core.tool import ToolResult
from agent_engine.tools.code_generator import CodeGeneratorTool
from agent_engine.tools.web_search import WebSearchTool


def _generation(text, usage=None):
    return SimpleNamespace(text=text, usage=usage or {"total_tokens": 1})


def _state():
    return SimpleNamespace(question_id=0, attachments=None, metadata={})


class StubProvider:
    """Minimal provider: both tools render their prompt through the provider's
    chat template before the batched call."""

    def __init__(self, label="stub", order=None):
        self.label = label
        self.order = order

    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return "".join(m["content"] for m in messages)

    def generate(self, prompts):
        if self.order is not None:
            self.order.append(self.label)
        return [_generation("out") for _ in prompts]


@pytest.fixture
def web():
    tool = WebSearchTool(api_key="key", search_cache={}, url_cache={}, model_provider=StubProvider())
    return tool


@pytest.fixture
def code():
    # return_code keeps execute() from running a subprocess.
    return CodeGeneratorTool(model_provider=StubProvider(), return_code=True)


# --- protocol conformance -------------------------------------------------


@pytest.mark.parametrize("cls, priority", [(WebSearchTool, 10), (CodeGeneratorTool, 20)])
def test_tools_satisfy_the_protocol_with_the_right_priority(cls, priority):
    """Priority order is load-bearing: web must flush before code, because web
    analyses populate a cache a code job in the same turn can read."""
    assert isinstance(cls, type) and issubclass(cls, object)
    assert cls.batch_priority == priority
    assert WebSearchTool.batch_priority < CodeGeneratorTool.batch_priority


def test_instances_are_recognised_as_batched(web, code):
    assert isinstance(web, BatchedTool)
    assert isinstance(code, BatchedTool)


# --- web: prepare ---------------------------------------------------------


def test_web_prepare_rejects_a_missing_query(web):
    result = web.prepare(_state(), {"name": "web_search", "arguments": {}}, {})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error == "Missing required web_search arguments"


def test_web_prepare_returns_a_cache_hit_without_searching(web, monkeypatch):
    web.analysis_cache["alpha"] = "CACHED ANALYSIS"
    monkeypatch.setattr(web, "search_and_format",
                        lambda q: pytest.fail("search_and_format ran on a cache hit"))

    result = web.prepare(_state(), {"name": "web_search"}, {"query": "alpha"})
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.output == "CACHED ANALYSIS"
    assert result.metadata == {"cached": True, "query": "alpha", "mode": "sub-agent"}


def test_web_prepare_turns_a_search_failure_into_a_failed_result(web, monkeypatch):
    def boom(query):
        raise RuntimeError("search backend exploded")

    monkeypatch.setattr(web, "search_and_format", boom)
    result = web.prepare(_state(), {"name": "web_search"}, {"query": "q"})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error == "search backend exploded"
    assert result.metadata == {"query": "q"}


def test_web_prepare_defers_on_success(web, monkeypatch):
    payload = {"results": [{"title": "t"}], "urls_to_fetch": [], "url_snippets": {}}
    monkeypatch.setattr(web, "search_and_format", lambda q: payload)

    job = web.prepare(_state(), {"name": "web_search"}, {"query": "q"})
    assert isinstance(job, BatchJob)
    assert job.payload["query"] == "q"
    assert job.payload["payload"] is payload


# --- web: finalize --------------------------------------------------------


def test_web_finalize_strips_thinking_and_populates_the_cache(web):
    job = BatchJob(_state(), {"name": "web_search"}, web, {"query": "q", "payload": {}})
    result = web.finalize(job, _generation("<think>hidden</think>ANALYSIS"))

    assert result.output == "ANALYSIS"
    # The cache write is what makes a later identical query a prepare() hit.
    assert web.analysis_cache["q"] == "ANALYSIS"


# --- code -----------------------------------------------------------------


def test_code_prepare_rejects_a_missing_task(code):
    result = code.prepare(_state(), {"name": "code_generator", "arguments": {}}, {})
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error == "Missing required code_generator arguments"


def test_code_prepare_defers_with_a_built_prompt(code):
    job = code.prepare(_state(), {"name": "code_generator"}, {"task": "add two numbers"})
    assert isinstance(job, BatchJob)
    assert "add two numbers" in job.payload["prompt"]
    assert code.batch_prompt(job) == job.payload["prompt"]


def test_code_finalize_strips_thinking_before_extracting_code(code):
    job = BatchJob(_state(), {"name": "code_generator"}, code, {"prompt": "p"})
    result = code.finalize(job, _generation("<think>planning</think>```python\nx = 1\n```"))

    assert result.success is True
    assert "<think>" not in result.output
    assert "planning" not in result.output


def test_code_finalize_strips_the_executed_output_too(code, monkeypatch):
    """There are *two* strips on this path, and they guard different text.

    The first cleans the model's generation before code extraction; this one
    cleans what execution returned.  With ``return_code`` set, the "output" is
    model-written code that can still carry think tags, so dropping the second
    strip leaks them into the orchestrator's context.  A test whose executed
    output happens to be clean cannot tell the two strips apart.
    """
    monkeypatch.setattr(code, "execute", lambda code=None, task=None: ToolResult(
        success=True, output="<think>leaked</think>RESULT", metadata={}))
    job = BatchJob(_state(), {"name": "code_generator"}, code, {"prompt": "p"})

    result = code.finalize(job, _generation("x = 1"))
    assert result.output == "RESULT"


def test_code_finalize_converts_an_exception_into_a_failed_result(code, monkeypatch):
    monkeypatch.setattr(code, "extract_code_from_llm_response",
                        lambda text: (_ for _ in ()).throw(ValueError("bad code")))
    job = BatchJob(_state(), {"name": "code_generator"}, code, {"prompt": "p"})
    result = code.finalize(job, _generation("whatever"))

    assert result.success is False
    assert result.error == "bad code"


# --- flush_batches ordering with the real tools ---------------------------


def test_flush_order_is_web_then_code(web, code, monkeypatch):
    """The single most load-bearing ordering property in the batching module."""
    order = []

    web.model_provider = StubProvider("web", order)
    code.model_provider = StubProvider("code", order)
    monkeypatch.setattr(web, "pre_batch", lambda jobs: order.append("web.pre_batch"))

    jobs_by_tool = {
        # deliberately code-first insertion order: priority must win, not order
        "code_generator": [BatchJob(_state(), {"name": "code_generator"}, code, {"prompt": "p"})],
        "web_search": [BatchJob(_state(), {"name": "web_search"}, web,
                                {"query": "q", "payload": {"results": []}})],
    }
    flush_batches(jobs_by_tool, lambda s, tc, text: None, lambda s, u: None)

    assert order == ["web.pre_batch", "web", "code"]


def test_a_tool_without_a_provider_drops_its_jobs_silently(code):
    """Preserved from the original: `generate` is skipped and `zip` yields
    nothing, so nothing is committed.

    Uses the code tool because its `batch_prompt` just returns the prepared
    string.  The web tool renders its prompt through the provider, so a missing
    provider raises there instead -- which is also what the original
    `_run_web_analysis_batch` did, since it built prompts before the
    `if provider` check.
    """
    code.model_provider = None
    committed = []
    job = BatchJob(_state(), {"name": "code_generator"}, code, {"prompt": "p"})

    flush_batches({"code_generator": [job]},
                  lambda s, tc, text: committed.append(text),
                  lambda s, u: None)

    assert committed == []


def test_web_without_a_provider_raises_as_it_always_did(web):
    """`_run_web_analysis_batch` built its prompts before checking the provider,
    so a missing provider surfaced as an AttributeError rather than a silent
    skip.  Locked so the collapse cannot quietly turn a crash into a no-op."""
    web.model_provider = None
    job = BatchJob(_state(), {"name": "web_search"}, web, {"query": "q", "payload": {"results": []}})

    with pytest.raises(AttributeError):
        flush_batches({"web_search": [job]}, lambda s, tc, text: None, lambda s, u: None)


def test_only_the_generation_usage_is_accumulated(code):
    """The deferred path must not count the finalize ToolResult's usage; the
    immediate path does the opposite.  Unifying them double-counts tokens."""
    usages = []

    class Provider(StubProvider):
        def generate(self, prompts):
            return [_generation("x = 1", usage={"total_tokens": 5}) for _ in prompts]

    code.model_provider = Provider()
    job = BatchJob(_state(), {"name": "code_generator"}, code, {"prompt": "p"})

    flush_batches({"code_generator": [job]},
                  lambda s, tc, text: None,
                  lambda s, u: usages.append(u))

    assert usages == [{"total_tokens": 5}]
