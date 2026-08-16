"""B3: the orchestrator's batching control flow must not change.

Locks the tool-call sequence, committed message text, structured-memory
contents, per-state token usage, and the *order* of all of it -- including
order across states, which no single state's history can show.

This is the gate for the batching collapse: that phase replaces the hardcoded
``_WebJob``/``_CodeJob`` paths with one protocol.  If this fixture stays green,
the collapse preserved behaviour.

Four details this scenario depends on, each learned from the orchestrator
rather than assumed:

1. ``_schedule_code_job`` (orchestrator.py:478) guards on
   ``hasattr(tool, "execute_code")``.  A fake without that attribute silently
   falls into the missing-arguments *error* branch and never defers, leaving
   the entire batched-code path untested.
2. ``ToolResult.usage`` defaults to ``None``.  The deferred code path counts
   only the generation's usage and never the finalize result's -- an asymmetry
   with the immediate path, which does the opposite.  Probing it requires a
   fake whose ``execute`` returns a non-zero usage.
3. Flush order (web before code) is only observable *across* states, since the
   two paths serve different states.  Hence ``EVENTS``.
4. ``strip_thinking_tags`` in ``_apply_immediate_results`` is only observable
   if some immediate result's output actually carries think tags.  Cached
   analyses do not -- they were stripped when written (orchestrator.py:560) --
   so a direct-mode tool supplies them.
"""

import json
from typing import Any, Dict, List

from agent_engine.core.batching import BatchJob
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import BaseTool, ToolRegistry, ToolResult
from agent_engine.utils.parsing import strip_thinking_tags

from .conftest import assert_matches_fixture
from .scripted_provider import ScriptedProvider

# Global, ordered record of side effects across all states.  Flush order is a
# cross-state property: swapping the web and code flushes leaves every
# individual state's message list byte-identical, so a per-state dump cannot
# see it.
EVENTS: List[str] = []


class FakeWebSearch(BaseTool):
    """Mimics WebSearchTool's sub-agent (deferred) contract."""

    direct_mode = False

    def __init__(self, model_provider):
        self.model_provider = model_provider
        self.url_cache: Dict[str, str] = {}
        self.use_jina = False
        self._analysis_cache: Dict[str, str] = {}
        self.pre_batch_batch_sizes: List[int] = []

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "fake web search"

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "web_search", "parameters": {}}}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="direct", metadata={})

    # -- BatchedTool protocol ----------------------------------------------
    # Mirrors WebSearchTool's real implementation.  The EVENTS strings are
    # unchanged from the pre-protocol version of this fake and fire at the same
    # points in the sequence, so the recorded fixture stays byte-identical
    # across the collapse.

    batch_priority = 10

    def prepare(self, state, tool_call, args):
        query = args.get("query", "")
        if not query:
            return ToolResult(success=False, output="", metadata={},
                              error="Missing required web_search arguments")
        if query in self._analysis_cache:
            return ToolResult(success=True, output=self._analysis_cache[query],
                              metadata={"cached": True, "query": query, "mode": "sub-agent"})
        EVENTS.append(f"web.prepare({query})")
        if query == "boom":
            return ToolResult(success=False, output="", metadata={"query": query},
                              error="search backend exploded")
        return BatchJob(state=state, tool_call=tool_call, tool=self,
                        payload={"query": query,
                                 "payload": {"results": [{"title": query}],
                                             "urls_to_fetch": [], "url_snippets": {}}})

    def pre_batch(self, jobs) -> None:
        # Deliberately emits no EVENT: the hook is new, and adding a line here
        # would change the recorded trace for a reason unrelated to behaviour.
        # Coverage comes from the pre_batch_batch_sizes assertion in the test.
        self.pre_batch_batch_sizes.append(len(jobs))

    def batch_prompt(self, job) -> str:
        # Emitted during the web *flush*, not during classification.  Without an
        # event on this side of the flush there is nothing for the code flush's
        # event to be ordered against, and swapping the two would be invisible.
        query = job.payload["query"]
        EVENTS.append(f"web.analyse({query})")
        return f"ANALYSE {query} :: RESULTS({query})"

    def finalize(self, job, generation) -> ToolResult:
        query = job.payload["query"]
        text = strip_thinking_tags(generation.text)
        self._analysis_cache[query] = text
        return ToolResult(success=True, output=text,
                          metadata={"query": query, "mode": "sub-agent"})


class FakeCodeGenerator(BaseTool):
    """Mimics CodeGeneratorTool's sub-agent (deferred) contract."""

    direct_mode = False

    def __init__(self, model_provider):
        self.model_provider = model_provider

    @property
    def name(self) -> str:
        return "code_generator"

    @property
    def description(self) -> str:
        return "fake code generator"

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "code_generator", "parameters": {}}}

    # -- BatchedTool protocol ----------------------------------------------
    batch_priority = 20

    def prepare(self, state, tool_call, args):
        task = args.get("task", "")
        if not task:
            return ToolResult(success=False, output="", metadata={},
                              error="Missing required code_generator arguments")
        EVENTS.append(f"code.prepare({task})")
        return BatchJob(state=state, tool_call=tool_call, tool=self,
                        payload={"prompt": f"WRITE CODE FOR {task}"})

    def batch_prompt(self, job) -> str:
        return job.payload["prompt"]

    def finalize(self, job, generation) -> ToolResult:
        code = strip_thinking_tags(generation.text).strip()
        tr = self.execute(code=code, task=None)
        return ToolResult(success=tr.success,
                          output=strip_thinking_tags(tr.output or ""),
                          metadata=tr.metadata, error=tr.error, usage=tr.usage)

    def execute(self, code=None, task=None) -> ToolResult:
        EVENTS.append(f"code.finalize({code!r})")
        # Non-zero usage on purpose: the deferred path must NOT count it.  With
        # the default None this asymmetry would be invisible.
        return ToolResult(
            success=True,
            output=f"RAN[{code}]",
            metadata={},
            usage={"prompt_tokens": 700, "completion_tokens": 70, "total_tokens": 770},
        )


class FakeTextInspector(BaseTool):
    """Immediate (non-deferred) path, returning output that carries think tags.

    ``text_inspector`` is attachment-gated: ``_inject_attachment_path``
    (orchestrator.py:871) short-circuits to an error unless the state carries a
    file with a ``_TEXT_EXTS`` extension, so ``execute`` is never reached.
    State 3 is given ``notes.txt`` for this reason, which also covers the
    attachment-injection branch.
    """

    direct_mode = True

    @property
    def name(self) -> str:
        return "text_inspector"

    @property
    def description(self) -> str:
        return "fake text inspector"

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "text_inspector", "parameters": {}}}

    def execute(self, **kwargs) -> ToolResult:
        EVENTS.append(f"direct.execute(full_file_path={kwargs.get('full_file_path')!r})")
        return ToolResult(
            success=True,
            output="<think>pondering the file</think>INSPECTED",
            metadata={},
            # Counted by the immediate path, unlike the deferred paths.
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )


def _call(name: str, **args) -> str:
    """Qwen3 tool-call format, which is what parse_tool_call matches first."""
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


def _serialise(states, events, providers) -> str:
    """Deterministic, diff-friendly dump of everything B3 locks."""
    out = []
    for s in sorted(states, key=lambda x: x.question_id):
        out.append(f"=== state {s.question_id} ===")
        out.append(f"finished={s.finished} turn={s.turn} answer={s.answer!r}")
        out.append(f"tool_counts={json.dumps(s.tool_counts, sort_keys=True)}")
        out.append(f"token_usage={json.dumps(s.metadata.get('token_usage', {}), sort_keys=True)}")
        out.append(f"query_analysis={s.query_analysis!r}")
        out.append("action_history:")
        for i, step in enumerate(s.action_history):
            out.append(f"  [{i}] {json.dumps(step, sort_keys=True)}")
        out.append("output_messages:")
        for i, msg in enumerate(s.output_messages):
            out.append(f"  [{i}] {msg['role']}: {msg['content']}")

    out.append("=== events (global order) ===")
    for i, event in enumerate(events):
        out.append(f"  [{i}] {event}")

    out.append("=== sub-agent batching ===")
    for label, provider in providers:
        for i, call in enumerate(provider.calls):
            out.append(f"  {label} call[{i}] n={len(call)} prompts={json.dumps(call)}")

    return "\n".join(out) + "\n"


def test_orchestrator_trace_unchanged(update_fixtures):
    EVENTS.clear()

    orch_outputs = (
        # turn 0: planning, one per state
        [f"PLAN for state {i}" for i in range(5)]
        # turn 1
        + [
            _call("web_search", query="alpha"),      # s0 -> deferred
            _call("code_generator", task="sum"),     # s1 -> deferred
            _call("web_search", query="boom"),       # s2 -> prepare raises
            _call("web_search"),                     # s3 -> missing argument
            "Final Answer: four",                    # s4 -> done
        ]
        # turn 2
        + [
            "Final Answer: zero",                    # s0
            "Final Answer: one",                     # s1
            _call("web_search", query="alpha"),      # s2 -> analysis_cache hit
            _call("text_inspector", question="what?"),  # s3 -> immediate, think tags
        ]
        # turn 3
        + [
            "Final Answer: two",                     # s2
            "Final Answer: three",                   # s3
        ]
    )

    orch_provider = ScriptedProvider(orch_outputs, name="orchestrator")
    web_provider = ScriptedProvider(["<think>searching</think>ANALYSIS OF ALPHA"], name="web-sub")
    code_provider = ScriptedProvider(["print(1 + 1)"], name="code-sub")

    web_tool = FakeWebSearch(web_provider)
    tools = ToolRegistry()
    tools.register(web_tool)
    tools.register(FakeCodeGenerator(code_provider))
    tools.register(FakeTextInspector())

    orchestrator = AgenticOrchestrator(
        model_provider=orch_provider,
        tool_registry=tools,
        max_turns=15,
        use_thinking=False,
        baseline=False,
    )

    # Record the global commit order.  Immediate results commit before the web
    # flush, which commits before the code flush -- an ordering spread across
    # different states, so no single state's message list reveals it.
    _original_commit = orchestrator._commit_tool_result

    def _recording_commit(state, tool_call, clean_output):
        EVENTS.append(f"commit(q{state.question_id}, {tool_call['name']})")
        return _original_commit(state, tool_call, clean_output)

    orchestrator._commit_tool_result = _recording_commit

    states = orchestrator.run_batch(
        questions=[f"question {i}" for i in range(5)],
        question_ids=list(range(5)),
        system_prompts=["SYSTEM"] * 5,
        # text_inspector is attachment-gated; without this state 3 errors out
        # before execute() and the immediate path stays uncovered.
        attachments=[None, None, None, ["notes.txt"], None],
    )

    # A changed generate() count is itself a behaviour change worth understanding.
    # Never pad the queue to make this pass.
    assert orch_provider.remaining == 0, (
        f"scenario drifted: {orch_provider.remaining} scripted turns unconsumed; "
        f"orchestrator made {len(orch_provider.calls)} generate() calls"
    )
    assert web_provider.remaining == 0, "web sub-agent script unconsumed"
    assert code_provider.remaining == 0, "code sub-agent script unconsumed"

    # The pre_batch hook runs once per flush, over ALL of that tool's jobs.
    # Asserted here rather than recorded in the fixture: the hook postdates the
    # fixture, so emitting an event for it would change the trace for a reason
    # that has nothing to do with behaviour.
    assert web_tool.pre_batch_batch_sizes == [1], (
        f"expected one web flush of one job, got {web_tool.pre_batch_batch_sizes}"
    )

    trace = _serialise(
        states,
        EVENTS,
        [("web", web_provider), ("code", code_provider)],
    )
    assert_matches_fixture("orchestrator_trace.txt", trace, update_fixtures)
