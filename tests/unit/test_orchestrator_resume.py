"""resume_batch continues a rebuilt state, optionally injecting one tool call."""

import json
from types import SimpleNamespace

import pytest

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry, ToolResult

CALL = '<tool_call>\n{"name": "web_search", "arguments": {"query": "%s"}}\n</tool_call>'
INJECTED = '<sub_goal>g</sub_goal>\n\n' + CALL % "q2"


class FakeModel:
    """Scripted provider that records every prompt it is asked to generate from."""

    def __init__(self, script):
        self.script = list(script)
        self.prompts_seen = []
        self.config = SimpleNamespace(supports_thinking=False, family="qwen3")

    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False):
        return "\n".join(m["content"] for m in messages)

    def generate(self, prompts):
        out = []
        for p in prompts:
            self.prompts_seen.append(p)
            text = self.script.pop(0) if self.script else "no answer, still thinking"
            out.append(SimpleNamespace(text=text, usage={}))
        return out

    def cleanup(self):
        pass


class FakeSearch:
    name = "web_search"
    def execute(self, query, **kw):
        return ToolResult(output="result for " + query, success=True)


def make_orch(script, max_turns=10):
    reg = ToolRegistry()
    reg.register(FakeSearch())
    model = FakeModel(script)
    return AgenticOrchestrator(model_provider=model, tool_registry=reg, max_turns=max_turns), model


def prefix_state(turn=1):
    return ExecutionState(
        question_id=0,
        question="Q?",
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q?"}],
        query_analysis="analysis",
        action_history=[
            {"tool_name": "web_search", "sub_goal": "s", "command": "{}", "result": "R1"}
        ],
        output_messages=[
            {"role": "assistant", "content": "call1"},
            {"role": "tool", "tool_name": "web_search", "content": "R1"},
        ],
        tool_calls=[{"name": "web_search", "arguments": {"query": "q1"}}],
        tool_counts={"web_search": 1},
        turn=turn,
        metadata={},
    )


def test_run_batch_output_unchanged_after_refactor():
    orch, _ = make_orch(["plan", CALL % "q1", "\\boxed{A}"])
    st = orch.run_batch(questions=["Q?"], question_ids=[0], system_prompts=["SYS"])[0]
    assert st.answer == "A"
    assert st.turn == 2
    assert len(st.action_history) == 1
    assert [m["role"] for m in st.output_messages] == ["assistant", "tool", "assistant"]


def test_resume_injects_call_then_continues():
    orch, model = make_orch(["\\boxed{X}"])
    st = orch.resume_batch([prefix_state()], [INJECTED])[0]
    assert st.answer == "X"
    assert st.turn == 3
    assert st.action_history[1]["sub_goal"] == "g"
    assert json.loads(st.action_history[1]["command"])["arguments"] == {"query": "q2"}
    assert st.output_messages[2] == {"role": "assistant", "content": INJECTED}
    assert st.tool_counts["web_search"] == 2
    assert "Action Step 2:" in model.prompts_seen[0]


def test_resume_without_injection_generates_from_prefix():
    orch, model = make_orch(["\\boxed{X}"])
    st = orch.resume_batch([prefix_state()], [None])[0]
    assert st.answer == "X"
    assert st.turn == 2
    assert "Action Step 1:" in model.prompts_seen[0]
    assert "Action Step 2:" not in model.prompts_seen[0]


def test_resume_at_last_turn_uses_synthesis():
    orch, model = make_orch(["\\boxed{S}"], max_turns=7)
    st = orch.resume_batch([prefix_state(turn=6)], [INJECTED])[0]
    assert st.metadata["max_turns_reached"] is True
    assert len(model.prompts_seen) == 1  # only the synthesis call, no extra turn
    assert st.answer == "S"
    assert st.turn == 7


def test_resume_rejects_bad_input():
    orch, _ = make_orch([], max_turns=7)
    with pytest.raises(ValueError):
        orch.resume_batch([prefix_state()], ["just prose, no call"])
    with pytest.raises(ValueError):
        orch.resume_batch([prefix_state(turn=7)], [INJECTED])
    with pytest.raises(ValueError):
        orch.resume_batch([prefix_state()], [])
    done = prefix_state()
    done.finished = True
    with pytest.raises(ValueError):
        orch.resume_batch([done], [None])


PLAN_SUFFIX = (
    "\n\nBefore using any tools, analyze this query to determine the approach needed.\n"
    "Instructions:\n"
    "1. Identify the main objectives in the query.\n"
    "2. List the necessary skills and tools.\n"
    "3. For each tool, explain how it helps address the query.\n"
    "4. Note any additional considerations.\n\n"
    "Be brief and precise. Do NOT call any tools yet."
)
SYNTH = (
    "You have used all available tool calls. Based on the information gathered so far, "
    "provide your final answer now."
)


def test_run_batch_single_prompt_bytes_unchanged():
    """One shared system prompt: turn and synthesis prompts match the pre-fix bytes."""
    orch, model = make_orch(["p1", "p2"] + [CALL % "q1"] * 4 + ["\\boxed{A}", "\\boxed{B}"], 2)
    orch.run_batch(questions=["Q?", "Q2?"], question_ids=[0, 1], system_prompts=["SYS", "SYS"])
    step1 = (
        "\n\n**Previous Steps:**\nAction Step 1:\n  - Tool: web_search\n  - Sub-goal: \n"
        '  - Command: {"name": "web_search", "arguments": {"query": "q1"}}\n'
        "  - Result: result for q1\n"
    )
    step2 = step1.replace("Action Step 1", "Action Step 2").replace("\n\n**Previous Steps:**", "")
    assert model.prompts_seen == [
        "SYS\nQ?" + PLAN_SUFFIX,
        "SYS\nQ2?" + PLAN_SUFFIX,
        "SYS\nQ?\n\n**Query Analysis:**\np1",
        "SYS\nQ2?\n\n**Query Analysis:**\np2",
        "SYS\nQ?\n\n**Query Analysis:**\np1" + step1,
        "SYS\nQ2?\n\n**Query Analysis:**\np2" + step1,
        "SYS\nQ?\n\n**Query Analysis:**\np1" + step1 + step2 + "\n" + SYNTH,
        "SYS\nQ2?\n\n**Query Analysis:**\np2" + step1 + step2 + "\n" + SYNTH,
    ]


def _assert_own_system_prompt(prompts, by_question):
    assert prompts
    for p in prompts:
        owners = [q for q in by_question if "\n" + q in p]
        assert len(owners) == 1, p
        assert p.startswith(by_question[owners[0]] + "\n"), p


def test_run_batch_mixed_system_prompts_each_state_keeps_its_own():
    orch, model = make_orch(["p1", "p2"] + [CALL % "q1"] * 4 + ["\\boxed{A}", "\\boxed{B}"], 2)
    orch.run_batch(
        questions=["alpha?", "beta?"], question_ids=[0, 1], system_prompts=["SYS_A", "SYS_B"]
    )
    assert len(model.prompts_seen) == 8  # planning, two turns, synthesis
    _assert_own_system_prompt(model.prompts_seen, {"alpha?": "SYS_A", "beta?": "SYS_B"})


def test_resume_batch_mixed_system_prompts_each_state_keeps_its_own():
    orch, model = make_orch([CALL % "q1"] * 3 + ["\\boxed{A}", "\\boxed{B}"], 3)
    a, b = prefix_state(), prefix_state()
    for s, qid, question, system in ((a, 0, "alpha?", "SYS_A"), (b, 1, "beta?", "MATH_SYS")):
        s.question_id, s.question = qid, question
        s.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
    orch.resume_batch([a, b], [INJECTED, None])
    # a: injected turn 2, generated turn 3, synthesis; b: generated turns 2 and 3, synthesis
    assert len(model.prompts_seen) == 5
    _assert_own_system_prompt(model.prompts_seen, {"alpha?": "SYS_A", "beta?": "MATH_SYS"})
