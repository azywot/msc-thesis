"""The fix: the prompt a reflector reads, its parsed reply, and the leakage guard.

For a failed trajectory the reflector returns one corrected orchestrator tool call at the first
step it judges wrong. `Fix.text` is in the orchestrator's own action format so that it can be
injected as that step's output.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from agent_engine.analysis.failure_modes import classify_failure
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.datasets.evaluators.metrics import normalize_answer
from agent_engine.utils.parsing import parse_tool_call, strip_thinking_tags
from rsi.executor import TrajectoryRecord
from rsi.leakage import GENERIC_ANSWERS, LeakageIndex
from rsi.reflector import restore_latex_escapes
from rsi.reflector_input import MIN_PER_CALL_CHARS, _render_trajectory

TOOL_ARGS = {"web_search": "query", "code_generator": "task"}
MAX_SUB_GOAL_CHARS = 500


@dataclass(frozen=True)
class Fix:
    step: int
    sub_goal: str
    tool_call: dict

    @property
    def text(self) -> str:
        call = json.dumps(self.tool_call, ensure_ascii=False)
        return f"<sub_goal>{self.sub_goal}</sub_goal>\n\n<tool_call>\n{call}\n</tool_call>"

    def kind(self, n_actions: int) -> str:
        if self.step < n_actions:
            return "rewrite"
        return "follow_up" if n_actions >= 1 else "insert"


class FixParseError(ValueError):
    pass


def _last_block(body: str, tag: str) -> str:
    blocks = re.findall(rf"<{tag}>(.*?)</{tag}>", body, re.DOTALL)
    if not blocks:
        raise FixParseError(f"no complete <{tag}> block")
    return blocks[-1].strip()


def parse_fix(text: str, n_actions: int, max_turns: int) -> Fix:
    body = strip_thinking_tags(text or "")
    if not body:
        raise FixParseError("empty reflector output")
    raw_step = _last_block(body, "step")
    try:
        step = int(raw_step)
    except ValueError:
        raise FixParseError(f"step is not an integer: {raw_step!r}")
    top = min(n_actions, max_turns - 1)
    if not 0 <= step <= top:
        raise FixParseError(f"step {step} outside [0, {top}]")
    sub_goal = _last_block(body, "sub_goal")
    if not sub_goal:
        raise FixParseError("empty sub-goal")
    if len(sub_goal) > MAX_SUB_GOAL_CHARS:
        raise FixParseError(f"sub-goal longer than {MAX_SUB_GOAL_CHARS} characters")
    try:
        call = json.loads(_last_block(body, "tool_call"))
    except json.JSONDecodeError as e:
        raise FixParseError(f"tool call is not valid JSON: {e}")
    if not isinstance(call, dict):
        raise FixParseError("tool call is not an object")
    call = restore_latex_escapes(call)
    name = call.get("name")
    if name not in TOOL_ARGS:
        raise FixParseError(f"unknown tool: {name!r}")
    args = call.get("arguments")
    key = TOOL_ARGS[name]
    if not isinstance(args, dict) or set(args) != {key}:
        raise FixParseError(f"{name} takes exactly one argument, {key!r}")
    if not isinstance(args[key], str) or not args[key].strip():
        raise FixParseError(f"empty {key!r}")
    fix = Fix(step, sub_goal, {"name": name, "arguments": {key: args[key]}})
    if parse_tool_call(fix.text) != fix.tool_call:
        raise FixParseError("injected text does not parse back to the same tool call")
    if AgenticOrchestrator._extract_sub_goal(fix.text) != fix.sub_goal:
        raise FixParseError("injected text does not parse back to the same sub-goal")
    return fix


def to_trajectory_record(record: dict, show_gold: bool) -> TrajectoryRecord:
    prediction = record.get("prediction") or ""
    correct = bool(record.get("correct"))
    label = None
    if not correct:
        label = classify_failure(
            {
                "action_history": record.get("action_history") or [],
                "prediction": prediction,
                "turns": record.get("turns") or 0,
                "question": record.get("question") or "",
            }
        )
    return TrajectoryRecord(
        question_id=record["question_id"],
        question=record.get("question") or "",
        gold=str(record.get("ground_truth")) if show_gold else "(hidden)",
        aliases=list(record.get("aliases") or []) if show_gold else [],
        prediction=prediction,
        correct=correct,
        failure_label=label,
        plan=record.get("raw_query_analysis") or record.get("query_analysis") or "",
        action_history=list(record.get("action_history") or []),
        subagent_calls=list(record.get("subagent_calls") or []),
        turns=record.get("turns") or 0,
    )


# Phase 1's LEAKAGE_INSTRUCTION forbids the question's entities and numbers, which a corrected
# query or task must carry; the leakage guard exempts the question's own text, so only the
# answer is forbidden here.
FIX_LEAKAGE_INSTRUCTION = (
    "The corrected call may use the question's own entities, numbers and wording. Do not put "
    "the gold answer, or anything that reveals it, in the sub-goal or the tool call; such fixes "
    "are rejected."
)

_SYSTEM = (
    "You are reviewing one failed run of a tool-using orchestrator that answers a question by "
    "calling tools step by step. Each action has a sub-goal and one tool call. Read the "
    "trajectory and find the first step that went wrong. Then return one corrected tool call "
    "for that step. It takes one of three kinds:\n"
    "- rewrite: replace the tool call the orchestrator made at step N (N is the action number).\n"
    "- follow_up: all actions were fine so far but one more call is needed; N is the number of "
    "actions taken.\n"
    "- insert: the run made no tool call but should have; N is 0.\n\n"
    "Use only these tools, each with exactly one argument:\n"
    '- web_search with {"query": "<search query>"}\n'
    '- code_generator with {"task": "<what the code should do>"}\n\n'
    "Reply in exactly this format:\n"
    "<step>N</step>\n"
    "<sub_goal>one sentence, at most 500 characters</sub_goal>\n"
    '<tool_call>{"name": "web_search", "arguments": {"query": "..."}}</tool_call>\n\n'
    + FIX_LEAKAGE_INSTRUCTION
)


def _earlier_line(previous):
    step = previous.get("step")
    if step is None:
        return "This fix was tried at the step it named:"
    return f"This fix was tried at step {step}:"


def _user_text(t, pcc, cut_actions, previous):
    text, _ = _render_trajectory(t, pcc, cut_actions, None)
    parts = ["## Failed trajectory", text]
    if previous:
        parts += ["", "## Earlier attempt", _earlier_line(previous), previous["fix_text"]]
        if previous.get("status") == "leakage":
            parts.append(
                "That fix was rejected because it revealed the answer. Propose a different fix."
            )
        else:
            parts += [
                f"The trajectory then answered: {previous['prediction']}",
                "That answer was wrong. Propose a different fix.",
            ]
    return "\n".join(parts)


def build_fix_messages(
    record: dict,
    *,
    show_gold: bool = True,
    previous: Optional[dict] = None,
    per_call_chars: int = 1500,
    count_tokens: Optional[Callable[[list], int]] = None,
    budget_tokens: int = 14000,
) -> list:
    t = to_trajectory_record(record, show_gold)

    def build(pcc, cut):
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _user_text(t, pcc, cut, previous)},
        ]

    pcc = per_call_chars
    msgs = build(pcc, False)
    if count_tokens is None:
        return msgs
    while count_tokens(msgs) > budget_tokens and pcc > MIN_PER_CALL_CHARS:
        pcc = max(pcc // 2, MIN_PER_CALL_CHARS)
        msgs = build(pcc, False)
    if count_tokens(msgs) > budget_tokens:
        msgs = build(pcc, True)
    return msgs


def _norm_tokens(text):
    return normalize_answer(text).split() if text else []


def _has_seq(hay, needle):
    m = len(needle)
    return any(hay[i : i + m] == needle for i in range(len(hay) - m + 1))


_LIST_NUMBER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def fix_leakage_errors(
    fix: Fix, question: str, answers: Sequence[str], index: LeakageIndex
) -> list:
    arg = next(iter(fix.tool_call["arguments"].values()))
    texts = [fix.sub_goal, arg]
    errors = list(index.check(texts, [], exempt_text=question))
    q_toks = _norm_tokens(question)
    text_toks = [_norm_tokens(_LIST_NUMBER.sub("", t)) for t in texts]
    for ans in answers or []:
        norm = normalize_answer(ans) if ans else ""
        ans_toks = norm.split()
        if not ans_toks or norm in GENERIC_ANSWERS or _has_seq(q_toks, ans_toks):
            continue
        if any(_has_seq(toks, ans_toks) for toks in text_toks):
            errors.append("fix contains a gold answer: '%s'" % norm)
    return errors
