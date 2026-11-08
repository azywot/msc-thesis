"""Reflector input: the chat messages the prompt-editing model reads.

The request holds the current prompts, the history of past edits and evidence trajectories
(including sub-agent inputs and outputs), fitted to a token budget. A returned request never
exceeds the budget; every cut made to fit is recorded in the stats.
"""

import json
from collections import Counter
from dataclasses import dataclass, field

from rsi.bundle import unit_text

MIN_PER_CALL_CHARS = 200
FALLBACK_FIELD_CAPS = (400, 200, 100)

LEAKAGE_INSTRUCTION = (
    "Write general strategies only. Do not copy question text, named entities, numbers or "
    "answers from the trajectories below, and do not build a worked example from any of these "
    "questions."
)

OUTPUT_SCHEMA_TEXT = """Reply with one JSON object and nothing else:
{"unit": "...", "rationale": "...", "expected_effect": "...", "replacement": ...}
- "unit" is one of the units you were asked to improve ("orchestrator", "web_search" or \
"code_generator").
- "rationale" says which recurring problem the edit targets and why it should help.
- "expected_effect" says what should change in the trajectories.
- "replacement" depends on the unit:
  - orchestrator: a partial dict holding only the parts you change. Allowed keys are \
"guidance_opening", "guidance_closing", "example", "turn0_prompt", "turn1n_prompt", \
"tool_descriptions" and "tool_use_note". "example" keeps the template shape shown in the current \
prompts. "tool_call" inside the example may be an object.
  - web_search or code_generator: the full new template string."""

PROTECTED_TEXT = (
    "Protected parts must not change: tool names, parameter names, types and the required "
    "list; the <tools> header and the <sub_goal>/<tool_call> format paragraph; the \\boxed{} "
    "final answer rule; attachment notes; the memory-turn structure. The rendered system prompt "
    "is shown read-only for context."
)


@dataclass
class HistoryEntry:
    label: str
    unit: str
    diff: str
    rationale: str
    screen_score: float = None
    heldout_score: float = None
    accepted: bool = False


@dataclass
class ReflectorContext:
    domain: str
    champion: object
    seed: object
    best: object
    history: list = field(default_factory=list)
    trajectories: list = field(default_factory=list)
    rendered_system_prompt: str = ""
    units: list = field(default_factory=list)
    rejected: list = field(default_factory=list)


def _system_message(domain):
    return (
        f"You improve the prompts of a small team of language-model agents (an orchestrator and "
        f"its sub-agents) for the {domain} domain. Each turn you edit exactly one unit of the "
        f"team's prompts so that the team answers more questions correctly. You see the current "
        f"prompts, past edits with their scores, and example runs, including what the "
        f"sub-agents received and returned.\n\n{LEAKAGE_INSTRUCTION}\n\n{PROTECTED_TEXT}\n\n"
        f"{OUTPUT_SCHEMA_TEXT}"
    )


def _excerpt(text, limit):
    text = "" if text is None else str(text)
    if limit is None or len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return f"{text[:half]}\n[...]\n{text[-half:]}"


def _fmt_score(value):
    return "n/a" if value is None else f"{value:.2f}"


def _render_history(history):
    if not history:
        return "(no earlier edits)"
    lines = []
    for h in history:
        verdict = "accepted" if h.accepted else "rejected"
        lines.append(
            f"- {h.label} [{h.unit}] {verdict}; screen {_fmt_score(h.screen_score)}, "
            f"held-out {_fmt_score(h.heldout_score)}"
        )
        lines.append(f"  Rationale: {h.rationale}")
        lines.append("  Diff:")
        lines.extend(f"    {line}" for line in (h.diff or "").splitlines())
    return "\n".join(lines)


def _render_call(call, pcc, cap):
    tool = call.get("tool")
    inputs = call.get("inputs") or {}
    lines = []
    if tool == "web_search":
        lines.append("    Sub-agent web_search")
        lines.append(f"    Query: {_excerpt(inputs.get('query'), cap)}")
        lines.append(f"    Search results: {_excerpt(inputs.get('formatted_results'), pcc)}")
        lines.append(f"    Response: {_excerpt(call.get('response'), pcc)}")
    elif tool == "code_generator":
        lines.append("    Sub-agent code_generator")
        lines.append(f"    Task: {_excerpt(inputs.get('task'), cap)}")
        lines.append(f"    Response: {_excerpt(call.get('response'), pcc)}")
        lines.append(f"    Output: {_excerpt(call.get('output'), pcc)}")
    else:
        lines.append(f"    Sub-agent {tool}")
        lines.append(f"    Response: {_excerpt(call.get('response'), pcc)}")
    return "\n".join(lines)


def _render_trajectory(t, pcc, cut_actions, cap):
    verdict = "CORRECT" if t.correct else "WRONG"
    lines = [
        f"### Trajectory {t.question_id} ({verdict})",
        f"Question: {_excerpt(t.question, cap)}",
        f"Gold answer: {t.gold}",
        f"Prediction: {_excerpt(t.prediction, cap)}",
        f"Result: {verdict}",
        f"Failure label: {t.failure_label or 'none'}",
        f"Plan: {_excerpt(t.plan, cap)}",
    ]
    by_index = {}
    for call in t.subagent_calls:
        by_index.setdefault(call.get("action_index"), []).append(call)
    indices = list(range(len(t.action_history)))
    omitted = 0
    if cut_actions and len(indices) > 2:
        omitted = len(indices) - 2
        indices = [indices[0], indices[-1]]
    if not indices:
        lines.append("Actions: none")
    for pos, i in enumerate(indices):
        if omitted and pos == 1:
            lines.append(f"[... {omitted} actions omitted ...]")
        a = t.action_history[i]
        lines.append(f"Action {i}:")
        lines.append(f"  Tool: {a.get('tool_name', a.get('tool', 'unknown'))}")
        lines.append(f"  Sub-goal: {_excerpt(a.get('sub_goal', ''), cap)}")
        lines.append(f"  Command: {_excerpt(a.get('command', ''), cap)}")
        calls = by_index.get(i, [])
        if not calls:
            # with a matched sub-agent call the result repeats its response or output
            lines.append(f"  Result: {_excerpt(a.get('result', ''), pcc)}")
        for call in calls:
            lines.append(_render_call(call, pcc, cap))
    return "\n".join(lines), omitted


def _render_prompts(bundle, units):
    parts = []
    for unit in units:
        parts.append(f"[{unit}]\n{unit_text(bundle, unit)}")
    return "\n\n".join(parts)


def _same(a, b):
    return a is not None and b is not None and a.to_dict() == b.to_dict()


def _render_user(ctx, shown, pcc, cut_actions, cap):
    units = ctx.units or ["orchestrator", "web_search", "code_generator"]
    out = ["## Current champion", "Rendered system prompt (read-only):", ctx.rendered_system_prompt]
    out += ["", "Editable units:", _render_prompts(ctx.champion, units)]
    if not _same(ctx.seed, ctx.champion):
        out += ["", "## Seed prompts (v0)", _render_prompts(ctx.seed, units)]
    if ctx.best is not None and not _same(ctx.best, ctx.champion) and not _same(ctx.best, ctx.seed):
        out += ["", "## Best prompts so far", _render_prompts(ctx.best, units)]
    out += ["", "## History of edits", _render_history(ctx.history)]
    if ctx.rejected:
        out += ["", "## Last round's rejected proposals"]
        out += [f"- {line}" for line in ctx.rejected]
    counts = Counter(t.failure_label for t in shown if t.failure_label)
    if counts:
        problems = ", ".join(f"{k} x{v}" for k, v in counts.most_common())
    else:
        problems = "none"
    out += ["", f"Recurring problems (failure labels over the trajectories below): {problems}"]
    out += ["", "## Trajectories"]
    cut_total = 0
    for t in shown:
        text, omitted = _render_trajectory(t, pcc, cut_actions, cap)
        cut_total += omitted
        out += ["", text]
    if not shown:
        out += ["", "(no trajectories fit the input budget)"]
    return "\n".join(out), cut_total


def _drop_one(shown):
    """Drop one trajectory from the larger of the correct and wrong groups."""
    correct = [i for i, t in enumerate(shown) if t.correct]
    wrong = [i for i, t in enumerate(shown) if not t.correct]
    group = correct if len(correct) >= len(wrong) else wrong
    return shown[: group[-1]] + shown[group[-1] + 1 :]


def build_reflector_messages(ctx, count_tokens, budget_tokens, per_call_chars):
    system = _system_message(ctx.domain)
    stats = {
        "trajectories_total": len(ctx.trajectories),
        "trajectories_shown": 0,
        "per_call_chars_used": per_call_chars,
        "actions_cut": 0,
        "input_tokens": 0,
        "text_cap": None,
    }

    def measure(shown, pcc, cut_actions, cap):
        user, cut_total = _render_user(ctx, shown, pcc, cut_actions, cap)
        tokens = max(count_tokens(system) + count_tokens(user), count_tokens(system + "\n" + user))
        return user, cut_total, tokens

    def finish(shown, pcc, cut_actions, cap):
        user, cut_total, tokens = measure(shown, pcc, cut_actions, cap)
        stats.update(
            trajectories_shown=len(shown),
            per_call_chars_used=pcc,
            actions_cut=cut_total,
            input_tokens=tokens,
            text_cap=cap,
        )
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return msgs, stats

    shown = list(ctx.trajectories)
    pcc = per_call_chars
    # 1. drop trajectories (alternately from the larger group) at full per-call detail
    while True:
        if measure(shown, pcc, False, None)[2] <= budget_tokens:
            return finish(shown, pcc, False, None)
        if len(shown) <= 1:
            break
        shown = _drop_one(shown)
    # 2. one trajectory left: halve the per-call excerpts
    while pcc > MIN_PER_CALL_CHARS:
        pcc = max(pcc // 2, MIN_PER_CALL_CHARS)
        if measure(shown, pcc, False, None)[2] <= budget_tokens:
            return finish(shown, pcc, False, None)
    # 3. cut the action list to its first and last actions
    if measure(shown, pcc, True, None)[2] <= budget_tokens:
        return finish(shown, pcc, True, None)
    # 4. cap every remaining free-text field
    for cap in FALLBACK_FIELD_CAPS:
        pcc = min(pcc, cap)
        if measure(shown, pcc, True, cap)[2] <= budget_tokens:
            return finish(shown, pcc, True, cap)
    # 5. last resort: show no trajectory at all
    if measure([], pcc, True, FALLBACK_FIELD_CAPS[-1])[2] <= budget_tokens:
        return finish([], pcc, True, FALLBACK_FIELD_CAPS[-1])
    raise ValueError(f"reflector prompts alone exceed the input budget of {budget_tokens} tokens")
