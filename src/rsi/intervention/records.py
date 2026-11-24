"""Helpers shared by trajectory collection and the tool-call intervention."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from agent_engine.core.state import ExecutionState
from rsi.data import DOMAIN_SOURCES


def trajectory_messages(state) -> List[Dict[str, str]]:
    """Combine initial system+user messages with the full conversation turns.

    state.messages holds [system, user] (AgentFlow rewrites it each turn but
    always keeps these two at indices 0 and 1).
    state.output_messages holds the accumulated assistant + tool turns for
    tool-call and final-answer steps, but NOT the planning turn.

    The planning turn (Turn 0) output is stored separately in
    state.raw_query_analysis (full text including <think> tags).
    We prepend it as an assistant message so the data includes the full
    planning, tool-call and answer trajectory.
    """
    initial = list(state.messages[:2])  # [system, user]
    turns: List[Dict[str, str]] = []

    # Turn 0: planning (only present in AgentFlow mode, not baseline)
    if state.raw_query_analysis:
        turns.append({"role": "assistant", "content": state.raw_query_analysis})

    turns.extend(state.output_messages)
    return initial + turns


def set_sampling_seed(providers: Iterable, seed: int) -> None:
    """Set ``config.seed`` once per distinct provider object (identity, not equality).

    VLLMProvider reads ``config.seed`` per request, so this changes sampling for
    every later request.
    """
    seen = set()
    for provider in providers:
        if provider is None or id(provider) in seen:
            continue
        seen.add(id(provider))
        provider.config.seed = seed


def load_collected(paths: Sequence[Path]) -> List[dict]:
    """Read JSONL record files, deduplicated by (question_id, pass); first wins."""
    seen = set()
    records: List[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec.get("question_id"), rec.get("pass"))
                if key in seen:
                    continue
                seen.add(key)
                records.append(rec)
    return records


def domain_of(data_source: str) -> str:
    for domain, sources in DOMAIN_SOURCES.items():
        if data_source in sources:
            return domain
    raise ValueError("unknown data_source %r" % (data_source,))


def intervention_candidates(records: List[dict], n_passes: int):
    """Pass-1 records of questions with exactly n_passes records, none correct.

    Error records and planning-answered records (turns == 0) are skipped and
    counted per domain.
    """
    by_qid: Dict[object, List[dict]] = {}
    for rec in records:
        by_qid.setdefault(rec.get("question_id"), []).append(rec)

    stats = {
        d: {"zero_of_n": 0, "skipped_error": 0, "skipped_planning_answer": 0, "candidates": 0}
        for d in DOMAIN_SOURCES
    }
    candidates: List[dict] = []
    for recs in by_qid.values():
        if len(recs) != n_passes or any(r.get("correct") for r in recs):
            continue
        first = next((r for r in recs if r.get("pass") == 1), None)
        if first is None:
            continue
        domain = domain_of(first["data_source"])
        stats[domain]["zero_of_n"] += 1
        if first.get("error") or not first.get("messages"):
            stats[domain]["skipped_error"] += 1
        elif first.get("turns", 0) == 0:
            stats[domain]["skipped_planning_answer"] += 1
        else:
            stats[domain]["candidates"] += 1
            candidates.append(first)
    return candidates, stats


def state_from_record(record: dict, step: int) -> ExecutionState:
    """Rebuild the ExecutionState after the first ``step`` actions of a record."""
    history = record.get("action_history") or []
    if not 0 <= step <= len(history):
        raise ValueError("step %d outside [0, %d]" % (step, len(history)))
    messages = record["messages"]
    raw_plan = record.get("raw_query_analysis")
    start = 3 if raw_plan else 2  # [system, user, (plan)]
    kept = history[:step]
    tool_calls = []
    for action in kept:
        try:
            call = json.loads(action["command"])
        except (TypeError, ValueError, KeyError):
            call = {"name": action.get("tool_name"), "arguments": {}}
        tool_calls.append(call)
    state = ExecutionState(
        question_id=record["question_id"],
        question=record["question"],
        messages=[dict(m) for m in messages[:2]],
        query_analysis=record.get("query_analysis", ""),
        raw_query_analysis=raw_plan,
        action_history=[dict(a) for a in kept],
        output_messages=[dict(m) for m in messages[start : start + 2 * step]],
        tool_calls=tool_calls,
        turn=step,
        metadata={
            "subagent_calls": [
                c for c in (record.get("subagent_calls") or []) if c["action_index"] < step
            ],
            "intervention_step": step,
        },
    )
    for name, n in Counter(c.get("name") for c in tool_calls).items():
        for _ in range(n):
            state.increment_tool_count(name)
    return state
