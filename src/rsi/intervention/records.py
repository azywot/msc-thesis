"""Helpers shared by trajectory collection and the tool-call intervention."""

from typing import Dict, Iterable, List


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
