import pytest

from rsi.bundle import PromptBundle
from rsi.executor import TrajectoryRecord
from rsi.reflector_input import (
    OUTPUT_SCHEMA_TEXT,
    HistoryEntry,
    ReflectorContext,
    build_reflector_messages,
)


def char_count(s):
    return len(s) // 4


def make_bundle(tag="v"):
    return PromptBundle(
        domain="search",
        orchestrator={"guidance_opening": f"open {tag}", "tool_use_note": "note"},
        subagents={"web_search": f"WS template {tag}", "code_generator": f"CG template {tag}"},
    )


def make_traj(i, correct=False, label="direct_reasoning_no_action", n_actions=2, big=0):
    actions = [
        {
            "tool_name": "web_search",
            "sub_goal": f"goal {i}-{k}",
            "command": '{"name": "web_search"}',
            "result": "R" * (50 + big),
        }
        for k in range(n_actions)
    ]
    calls = [
        {
            "tool": "web_search",
            "inputs": {
                "query": "grand canyon different view editor",
                "formatted_results": "H" * (1000 + big) + "T" * 1000,
            },
            "response": "resp",
            "output": "out",
            "action_index": 0,
        },
        {
            "tool": "code_generator",
            "inputs": {"task": "compute thing", "context": "ctx"},
            "response": "print(1)",
            "output": "1",
            "action_index": 1,
        },
    ]
    return TrajectoryRecord(
        question_id=i,
        question=f"Question number {i}?",
        gold="Tom Vail",
        prediction="Someone",
        correct=correct,
        failure_label=None if correct else label,
        plan="plan " * (5 + big // 10),
        action_history=actions,
        subagent_calls=calls,
    )


def make_ctx(trajs):
    history = [
        HistoryEntry(
            label="r1c0",
            unit="orchestrator",
            diff="-  - Find links\n+  - Find each link",
            rationale="be specific",
            screen_score=0.5,
            heldout_score=0.42,
            accepted=True,
        )
    ]
    return ReflectorContext(
        domain="search",
        champion=make_bundle("champ"),
        seed=make_bundle("seed"),
        best=None,
        history=history,
        trajectories=trajs,
        rendered_system_prompt="SYSTEM PROMPT TEXT",
        units=["orchestrator", "web_search", "code_generator"],
    )


@pytest.fixture
def ctx():
    return make_ctx([make_traj(0), make_traj(1, correct=True)])


@pytest.fixture
def ctx_many():
    return make_ctx([make_traj(i, correct=i % 2 == 0) for i in range(20)])


@pytest.fixture
def ctx_one_huge():
    return make_ctx([make_traj(0, n_actions=60, big=20_000)])


def test_contains_gold_failure_label_and_subagent_inputs(ctx):
    msgs, _ = build_reflector_messages(ctx, char_count, budget_tokens=10_000, per_call_chars=400)
    user = msgs[1]["content"]
    assert "Gold answer: Tom Vail" in user and "direct_reasoning_no_action" in user
    assert "Query: grand canyon different view editor" in user and "[...]" in user
    assert "CORRECT" in user and "WRONG" in user
    assert "Task: compute thing" in user


def test_leakage_instruction_present(ctx):
    msgs, _ = build_reflector_messages(ctx, char_count, 10_000, 400)
    assert "Do not copy question text, named entities, numbers or answers" in msgs[0]["content"]
    assert OUTPUT_SCHEMA_TEXT in msgs[0]["content"]


def test_drops_trajectories_before_cutting_harder(ctx_many):
    _, stats = build_reflector_messages(
        ctx_many, char_count, budget_tokens=3_000, per_call_chars=400
    )
    assert stats["trajectories_shown"] < stats["trajectories_total"]
    assert stats["per_call_chars_used"] == 400


def test_single_oversized_trajectory_is_cut_to_budget(ctx_one_huge):
    msgs, stats = build_reflector_messages(
        ctx_one_huge, char_count, budget_tokens=2_000, per_call_chars=4_000
    )
    assert char_count(msgs[0]["content"] + msgs[1]["content"]) <= 2_000
    assert stats["trajectories_shown"] == 1 and (
        stats["per_call_chars_used"] < 4_000 or stats["actions_cut"]
    )
    assert stats["actions_cut"] > 0
    assert stats["input_tokens"] <= 2_000


def test_history_shows_diff_and_scores(ctx):
    user = build_reflector_messages(ctx, char_count, 10_000, 400)[0][1]["content"]
    assert "held-out 0.42" in user and "+  - Find each link" in user


def test_alternating_drop_keeps_both_groups(ctx_many):
    msgs, stats = build_reflector_messages(ctx_many, char_count, 3_000, 400)
    user = msgs[1]["content"]
    assert "CORRECT" in user and "WRONG" in user
    assert "Recurring problems" in user


def test_seed_and_best_only_when_different():
    c = make_ctx([make_traj(0)])
    c.best = make_bundle("best")
    user = build_reflector_messages(c, char_count, 10_000, 400)[0][1]["content"]
    assert "WS template seed" in user and "WS template best" in user
    c.best = make_bundle("champ")
    user = build_reflector_messages(c, char_count, 10_000, 400)[0][1]["content"]
    assert "WS template best" not in user


def test_never_over_budget_even_if_tiny():
    c = make_ctx([make_traj(0, big=50_000, n_actions=200)])
    msgs, stats = build_reflector_messages(c, char_count, 1_500, 4_000)
    assert stats["input_tokens"] <= 1_500
    assert char_count(msgs[0]["content"] + msgs[1]["content"]) <= 1_500
