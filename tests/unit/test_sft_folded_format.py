"""Tests for the memory-folded SFT row format (`build_sft_parquet.py --format folded`).

The orchestrator never sees a native multi-turn conversation at inference: every
non-baseline turn is rebuilt as a fresh ``[system, user]`` memory prompt by
``AgenticOrchestrator._build_memory_prompt``. SFT rows must therefore be folded into
single-decision rows so that what the adapter is trained on equals what it is sampled
from.

Covers:
  - row expansion and shape (one row per orchestrator decision)
  - history accumulation across action rows
  - the planning turn and its two edge cases
  - byte-identity against the orchestrator's own formatter
  - loss-bearing invariants (no thinking, no tool output, tool call retained)
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS

import build_sft_parquet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _call(name: str, args: dict) -> str:
    import json
    return '{"name": "%s", "arguments": %s}' % (name, json.dumps(args))


def _action(sub_goal: str, name: str, args: dict) -> str:
    return (
        f"<sub_goal>{sub_goal}</sub_goal>\n"
        f"<tool_call>\n{_call(name, args)}\n</tool_call>"
    )


def _two_tool_trajectory() -> list:
    """[system, user, plan, action, tool, action, tool, answer] — the common case."""
    return [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "THE PLAN"},
        {"role": "assistant", "content": _action("SG1", "web_search", {"query": "q1"})},
        {"role": "tool", "tool_name": "web_search", "content": "RESULT ONE"},
        {"role": "assistant", "content": _action("SG2", "code_generator", {"task": "t2"})},
        {"role": "tool", "tool_name": "code_generator", "content": "RESULT TWO"},
        {"role": "assistant", "content": "FINAL \\boxed{42}"},
    ]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_two_tool_trajectory_folds_to_one_row_per_decision():
    """plan + 2 actions + answer = 4 rows, each a 3-message single-turn row."""
    rows = build_sft_parquet._fold_trajectory(
        _two_tool_trajectory(), planning_suffix=_DEFAULT_PLANNING_SUFFIX_TOOLS
    )

    assert len(rows) == 4
    for row in rows:
        assert [m["role"] for m in row] == ["system", "user", "assistant"]


# ---------------------------------------------------------------------------
# Train/inference identity — the reason this whole change exists
# ---------------------------------------------------------------------------

_QWEN3 = "Qwen/Qwen3-8B"


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(_QWEN3)
    except Exception as exc:  # not in the local HF cache
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")


def _folded_parquet(tmp_path, rows):
    pd = pytest.importorskip("pandas")
    path = tmp_path / "folded.parquet"
    pd.DataFrame([{"messages": r} for r in rows]).to_parquet(path, index=False)
    return path


def _dataset(parquet_path, tokenizer, **overrides):
    from verl_ext.folded_sft_dataset import FoldedSFTDataset

    config = {"messages_key": "messages", "max_length": 16384, "truncation": "right"}
    config.update(overrides)
    return FoldedSFTDataset(
        parquet_files=[str(parquet_path)], tokenizer=tokenizer, config=config
    )


def test_training_prompt_is_token_identical_to_the_inference_prompt(tmp_path):
    """The span the model is conditioned on must equal what the orchestrator sends.

    This is the defect the folded format exists to fix: MultiTurnSFTDataset tokenises
    each turn in isolation, so targets begin right after ``<|im_start|>assistant\\n``,
    while inference samples after ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.
    """
    tok = _tokenizer()
    rows = build_sft_parquet._fold_trajectory(
        _two_tool_trajectory(), planning_suffix=_DEFAULT_PLANNING_SUFFIX_TOOLS
    )
    ds = _dataset(_folded_parquet(tmp_path, rows), tok)

    for k, row in enumerate(rows):
        item = ds[k]
        length = int(item["attention_mask"].sum())
        input_ids = item["input_ids"][:length].tolist()
        loss_mask = item["loss_mask"][:length].tolist()
        prompt_ids = input_ids[: loss_mask.index(1)]

        # Exactly what VLLMProvider._render_messages builds for Qwen3 (vllm_provider.py:338-343)
        expected = tok.apply_chat_template(
            row[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        assert prompt_ids == expected, f"row {k} prompt diverges from the inference prompt"


def test_prompt_identity_holds_in_no_padding_mode(tmp_path):
    """`pad_mode: no_padding` is VERL's SFT default, so it is the mode that actually runs.

    It returns unpadded tensors and no attention_mask, matching MultiTurnSFTDataset.
    """
    tok = _tokenizer()
    rows = build_sft_parquet._fold_trajectory(
        _two_tool_trajectory(), planning_suffix=_DEFAULT_PLANNING_SUFFIX_TOOLS
    )
    ds = _dataset(_folded_parquet(tmp_path, rows), tok, pad_mode="no_padding")

    for k, row in enumerate(rows):
        item = ds[k]
        assert "attention_mask" not in item
        assert set(item) == {"input_ids", "position_ids", "loss_mask"}

        input_ids = item["input_ids"].tolist()
        loss_mask = item["loss_mask"].tolist()
        assert len(input_ids) == len(loss_mask) == len(item["position_ids"])

        prompt_ids = input_ids[: loss_mask.index(1)]
        expected = tok.apply_chat_template(
            row[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        assert prompt_ids == expected
        supervised = [t for t, m in zip(input_ids, loss_mask) if m == 1]
        assert tok.decode(supervised) == row[2]["content"] + "<|im_end|>"


def test_rows_that_are_not_single_turn_are_rejected(tmp_path):
    """A native parquet fed to this class must fail loudly, not train on garbage."""
    tok = _tokenizer()
    native = [_two_tool_trajectory()]

    with pytest.raises(ValueError, match="expects \\[system, user, assistant\\] rows"):
        _dataset(_folded_parquet(tmp_path, native), tok)


def test_supervised_span_is_exactly_the_target_plus_eos(tmp_path):
    tok = _tokenizer()
    rows = build_sft_parquet._fold_trajectory(
        _two_tool_trajectory(), planning_suffix=_DEFAULT_PLANNING_SUFFIX_TOOLS
    )
    ds = _dataset(_folded_parquet(tmp_path, rows), tok)

    for k, row in enumerate(rows):
        item = ds[k]
        length = int(item["attention_mask"].sum())
        input_ids = item["input_ids"][:length].tolist()
        loss_mask = item["loss_mask"][:length].tolist()
        supervised = [t for t, m in zip(input_ids, loss_mask) if m == 1]

        assert tok.decode(supervised) == row[2]["content"] + "<|im_end|>"


# ---------------------------------------------------------------------------
# History accumulation and the planning turn
# ---------------------------------------------------------------------------

def _fold(messages, **kwargs):
    kwargs.setdefault("planning_suffix", _DEFAULT_PLANNING_SUFFIX_TOOLS)
    return build_sft_parquet._fold_trajectory(messages, **kwargs)


def test_action_row_k_sees_steps_before_it_and_not_its_own():
    rows = _fold(_two_tool_trajectory())
    first_action, second_action = rows[1][1]["content"], rows[2][1]["content"]

    assert "**Previous Steps:**" not in first_action
    assert "RESULT ONE" in second_action
    assert "RESULT TWO" not in second_action


def test_answer_row_sees_every_step():
    answer_user = _fold(_two_tool_trajectory())[3][1]["content"]

    assert "RESULT ONE" in answer_user and "RESULT TWO" in answer_user


def test_planning_row_appends_the_suffix_and_targets_the_plan():
    plan_row = _fold(_two_tool_trajectory())[0]

    assert plan_row[1]["content"].endswith(_DEFAULT_PLANNING_SUFFIX_TOOLS)
    assert "**Query Analysis:**" not in plan_row[1]["content"]
    assert plan_row[2]["content"] == "THE PLAN"


def test_every_non_planning_row_carries_the_query_analysis():
    rows = _fold(_two_tool_trajectory())

    for row in rows[1:]:
        assert "**Query Analysis:**\nTHE PLAN" in row[1]["content"]


def test_folded_user_turn_matches_the_orchestrators_own_formatter():
    """Guard against the fold and `_build_memory_prompt` drifting apart."""
    from agent_engine.core.orchestrator import AgenticOrchestrator

    history = [{
        "tool_name": "web_search",
        "sub_goal": "SG1",
        "command": '{"name": "web_search", "arguments": {"query": "q1"}}',
        "result": "RESULT ONE",
    }]
    expected = "\n".join([
        "QUESTION",
        "\n**Query Analysis:**\nTHE PLAN",
        f"\n**Previous Steps:**\n{AgenticOrchestrator._format_action_history(history)}",
    ])

    assert _fold(_two_tool_trajectory())[2][1]["content"] == expected


# ---------------------------------------------------------------------------
# Walker edge cases (109 and 175 real trajectories respectively)
# ---------------------------------------------------------------------------

def test_plan_that_emitted_a_tool_call_keeps_the_plan_and_truncates_the_analysis():
    """109/968 real trajectories. The stored plan holds the full generation, but the
    value inference folds is only the text before the call (orchestrator.py:757-776)."""
    messages = _two_tool_trajectory()
    messages[2]["content"] = "THINKING OUT LOUD\n" + _action("SG0", "web_search", {"query": "q0"})

    rows = _fold(messages)

    assert len(rows) == 4, "the plan must not be silently dropped"
    assert rows[0][2]["content"] == messages[2]["content"]
    for row in rows[1:]:
        assert "**Query Analysis:**\nTHINKING OUT LOUD" in row[1]["content"]
        assert "<tool_call>" not in row[1]["content"]


def test_plan_that_emitted_the_final_answer_yields_one_row_and_no_answer_row():
    """175/968 real trajectories: finished=True at turn 0 (orchestrator.py:777-783)."""
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "DIRECT \\boxed{7}"},
    ]

    rows = _fold(messages)

    assert len(rows) == 1
    assert rows[0][2]["content"] == "DIRECT \\boxed{7}"
    assert rows[0][1]["content"].endswith(_DEFAULT_PLANNING_SUFFIX_TOOLS)


def test_drop_planning_answers_removes_turn_zero_answers_entirely():
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "DIRECT \\boxed{7}"},
    ]

    assert _fold(messages, drop_planning_answers=True) == []
    # a normal trajectory is untouched by the flag
    assert len(_fold(_two_tool_trajectory(), drop_planning_answers=True)) == 4


def test_tool_free_trajectory_yields_planning_and_answer_rows_only():
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "QUESTION"},
        {"role": "assistant", "content": "THE PLAN"},
        {"role": "assistant", "content": "FINAL \\boxed{42}"},
    ]

    rows = _fold(messages)

    assert len(rows) == 2
    assert rows[1][2]["content"] == "FINAL \\boxed{42}"
    assert "**Previous Steps:**" not in rows[1][1]["content"]


def test_row_count_equals_plan_plus_actions_plus_answer():
    """No silent drops: a misclassified turn must fail loudly, not vanish."""
    messages = _two_tool_trajectory()
    n_actions = sum(
        1 for i, m in enumerate(messages)
        if m["role"] == "assistant" and i + 1 < len(messages) and messages[i + 1]["role"] == "tool"
    )

    assert len(_fold(messages)) == 1 + n_actions + 1


# ---------------------------------------------------------------------------
# Loss-bearing invariants
# ---------------------------------------------------------------------------

def test_tool_results_never_appear_in_a_target():
    for row in _fold(_two_tool_trajectory()):
        assert "RESULT ONE" not in row[2]["content"]
        assert "RESULT TWO" not in row[2]["content"]


def test_action_targets_keep_their_tool_call():
    """The converse guard: masking more must not silently kill the training signal."""
    rows = _fold(_two_tool_trajectory())

    for row in (rows[1], rows[2]):
        assert "<tool_call>" in row[2]["content"]


def test_no_thinking_survives_into_a_folded_row():
    messages = _two_tool_trajectory()
    messages[2]["content"] = "THE PLAN"

    for row in _fold(messages):
        blob = "".join(m["content"] for m in row)
        assert "<think>" not in blob and "</think>" not in blob


# ---------------------------------------------------------------------------
# CLI: --from-parquet refolds an already-built native parquet in place
# ---------------------------------------------------------------------------
#
# This is the path the retrain uses. Rebuilding from collected_*.jsonl instead would
# need data/training/train/combined_train.parquet for the math:search ratio, which is
# not readable; refolding the shipped parquet inherits every control already applied
# (strip-thinking, strip-suffix, one-per-question, balance, split) and keeps the
# trajectory set bit-identical to the native run it is compared against.


def _native_parquet(tmp_path, name="sft_train.parquet"):
    pd = pytest.importorskip("pandas")
    rows = [
        {
            "data_source": "deepmath",
            "question": "QUESTION",
            "result": "42",
            "extra_info": {"idx": 7},
            "messages": _two_tool_trajectory(),
        },
        {
            "data_source": "hotpotqa",
            "question": "QUESTION",
            "result": "7",
            "extra_info": {"idx": 9},
            "messages": [
                {"role": "system", "content": "SYSTEM PROMPT"},
                {"role": "user", "content": "QUESTION"},
                {"role": "assistant", "content": "DIRECT \\boxed{7}"},
            ],
        },
    ]
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path, index=False)
    # the sibling val split the refold must pick up automatically
    pd.DataFrame(rows[:1]).to_parquet(tmp_path / "sft_val.parquet", index=False)
    return path


def _run_cli(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_sft_parquet.py"] + argv)
    build_sft_parquet.main()


def test_from_parquet_folds_both_splits_and_preserves_metadata(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    native = _native_parquet(tmp_path)

    _run_cli([
        "--from-parquet", str(native),
        "--output-dir", str(tmp_path),
        "--output-name", "sft_folded_train.parquet",
    ], monkeypatch)

    train = pd.read_parquet(tmp_path / "sft_folded_train.parquet")
    val = pd.read_parquet(tmp_path / "sft_folded_val.parquet")

    # 4 rows from the 2-tool trajectory + 1 from the turn-0 answer
    assert len(train) == 5
    assert len(val) == 4
    for msgs in train["messages"]:
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert set(train.columns) == {"data_source", "question", "result", "extra_info", "messages"}
    assert train["extra_info"].notna().all()
    assert sorted(train["data_source"].unique()) == ["deepmath", "hotpotqa"]


def test_from_parquet_does_not_reselect_or_rebalance(tmp_path, monkeypatch):
    """The input is already the final dataset: every question must survive verbatim."""
    pd = pytest.importorskip("pandas")
    native = _native_parquet(tmp_path)
    before = pd.read_parquet(native)

    _run_cli([
        "--from-parquet", str(native),
        "--output-dir", str(tmp_path),
        "--output-name", "sft_folded_train.parquet",
    ], monkeypatch)

    after = pd.read_parquet(tmp_path / "sft_folded_train.parquet")
    assert set(before["data_source"]) == set(after["data_source"])
    assert len(after) > len(before)


def test_drop_planning_answers_flag_reaches_the_fold(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    native = _native_parquet(tmp_path)

    _run_cli([
        "--from-parquet", str(native),
        "--output-dir", str(tmp_path),
        "--output-name", "sft_folded_train.parquet",
        "--drop-planning-answers",
    ], monkeypatch)

    train = pd.read_parquet(tmp_path / "sft_folded_train.parquet")
    assert len(train) == 4  # the turn-0 answer trajectory is gone
    assert "hotpotqa" not in set(train["data_source"])


def _collected_jsonl(tmp_path):
    import json as _json

    path = tmp_path / "collected_test.jsonl"
    records = [
        {
            "question_id": 0, "pass": 1, "question": "QUESTION", "data_source": "deepmath",
            "ground_truth": "42", "prediction": "42", "correct": True,
            "messages": _two_tool_trajectory(),
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(_json.dumps(r) + "\n")
    return path


def test_format_folded_applies_to_the_jsonl_build_path(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")

    _run_cli([
        str(_collected_jsonl(tmp_path)),
        "--output-dir", str(tmp_path),
        "--output-name", "sft_folded_train.parquet",
        "--format", "folded",
        "--reference-parquet", "",
        "--collection-config", "",
        "--val-fraction", "0",
    ], monkeypatch)

    train = pd.read_parquet(tmp_path / "sft_folded_train.parquet")
    assert len(train) == 4
    for msgs in train["messages"]:
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_format_native_is_still_the_default(tmp_path, monkeypatch):
    """Existing behaviour must be unchanged when the flag is not passed."""
    pd = pytest.importorskip("pandas")

    _run_cli([
        str(_collected_jsonl(tmp_path)),
        "--output-dir", str(tmp_path),
        "--output-name", "sft_native_train.parquet",
        "--reference-parquet", "",
        "--collection-config", "",
        "--val-fraction", "0",
    ], monkeypatch)

    train = pd.read_parquet(tmp_path / "sft_native_train.parquet")
    assert len(train) == 1
    assert [m["role"] for m in train["messages"][0]] == [
        "system", "user", "assistant", "assistant", "tool", "assistant", "tool", "assistant"
    ]
