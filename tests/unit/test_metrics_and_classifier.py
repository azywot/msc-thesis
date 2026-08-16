"""``classify_failure`` and ``compute_metrics`` over a hermetic synthetic corpus.

**Why this exists alongside the replay fixtures.** B4 and B5
(`tests/characterization/test_metrics_replay.py`,
`test_failure_modes_replay.py`) replay these same two functions over real
recorded runs under `experiments/results/`. That directory is gitignored,
multi-gigabyte, and contains ground-truth answers for gated datasets, so it
cannot be committed — and both tests therefore **skip on a fresh clone**. They
protect the thesis numbers on the machine that produced them and offer a new
researcher nothing.

Every row below is hand-built and synthetic: no real questions, no real
answers, nothing derived from a gated dataset. These run everywhere.

The two are complementary and both are worth keeping. The replay fixtures
guard *the numbers* — if aggregation drifts, recorded runs re-score
differently. These guard *the logic* — they state the taxonomy's rules
explicitly, so a rule that is quietly weakened fails here even though no
recorded run happened to exercise it.

``classify_failure`` is **frozen**; these tests are written against its
documented rules, not against whatever it happens to return.
"""

import pytest

from agent_engine.analysis.failure_modes import (
    MAX_TURNS,
    MIN_LOOP_REPEATS,
    classify_failure,
)
from agent_engine.runner.metrics import compute_metrics


def test_the_thresholds_themselves_are_pinned():
    """The rest of this file is written in terms of ``MAX_TURNS`` and
    ``MIN_LOOP_REPEATS``, which keeps it readable but means every case moves
    with the constant: changing ``MIN_LOOP_REPEATS`` to 4 leaves all of them
    green while silently re-labelling three-call traces across every published
    breakdown.

    So the literals are pinned here, once. If you are changing one on purpose,
    this is the test to update — and the thesis figures need regenerating.
    """
    assert MAX_TURNS == 15
    assert MIN_LOOP_REPEATS == 3


def _step(tool_name, sub_goal="", result="ok"):
    return {"tool_name": tool_name, "sub_goal": sub_goal, "command": "{}", "result": result}


def _record(action_history=None, prediction="an answer", turns=1, question="a question"):
    return {
        "question": question,
        "prediction": prediction,
        "turns": turns,
        "action_history": action_history or [],
    }


# --- priority 1: modality_tool_gap ----------------------------------------


@pytest.mark.parametrize("tool", ["image_inspector", "video_analysis"])
def test_calling_a_visual_only_tool_is_a_modality_gap(tool):
    """Signal A. Either tool signals a modality gap whatever the result was —
    `video_analysis` was never wired in, `image_inspector` is image-specific."""
    assert classify_failure(_record([_step(tool)])) == "modality_tool_gap"


def test_repeated_empty_text_inspector_calls_about_an_image_are_a_modality_gap():
    """Signal B needs all three: >= 2 text_inspector calls, >= 1 empty result,
    and a visual keyword in a sub-goal."""
    record = _record(
        [
            _step("text_inspector", sub_goal="read the image caption", result=""),
            _step("text_inspector", sub_goal="read the rest", result="some text"),
        ]
    )
    assert classify_failure(record) == "modality_tool_gap"


def test_two_text_inspector_calls_without_an_empty_result_are_not_a_modality_gap():
    record = _record(
        [
            _step("text_inspector", sub_goal="read the image caption", result="text"),
            _step("text_inspector", sub_goal="read the rest", result="more text"),
        ]
    )
    assert classify_failure(record) != "modality_tool_gap"


def test_two_empty_text_inspector_calls_without_a_visual_keyword_are_not_a_modality_gap():
    record = _record(
        [
            _step("text_inspector", sub_goal="read the document", result=""),
            _step("text_inspector", sub_goal="read the appendix", result=""),
        ]
    )
    assert classify_failure(record) != "modality_tool_gap"


def test_a_visual_question_answered_with_no_tools_is_a_modality_gap():
    """Signal C: the upstream cause is the missing modality, not a reasoning
    choice."""
    record = _record([], prediction="a guess", question="What colour is the car in the photo?")
    assert classify_failure(record) == "modality_tool_gap"


def test_a_visual_question_with_an_empty_prediction_is_not_a_modality_gap():
    """Signal C requires a non-empty prediction; empty-prediction cases belong
    to priority 2 regardless of visual content. This boundary is stated
    explicitly in the classifier and is easy to erase by reordering."""
    record = _record([], prediction="   ", question="What colour is the car in the photo?")
    assert classify_failure(record) == "tool_loop_or_empty_final"


# --- priority 2: tool_loop_or_empty_final ---------------------------------


@pytest.mark.parametrize("prediction", ["", "   ", "\n"])
def test_an_empty_prediction_is_a_loop_or_empty_final(prediction):
    assert classify_failure(_record([_step("web_search")], prediction=prediction)) == (
        "tool_loop_or_empty_final"
    )


def test_exhausting_the_turn_budget_is_a_loop_or_empty_final():
    assert classify_failure(_record([_step("web_search")], turns=MAX_TURNS)) == (
        "tool_loop_or_empty_final"
    )


def test_one_turn_below_the_budget_is_not():
    """Pins the boundary as `>=`, not `>`."""
    assert classify_failure(_record([_step("web_search")], turns=MAX_TURNS - 1)) != (
        "tool_loop_or_empty_final"
    )


def test_repeating_one_tool_enough_times_is_a_loop():
    history = [_step("web_search")] * MIN_LOOP_REPEATS
    assert classify_failure(_record(history)) == "tool_loop_or_empty_final"


def test_one_repeat_below_the_threshold_is_not_a_loop():
    history = [_step("web_search")] * (MIN_LOOP_REPEATS - 1)
    assert classify_failure(_record(history)) != "tool_loop_or_empty_final"


def test_the_loop_threshold_applies_per_tool_not_in_total():
    """Two different tools twice each is four calls but no loop."""
    history = [
        _step("web_search"),
        _step("web_search"),
        _step("code_generator"),
        _step("code_generator"),
    ]
    assert classify_failure(_record(history)) != "tool_loop_or_empty_final"


# --- priority 3: direct_reasoning_no_action -------------------------------


def test_answering_with_no_tools_at_all_is_direct_reasoning():
    """The dominant failure mode in the thesis analysis, and the one the RL
    work targets."""
    assert classify_failure(_record([], prediction="42")) == "direct_reasoning_no_action"


# --- priority 4: computational_subgoal_error ------------------------------


def test_two_code_generator_calls_are_a_computational_subgoal_error():
    history = [_step("code_generator"), _step("code_generator")]
    assert classify_failure(_record(history)) == "computational_subgoal_error"


def test_a_single_code_generator_call_is_single_shot_trust_instead():
    """Documented distinction: one call blindly trusted is not the same failure
    as multiple wrong computational sub-goals."""
    assert classify_failure(_record([_step("code_generator")])) == "single_shot_tool_trust"


# --- priority 5: retrieval_evidence_failure -------------------------------


def test_two_web_searches_are_a_retrieval_failure():
    history = [_step("web_search"), _step("web_search")]
    assert classify_failure(_record(history)) == "retrieval_evidence_failure"


def test_a_single_web_search_is_single_shot_trust_instead():
    assert classify_failure(_record([_step("web_search")])) == "single_shot_tool_trust"


# --- priority 6: catch-all ------------------------------------------------


def test_an_unremarkable_single_tool_trace_is_single_shot_trust():
    assert classify_failure(_record([_step("text_inspector")])) == "single_shot_tool_trust"


# --- first-match-wins ordering --------------------------------------------
#
# The ordering is the part a careless edit breaks, and no individual rule test
# would notice. Each case below matches *two* rules; the higher-priority label
# must win.


@pytest.mark.parametrize(
    "record, expected, also_matches",
    [
        (
            _record([_step("image_inspector")], prediction=""),
            "modality_tool_gap",
            "empty prediction (P2)",
        ),
        (
            _record([_step("image_inspector")] * MIN_LOOP_REPEATS),
            "modality_tool_gap",
            "tool loop (P2)",
        ),
        (
            _record([_step("code_generator")] * MIN_LOOP_REPEATS),
            "tool_loop_or_empty_final",
            "2+ code_generator (P4)",
        ),
        (
            _record([_step("web_search")] * MIN_LOOP_REPEATS),
            "tool_loop_or_empty_final",
            "2+ web_search (P5)",
        ),
        (
            _record(
                [
                    _step("web_search"),
                    _step("web_search"),
                    _step("code_generator"),
                    _step("code_generator"),
                ]
            ),
            "computational_subgoal_error",
            "2+ web_search (P5)",
        ),
        (_record([], prediction=""), "tool_loop_or_empty_final", "no tools (P3)"),
    ],
    ids=[
        "visual-beats-empty",
        "visual-beats-loop",
        "loop-beats-code",
        "loop-beats-retrieval",
        "code-beats-retrieval",
        "empty-beats-no-action",
    ],
)
def test_the_higher_priority_rule_wins(record, expected, also_matches):
    assert classify_failure(record) == expected, f"should outrank {also_matches}"


def test_every_mode_is_reachable():
    """A rule made unreachable by an ordering change is otherwise invisible:
    its own test still passes while the mode never appears in any breakdown."""
    reached = {
        classify_failure(_record([_step("image_inspector")])),
        classify_failure(_record([], prediction="")),
        classify_failure(_record([], prediction="42")),
        classify_failure(_record([_step("code_generator")] * 2)),
        classify_failure(_record([_step("web_search")] * 2)),
        classify_failure(_record([_step("text_inspector")])),
    }
    assert reached == {
        "modality_tool_gap",
        "tool_loop_or_empty_final",
        "direct_reasoning_no_action",
        "computational_subgoal_error",
        "retrieval_evidence_failure",
        "single_shot_tool_trust",
    }


# --- compute_metrics ------------------------------------------------------


class _Example:
    """Minimal stand-in for DatasetExample — `compute_metrics` reads `.metadata`."""

    def __init__(self, metadata=None):
        self.metadata = metadata or {}


def _result(accuracy, tools=None, tokens=0):
    return {
        "evaluation": {"accuracy": accuracy, "em": accuracy, "f1": accuracy},
        "tool_counts": tools or {},
        "token_usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
    }


def test_an_unstratified_dataset_has_no_per_level_block():
    examples = [_Example({"level": "1"}), _Example({"level": "2"})]
    metrics = compute_metrics([_result(1.0), _result(0.0)], examples, "musique")

    assert metrics["overall"]["accuracy"] == 0.5
    assert "per_level" not in metrics


def test_a_stratified_dataset_reports_per_level_accuracy():
    examples = [_Example({"level": "1"}), _Example({"level": "1"}), _Example({"level": "2"})]
    results = [_result(1.0), _result(0.0), _result(1.0)]

    metrics = compute_metrics(results, examples, "gaia")

    assert metrics["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["overall"]["num_correct"] == "2 of 3"
    assert metrics["per_level"]["1"]["accuracy"] == 0.5
    assert metrics["per_level"]["1"]["num_correct"] == "1 of 2"
    assert metrics["per_level"]["2"]["accuracy"] == 1.0


def test_token_usage_and_tool_counts_sum_overall_and_per_level():
    examples = [_Example({"level": "1"}), _Example({"level": "2"})]
    results = [
        _result(1.0, tools={"web_search": 2}, tokens=100),
        _result(0.0, tools={"web_search": 1, "code_generator": 3}, tokens=50),
    ]

    metrics = compute_metrics(results, examples, "gaia")

    assert metrics["tool_usage"] == {"web_search": 3, "code_generator": 3}
    assert metrics["overall"]["token_usage"]["total_tokens"] == 150
    assert metrics["per_level"]["1"]["token_usage"]["total_tokens"] == 100
    assert metrics["per_level"]["2"]["tool_usage"] == {"web_search": 1, "code_generator": 3}


def test_a_missing_level_field_buckets_as_unknown():
    metrics = compute_metrics([_result(1.0)], [_Example({})], "gaia")
    assert list(metrics["per_level"]) == ["unknown"]


def test_a_present_but_null_level_buckets_as_the_string_none():
    """`dict.get(key, default)` semantics: a key present with value ``None``
    yields ``"None"``, it does not fall through to the default. Preserved
    deliberately when this moved into `DatasetSpec`."""
    metrics = compute_metrics([_result(1.0)], [_Example({"level": None})], "gaia")
    assert list(metrics["per_level"]) == ["None"]


def test_the_fallback_field_is_used_only_when_the_primary_is_absent():
    """math500 declares `level_field="difficulty"` with `level_fallback_field="year"`."""
    absent = compute_metrics([_result(1.0)], [_Example({"year": "2024"})], "math500")
    assert list(absent["per_level"]) == ["2024"]

    present = compute_metrics(
        [_result(1.0)], [_Example({"difficulty": "5", "year": "2024"})], "math500"
    )
    assert list(present["per_level"]) == ["5"]


def test_more_examples_than_results_does_not_crash():
    """`compute_metrics` tolerates a short results list — the partial-checkpoint
    case, where a run died mid-way and `raw_results.partial.json` is analysed."""
    examples = [_Example({"level": "1"}), _Example({"level": "2"})]
    metrics = compute_metrics([_result(1.0)], examples, "gaia")

    assert metrics["overall"]["num_correct"] == "1 of 2"
