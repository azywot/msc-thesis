"""``gaia_scorer`` — the function that decides every GAIA number in the thesis.

This module is documented in its own header as an "exact copy from
multi-agent-tools ... to ensure consistent scoring".  That makes it the one
file in the repo where matching upstream matters more than being right, so
these tests are strictly characterization: they pin the behaviour, including
the sharp edges, so nobody tidies it up and silently moves a headline result.

Two of those edges are recorded explicitly below —
``test_a_none_ground_truth_raises`` and
``test_letters_in_order_does_not_advance_past_a_match``.
"""

import warnings

import pytest

from agent_engine.datasets.evaluators.gaia_scorer import (
    check_close_call,
    check_prediction_contains_answer_letters_in_order,
    is_float,
    normalize_number_str,
    normalize_str,
    question_scorer,
    split_string,
)

# --- normalize_number_str -------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("42", 42.0),
        ("$1,000", 1000.0),
        ("17%", 17.0),
        ("$1,234.56", 1234.56),
        ("  3 ", 3.0),  # float() tolerates surrounding whitespace
        ("-5", -5.0),
        ("1e3", 1000.0),
    ],
)
def test_normalize_number_str_strips_units_and_separators(raw, expected):
    assert normalize_number_str(raw) == expected


def test_an_unparseable_number_becomes_infinity():
    """Not an exception and not None: ``inf``, chosen so the equality check in
    ``question_scorer`` simply returns False."""
    assert normalize_number_str("about twelve") == float("inf")


def test_infinity_never_equals_a_finite_ground_truth():
    assert question_scorer("about twelve", "12") is False


# --- is_float -------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [("3", True), (3, True), ("3.5", True), ("x", False)])
def test_is_float(value, expected):
    assert is_float(value) is expected


def test_is_float_only_catches_valueerror():
    """``float(None)`` raises TypeError, which is *not* caught.  Recorded, not
    fixed: it is the reason a None ground truth propagates as a crash."""
    with pytest.raises(TypeError):
        is_float(None)


# --- split_string ---------------------------------------------------------


def test_split_string_defaults_to_comma_and_semicolon():
    assert split_string("a,b;c") == ["a", "b", "c"]


def test_split_string_accepts_custom_delimiters():
    assert split_string("a|b", ["|"]) == ["a", "b"]


def test_split_string_does_not_strip_whitespace():
    """Elements keep their spaces; ``normalize_str`` removes them later."""
    assert split_string("a, b") == ["a", " b"]


# --- normalize_str --------------------------------------------------------


def test_normalize_str_removes_whitespace_punctuation_and_case():
    assert normalize_str("Sea Gull!") == "seagull"


def test_normalize_str_can_keep_punctuation():
    assert normalize_str("Sea Gull!", remove_punct=False) == "seagull!"


def test_normalization_makes_seagull_match_sea_gull():
    """The case the upstream comment calls out by name."""
    assert question_scorer("sea gull", "seagull") is True


# --- question_scorer: numeric ---------------------------------------------


@pytest.mark.parametrize(
    "prediction, truth, expected",
    [
        ("42", "42", True),
        ("42.0", "42", True),
        ("$1,000", "1000", True),
        ("1,000", "1000", True),
        ("43", "42", False),
        ("forty-two", "42", False),
    ],
)
def test_a_numeric_ground_truth_compares_as_a_float(prediction, truth, expected):
    assert question_scorer(prediction, truth) is expected


def test_a_numeric_prediction_is_coerced_to_str_first():
    """``str(model_answer)`` — so a genuine int prediction still scores."""
    assert question_scorer(42, "42") is True


# --- question_scorer: lists -----------------------------------------------


def test_a_list_ground_truth_compares_elementwise():
    assert question_scorer("a, b, c", "a,b,c") is True


def test_list_order_matters():
    assert question_scorer("b,a", "a,b") is False


def test_a_length_mismatch_warns_and_returns_false():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = question_scorer("a,b", "a,b,c")

    assert result is False
    assert any("different lengths" in str(w.message) for w in caught)


def test_numeric_elements_inside_a_list_are_normalised():
    assert question_scorer("$1000; 2", "1000;2") is True


def test_a_thousands_separator_inside_a_list_answer_splits_the_number():
    """Recorded, not fixed.

    ``split_string`` treats "," as a delimiter, and it runs *before* the
    per-element ``normalize_number_str`` that would have stripped it.  So
    "$1,000; 2" splits into three elements and fails the length check against a
    two-element truth — a formatting choice in the prediction, scored as wrong.

    It only bites when the ground truth is itself a list; a bare "$1,000"
    against "1000" takes the numeric path and passes (asserted above).
    """
    assert split_string("$1,000; 2") == ["$1", "000", " 2"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert question_scorer("$1,000; 2", "1000;2") is False


def test_list_elements_keep_their_punctuation():
    """Element comparison passes ``remove_punct=False``, so punctuation inside
    an element is significant even though it is ignored for a plain string
    answer."""
    assert question_scorer("a.b, c", "ab,c") is False
    assert question_scorer("a.b", "ab") is True  # same text, scored as a string


def test_a_semicolon_in_the_ground_truth_selects_the_list_path():
    assert question_scorer("x;y", "x;y") is True


# --- question_scorer: strings ---------------------------------------------


@pytest.mark.parametrize(
    "prediction, truth, expected",
    [
        ("Paris", "paris", True),
        ("  Paris  ", "Paris", True),
        ("Paris!", "Paris", True),
        ("Lyon", "Paris", False),
        ("", "", True),
    ],
)
def test_a_string_ground_truth_compares_normalised(prediction, truth, expected):
    assert question_scorer(prediction, truth) is expected


def test_a_none_ground_truth_raises():
    """Recorded, not fixed.

    ``question_scorer`` calls ``is_float(ground_truth)`` first, and that raises
    TypeError on None rather than returning False.  A dataset row with a null
    answer therefore aborts scoring instead of counting as wrong.  Every loader
    in the repo supplies a string today, which is why this has never fired.
    """
    with pytest.raises(TypeError):
        question_scorer("anything", None)


# --- close-call helpers ---------------------------------------------------


def test_letters_in_order_rejects_a_much_longer_prediction():
    """The length guard: more than 3x the truth and it is not a close call, no
    matter what letters it contains."""
    assert check_prediction_contains_answer_letters_in_order("x" * 10 + "ab", "ab") is False


def test_letters_in_order_accepts_letters_appearing_in_sequence():
    assert check_prediction_contains_answer_letters_in_order("a-b", "ab") is True


def test_letters_in_order_is_case_insensitive():
    assert check_prediction_contains_answer_letters_in_order("AB", "ab") is True


def test_letters_in_order_rejects_a_reordered_answer():
    assert check_prediction_contains_answer_letters_in_order("ba", "ab") is False


def test_letters_in_order_does_not_advance_past_a_match():
    """Recorded, not fixed.

    The index advances by ``prediction[i:].index(letter)`` with no ``+ 1``, so
    a single character in the prediction can satisfy the same letter of the
    truth repeatedly: "a" contains "aa" by this check.  It only ever feeds the
    close-call *diagnostic*, never ``correct``, so the blast radius is limited
    to reported close-call counts.
    """
    assert check_prediction_contains_answer_letters_in_order("a", "aa") is True


def test_a_correct_answer_is_trivially_a_close_call():
    assert check_close_call("anything", "unrelated", is_correct=True) is True


def test_a_wrong_numeric_answer_is_never_a_close_call():
    assert check_close_call("41", "42", is_correct=False) is False


def test_a_wrong_string_answer_within_the_length_window_is_a_close_call():
    assert check_close_call("Pariss", "Paris", is_correct=False) is True


def test_letters_alone_are_not_enough_the_length_must_also_fit():
    """Length must sit in [0.5x, 2x] of the truth *even when* the letters line
    up, so both halves of the condition are load-bearing.

    Both predictions below pass ``check_prediction_contains_answer_letters_in_order``
    — asserted here so the test cannot pass for the wrong reason — and are
    rejected only by the length window: one too long, one too short.
    """
    too_long, too_short = "axxbx", "a"

    assert check_prediction_contains_answer_letters_in_order(too_long, "ab") is True
    assert check_close_call(too_long, "ab", is_correct=False) is False

    assert check_prediction_contains_answer_letters_in_order(too_short, "aaaa") is True
    assert check_close_call(too_short, "aaaa", is_correct=False) is False


def test_a_prediction_with_the_wrong_letters_is_not_a_close_call():
    assert check_close_call("P", "Paris", is_correct=False) is False
