"""Dataset evaluators and metrics."""

from .metrics import (
    exact_match,
    normalized_match,
    contains_match,
    numeric_match,
    evaluate_gaia,
    evaluate_gpqa,
    evaluate_math,
    evaluate_qa,
)

# GAIA official scorer functions (from multi-agent-tools)
from .gaia_scorer import (
    question_scorer,
    normalize_str,
    normalize_number_str,
    check_prediction_contains_answer_letters_in_order,
    check_close_call,
    is_float,
    split_string,
)

__all__ = [
    # Generic metrics
    "exact_match",
    "normalized_match",
    "contains_match",
    "numeric_match",
    # Dataset-specific evaluators
    "evaluate_gaia",
    "evaluate_gpqa",
    "evaluate_math",
    "evaluate_qa",
    # GAIA official scorer
    "question_scorer",
    "normalize_str",
    "normalize_number_str",
    "check_prediction_contains_answer_letters_in_order",
    "check_close_call",
    "is_float",
    "split_string",
]
