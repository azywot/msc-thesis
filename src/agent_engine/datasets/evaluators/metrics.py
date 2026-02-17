"""Evaluation metrics for different datasets.

This module provides various evaluation metrics used across datasets.
"""

import re
import string
from typing import Any, Dict, List


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Normalization steps:
    - Convert to lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace

    Args:
        answer: Answer string

    Returns:
        Normalized answer
    """
    if not answer:
        return ""

    # Convert to lowercase
    answer = answer.lower()

    # Remove punctuation
    answer = answer.translate(str.maketrans('', '', string.punctuation))

    # Remove articles
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    return answer.strip()


def exact_match(prediction: str, ground_truth: str, case_sensitive: bool = False) -> bool:
    """Check if prediction exactly matches ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        case_sensitive: Whether to use case-sensitive comparison

    Returns:
        True if exact match
    """
    if not case_sensitive:
        prediction = prediction.lower()
        ground_truth = ground_truth.lower()

    return prediction.strip() == ground_truth.strip()


def normalized_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction matches normalized ground truth.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if normalized match
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    """Check if ground truth is contained in prediction.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if ground truth is in prediction
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    return gt_norm in pred_norm


def numeric_match(prediction: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """Check if prediction matches ground truth numerically.

    Tries to extract and compare numbers from strings.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Numerical tolerance for comparison

    Returns:
        True if numeric match
    """
    try:
        # Try to parse as floats
        pred_num = float(prediction.strip())
        gt_num = float(ground_truth.strip())
        return abs(pred_num - gt_num) <= tolerance
    except (ValueError, TypeError):
        return False


def evaluate_multiple_choice(
    prediction: str,
    ground_truth: str,
    choices: List[str]
) -> Dict[str, Any]:
    """Evaluate multiple choice prediction.

    Args:
        prediction: Predicted answer (A, B, C, D or full text)
        ground_truth: Ground truth answer
        choices: List of choice texts

    Returns:
        Evaluation results
    """
    # Normalize ground truth
    gt_norm = ground_truth.strip().upper()

    # Try to extract choice letter from prediction
    pred_letter = None
    pred_norm = prediction.strip().upper()

    # Check if prediction is just a letter
    if len(pred_norm) == 1 and pred_norm in ['A', 'B', 'C', 'D', 'E']:
        pred_letter = pred_norm
    else:
        # Try to find letter pattern like "A)", "(A)", "A.", "A:"
        match = re.search(r'[(]?([A-E])[).:)]?', pred_norm)
        if match:
            pred_letter = match.group(1)

    # Check if letters match
    correct = pred_letter == gt_norm if pred_letter else False

    # Also check if prediction text matches any choice
    if not correct and choices:
        pred_text_norm = normalize_answer(prediction)
        for i, choice in enumerate(choices):
            if normalize_answer(choice) == pred_text_norm:
                choice_letter = chr(ord('A') + i)
                correct = choice_letter == gt_norm
                break

    return {
        "correct": correct,
        "predicted_letter": pred_letter,
        "ground_truth_letter": gt_norm,
        "exact_match": exact_match(prediction, ground_truth),
    }


def evaluate_gaia(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Evaluate GAIA prediction.

    GAIA uses normalized matching for most answers.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Evaluation results
    """
    # Try exact match first
    exact = exact_match(prediction, ground_truth, case_sensitive=False)

    # Try normalized match
    normalized = normalized_match(prediction, ground_truth)

    # Try numeric match
    numeric = numeric_match(prediction, ground_truth)

    # Try contains match
    contains = contains_match(prediction, ground_truth)

    # GAIA considers it correct if any of these match
    correct = exact or normalized or numeric

    return {
        "correct": correct,
        "exact_match": exact,
        "normalized_match": normalized,
        "numeric_match": numeric,
        "contains_match": contains,
        "score": 1.0 if correct else 0.0,
    }


def evaluate_gpqa(
    prediction: str,
    ground_truth: str,
    choices: List[str]
) -> Dict[str, Any]:
    """Evaluate GPQA prediction.

    GPQA is multiple choice with specific format.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer (letter)
        choices: List of choice texts

    Returns:
        Evaluation results
    """
    result = evaluate_multiple_choice(prediction, ground_truth, choices)
    result["score"] = 1.0 if result["correct"] else 0.0
    return result


def evaluate_math(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Evaluate math dataset prediction.

    Focuses on numeric matching with some tolerance.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Evaluation results
    """
    # Try exact match
    exact = exact_match(prediction, ground_truth, case_sensitive=False)

    # Try normalized match
    normalized = normalized_match(prediction, ground_truth)

    # Try numeric match (primary for math)
    numeric = numeric_match(prediction, ground_truth, tolerance=1e-4)

    # Math is correct if normalized or numeric match
    correct = normalized or numeric

    return {
        "correct": correct,
        "exact_match": exact,
        "normalized_match": normalized,
        "numeric_match": numeric,
        "score": 1.0 if correct else 0.0,
    }


def evaluate_qa(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Evaluate QA dataset prediction.

    Uses normalized matching like SQuAD.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Evaluation results
    """
    # Normalize both answers
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    # Check normalized match
    correct = pred_norm == gt_norm

    # Also check if prediction contains ground truth
    contains = gt_norm in pred_norm

    return {
        "correct": correct,
        "normalized_match": correct,
        "contains_match": contains,
        "score": 1.0 if correct else 0.0,
    }
