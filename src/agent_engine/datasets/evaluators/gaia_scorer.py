"""GAIA evaluation functions - exact copy from multi-agent-tools.

This module contains the exact evaluation logic used in the original
multi-agent-tools repository to ensure consistent scoring.
"""

import json
import re
import string
import warnings
from typing import Union, List

from ...utils.logging import get_logger

logger = get_logger(__name__)


def normalize_number_str(number_str: str) -> float:
    """Normalize a number string by removing common units and formatting.

    Removes: $, %, commas

    Args:
        number_str: String representation of a number

    Returns:
        Float value, or float('inf') if cannot parse
    """
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        logger.warning("String %s cannot be normalized to number str.", number_str)
        return float("inf")


def split_string(
    s: str,
    char_list: List[str] = [",", ";"],
) -> List[str]:
    """Split string by multiple delimiters.

    Args:
        s: String to split
        char_list: List of delimiter characters

    Returns:
        List of split strings
    """
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def is_float(element) -> bool:
    """Check if element can be converted to float.

    Args:
        element: Any element to check

    Returns:
        True if can convert to float
    """
    try:
        float(element)
        return True
    except ValueError:
        return False


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    """Score a GAIA question answer.

    This is the main evaluation function that handles:
    - Numeric answers (with normalization)
    - List answers (comma/semicolon separated)
    - String answers (with normalization)

    Args:
        model_answer: The model's predicted answer
        ground_truth: The correct answer

    Returns:
        True if correct, False otherwise
    """
    # if gt is a number
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.", UserWarning
            )
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        return normalize_str(model_answer) == normalize_str(ground_truth)


def check_prediction_contains_answer_letters_in_order(prediction, true_answer):
    """Check if prediction contains answer letters in order.

    Used for edge case detection where answer might be embedded in longer text.

    Args:
        prediction: Model prediction
        true_answer: Ground truth answer

    Returns:
        True if letters appear in order
    """
    prediction = prediction.lower()
    true_answer = true_answer.lower()
    if len(prediction) > len(true_answer) * 3:
        return False
    i = 0
    for letter in true_answer:
        if letter in prediction[i:]:
            i += prediction[i:].index(letter)
        else:
            return False
    return True


def check_close_call(prediction, true_answer, is_correct):
    """Check if a failed prediction is a close call.

    Identifies borderline cases where the answer is technically wrong
    but very close to correct.

    Args:
        prediction: Model prediction
        true_answer: Ground truth answer
        is_correct: Whether already marked correct

    Returns:
        True if close call, False otherwise
    """
    if is_correct:
        return True
    else:
        if is_float(true_answer):
            return is_correct
        else:
            if check_prediction_contains_answer_letters_in_order(str(prediction), str(true_answer)) and len(str(true_answer)) * 0.5 <= len(str(prediction)) <= len(str(true_answer))*2:
                logger.info("Close call: %s vs %s", prediction, true_answer)
                return True
            else:
                return False


def normalize_str(input_str, remove_punct=True) -> str:
    """Normalize a string for comparison.

    Normalization steps:
    - Remove all white spaces
    - Optionally remove punctuation (if remove_punct is True)
    - Convert to lowercase

    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)

    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()
