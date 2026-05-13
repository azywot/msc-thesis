"""Evaluation metrics for all datasets.

Single entry point: `evaluate_answer(prediction, ground_truth, choices)`.

Accuracy:
  - Multiple-choice answers  → letter matching
  - Numerical / math answers → Math-Verify (parse + verify)
  - Plain text answers       → GAIA-style normalised string comparison

EM and F1 are computed the same way for every dataset (SQuAD-style).
"""

import re
import string
from collections import Counter
from math_verify import parse, verify
from typing import Any, Dict, List, Optional
from .gaia_scorer import question_scorer


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def normalize_answer(answer: str) -> str:
    """Lowercase, strip punctuation, articles, and extra whitespace."""
    if not answer:
        return ""
    # Strip common LaTeX wrappers like \text{No} or \boxed{42}.
    answer = strip_latex_wrappers(answer)
    answer = answer.lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Canonicalize boolean synonyms as whole words so "true"/"yes" and "false"/"no"
    # compare equal whether standalone or embedded ("the answer is true" == "the answer is yes").
    answer = re.sub(r"\btrue\b", "yes", answer)
    answer = re.sub(r"\bfalse\b", "no", answer)
    return " ".join(answer.split()).strip()


def strip_latex_wrappers(ans: str) -> str:
    """Remove simple LaTeX wrappers that often surround plain answers.

    Examples:
      '\\text{No}'     -> 'No'
      '\\boxed{42}'    -> '42'
      '$42$'           -> '42'
      '\\(No\\)'       -> 'No'
    """
    if not ans:
        return ans

    s = ans.strip()

    # Strip surrounding $...$ or $$...$$
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()

    # Iteratively strip simple single-argument wrappers like \text{...}, \boxed{...}
    for _cmd in ("text", "boxed"):
        m = re.fullmatch(rf"\s*\\{_cmd}\{{(.+)\}}\s*", s)
        if m:
            s = m.group(1).strip()
            break
        prefix = f"\\{_cmd}" + "{"
        if s.startswith(prefix):
            # Drop leading "\cmd{" and keep the rest, even if the closing brace is missing.
            s = s[len(prefix):].strip()
            break

    return s


def exact_match(prediction: str, ground_truth: str, case_sensitive: bool = False) -> bool:
    if not case_sensitive:
        prediction, ground_truth = prediction.lower(), ground_truth.lower()
    return prediction.strip() == ground_truth.strip()


def normalized_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if gt_norm in pred_norm:
        return True
    gt_no_initials = " ".join(t for t in gt_norm.split() if len(t) > 1)
    pred_no_initials = " ".join(t for t in pred_norm.split() if len(t) > 1)
    return bool(gt_no_initials) and gt_no_initials in pred_no_initials


def token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-level F1 on normalised tokens."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Math-Verify based accuracy
# ---------------------------------------------------------------------------

MATH_TOKEN_PATTERN = re.compile(
    r"""
    ^
    \s*
    (                                    # valid math expression tokens
        (?:[+\-*/^%]|\d+(?:\.\d+)?       # operator or number
        |[a-zA-Z]+                       # variable/function name
        |\(|\)
        |\\frac|\\sqrt|\\sin|\\cos|\\tan # common LaTeX/math funcs
        |pi|e
        |\\[a-zA-Z]+                     # any LaTeX command
        |\s+
        )+
    )
    \s*$
    """,
    re.VERBOSE,
)


def is_math_answer(s: str) -> bool:
    """Return True if *s* should be scored with Math-Verify / symbolic rules.

    Numeric strings and expressions with operators, digits, or LaTeX use the math
    path. Multi-letter alphabetic-only strings (e.g. *Paris*, *yes*) are treated
    as plain-text QA answers so they are not mis-routed to Math-Verify.

    Single-letter tokens (e.g. *x*, *n*) keep the math path for short symbolic
    gold answers.
    """
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass

    if len(s) >= 2 and re.fullmatch(r"[A-Za-z]+", s) is not None:
        return False

    return bool(MATH_TOKEN_PATTERN.match(s))


def evaluate_with_math_verify(prediction: str, ground_truth: str) -> bool:
    """Return True if prediction matches ground_truth via Math-Verify.

    Falls back to normalised string comparison when Math-Verify cannot parse
    either expression (e.g. purely textual answers).
    """
    try:
        gold = parse(ground_truth)
        answer = parse(prediction)
        if gold and answer:
            return bool(verify(gold, answer))
    except Exception:
        pass
    return normalized_match(prediction, ground_truth)


# ---------------------------------------------------------------------------
# Unified evaluator (single entry point for all datasets)
# ---------------------------------------------------------------------------

def evaluate_answer(
    prediction: str,
    ground_truth: str,
    choices: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a prediction against its ground truth.

    Returns a dict with keys:
        correct   bool  – primary correctness flag (= accuracy > 0)
        accuracy  float – 1.0 / 0.0 (math_verify where applicable)
        em        float – exact match after normalisation
        f1        float – SQuAD-style token F1
    """
    pred = (prediction or "").strip()
    gt = (ground_truth or "").strip()

    # Fast path: identical after stripping common LaTeX/text wrappers and normalising.
    # This handles cases like prediction='\\text{No}' vs ground_truth='No'.
    stripped_pred = strip_latex_wrappers(pred)
    stripped_gt = strip_latex_wrappers(gt)
    if normalize_answer(stripped_pred) == normalize_answer(stripped_gt):
        return {
            "correct": True,
            "accuracy": 1.0,
            "em": 1.0,
            "f1": 1.0,
        }

    if choices is not None:
        # Multiple-choice: letter matching (GPQA style)
        score = float(_mc_correct(pred, gt, choices))
        em_score = score
        f1_score = score
    elif is_math_answer(gt):
        # Numerical / LaTeX / symbolic answer → Math-Verify
        score = float(evaluate_with_math_verify(pred, gt))
        em_score = float(exact_match(pred, gt, case_sensitive=False))
        f1_score = token_f1(pred, gt)
    else:
        # Plain text → GAIA-style normalised comparison, see: https://github.com/aymeric-roucher/GAIA/blob/main/scripts/evaluation/gaia_scorer.py
        score = float(question_scorer(pred, gt))
        em_score = float(exact_match(pred, gt, case_sensitive=False))
        f1_score = token_f1(pred, gt)

    if choices is None and contains_match(pred, gt):
        score = max(score, 1.0)
        em_score = max(em_score, 1.0)
        f1_score = max(f1_score, token_f1(pred, gt))

    return {
        "correct": score > 0,
        "accuracy": score,
        "em": em_score,
        "f1": f1_score,
    }


def _mc_correct(prediction: str, ground_truth: str, choices: List[str]) -> bool:
    """Return True if prediction matches the correct multiple-choice letter."""
    gt_norm = ground_truth.strip().upper()
    pred_norm = prediction.strip().upper()

    pred_letter: Optional[str] = None
    if len(pred_norm) == 1 and pred_norm in 'ABCDE':
        pred_letter = pred_norm
    else:
        m = re.search(r'\b([A-E])\b', pred_norm)
        if m:
            pred_letter = m.group(1)

    if pred_letter == gt_norm:
        return True

    if choices:
        pred_text_norm = normalize_answer(prediction)
        for i, choice in enumerate(choices):
            if normalize_answer(choice) == pred_text_norm:
                return chr(ord('A') + i) == gt_norm

    return False


# ---------------------------------------------------------------------------
# Dataset-specific wrappers (semantic aliases used by dataset loaders)
# ---------------------------------------------------------------------------

def evaluate_gaia(prediction: str, ground_truth: str) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth)


def evaluate_math(prediction: str, ground_truth: str) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth)


def evaluate_gpqa(prediction: str, ground_truth: str, choices: List[str]) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth, choices=choices)


def evaluate_qa(prediction: str, ground_truth: str) -> Dict[str, Any]:
    return evaluate_answer(prediction, ground_truth)


def evaluate_musique(
    prediction: str,
    ground_truth: str,
    answer_aliases: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate MuSiQue prediction with support for answer aliases.

    A prediction is considered correct if it matches either the canonical
    ground-truth answer or any of the aliases (after normalisation).
    """
    # First, evaluate against the canonical answer using the unified logic.
    base_result = evaluate_answer(prediction, ground_truth)
    if base_result.get("correct", False):
        return base_result

    aliases = answer_aliases or []
    for alias in aliases:
        if not alias:
            continue
        # Reuse the full evaluation machinery for each alias so that
        # normalisation, GAIA scoring, and Math-Verify behave consistently.
        alias_result = evaluate_answer(prediction, alias)
        if alias_result.get("correct", False):
            return alias_result

    # No alias matched; fall back to the canonical evaluation result.
    return base_result
