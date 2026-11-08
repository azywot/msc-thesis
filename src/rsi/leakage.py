"""Leakage guard: reject candidate prompt text that copies training questions or answers."""

import re

from agent_engine.datasets.evaluators.metrics import normalize_answer

GENERIC_ANSWER_MAX_CHARS = 3
GENERIC_ANSWERS = frozenset({"yes", "no", "true", "false"})


def _tokens(text):
    return re.findall(r"\w+", (text or "").lower())


def _find_seq(haystack, needle):
    """Index of the first occurrence of token list `needle` in `haystack`, or -1."""
    m = len(needle)
    if m == 0:
        return -1
    for i in range(len(haystack) - m + 1):
        if haystack[i : i + m] == needle:
            return i
    return -1


class LeakageIndex:
    """Rejects text that copies a training-question n-gram or a shown gold answer.

    Exemption: answers whose normalized form is at most GENERIC_ANSWER_MAX_CHARS characters
    (ignoring spaces) or is a generic word (yes, no, true, false) are not checked. On the real
    training data many answers are such low-information tokens ("0", "1", "yes", "b"), and
    banning them would reject almost any candidate containing "Step 1" or a small number.
    All other answers match as a token sequence anywhere in the normalized text. The question
    n-gram rule is unaffected.
    """

    def __init__(self, questions, n=8):
        self.n = n
        self._grams = set()
        for q in questions or []:
            toks = _tokens(q)
            for i in range(len(toks) - n + 1):
                self._grams.add(tuple(toks[i : i + n]))

    def check(self, texts, shown_answers):
        errors = []
        for text in texts or []:
            toks = _tokens(text)
            for i in range(len(toks) - self.n + 1):
                gram = tuple(toks[i : i + self.n])
                if gram in self._grams:
                    errors.append(
                        "%d-gram match with a training question: '%s'" % (self.n, " ".join(gram))
                    )
                    break
        norm_texts = None
        for ans in shown_answers or []:
            norm = normalize_answer(ans) if ans else ""
            ans_toks = _tokens(norm)
            if not ans_toks:
                continue
            if len(norm.replace(" ", "")) <= GENERIC_ANSWER_MAX_CHARS or norm in GENERIC_ANSWERS:
                continue
            if norm_texts is None:
                norm_texts = [_tokens(normalize_answer(t) if t else "") for t in texts or []]
            for text_toks in norm_texts:
                if _find_seq(text_toks, ans_toks) >= 0:
                    errors.append("text contains a shown gold answer: '%s'" % " ".join(ans_toks))
                    break
        return errors
