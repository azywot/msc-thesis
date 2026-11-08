from rsi.leakage import LeakageIndex

Q = ["Who was the editor of the book Grand Canyon A Different View that Kurt Wise contributed to?"]


def test_long_ngram_from_any_training_question_is_rejected():
    errs = LeakageIndex(Q, n=8).check(
        ["Example: the editor of the book Grand Canyon A Different View"], []
    )
    assert errs and "grand canyon" in errs[0]


def test_seven_word_overlap_passes_at_n8():
    assert LeakageIndex(Q, n=8).check(["editor of the book Grand Canyon A"], []) == []


def test_shown_answer_is_rejected():
    assert LeakageIndex([], n=8).check(["The answer is Tom Vail, remember it."], ["Tom Vail"])


def test_long_and_numeric_answers_match_only_as_tokens():
    idx = LeakageIndex([], n=8)
    assert idx.check(["Say london or paris."], ["london"])
    assert idx.check(["A londoner said hi."], ["london"]) == []
    assert idx.check(["in 1989 the"], ["1989"])
    assert idx.check(["Use 19890 searches."], ["1989"]) == []


def test_low_information_answers_are_exempt():
    idx = LeakageIndex([], n=8)
    assert idx.check(["Step 1: answer yes or no, then 0"], ["1", "yes", "0"]) == []


def test_empty_and_none_inputs_are_ignored():
    idx = LeakageIndex(["", None, "short q"], n=8)
    assert idx.check(["", None], ["", None, "   "]) == []
