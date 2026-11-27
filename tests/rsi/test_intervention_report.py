import json

from rsi.intervention.report import blind_sample, summarize, write_gate

COUNTS = {
    "search": {"zero_of_n": 12, "skipped_error": 1, "skipped_planning_answer": 1, "candidates": 10},
    "math": {"zero_of_n": 10, "skipped_error": 0, "skipped_planning_answer": 0, "candidates": 10},
}


def _cands(domain, n, start):
    src = "hotpotqa" if domain == "search" else "deepmath"
    return [{"question_id": start + i, "data_source": src} for i in range(n)]


def _row(c, attempt=1, correct=False, control=False, status="ok", kind="rewrite", reasons=()):
    return {
        "question_id": c["question_id"],
        "data_source": c["data_source"],
        "domain": "search" if c["data_source"] == "hotpotqa" else "math",
        "attempt": attempt,
        "status": status,
        "reasons": list(reasons),
        "kind": kind if status == "ok" else None,
        "correct": correct,
        "control_correct": control,
    }


def _setup():
    search, math = _cands("search", 10, 0), _cands("math", 10, 100)
    rows = []
    for i, c in enumerate(search):
        rows.append(_row(c, correct=i < 4, control=i == 0))
    for i, c in enumerate(math):
        rows.append(_row(c, correct=i == 0, control=i == 0))
    return search + math, rows


def test_summarize_net_points_and_gate():
    cands, rows = _setup()
    s = summarize(cands, rows, COUNTS)
    assert s["search"]["candidates"] == 10 and s["search"]["rescued"] == 4
    # rates are over the 12 zero-of-n questions, not the 10 candidates
    assert s["search"]["rescue_rate"] == 33.333333 and s["search"]["control_rate"] == 8.333333
    assert s["search"]["net_points"] == 25.0
    assert s["search"]["rescue_rate_candidates"] == 40.0
    assert s["search"]["control_rate_candidates"] == 10.0
    assert s["search"]["net_points_candidates"] == 30.0
    assert s["math"]["net_points"] == 0.0 and s["math"]["net_points_candidates"] == 0.0
    assert s["gate"] == {"threshold_points": 10.0, "pass": True, "passing_domains": ["search"]}
    assert s["search"]["skipped_error"] == 1 and s["search"]["zero_of_n"] == 12
    assert "blind" not in s


def test_summarize_counts_attempts_kinds_and_rejections():
    cands = _cands("search", 5, 0)
    rows = [
        _row(cands[0], 1, correct=True, kind="rewrite"),
        _row(cands[1], 1, correct=False, kind="follow_up"),
        _row(cands[1], 2, correct=True, kind="follow_up", control=True),
        _row(cands[2], 1, status="parse_error", reasons=["no step"]),
        _row(cands[2], 2, status="leakage", reasons=["contains answer", "ngram"]),
        _row(cands[3], 1, correct=True, kind="insert"),
        _row(cands[4], 1, status="leakage", reasons=["contains answer"]),
    ]
    s = summarize(cands, rows, {"search": COUNTS["search"]})["search"]
    assert s["rescued"] == 3 and s["control_solved"] == 1
    assert s["attempts_used"] == {"1": 2, "2": 1}
    assert s["kinds"] == {"rewrite": 1, "follow_up": 1, "insert": 1}
    assert s["parse_errors"] == 1 and s["leakage_rejections"] == 2
    assert s["leakage_by_reason"] == {"contains answer": 2, "ngram": 1}


def test_gate_fails_below_threshold():
    search, math = _cands("search", 1000, 0), _cands("math", 1000, 5000)
    rows = [_row(c, correct=i < 109, control=i < 10) for i, c in enumerate(search)]
    rows += [_row(c, correct=i < 109, control=i < 10) for i, c in enumerate(math)]
    counts = {d: dict(COUNTS[d], zero_of_n=1000, candidates=1000) for d in COUNTS}
    s = summarize(search + math, rows, counts)
    assert s["search"]["net_points"] < 10.0
    assert s["gate"]["pass"] is False and s["gate"]["passing_domains"] == []


def test_blind_summary_and_write_gate(tmp_path):
    cands, rows = _setup()
    blind = [
        _row(cands[0], correct=True),
        _row(cands[1], correct=False, control=True),
        _row(cands[2], correct=False),
        _row(cands[3], correct=False),
    ]
    s = summarize(cands, rows, COUNTS, blind_rows=blind)
    assert s["blind"] == {
        "n": 4,
        "rescued": 1,
        "rate": 25.0,
        "control_rescued": 1,
        "control_rate": 25.0,
    }
    write_gate(tmp_path, s)
    assert json.loads((tmp_path / "gate.json").read_text())["gate"]["pass"] is True
    md = (tmp_path / "gate.md").read_text()
    assert md.index("PASS") < md.index("search") and "blind" in md.lower()
    assert "1 of 4 previously rescued questions rescued again (25.0%)" in md
    assert "no-fix control: 1 of 4 (25.0%)" in md
    assert "| rescue_rate_candidates | 40.0 |" in md and "| net_points | 25.0 |" in md
    assert "over the domain's 0-of-N questions" in md


def test_gate_uses_zero_of_n_denominator_not_candidates():
    cands = _cands("search", 10, 0)
    rows = [_row(c, correct=i < 2) for i, c in enumerate(cands)]
    counts = {"search": dict(COUNTS["search"], zero_of_n=25)}
    s = summarize(cands, rows, counts)
    assert s["search"]["net_points_candidates"] == 20.0
    assert s["search"]["net_points"] == 8.0
    assert s["gate"]["pass"] is False and s["gate"]["passing_domains"] == []


def test_blind_sample_deterministic_and_rescued_only():
    cands, rows = _setup()
    rows.append(_row(cands[5], 2, correct=True))
    ids = blind_sample(rows, 3, seed=1)
    assert ids == blind_sample(rows, 3, seed=1) and len(ids) == 3
    rescued = {r["question_id"] for r in rows if r["correct"]}
    assert set(ids) <= rescued
    assert sorted(blind_sample(rows, 1000)) == sorted(rescued)


def test_rescued_earlier_rows_count_toward_control_only():
    cands = _cands("search", 2, 0)
    rows = [
        _row(cands[0], 1, correct=True),
        _row(cands[0], 2, status="rescued_earlier", control=True),
        _row(cands[1], 1, correct=False),
        _row(cands[1], 2, correct=False),
    ]
    counts = {"search": dict(COUNTS["search"], zero_of_n=2, candidates=2)}
    s = summarize(cands, rows, counts)["search"]
    assert s["rescued"] == 1 and s["attempts_used"] == {"1": 1}
    assert s["control_solved"] == 1 and s["kinds"] == {"rewrite": 1}
    assert s["parse_errors"] == 0 and s["leakage_rejections"] == 0
