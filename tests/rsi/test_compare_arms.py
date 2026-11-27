import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_arms.py"
spec = importlib.util.spec_from_file_location("compare_arms", SCRIPT)
compare_arms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_arms)


def _row(qid, correct, **tools):
    return {"question_id": qid, "correct": correct, "tool_counts": tools}


def _write(root, bench, rows, run="run_a"):
    d = root / bench / "variant" / run
    d.mkdir(parents=True)
    (d / "raw_results.json").write_text(json.dumps(rows))


def test_compare_mean_mcnemar_and_behaviour(tmp_path):
    base = {i: _row(i, i < 2, code_generator=0, web_search=1) for i in range(10)}
    arm_c = {i: _row(i, i < 3, code_generator=0, web_search=1) for i in range(10)}
    arm_i = {i: _row(i, i < 8, code_generator=1, web_search=2) for i in range(10)}
    r = compare_arms.compare(base, arm_c, arm_i, "aime")
    assert r["accuracy"] == {"base": 0.2, "arm_c": 0.3, "arm_i": 0.8}
    assert r["n_shared"] == 10
    # arm I only correct on 5 questions, arm C only on 0
    assert r["mcnemar_p"] == 0.0625
    assert r["behaviour"] == {"code_call_rate": {"base": 0.0, "arm_c": 0.0, "arm_i": 1.0}}
    m = compare_arms.compare(base, arm_c, arm_i, "musique")
    assert m["behaviour"]["multi_search_rate"] == {"base": 0.0, "arm_c": 0.0, "arm_i": 1.0}
    assert (
        "behaviour" not in compare_arms.compare(base, arm_c, arm_i, "gpqa")
        or not compare_arms.compare(base, arm_c, arm_i, "gpqa")["behaviour"]
    )


def test_cli_writes_outputs_and_verdict(tmp_path):
    for name, k in (("base", 1), ("c", 2), ("i", 4)):
        for b in compare_arms.BENCHMARKS:
            rows = [_row(q, q < k, code_generator=1, web_search=2) for q in range(5)]
            _write(tmp_path / name, b, [_row(9, False)], run="2020-old")
            _write(tmp_path / name, b, rows, run="2030-new")
    out = tmp_path / "out"
    compare_arms.main(
        [
            "--base-root",
            str(tmp_path / "base"),
            "--arm-c-root",
            str(tmp_path / "c"),
            "--arm-i-root",
            str(tmp_path / "i"),
            "--out-dir",
            str(out),
        ]
    )
    data = json.loads((out / "comparison.json").read_text())
    assert data["mean_accuracy"] == {"base": 0.2, "arm_c": 0.4, "arm_i": 0.8}
    assert data["verdict"] == "arm I mean > arm C mean: True"
    assert "arm I mean > arm C mean" in (out / "comparison.md").read_text()
