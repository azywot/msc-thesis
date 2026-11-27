import importlib.util
import json
from pathlib import Path

import pytest

from rsi.intervention.records import intervention_candidates
from rsi.intervention.report import summarize

from .test_intervention_rescue import _cand

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "report_intervention", REPO / "scripts/report_intervention.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load()

SEARCH_FIGS = (
    "rescue_vs_control",
    "ceiling",
    "funnel",
    "fix_kinds",
    "fix_step",
    "behaviour",
)


def _recs(qid, source, correct_by_pass, n_actions=2, tool="web_search", **kw):
    out = []
    for p, ok in enumerate(correct_by_pass, start=1):
        rec = _cand(qid, n_actions=n_actions)
        rec.update({"pass": p, "correct": ok, "data_source": source, **kw})
        if tool != "web_search":
            for a in rec["action_history"]:
                a["command"] = json.dumps({"name": tool, "arguments": {}})
        out.append(rec)
    return out


def _call_msgs(name, n=1):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msgs.append({"role": "assistant", "content": "plan"})
    for i in range(n):
        call = json.dumps({"name": name, "arguments": {"query": f"x{i}"}})
        msgs.append({"role": "assistant", "content": f"<tool_call>\n{call}\n</tool_call>"})
        msgs.append({"role": "tool", "content": "res"})
    msgs.append({"role": "assistant", "content": "final"})
    return msgs


def _row(qid, domain, status="ok", attempt=1, correct=False, kind="rewrite", step=0, **kw):
    fix = {"name": "web_search", "arguments": {"query": "a|b"}}
    row = {
        "question_id": qid,
        "data_source": "hotpotqa" if domain == "search" else "deepmath",
        "domain": domain,
        "question": "Q | pipe " + "x" * 300,
        "ground_truth": "Paris",
        "aliases": [],
        "attempt": attempt,
        "seed": attempt,
        "status": status,
        "reasons": [],
        "reflector_text": "",
        "step": step if status in ("ok", "leakage") else None,
        "kind": kind if status in ("ok", "leakage") else None,
        "fix_text": (
            "<sub_goal>g</sub_goal>\n\n<tool_call>\n" + json.dumps(fix) + "\n</tool_call>"
            if status in ("ok", "leakage")
            else None
        ),
        "correct": correct,
        "prediction": "Paris" if correct else "",
        "messages": _call_msgs("web_search", 2) if status == "ok" else [],
        "control_step": 0,
        "control_correct": False,
        "control_prediction": "Lyon",
    }
    row.update(kw)
    return row


def make_inputs(tmp_path, with_math=True, blind=False):
    recs = []
    recs += _recs(1, "hotpotqa", [False] * 3)  # rescued, rewrite
    recs += _recs(2, "hotpotqa", [False] * 3)  # parse error
    recs += _recs(3, "hotpotqa", [False] * 3)  # leakage
    recs += _recs(4, "hotpotqa", [True, True, True])  # solved
    recs += _recs(5, "hotpotqa", [False, True, False])  # pass@3 only
    recs += _recs(6, "hotpotqa", [False] * 3)  # rescued at attempt 2, control solves too
    rows = [
        _row(1, "search", correct=True),
        _row(2, "search", status="parse_error"),
        _row(3, "search", status="leakage", reasons=["gold"]),
        _row(6, "search", attempt=1, correct=False, control_correct=True),
        _row(6, "search", attempt=2, correct=True, kind="follow_up", step=2),
        _row(1, "search", status="rescued_earlier", attempt=2),
    ]
    if with_math:
        m = _recs(10, "deepmath", [False] * 3, n_actions=0, turns=1)
        for r in m:
            r["action_history"] = []
        recs += m
        recs += _recs(11, "deepmath", [False] * 3, error="boom")
        mrow = _row(10, "math", correct=True, kind="insert", step=0)
        mrow["messages"] = _call_msgs("code_generator", 1)
        rows.append(mrow)
    collected = tmp_path / "collected.jsonl"
    collected.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    cands, counts = intervention_candidates(R.load_collected([collected]), 3)
    blind_rows = None
    if blind:
        blind_rows = [_row(1, "search", correct=True, control_correct=False)]
        (tmp_path / "rescue").mkdir(exist_ok=True)
    summary = summarize(cands, rows, counts, 10.0, blind_rows)
    rdir = tmp_path / "rescue"
    rdir.mkdir(exist_ok=True)
    (rdir / "gate.json").write_text(json.dumps(summary))
    (rdir / "rescue.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if blind:
        (rdir / "blind.jsonl").write_text("\n".join(json.dumps(r) for r in blind_rows) + "\n")
    return collected, rdir, summary


def _comparison():
    def bench(name, beh):
        return {
            "benchmark": name,
            "accuracy": {"base": 0.2, "arm_c": 0.25, "arm_i": 0.3},
            "n": {"base": 10, "arm_c": 10, "arm_i": 10},
            "n_shared": 10,
            "discordant": {"arm_i_only": 2, "arm_c_only": 1},
            "mcnemar_p": 0.5,
            "behaviour": beh,
        }

    three = {"base": 0.1, "arm_c": 0.2, "arm_i": 0.4}
    return {
        "benchmarks": [
            bench("gaia", {"multi_search_rate": three}),
            bench("gpqa", {}),
            bench("aime", {"code_call_rate": three}),
            bench("musique", {"multi_search_rate": three}),
            bench("hle", {}),
        ],
        "mean_accuracy": {"base": 0.2, "arm_c": 0.25, "arm_i": 0.3},
        "verdict": "arm I mean > arm C mean: True",
    }


def test_report_writes_all_outputs(tmp_path):
    collected, rdir, _ = make_inputs(tmp_path)
    out = tmp_path / "out"
    R.build_report([collected], rdir, out, None, 8)
    for name in SEARCH_FIGS:
        for ext in ("png", "pdf"):
            assert (out / "figures" / f"{name}.{ext}").exists(), name
    for name in ("blind", "benchmarks", "benchmark_behaviour"):
        assert not (out / "figures" / f"{name}.png").exists()
    md = (out / "report.md").read_text()
    assert "No blind check" in md and "No comparison.json" in md
    assert (out / "report.json").exists()


def test_report_with_blind_and_comparison(tmp_path):
    collected, rdir, _ = make_inputs(tmp_path, blind=True)
    comp = tmp_path / "comparison.json"
    comp.write_text(json.dumps(_comparison()))
    out = tmp_path / "out"
    R.build_report([collected], rdir, out, comp, 8)
    for name in ("blind", "benchmarks", "benchmark_behaviour"):
        for ext in ("png", "pdf"):
            assert (out / "figures" / f"{name}.{ext}").exists(), name
    md = (out / "report.md").read_text()
    assert "figures/benchmarks.png" in md and "figures/blind.png" in md


def test_ceiling_numbers(tmp_path):
    collected, rdir, gate = make_inputs(tmp_path)
    data = R.build_report([collected], rdir, tmp_path / "out")
    s = data["domains"]["search"]
    # 6 questions: pass@1 = q4 only; pass@3 = q4 and q5
    assert s["questions"] == 6 and s["pass_at_1"] == 1 and s["pass_at_3"] == 2
    # rescued q1, q6 = 2; control solved q6 = 1
    assert s["rescued"] == 2 and s["control_solved"] == 1
    assert s["ceiling_net_pct"] == pytest.approx(100 * (2 + 2 - 1) / 6)
    assert s["ceiling_gross_pct"] == pytest.approx(100 * (2 + 2) / 6)
    on_disk = json.loads((tmp_path / "out" / "report.json").read_text())
    assert on_disk["domains"]["search"]["pass_at_3_pct"] == pytest.approx(100 * 2 / 6)


def test_rescue_and_control_rates_match_gate(tmp_path):
    collected, rdir, gate = make_inputs(tmp_path)
    data = R.build_report([collected], rdir, tmp_path / "out")
    for d in ("search", "math"):
        for key in ("rescue_rate", "control_rate", "net_points", "zero_of_n"):
            assert data["domains"][d][key] == gate[d][key]


@pytest.mark.parametrize(
    "k,n,lo,hi",
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0)],
)
def test_wilson_interval_known_values(k, n, lo, hi):
    got = R.wilson(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-3) and got[1] == pytest.approx(hi, abs=1e-3)
    assert R.wilson(0, 0) == (0.0, 0.0)


def test_behaviour_counts(tmp_path):
    collected, rdir, _ = make_inputs(tmp_path)
    data = R.build_report([collected], rdir, tmp_path / "out")
    b = data["domains"]["math"]["behaviour"]
    assert b["n_rescued"] == 1 and b["original"] == 0.0 and b["rescued"] == 100.0
    s = data["domains"]["search"]["behaviour"]
    # originals have two web_search calls each; rescued trajectories carry two as well
    assert s["original"] == 2.0 and s["rescued"] == 2.0


def test_examples_table_balanced_and_escaped(tmp_path):
    collected, rdir, _ = make_inputs(tmp_path)
    data = R.build_report([collected], rdir, tmp_path / "out", n_examples=3)
    assert [e["domain"] for e in data["examples"]] == ["search", "math", "search"]
    assert [e["question_id"] for e in data["examples"]] == [1, 10, 6]
    md = (tmp_path / "out" / "report.md").read_text()
    table = md.split("## Examples of rescued questions")[1]
    assert "a\\|b" in table and "pipe" in table
    assert "..." in table  # question truncated to 200 chars
    first = data["examples"][0]
    assert first["original_call"].startswith('{"name": "web_search"')
    assert data["examples"][1]["original_call"] == "none"
    for line in table.splitlines():
        if line.startswith("| search") or line.startswith("| math"):
            assert line.replace("\\|", "").count("|") == 9


def test_empty_domain_renders(tmp_path):
    collected, rdir, _ = make_inputs(tmp_path, with_math=False)
    out = tmp_path / "out"
    data = R.build_report([collected], rdir, out)
    assert data["domains"]["math"]["zero_of_n"] == 0
    assert data["domains"]["math"]["rescue_rate"] == 0.0
    for name in SEARCH_FIGS:
        assert (out / "figures" / f"{name}.png").exists()
    assert "Domain math: no data" in (out / "report.md").read_text()


def test_no_em_dashes_in_report(tmp_path):
    collected, rdir, _ = make_inputs(tmp_path, blind=True)
    comp = tmp_path / "comparison.json"
    comp.write_text(json.dumps(_comparison()))
    R.build_report([collected], rdir, tmp_path / "out", comp)
    assert "—" not in (tmp_path / "out" / "report.md").read_text()
    assert "—" not in Path(R.__file__).read_text()
