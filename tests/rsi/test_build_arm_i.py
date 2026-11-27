import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    import sys

    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("build_arm_i_parquet")
import build_sft_parquet as sft  # noqa: E402

SUFFIX = "\nPLAN SUFFIX"


def _call(q):
    return (
        '<sub_goal>g</sub_goal>\n<tool_call>\n{"name": "web_search", "arguments": {"query": "%s"}}\n</tool_call>'
        % q
    )


def _traj(calls, answer="ANS \\boxed{1}"):
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "PLAN"},
    ]
    for c in calls:
        msgs += [
            {"role": "assistant", "content": c},
            {"role": "tool", "tool_name": "web_search", "content": "R"},
        ]
    msgs.append({"role": "assistant", "content": answer})
    return msgs


def _rescue(qid=1, step=1, calls=None, **kw):
    calls = calls or [_call("a"), _call("fixed")]
    row = {
        "question_id": qid,
        "data_source": "hotpotqa",
        "domain": "search",
        "status": "ok",
        "correct": True,
        "kind": "rewrite",
        "step": step,
        "fix_text": calls[step] if step < len(calls) else "x",
        "messages": _traj(calls),
        "question": f"Q{qid}",
        "ground_truth": "1",
    }
    row.update(kw)
    return row


def _arm_c():
    return pd.DataFrame(
        [
            {
                "data_source": "deepmath",
                "question": "old math",
                "result": "2",
                "extra_info": {"idx": 7},
                "messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "U"},
                    {"role": "assistant", "content": "no tools \\boxed{2}"},
                ],
            }
        ]
    )


def test_rescued_rows_start_at_fix():
    calls = [_call("a"), _call("fixed")]
    out = M.rescued_rows(_rescue(step=1, calls=calls), SUFFIX)
    assert len(out) == 2  # fixed call + answer
    assert out[0][2]["content"] == calls[1]
    assert "a" in out[0][1]["content"]  # history of the first call is in the context


def test_rescued_rows_insert_at_zero():
    calls = [_call("first")]
    out = M.rescued_rows(_rescue(step=0, calls=calls, kind="insert"), SUFFIX)
    assert len(out) == 2
    assert out[0][2]["content"] == calls[0]
    assert out[0][1]["content"] != "Q" + SUFFIX  # the planning row is dropped


def test_misaligned_trajectory_returns_none():
    assert M.rescued_rows(_rescue(fix_text="something else"), SUFFIX) is None
    assert M.rescued_rows(_rescue(step=5), SUFFIX) is None


def test_build_arm_i_appends_and_keeps_arm_c():
    arm_c = _arm_c()
    arm_i, stats = M.build_arm_i(arm_c, [_rescue()], {1: {"idx": 99}}, SUFFIX)
    assert arm_i.iloc[:1].reset_index(drop=True).equals(arm_c)
    assert len(arm_i) == 3
    assert list(arm_i.columns) == list(arm_c.columns)
    assert arm_i.iloc[1]["extra_info"] == {"idx": 99}
    assert arm_i.iloc[1]["result"] == "1"
    assert stats["arm_c"]["rows"] == 1


def test_one_trajectory_per_rescued_question():
    first = _rescue(calls=[_call("a"), _call("fixed1")], attempt=0)
    second = _rescue(calls=[_call("a"), _call("fixed2")], attempt=1)
    arm_i, stats = M.build_arm_i(_arm_c(), [first, second], {}, SUFFIX)
    assert stats["arm_i"]["rescued_questions"] == 1
    assert "fixed1" in arm_i.iloc[1]["messages"][2]["content"]


def test_overlap_with_arm_c_raises():
    with pytest.raises(ValueError):
        M.build_arm_i(_arm_c(), [_rescue(question="old math")], {}, SUFFIX)
    with pytest.raises(ValueError):
        M.build_arm_i(_arm_c(), [_rescue()], {1: {"idx": 7}}, SUFFIX)


def test_stats_counts(monkeypatch):
    rows = [
        _rescue(qid=1),
        _rescue(
            qid=2, kind="insert", step=0, calls=[_call("x")], domain="math", data_source="deepmath"
        ),
        _rescue(qid=3, fix_text="bad"),
        _rescue(qid=4, status="parse_error"),
        _rescue(qid=5, correct=False),
    ]
    arm_i, stats = M.build_arm_i(_arm_c(), rows, {}, SUFFIX)
    s = stats["arm_i"]
    assert s["rescued_questions"] == 2
    assert s["rescued_rows"] == 4
    assert s["by_kind"] == {"rewrite": 1, "insert": 1}
    assert s["by_domain"] == {"search": 1, "math": 1}
    assert s["misaligned"] == 1
    assert s["no_tool_rows"] == 3  # arm C row + two answers
    assert s["math_rows"] == 3 and s["search_rows"] == 2
    assert s["rows"] == 5 and s["questions"] == 3


def test_over_length_applies_to_new_rows_only(monkeypatch):
    seen = {}

    def fake(folded, max_length, tok):
        seen["n"] = len(folded)
        return folded[:1]

    monkeypatch.setattr(sft, "_drop_over_length", fake)
    arm_i, stats = M.build_arm_i(_arm_c(), [_rescue()], {}, SUFFIX, 100, "tok")
    assert seen["n"] == 2
    assert len(arm_i) == 2
    assert stats["arm_i"]["dropped_over_length"] == 1


def test_cli_copies_val_and_writes_stats(tmp_path, monkeypatch):
    arm_c_dir = tmp_path / "c"
    arm_c_dir.mkdir()
    _arm_c().to_parquet(arm_c_dir / M.ARM_C_TRAIN, index=False)
    _arm_c().to_parquet(arm_c_dir / M.ARM_C_VAL, index=False)
    rescue = tmp_path / "r.jsonl"
    rescue.write_text(json.dumps(_rescue()) + "\n")
    ref = tmp_path / "ref.parquet"
    pd.DataFrame([{"extra_info": {"idx": 0}}, {"extra_info": {"idx": 1}}]).to_parquet(ref)
    out = tmp_path / "o"
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "--arm-c-dir",
            str(arm_c_dir),
            "--rescue",
            str(rescue),
            "--reference-parquet",
            str(ref),
            "--output-dir",
            str(out),
            "--planning-suffix",
            SUFFIX,
        ],
    )
    M.main()
    assert (out / "arm_i_folded_val.parquet").read_bytes() == (arm_c_dir / M.ARM_C_VAL).read_bytes()
    assert json.loads((out / "arm_stats.json").read_text())["arm_i"]["rescued_rows"] == 2
    assert len(pd.read_parquet(out / "arm_i_folded_train.parquet")) == 3
