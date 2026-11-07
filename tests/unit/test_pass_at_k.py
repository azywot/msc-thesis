import json
from math import comb

import pytest

from agent_engine.analysis.pass_at_k import (
    PassAtKInputError,
    compute_pass_at_k,
    load_run,
    pass_at_k,
    validate_runs,
)


def make_runs(
    tmp_path,
    correct,
    fingerprints=None,
    seeds=None,
    temperature=0.6,
    sampling_by_thinking=None,
    levels=None,
    dataset="gaia",
    drop=None,
    questions=None,
    model_seeds=False,
):
    """correct[q][r] -> bool for question q in run r. drop = {(q, r)} omitted."""
    n_q, n_r = len(correct), len(correct[0])
    drop = drop or set()
    runs = []
    for r in range(n_r):
        d = tmp_path / f"run{r}"
        d.mkdir()
        cfg = {
            "seed": seeds[r] if seeds else r,
            "dataset": {"name": dataset, "split": "validation", "subset_num": None},
            "models": {
                "orchestrator": {
                    "temperature": temperature,
                    "sampling_by_thinking": sampling_by_thinking,
                    **({"seed": r} if model_seeds else {}),
                }
            },
            "config_fingerprint": fingerprints[r] if fingerprints else "abc",
        }
        if fingerprints and fingerprints[r] is None:
            del cfg["config_fingerprint"]
        (d / "config.json").write_text(json.dumps(cfg))
        res = [
            {
                "question_id": q,
                "question": questions[q] if questions and r == n_r - 1 else f"q{q}",
                "prediction": f"x{r}",
                "ground_truth": "y",
                "correct": bool(correct[q][r]),
                "evaluation": {},
                "metadata": {"level": levels[q] if levels else 1},
            }
            for q in range(n_q)
            if (q, r) not in drop
        ]
        (d / "raw_results.json").write_text(json.dumps(res))
        acc = sum(x["correct"] for x in res) / len(res)
        (d / "metrics.json").write_text(json.dumps({"overall": {"accuracy": acc}}))
        runs.append(load_run(d))
    return runs


def test_estimator_known_values():
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(5, 0, 3) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 2, 3) == pytest.approx(1 - comb(3, 3) / comb(5, 3))
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def test_pass1_equals_mean_accuracy(tmp_path):
    runs = make_runs(tmp_path, correct=[[1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1]])
    out = compute_pass_at_k(runs, [1, 3, 5])
    assert out["overall"]["pass@1"] == pytest.approx(out["accuracy_mean"])
    assert out["accuracy_std"] > 0


def test_pass5_is_any_correct(tmp_path):
    runs = make_runs(tmp_path, correct=[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    out = compute_pass_at_k(runs, [5])
    assert out["overall"]["pass@5"] == pytest.approx(0.5)


def test_by_level_uses_spec_level_field(tmp_path):
    runs = make_runs(tmp_path, correct=[[1, 1], [0, 1], [0, 0]], levels=[1, 1, 2])
    out = compute_pass_at_k(runs, [1, 2], by_level=True)
    assert set(out["per_level"]) == {"1", "2"}
    assert out["per_level"]["1"]["num_questions"] == 2
    assert out["per_level"]["1"]["pass@2"] == pytest.approx(1.0)
    assert out["per_level"]["2"]["pass@2"] == 0.0


def test_config_mismatch_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], fingerprints=["a", "b"])
    with pytest.raises(PassAtKInputError, match="fingerprint"):
        validate_runs(runs)


def test_duplicate_seeds_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], seeds=[3, 3])
    with pytest.raises(PassAtKInputError, match="seed"):
        validate_runs(runs)


def test_runs_without_fingerprint_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], fingerprints=[None, None])
    with pytest.raises(PassAtKInputError, match="fingerprint"):
        validate_runs(runs)


def test_single_run_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1]])
    with pytest.raises(PassAtKInputError, match="at least 2"):
        validate_runs(runs)


def test_question_set_mismatch_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]], drop={(1, 1)})
    with pytest.raises(PassAtKInputError, match="question sets"):
        validate_runs(runs)


def test_question_text_mismatch_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]], questions=["q0", "changed"])
    with pytest.raises(PassAtKInputError, match="text differs"):
        validate_runs(runs)


def test_allow_mismatch_returns_warnings(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]], drop={(1, 1)})
    warnings = validate_runs(runs, allow_mismatch=True)
    assert warnings and "question sets" in warnings[0]


def test_greedy_run_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], temperature=0.0)
    with pytest.raises(PassAtKInputError, match="temperature 0"):
        validate_runs(runs)


def test_greedy_rejected_even_with_allow_mismatch(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], temperature=0.0)
    with pytest.raises(PassAtKInputError, match="temperature 0"):
        validate_runs(runs, allow_mismatch=True)


def test_greedy_in_either_thinking_block_rejected(tmp_path):
    sbt = {
        "thinking": {"temperature": 0.6},
        "non_thinking": {"temperature": 0.0},
    }
    runs = make_runs(tmp_path, [[1, 0]], temperature=0.7, sampling_by_thinking=sbt)
    with pytest.raises(PassAtKInputError, match="temperature 0"):
        validate_runs(runs)


def test_sampling_by_thinking_overrides_base_temperature(tmp_path):
    sbt = {"thinking": {"temperature": 0.6}, "non_thinking": {"temperature": 0.7}}
    runs = make_runs(tmp_path, [[1, 0]], temperature=0.0, sampling_by_thinking=sbt)
    assert validate_runs(runs) == []


def test_missing_raw_results_rejected(tmp_path):
    d = tmp_path / "crashed"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "raw_results.partial.json").write_text("[]")
    with pytest.raises(PassAtKInputError, match="raw_results.json"):
        load_run(d)


def test_load_run_accepts_raw_results_path(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]])
    again = load_run(runs[0].run_dir / "raw_results.json")
    assert again.run_dir == runs[0].run_dir and again.metrics is not None


def test_question_short_of_k_excluded_and_counted(tmp_path):
    correct = [[1, 0, 1, 0, 1], [0, 1, 0, 0, 1], [1, 1, 1, 1, 1]]
    runs = make_runs(tmp_path, correct, drop={(1, 4)})
    out = compute_pass_at_k(runs, [1, 3, 5])
    assert out["overall"]["excluded_short_of_k"] == {1: 0, 3: 0, 5: 1}
    assert out["overall"]["num_questions"] == 3


def test_metrics_disagreement_is_warned_not_fatal(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]])
    runs[0].metrics = {"overall": {"accuracy": 0.99}}
    out = compute_pass_at_k(runs, [1])
    assert "warning" in out["runs"][0]
    assert "warning" not in out["runs"][1]


def _rewrite(run, fn):
    path = run.run_dir / "raw_results.json"
    path.write_text(json.dumps(fn(json.loads(path.read_text()))))
    return load_run(run.run_dir)


def _rewrite_cfg(run, fn):
    path = run.run_dir / "config.json"
    path.write_text(json.dumps(fn(json.loads(path.read_text()))))
    return load_run(run.run_dir)


def test_pass1_and_pass5_on_five_runs(tmp_path):
    correct = [[1, 0, 1, 0, 1], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    out = compute_pass_at_k(make_runs(tmp_path, correct), [1, 5])
    assert out["overall"]["pass@1"] == pytest.approx(out["accuracy_mean"])
    assert out["overall"]["pass@5"] == pytest.approx(0.8)


def test_duplicate_question_id_in_run_rejected(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]])
    runs[0] = _rewrite(runs[0], lambda rows: rows + [rows[0]])
    with pytest.raises(PassAtKInputError, match="duplicate question_id"):
        validate_runs(runs, allow_mismatch=True)
    with pytest.raises(PassAtKInputError, match="duplicate question_id"):
        compute_pass_at_k(runs, [1])


@pytest.mark.parametrize("field", ["correct", "question_id"])
def test_missing_result_field_rejected(tmp_path, field):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]])
    runs[1] = _rewrite(runs[1], lambda rows: [{k: v for k, v in rows[0].items() if k != field}])
    with pytest.raises(PassAtKInputError, match=field):
        validate_runs(runs, allow_mismatch=True)
    with pytest.raises(PassAtKInputError, match=field):
        compute_pass_at_k(runs, [1])


def test_single_run_never_waivable(tmp_path):
    runs = make_runs(tmp_path, [[1]])
    with pytest.raises(PassAtKInputError, match="at least 2"):
        validate_runs(runs, allow_mismatch=True)


def test_duplicate_seeds_never_waivable(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], seeds=[3, 3])
    with pytest.raises(PassAtKInputError, match="duplicate seeds"):
        validate_runs(runs, allow_mismatch=True)


def test_duplicate_run_dir_never_waivable(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]])
    with pytest.raises(PassAtKInputError, match="same run directory"):
        validate_runs([runs[0], runs[0]], allow_mismatch=True)


def test_duplicate_model_seeds_never_waivable(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], model_seeds=True)

    def fn(cfg):
        cfg["models"]["orchestrator"]["seed"] = 7
        return cfg

    runs = [_rewrite_cfg(r, fn) for r in runs]
    with pytest.raises(PassAtKInputError, match="per-model seeds"):
        validate_runs(runs, allow_mismatch=True)


def test_distinct_model_seeds_accepted(tmp_path):
    runs = make_runs(tmp_path, [[1, 0]], model_seeds=True)
    assert validate_runs(runs) == []


def test_identical_predictions_warn(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]])
    runs = [_rewrite(r, lambda rows: [{**x, "prediction": "same"} for x in rows]) for r in runs]
    warnings = validate_runs(runs)
    assert len(warnings) == 1 and "identical predictions" in warnings[0]


def test_errored_result_reported(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]])
    runs[0] = _rewrite(runs[0], lambda rows: [{"error": "CUDA OOM"}] + rows[1:])
    with pytest.raises(PassAtKInputError, match=r"errored on result #0: CUDA OOM"):
        validate_runs(runs)


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_rejected(tmp_path, k):
    runs = make_runs(tmp_path, [[1, 0]])
    with pytest.raises(PassAtKInputError, match="at least 1"):
        compute_pass_at_k(runs, [k])


def test_single_run_std_is_none_when_computed_directly(tmp_path):
    runs = make_runs(tmp_path, [[1], [0]])[:1]
    out = compute_pass_at_k(runs, [1])
    assert out["accuracy_std"] is None
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("models", [None, {}])
def test_missing_or_empty_models_rejected(tmp_path, models):
    runs = make_runs(tmp_path, [[1, 0]])

    def fn(cfg):
        cfg.pop("models")
        if models is not None:
            cfg["models"] = models
        return cfg

    runs = [_rewrite_cfg(r, fn) for r in runs]
    with pytest.raises(PassAtKInputError, match="models"):
        validate_runs(runs, allow_mismatch=True)


def test_thinking_block_none_or_without_temperature_uses_base(tmp_path):
    sbt = {"thinking": None, "non_thinking": {"top_p": 0.8}}
    (tmp_path / "a").mkdir()
    ok = make_runs(tmp_path / "a", [[1, 0]], temperature=0.6, sampling_by_thinking=sbt)
    assert validate_runs(ok) == []
    (tmp_path / "b").mkdir()
    bad = make_runs(tmp_path / "b", [[1, 0]], temperature=0.0, sampling_by_thinking=sbt)
    with pytest.raises(PassAtKInputError, match="temperature 0"):
        validate_runs(bad, allow_mismatch=True)


def test_level_fallback_and_unknown(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1], [1, 1]], dataset="math500")

    def fn(rows):
        rows[0]["metadata"] = {"difficulty": 3, "year": 2020}
        rows[1]["metadata"] = {"year": 2021}
        rows[2]["metadata"] = {}
        return rows

    runs = [_rewrite(r, fn) for r in runs]
    out = compute_pass_at_k(runs, [1], by_level=True)
    assert set(out["per_level"]) == {"3", "2021", "unknown"}


def test_level_all_when_dataset_has_no_level_field(tmp_path):
    runs = make_runs(tmp_path, [[1, 0], [0, 1]], dataset="gpqa")
    out = compute_pass_at_k(runs, [1], by_level=True)
    assert set(out["per_level"]) == {"all"}
