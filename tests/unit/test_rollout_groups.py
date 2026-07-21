"""Unit tests for the GRPO rollout data layer (group reconstruction + inference).

Group size is the training hyperparameter ``actor_rollout_ref.rollout.n`` (8 in
config.yaml, 2 in the smoke configs), so these tests cover inference across
several G rather than pinning the historical 8.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts/failure_modes/fine_tuning"))

import pytest
from rollout_groups import (
    domain_order,
    fmt_pct,
    infer_group_size,
    iter_groups,
    load_rollout,
    pct,
    resolve_rollout_dir,
    rollout_counts,
)


def _write_rollouts(idx_dir, outcomes):
    """outcomes: list of (reward, uses_tool) -> one rollout file each."""
    idx_dir.mkdir(parents=True, exist_ok=True)
    for k, (reward, tool) in enumerate(outcomes):
        messages = [{"role": "assistant", "content": "x"}]
        if tool:
            messages.append({"role": "tool", "content": "t"})
        (idx_dir / f"{k}_rollout_{k:04d}.json").write_text(
            json.dumps({"reward": reward, "output_messages": messages})
        )


def _build(root, spec):
    """spec: {domain: [n_rollouts_per_question, ...]} -> tree of correct rollouts."""
    for domain, counts in spec.items():
        for q, n in enumerate(counts):
            _write_rollouts(root / f"idx_{domain}_{q}", [(1.0, False)] * n)
    return root


# --------------------------------------------------------------------------- #
# rollout parsing                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "reward,expected",
    [
        (1.0, True),
        (0.0, False),
        (1, True),
        (0, False),
        (0.5, True),  # threshold is >= 0.5, for float noise
        (0.49, False),
        ("1.0", True),  # stringified rewards still parse
        (None, False),  # missing/garbage reward is not a success
        ("nonsense", False),
    ],
)
def test_reward_parsing(tmp_path, reward, expected):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"reward": reward, "output_messages": []}))
    assert load_rollout(p).correct is expected


def test_tool_use_detected_from_tool_role(tmp_path):
    tool = tmp_path / "a.json"
    tool.write_text(
        json.dumps(
            {
                "reward": 0.0,
                "output_messages": [{"role": "assistant"}, {"role": "tool"}],
            }
        )
    )
    no_tool = tmp_path / "b.json"
    no_tool.write_text(json.dumps({"reward": 0.0, "output_messages": [{"role": "assistant"}]}))
    assert load_rollout(tool).tool is True
    assert load_rollout(no_tool).tool is False


def test_missing_output_messages_is_not_tool_using(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"reward": 1.0}))
    assert load_rollout(p).tool is False


# --------------------------------------------------------------------------- #
# group size inference                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("group_size", [2, 4, 8, 16])
def test_infers_group_size_from_visit_counts(tmp_path, group_size):
    """Each question dir holds s x G rollouts; the GCD recovers G."""
    g = group_size
    _build(tmp_path, {"deepmath": [g, 2 * g, 3 * g], "nq": [g, 2 * g]})
    inferred, evidence = infer_group_size(tmp_path)
    assert inferred == group_size
    assert evidence["method"] == "gcd"
    assert evidence["dirs_not_multiple"] == 0


def test_truncated_dir_falls_back_to_most_common_count(tmp_path):
    """A partial dir collapses the GCD to 1; the modal count still recovers G."""
    _build(tmp_path, {"deepmath": [8, 16, 8, 5], "nq": [8, 8]})
    inferred, evidence = infer_group_size(tmp_path)
    assert inferred == 8
    assert evidence["gcd"] == 1
    assert evidence["method"] == "most_common_count"
    assert evidence["dirs_not_multiple"] == 1  # the 5-rollout dir


def test_uniform_visit_count_is_ambiguous(tmp_path):
    """Documented limitation: if every question was visited exactly twice, counts
    alone cannot distinguish G from 2G. The override exists for this case."""
    _build(tmp_path, {"deepmath": [16, 16], "nq": [16]})
    inferred, _ = infer_group_size(tmp_path)
    assert inferred == 16  # not the true G=8 -- caller must override


def test_empty_dir_uses_fallback(tmp_path):
    inferred, evidence = infer_group_size(tmp_path, fallback=8)
    assert inferred == 8
    assert evidence["method"] == "fallback"
    assert evidence["n_question_dirs"] == 0


def test_evidence_reports_totals(tmp_path):
    _build(tmp_path, {"deepmath": [8, 16], "nq": [8]})
    _, evidence = infer_group_size(tmp_path)
    assert evidence["n_question_dirs"] == 3
    assert evidence["total_rollouts"] == 32
    assert evidence["count_histogram"] == {8: 2, 16: 1}


# --------------------------------------------------------------------------- #
# group reconstruction                                                         #
# --------------------------------------------------------------------------- #


def test_chunks_multi_visit_dirs_into_separate_groups(tmp_path):
    _build(tmp_path, {"deepmath": [24]})  # 3 visits at G=8
    groups = list(iter_groups(tmp_path, 8))
    assert len(groups) == 3
    assert all(len(g) == 8 for _, g in groups)


def test_partial_trailing_block_dropped(tmp_path):
    _build(tmp_path, {"deepmath": [11]})  # 8 complete + 3 leftover
    groups = list(iter_groups(tmp_path, 8))
    assert len(groups) == 1


def test_dir_smaller_than_group_yields_nothing(tmp_path):
    _build(tmp_path, {"deepmath": [5]})
    assert list(iter_groups(tmp_path, 8)) == []


def test_domain_parsed_from_dir_name(tmp_path):
    _build(tmp_path, {"deepmath": [8], "hotpotqa": [8], "nq": [8]})
    assert sorted(d for d, _ in iter_groups(tmp_path, 8)) == ["deepmath", "hotpotqa", "nq"]


def test_non_group_files_and_dirs_ignored(tmp_path):
    _build(tmp_path, {"deepmath": [8]})
    (tmp_path / "README.txt").write_text("noise")
    (tmp_path / "not_a_group_dir").mkdir()
    (tmp_path / "idx_deepmath_0" / "summary.json").write_text("{}")  # no <k>_rollout_ prefix
    assert len(list(iter_groups(tmp_path, 8))) == 1
    assert rollout_counts(tmp_path) == [8]


def test_rollouts_ordered_by_index_not_filename(tmp_path):
    """Order must be numeric on k, else 10 sorts before 2 and groups get shuffled."""
    d = tmp_path / "idx_deepmath_0"
    d.mkdir(parents=True)
    for k in range(16):
        # correct only in the second block, so a mis-sort changes the group outcome
        (d / f"{k}_rollout_x.json").write_text(
            json.dumps({"reward": 1.0 if k >= 8 else 0.0, "output_messages": []})
        )
    groups = list(iter_groups(tmp_path, 8))
    assert [sum(r.correct for r in g) for _, g in groups] == [0, 8]


def test_group_size_must_be_positive(tmp_path):
    _build(tmp_path, {"deepmath": [8]})
    with pytest.raises(ValueError):
        list(iter_groups(tmp_path, 0))


# --------------------------------------------------------------------------- #
# path resolution + formatting helpers                                         #
# --------------------------------------------------------------------------- #


def test_resolves_train_dir_directly(tmp_path):
    _build(tmp_path, {"deepmath": [8]})
    assert resolve_rollout_dir(tmp_path) == tmp_path


def test_resolves_timestamped_run_root(tmp_path):
    train = tmp_path / "29-05-2026_11-36-23210365" / "rollout_data" / "train"
    _build(train, {"deepmath": [8]})
    assert resolve_rollout_dir(tmp_path) == train


def test_resolve_picks_newest_run(tmp_path):
    for stamp in ("01-01-2026_00-00-1", "29-05-2026_11-36-2"):
        _build(tmp_path / stamp / "rollout_data" / "train", {"deepmath": [8]})
    assert resolve_rollout_dir(tmp_path).parents[1].name == "29-05-2026_11-36-2"


def test_resolve_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_rollout_dir(tmp_path / "nope")


def test_domain_order_puts_known_first_extras_after_and_all_last():
    assert domain_order({"nq", "zzz", "deepmath", "ALL"}) == ["deepmath", "nq", "zzz", "ALL"]


def test_pct_and_fmt_handle_empty_denominator():
    assert pct(1, 4) == 25.0
    assert pct(1, 0) is None
    assert "25.0%" in fmt_pct(1, 4)
    assert "--" in fmt_pct(1, 0)
