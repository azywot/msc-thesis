"""Unit tests for the all-wrong / all-correct group analysis.

The two senses of "all-wrong" are easy to conflate, so they are tested against
each other explicitly: a group can be mixed at the whole-group level while its
tool-using subgroup is entirely wrong.
"""

import json
from pathlib import Path

import pytest
from agent_engine.analysis.fine_tuning.all_wrong import collect, latex_composition, with_rates
from agent_engine.analysis.fine_tuning.rollout_groups import domain_order


def _group(root, domain, question, outcomes):
    """outcomes: list of (correct, uses_tool) -> one group's worth of rollouts."""
    d = root / f"idx_{domain}_{question}"
    d.mkdir(parents=True, exist_ok=True)
    start = len(list(d.glob("*.json")))
    for i, (correct, tool) in enumerate(outcomes):
        messages = [{"role": "assistant"}]
        if tool:
            messages.append({"role": "tool"})
        (d / f"{start + i}_rollout_x.json").write_text(
            json.dumps({"reward": 1.0 if correct else 0.0, "output_messages": messages})
        )


# --------------------------------------------------------------------------- #
# whole-group composition                                                      #
# --------------------------------------------------------------------------- #


def test_all_wrong_group(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    stats = collect(tmp_path, 4)
    assert (stats["ALL"]["all_wrong"], stats["ALL"]["all_correct"], stats["ALL"]["mixed"]) == (1, 0, 0)


def test_all_correct_group(tmp_path):
    _group(tmp_path, "deepmath", 0, [(True, False)] * 4)
    stats = collect(tmp_path, 4)
    assert (stats["ALL"]["all_wrong"], stats["ALL"]["all_correct"], stats["ALL"]["mixed"]) == (0, 1, 0)


def test_mixed_group(tmp_path):
    _group(tmp_path, "deepmath", 0, [(True, False), (False, False), (False, False), (False, False)])
    stats = collect(tmp_path, 4)
    assert (stats["ALL"]["all_wrong"], stats["ALL"]["all_correct"], stats["ALL"]["mixed"]) == (0, 0, 1)


def test_single_wrong_rollout_makes_group_mixed_not_all_correct(tmp_path):
    """Boundary: all-correct requires *every* rollout correct."""
    _group(tmp_path, "deepmath", 0, [(True, False)] * 3 + [(False, False)])
    assert collect(tmp_path, 4)["ALL"]["mixed"] == 1


def test_composition_partitions_all_groups(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    _group(tmp_path, "deepmath", 1, [(True, False)] * 4)
    _group(tmp_path, "nq", 0, [(True, False), (False, False), (False, False), (False, False)])
    s = collect(tmp_path, 4)["ALL"]
    assert s["all_wrong"] + s["all_correct"] + s["mixed"] == s["n_groups"] == 3


# --------------------------------------------------------------------------- #
# AXPO tool-using subgroup -- the *other* definition of all-wrong              #
# --------------------------------------------------------------------------- #


def test_mixed_group_can_have_all_wrong_tool_subgroup(tmp_path):
    """The distinction the analysis exists to make: the group carries a learning
    signal overall, yet every tool-using rollout failed."""
    _group(
        tmp_path,
        "deepmath",
        0,
        [(True, False), (True, False), (False, True), (False, True)],
    )
    s = collect(tmp_path, 4)["ALL"]
    assert s["mixed"] == 1  # whole-group: mixed
    assert s["tool_subgroup_all_wrong"] == 1  # tool-using subgroup: all wrong
    assert s["notool_subgroup_all_wrong"] == 0


def test_tool_subgroup_not_all_wrong_when_one_tool_rollout_succeeds(tmp_path):
    _group(tmp_path, "deepmath", 0, [(True, True), (False, True), (False, False), (False, False)])
    s = collect(tmp_path, 4)["ALL"]
    assert s["groups_with_tool_subgroup"] == 1
    assert s["tool_subgroup_all_wrong"] == 0


def test_group_without_tool_rollouts_excluded_from_tool_denominator(tmp_path):
    """Denominator is groups with a non-empty tool-using subgroup, not all groups."""
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    s = collect(tmp_path, 4)["ALL"]
    assert s["n_groups"] == 1
    assert s["groups_with_tool_subgroup"] == 0
    assert s["groups_with_notool_subgroup"] == 1
    assert s["notool_subgroup_all_wrong"] == 1


def test_all_tool_group_excluded_from_notool_denominator(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, True)] * 4)
    s = collect(tmp_path, 4)["ALL"]
    assert s["groups_with_notool_subgroup"] == 0
    assert s["groups_with_tool_subgroup"] == 1


def test_tool_use_rate_counts_rollouts_not_groups(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, True), (False, True), (False, False), (False, False)])
    s = collect(tmp_path, 4)["ALL"]
    assert s["n_rollouts"] == 4
    assert s["n_tool_rollouts"] == 2


# --------------------------------------------------------------------------- #
# aggregation across domains                                                   #
# --------------------------------------------------------------------------- #


def test_domains_aggregate_into_all(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    _group(tmp_path, "nq", 0, [(True, False)] * 4)
    stats = collect(tmp_path, 4)
    assert stats["deepmath"]["n_groups"] == 1
    assert stats["nq"]["n_groups"] == 1
    assert stats["ALL"]["n_groups"] == 2
    assert stats["ALL"]["all_wrong"] == 1
    assert stats["ALL"]["all_correct"] == 1


def test_multiple_visits_counted_as_separate_groups(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    _group(tmp_path, "deepmath", 0, [(True, False)] * 4)  # same question, second visit
    s = collect(tmp_path, 4)["ALL"]
    assert s["n_groups"] == 2
    assert s["all_wrong"] == 1 and s["all_correct"] == 1


# --------------------------------------------------------------------------- #
# derived rates + LaTeX                                                        #
# --------------------------------------------------------------------------- #


def test_with_rates_computes_percentages(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    _group(tmp_path, "deepmath", 1, [(True, False)] * 4)
    _group(tmp_path, "deepmath", 2, [(True, False), (False, False), (False, False), (False, False)])
    _group(tmp_path, "deepmath", 3, [(True, False), (True, False), (False, False), (False, False)])
    r = with_rates(collect(tmp_path, 4))["ALL"]
    assert r["all_wrong_pct"] == 25.0
    assert r["all_correct_pct"] == 25.0
    assert r["mixed_pct"] == 50.0
    assert r["dead_pct"] == 50.0  # all-wrong + all-correct


def test_rates_are_none_when_denominator_empty(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)  # no tool rollouts at all
    r = with_rates(collect(tmp_path, 4))["ALL"]
    assert r["tool_subgroup_all_wrong_pct"] is None
    assert r["tool_use_rate_pct"] == 0.0


def test_latex_table_is_well_formed(tmp_path):
    _group(tmp_path, "deepmath", 0, [(False, False)] * 4)
    _group(tmp_path, "nq", 0, [(True, False)] * 4)
    stats = collect(tmp_path, 4)
    out = latex_composition(stats, domain_order(stats), 4)
    assert out.count("\\begin{tabular}") == out.count("\\end{tabular}") == 1
    assert "DeepMath" in out and "Natural Questions" in out and "Overall" in out
    assert "4-rollout groups" in out  # caption reflects the actual group size
    assert out.rstrip().endswith("\\end{table}")


def test_latex_caption_reports_actual_group_count(tmp_path):
    for q in range(3):
        _group(tmp_path, "deepmath", q, [(True, False)] * 8)
    stats = collect(tmp_path, 8)
    assert "over all 3 complete 8-rollout groups" in latex_composition(stats, domain_order(stats), 8)
