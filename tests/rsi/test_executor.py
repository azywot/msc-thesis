import pytest

from agent_engine.core.tool import ToolRegistry
from agent_engine.datasets.base import DatasetExample
from agent_engine.models.base import ToolCallFormat
from rsi.bundle import replace_unit, seed_bundle
from rsi.executor import BundleExecutor, TrajectoryRecord, accuracy
from tests.characterization.scripted_provider import ScriptedProvider


def mk_example(qid, question, answer):
    return DatasetExample(
        question_id=qid,
        question=question,
        answer=answer,
        metadata={"data_source": "hotpotqa", "answer_aliases": []},
    )


@pytest.fixture
def seed_b(SCHEMAS):
    return seed_bundle("search", SCHEMAS)


def test_scores_and_labels(seed_b):
    # run_batch plans both questions in one generate() call, then answers both.
    prov = ScriptedProvider(["plan", "plan", "\\boxed{Paris}", "\\boxed{Rome}"])
    ex = [
        mk_example(1, "Capital of France?", "Paris"),
        mk_example(2, "Capital of Spain?", "Madrid"),
    ]
    recs = BundleExecutor(prov, ToolRegistry(), ToolCallFormat.JSON).run(seed_b, ex)
    assert [r.correct for r in recs] == [True, False]
    assert recs[0].failure_label is None and recs[1].failure_label == "direct_reasoning_no_action"
    assert recs[0].plan == "plan" and recs[1].prediction == "Rome"


def test_bundle_parts_reach_the_model(seed_b):
    from agent_engine.tools.web_search import WebSearchTool
    from tests.rsi.conftest import FakeProvider

    reg = ToolRegistry()
    reg.register(
        WebSearchTool(api_key="k", search_cache={}, url_cache={}, model_provider=FakeProvider())
    )
    b = replace_unit(
        seed_b,
        "orchestrator",
        {"guidance_opening": "You are a coordinator.", "turn0_prompt": "Plan the links."},
    )
    prov = ScriptedProvider(["plan", "\\boxed{x}"])
    BundleExecutor(prov, reg, ToolCallFormat.JSON).run(b, [mk_example(1, "Q?", "x")])
    first = prov.prompts_seen[0]
    assert "You are a coordinator." in first
    assert "Q?\n\nPlan the links." in first


def test_rollout_counter(seed_b):
    ex_ = BundleExecutor(
        ScriptedProvider(["p"] * 3 + ["\\boxed{a}"] * 3), ToolRegistry(), ToolCallFormat.JSON
    )
    ex_.run(seed_b, [mk_example(i, "q", "a") for i in range(3)])
    assert ex_.rollouts == 3


def test_record_roundtrip_and_accuracy(seed_b):
    prov = ScriptedProvider(["p", "\\boxed{a}"])
    recs = BundleExecutor(prov, ToolRegistry(), ToolCallFormat.JSON).run(
        seed_b, [mk_example(1, "q", "a")]
    )
    assert TrajectoryRecord.from_dict(recs[0].to_dict()) == recs[0]
    assert accuracy(recs) == 1.0 and accuracy([]) == 0.0
