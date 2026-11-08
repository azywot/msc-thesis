import random

import pandas as pd
import pytest

from rsi.data import Pools, PoolSizes, load_domain_examples, make_pools, sample_minibatch


@pytest.fixture
def tiny_parquet(tmp_path):
    rows = ["hotpotqa"] * 40 + ["nq"] * 8 + ["deepmath"] * 12
    random.Random(0).shuffle(rows)
    data = []
    for i, src in enumerate(rows):
        golden = None if src == "deepmath" else [f"ans{i}", f"alias{i}"]
        data.append(
            {
                "data_source": src,
                "question": f"question {i}",
                "result": f"ans{i}",
                "extra_info": {"idx": i, "golden_answers": golden},
            }
        )
    path = tmp_path / "combined_train.parquet"
    pd.DataFrame(data).to_parquet(path)
    return path


@pytest.fixture
def search_examples(tiny_parquet):
    return load_domain_examples(tiny_parquet, "search")


def share_of(examples, source):
    return sum(e.metadata["data_source"] == source for e in examples) / len(examples)


def share_of_ids(ids, source, examples):
    by_id = {e.question_id: e for e in examples}
    return sum(by_id[i].metadata["data_source"] == source for i in ids) / len(ids)


def test_search_domain_rows_and_aliases(tiny_parquet):
    ex = load_domain_examples(tiny_parquet, "search")
    assert {e.metadata["data_source"] for e in ex} == {"hotpotqa", "nq"}
    assert len(ex) == 48
    assert ex[0].metadata["answer_aliases"]


def test_math_domain_handles_missing_golden_answers_and_keeps_positions(tiny_parquet):
    ex = load_domain_examples(tiny_parquet, "math")
    assert len(ex) == 12
    assert all(e.metadata["answer_aliases"] == [] for e in ex)
    df = pd.read_parquet(tiny_parquet)
    for e in ex:
        assert df.iloc[e.question_id]["question"] == e.question


def test_unknown_domain_raises(tiny_parquet):
    with pytest.raises(ValueError):
        load_domain_examples(tiny_parquet, "nope")


def test_pools_are_disjoint_deterministic_and_stratified(search_examples):
    p1 = make_pools(search_examples, PoolSizes(30, 10, 5), seed=0)
    p2 = make_pools(search_examples, PoolSizes(30, 10, 5), seed=0)
    assert p1 == p2
    a, b, c = set(p1.shown), set(p1.screen), set(p1.heldout)
    assert not (a & b or a & c or b & c)
    assert [len(p1.shown), len(p1.screen), len(p1.heldout)] == [30, 10, 5]
    nq_share = share_of(search_examples, "nq")
    assert abs(share_of_ids(p1.shown, "nq", search_examples) - nq_share) < 0.05


def test_different_seed_changes_pools(search_examples):
    assert make_pools(search_examples, PoolSizes(10, 5, 5), 0) != make_pools(
        search_examples, PoolSizes(10, 5, 5), 1
    )


def test_pools_too_large_raise(search_examples):
    with pytest.raises(ValueError):
        make_pools(search_examples, PoolSizes(10_000, 1, 1), seed=0)


def test_pools_roundtrip_dict(search_examples):
    p = make_pools(search_examples, PoolSizes(10, 5, 5), seed=0)
    assert Pools.from_dict(p.to_dict()) == p


def test_sample_minibatch_without_replacement():
    pool = list(range(20))
    mb = sample_minibatch(pool, 8, random.Random(0))
    assert len(mb) == len(set(mb)) == 8 and set(mb) <= set(pool)
    assert len(sample_minibatch(pool, 50, random.Random(0))) == 20
