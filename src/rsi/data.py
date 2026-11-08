"""Domain data loading and disjoint, stratified pool construction."""

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from agent_engine.datasets.base import DatasetExample

DOMAIN_SOURCES = {"search": ("hotpotqa", "nq"), "math": ("deepmath",)}


def load_domain_examples(parquet_path, domain):
    if domain not in DOMAIN_SOURCES:
        raise ValueError("unknown domain %r, expected one of %s" % (domain, sorted(DOMAIN_SOURCES)))
    sources = DOMAIN_SOURCES[domain]
    df = pd.read_parquet(Path(parquet_path))
    examples = []
    for pos, row in enumerate(df.to_dict("records")):
        if row["data_source"] not in sources:
            continue
        extra = row.get("extra_info")
        golden = extra.get("golden_answers") if isinstance(extra, dict) else None
        aliases = [str(a) for a in golden] if golden is not None else []
        examples.append(
            DatasetExample(
                question_id=pos,
                question=str(row["question"]),
                answer=str(row["result"]),
                metadata={"data_source": row["data_source"], "answer_aliases": aliases},
            )
        )
    return examples


@dataclass
class PoolSizes:
    shown: int = 300
    screen: int = 200
    heldout: int = 50


@dataclass
class Pools:
    shown: list = field(default_factory=list)
    screen: list = field(default_factory=list)
    heldout: list = field(default_factory=list)

    def to_dict(self):
        return {
            "shown": list(self.shown),
            "screen": list(self.screen),
            "heldout": list(self.heldout),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(shown=list(d["shown"]), screen=list(d["screen"]), heldout=list(d["heldout"]))


def _largest_remainder(counts, total):
    """Split `total` across keys in proportion to `counts` (dict), largest remainder rounding."""
    n = sum(counts.values())
    quotas = {k: total * v / n for k, v in counts.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    left = total - sum(alloc.values())
    order = sorted(counts, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:left]:
        alloc[k] += 1
    return alloc


def make_pools(examples, sizes, seed):
    total = sizes.shown + sizes.screen + sizes.heldout
    if min(sizes.shown, sizes.screen, sizes.heldout) < 0 or total > len(examples):
        raise ValueError("pool sizes (%d) exceed the domain (%d examples)" % (total, len(examples)))
    by_source = {}
    for e in examples:
        by_source.setdefault(e.metadata.get("data_source"), []).append(e.question_id)
    counts = {k: len(v) for k, v in by_source.items()}
    allocs = [_largest_remainder(counts, n) for n in (sizes.shown, sizes.screen, sizes.heldout)]
    rng = random.Random(seed)
    pools = ([], [], [])
    for src in sorted(by_source, key=str):
        ids = sorted(by_source[src])
        need = sum(a[src] for a in allocs)
        if need > len(ids):
            raise ValueError("source %r has %d examples but pools need %d" % (src, len(ids), need))
        rng.shuffle(ids)
        start = 0
        for pool, a in zip(pools, allocs):
            pool.extend(ids[start : start + a[src]])
            start += a[src]
    for pool in pools:
        rng.shuffle(pool)
    return Pools(shown=pools[0], screen=pools[1], heldout=pools[2])


def sample_minibatch(pool, size, rng):
    return rng.sample(list(pool), min(size, len(pool)))
