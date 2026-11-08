"""Confirmation run: the seed and the champion on fresh questions that no pool used.

The champion is picked by its score on the held-out batch, and `beats_seed` is measured on that
same batch, so it is optimistic (winner's curse). This re-scores v0 and the champion on N
questions of the domain outside the shown, screen and held-out pools.
"""

import random
from pathlib import Path

from rsi.bundle import load_bundle
from rsi.data import Pools
from rsi.executor import TrajectoryRecord, accuracy
from rsi.search import _read_json, _write_json_atomic


def confirm_question_ids(all_ids, pools, n, pool_seed):
    """`n` ids outside every pool, a deterministic sample seeded by `pool_seed`."""
    used = set(pools.shown) | set(pools.screen) | set(pools.heldout)
    unused = sorted(set(all_ids) - used)
    if n < 1 or n > len(unused):
        raise ValueError(f"--confirm {n}: the domain has {len(unused)} unused questions")
    return sorted(random.Random(pool_seed).sample(unused, n))


def _run_version(out_dir, label, executor, examples, ids):
    """Records of bundle `label` on `ids`, reusing confirm/<label>.json when it matches."""
    path = out_dir / "confirm" / f"{label}.json"
    if path.exists():
        records = [TrajectoryRecord.from_dict(d) for d in _read_json(path)]
        if [r.question_id for r in records] == list(ids):
            return records
    bundle = load_bundle(out_dir / "bundles" / f"{label}.json")
    records = executor.run(bundle, [examples[i] for i in ids])
    _write_json_atomic(path, [r.to_dict() for r in records])
    return records


def run_confirm(out_dir, executor, examples, n):
    """Score v0 and the current champion on `n` unused questions; write confirm.json."""
    out_dir = Path(out_dir)
    state_path = out_dir / "state.json"
    if not state_path.exists():
        raise ValueError(f"{state_path} not found: run the search before --confirm")
    state = _read_json(state_path)
    pools = Pools.from_dict(_read_json(out_dir / "pools.json"))
    pool_seed = _read_json(out_dir / "config.json").get("pool_seed", 0)
    ids = confirm_question_ids(examples.keys(), pools, n, pool_seed)

    champion = state["champion"]
    seed_records = _run_version(out_dir, "v0", executor, examples, ids)
    if champion == "v0":
        champ_records = seed_records
    else:
        champ_records = _run_version(out_dir, champion, executor, examples, ids)

    seed_ok = {r.question_id: r.correct for r in seed_records}
    champ_ok = {r.question_id: r.correct for r in champ_records}
    paired = {"both": 0, "seed_only": 0, "champion_only": 0, "neither": 0}
    for q in ids:
        s, c = seed_ok[q], champ_ok[q]
        key = "both" if s and c else "seed_only" if s else "champion_only" if c else "neither"
        paired[key] += 1
    result = {
        "n": n,
        "question_ids": list(ids),
        "seed_acc": accuracy(seed_records),
        "champion_acc": accuracy(champ_records),
        "champion": champion,
        "paired": paired,
    }
    _write_json_atomic(out_dir / "confirm.json", result)
    return result
