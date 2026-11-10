"""Confirmation run: the seed and the champion on fresh questions that no pool used.

The champion is picked by its score on the held-out batch, and `beats_seed` is measured on that
same batch, so it is optimistic (winner's curse). This re-scores v0 and the champion on N
questions of the domain outside the shown, screen and held-out pools.
"""

import random
from math import comb
from pathlib import Path

from rsi.bundle import is_full, load_bundle
from rsi.data import Pools
from rsi.executor import TrajectoryRecord, accuracy
from rsi.search import _read_json, _write_json_atomic
from rsi.team import HEADERS, apply_team, team_sections


def mcnemar_p(seed_only, champion_only):
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = seed_only + champion_only
    if n == 0:
        return 1.0
    k = min(seed_only, champion_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def confirm_question_ids(all_ids, pools, n, pool_seed):
    """`n` ids outside every pool, a deterministic sample seeded by `pool_seed`."""
    used = set(pools.shown) | set(pools.screen) | set(pools.heldout)
    unused = sorted(set(all_ids) - used)
    if n < 1 or n > len(unused):
        raise ValueError(f"--confirm {n}: the domain has {len(unused)} unused questions")
    return sorted(random.Random(pool_seed).sample(unused, n))


def _run_version(out_dir, label, executor, examples, ids, bundle=None):
    """Records of bundle `label` on `ids`, reusing confirm/<label>.json when it matches."""
    path = out_dir / "confirm" / f"{label}.json"
    if path.exists():
        records = [TrajectoryRecord.from_dict(d) for d in _read_json(path)]
        if [r.question_id for r in records] == list(ids):
            return records
    if bundle is None:
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
        "mcnemar_p": mcnemar_p(paired["seed_only"], paired["champion_only"]),
    }
    _write_json_atomic(out_dir / "confirm.json", result)
    return result


def run_ablation(out_dir, executor, examples):
    """Score the champion with each changed section reverted to the seed, one at a time.

    Uses the questions of confirm.json, so the numbers compare directly with the confirmation.
    Needs full-mode bundles (orchestrator_mode: full). Writes ablation.json and
    confirm/<champion>_revert_<section>.json.
    """
    out_dir = Path(out_dir)
    confirm_path = out_dir / "confirm.json"
    if not confirm_path.exists():
        raise ValueError(f"{confirm_path} not found: run --confirm before --ablate")
    confirm = _read_json(confirm_path)
    state = _read_json(out_dir / "state.json")
    champion_label = state["champion"]
    if confirm["champion"] != champion_label:
        raise ValueError(
            f"confirm.json is for {confirm['champion']}, not {champion_label}: rerun --confirm"
        )
    seed = load_bundle(out_dir / "bundles" / "v0.json")
    champion = load_bundle(out_dir / "bundles" / f"{champion_label}.json")
    if not (is_full(seed) and is_full(champion)):
        raise ValueError("--ablate needs a run with orchestrator_mode: full")
    ids = confirm["question_ids"]
    seed_secs, champ_secs = team_sections(seed), team_sections(champion)
    changed = [k for k in champ_secs if champ_secs[k] != seed_secs.get(k)]
    champ_ok = {
        r.question_id: r.correct
        for r in _run_version(out_dir, champion_label, executor, examples, ids)
    }
    variants = {}
    for key in changed:
        bundle = apply_team(champion, dict(champ_secs, **{key: seed_secs[key]}))
        records = _run_version(
            out_dir, f"{champion_label}_revert_{key}", executor, examples, ids, bundle=bundle
        )
        ok = {r.question_id: r.correct for r in records}
        lost = sum(1 for q in ids if champ_ok[q] and not ok[q])
        gained = sum(1 for q in ids if ok[q] and not champ_ok[q])
        variants[key] = {
            "section": HEADERS[key],
            "acc": accuracy(records),
            "lost_vs_champion": lost,
            "gained_vs_champion": gained,
        }
    result = {
        "n": len(ids),
        "champion": champion_label,
        "seed_acc": confirm["seed_acc"],
        "champion_acc": confirm["champion_acc"],
        "reverted": variants,
    }
    _write_json_atomic(out_dir / "ablation.json", result)
    return result
