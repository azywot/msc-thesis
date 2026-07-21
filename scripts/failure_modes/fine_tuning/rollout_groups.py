"""Data layer for the GRPO training-rollout analyses.

Loads the per-rollout JSON files written during Flow-GRPO training and
reconstructs the sampling groups that GRPO normalises advantages over. Kept
separate from the analysis scripts so the group reconstruction -- the one step
that can silently change every headline number -- has a single definition.

Source layout (written by ``src/fine_tuning/rollout.py::_save_rollout``)::

    <rollout_dir>/idx_<domain>_<n>/<k>_rollout_<hash>.json

Each rollout JSON carries ``data_source``, ``reward`` (binary 0.0/1.0 on answer
correctness) and ``output_messages``. A rollout is *tool-using* iff its
``output_messages`` contain at least one ``role == "tool"`` message (the
orchestrator dispatched a sub-agent); a lone assistant message is the
"direct reasoning, no action" mode.

Group size
----------
The group size G is the training hyperparameter ``actor_rollout_ref.rollout.n``,
NOT a constant: it is 8 in ``experiments/configs/fine_tuning/config.yaml`` (the
run behind the thesis numbers) but 2 in the smoke configs, and any future run may
differ. Hardcoding 8 would silently mis-chunk those runs, so G defaults to being
inferred from the data (:func:`infer_group_size`) and can be overridden.

Inference: each ``idx_*`` directory holds ``s x G`` rollouts, where s is the
number of optimizer steps that visited the question. The GCD of the per-directory
counts is therefore a multiple of G, and equals G as soon as any two visit counts
are coprime (in practice, as soon as one question was visited exactly once). We
take the GCD when it is >= 2 and falls back to the most common directory count
otherwise; the chosen value, the evidence, and the number of directories that are
not an exact multiple of it are all reported so a bad inference is visible rather
than silent.

Group reconstruction: within one directory the files are named
``<k>_rollout_<hash>.json`` with k = 0, 1, 2, ... A group is the G rollouts
sampled for that question at one optimizer step; a question visited at several
steps yields several groups, so we sort by k and chunk into consecutive blocks of
G. A *complete* group has exactly G rollouts; trailing partial blocks are dropped
(they are not used for advantage normalisation in a full group). Reconstruction
is validated by reproducing the report's 2,106 complete groups at G=8.

Caveat: ``k`` is assigned as ``len(existing files)`` at write time, so it records
write order, not the optimizer step, and concurrent rollout workers can assign the
same k twice (distinct uuid filenames, so nothing is overwritten). Chunking is
made deterministic by sorting on (k, filename), but a group boundary can in
principle straddle two steps. This is a property of the recorded data, not of this
code -- writing the step index into the record would remove the ambiguity.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: Training run whose rollouts back the reported reward-composition and
#: AXPO-comparison numbers.
DEFAULT_TRAIN_RUN = ROOT / "experiments/results/fine_tuning/qwen3-8b-grpo-search-math-v2"

#: Group size of the thesis run (``rollout.n`` in config.yaml). Used only as a
#: last-resort fallback; prefer :func:`infer_group_size`.
DEFAULT_GROUP_SIZE = 8

#: Report order; domains found in the data but absent here are appended.
DOMAIN_ORDER = ["deepmath", "hotpotqa", "nq"]
DOMAIN_LABELS = {
    "deepmath": "DeepMath",
    "hotpotqa": "HotpotQA",
    "nq": "Natural Questions",
    "ALL": "Overall",
}

_IDX_RE = re.compile(r"^idx_(?P<domain>.+)_\d+$")
_PREFIX_RE = re.compile(r"^(?P<k>\d+)_rollout_")


@dataclass(frozen=True)
class Rollout:
    """One sampled trajectory: was it correct, and did it call a tool."""

    correct: bool
    tool: bool


def resolve_rollout_dir(path):
    """Accept either a ``rollout_data/train`` dir or a training-run root.

    Training runs nest the rollouts under a timestamped sub-directory, so
    callers should not have to paste the timestamp. Given a run root, pick the
    newest ``<timestamp>/rollout_data/train`` under it.
    """
    path = Path(path)
    if (path / "idx_deepmath_0").exists() or any(path.glob("idx_*_*")):
        return path
    candidates = sorted(path.glob("*/rollout_data/train"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No rollout groups under {path}. Expected idx_<domain>_<n> directories, "
        f"or a training-run root containing <timestamp>/rollout_data/train."
    )


def _is_correct(reward):
    """Binary outcome reward; treat >=0.5 as correct to be robust to float noise."""
    try:
        return float(reward) >= 0.5
    except (TypeError, ValueError):
        return False


def _uses_tool(record):
    return any(m.get("role") == "tool" for m in (record.get("output_messages") or []))


def load_rollout(path):
    with open(path) as fh:
        record = json.load(fh)
    return Rollout(correct=_is_correct(record.get("reward")), tool=_uses_tool(record))


def iter_question_dirs(rollout_dir):
    """Yield ``(domain, idx_dir, [path, ...])`` with rollout files in write order.

    Sorted on (k, filename): k alone is not unique, because concurrent workers can
    assign the same write-order index (see module docstring).
    """
    for idx_dir in sorted(Path(rollout_dir).iterdir()):
        if not idx_dir.is_dir():
            continue
        match = _IDX_RE.match(idx_dir.name)
        if not match:
            continue
        numbered = []
        for f in idx_dir.glob("*.json"):
            km = _PREFIX_RE.match(f.name)
            if km:
                numbered.append((int(km.group("k")), f.name, f))
        numbered.sort()
        yield match.group("domain"), idx_dir, [f for _, _, f in numbered]


def rollout_counts(rollout_dir):
    """Number of rollout files per question directory."""
    return [len(files) for _, _, files in iter_question_dirs(rollout_dir)]


def infer_group_size(rollout_dir, fallback=DEFAULT_GROUP_SIZE):
    """Infer the GRPO group size G from the per-question rollout counts.

    Returns ``(group_size, evidence)``. See the module docstring for why the GCD
    identifies G. ``evidence`` carries the competing estimates and the number of
    directories that are not an exact multiple of the chosen G, so a questionable
    inference shows up in the report instead of silently reshaping every group.
    """
    counts = [c for c in rollout_counts(rollout_dir) if c > 0]
    evidence = {
        "n_question_dirs": len(counts),
        "total_rollouts": sum(counts),
        "count_histogram": dict(sorted(Counter(counts).items())),
        "gcd": None,
        "most_common_count": None,
        "method": "fallback",
        "dirs_not_multiple": None,
    }
    if not counts:
        evidence["chosen"] = fallback
        return fallback, evidence

    common = Counter(counts).most_common(1)[0][0]
    divisor = 0
    for c in counts:
        divisor = gcd(divisor, c)
    evidence["gcd"] = divisor
    evidence["most_common_count"] = common

    if divisor >= 2:
        chosen, method = divisor, "gcd"
    elif common >= 2:
        chosen, method = common, "most_common_count"
    else:
        chosen, method = fallback, "fallback"

    evidence["method"] = method
    evidence["chosen"] = chosen
    evidence["dirs_not_multiple"] = sum(1 for c in counts if c % chosen)
    return chosen, evidence


def iter_groups(rollout_dir, group_size):
    """Yield ``(domain, [Rollout, ...])`` for every complete group.

    ``group_size`` is required: there is no correct default, since G is a training
    hyperparameter. Callers should pass :func:`infer_group_size` or an override.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    for domain, _idx_dir, files in iter_question_dirs(rollout_dir):
        rollouts = [load_rollout(f) for f in files]
        for start in range(0, len(rollouts) - group_size + 1, group_size):
            yield domain, rollouts[start : start + group_size]


def domain_order(domains):
    """Report order: known domains first, then any extras, then ``ALL``."""
    known = [d for d in DOMAIN_ORDER if d in domains]
    extra = sorted(d for d in domains if d not in DOMAIN_ORDER and d != "ALL")
    return known + extra + ["ALL"]


def label(domain):
    return DOMAIN_LABELS.get(domain, domain)


def pct(numerator, denominator, digits=1):
    """Percentage, or ``None`` when the denominator is empty."""
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, digits)


def fmt_pct(numerator, denominator, width=5):
    value = pct(numerator, denominator)
    return f"{value:{width}.1f}%" if value is not None else " " * (width - 2) + "-- "
