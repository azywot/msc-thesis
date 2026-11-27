"""Rescue loop: one reflector fix per failed question, with a paired no-fix control.

For each attempt the orchestrator resumes from the fix's step with the fix injected, and, with
the same sampling seed, from the same step with nothing injected. The difference in rescue rate
between the two is the effect of the fix.

Every candidate gets exactly ``max_attempts`` control draws. A question already rescued before
attempt ``a`` gets no reflector call and no fix continuation at ``a``: its row (status
``"rescued_earlier"``) carries the control draw only, resumed at the rescuing fix's step.

Each attempt runs in chunks of ``chunk_size`` candidates and appends its rows after every chunk,
so a killed job loses at most one chunk; a rerun skips the (question_id, attempt) pairs on disk.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from agent_engine.datasets.evaluators.metrics import evaluate_answer
from rsi.intervention.fix import (
    FixParseError,
    build_fix_messages,
    fix_leakage_errors,
    parse_fix,
)
from rsi.intervention.records import (
    domain_of,
    set_sampling_seed,
    state_from_record,
    trajectory_messages,
)
from rsi.leakage import LeakageIndex
from rsi.reflector import is_error_item

logger = logging.getLogger(__name__)


class RescueSettings(BaseModel):
    max_attempts: int = 2
    attempt_seed_base: int = 1000
    per_call_chars: int = 1500
    input_budget_tokens: int = 14000
    chunk_size: int = Field(default=100, ge=1)


def _read_rows(path: Path) -> List[dict]:
    """Read JSONL rows. A truncated final line (job killed mid-write) is dropped and logged;
    a malformed line anywhere else is real corruption and raises."""
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            logger.warning("ignoring truncated last line of %s", path)
    return rows


def _repair_tail(path: Path) -> None:
    """Cut the file back to its last complete line so an append starts on a fresh line."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    path.write_bytes(data[: data.rfind(b"\n") + 1])


def _append_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_tail(path)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _grade(prediction: str, record: dict) -> bool:
    result = evaluate_answer(
        prediction=prediction or "",
        ground_truth=str(record["ground_truth"]),
        answer_aliases=list(record.get("aliases") or []),
    )
    return bool(result.get("correct", False))


def _previous_for_prompt(prev_row: Optional[dict]) -> Optional[dict]:
    """What the reflector is told about its previous attempt; nothing after a parse error."""
    if not prev_row or prev_row.get("status") not in ("ok", "leakage"):
        return None
    if not prev_row.get("fix_text"):
        return None
    return {
        "status": prev_row["status"],
        "fix_text": prev_row["fix_text"],
        "prediction": prev_row.get("prediction") or "",
        "step": prev_row.get("step"),
    }


def _fix_entry(rec: dict, item: dict, index: LeakageIndex, max_turns: int) -> dict:
    n_actions = len(rec["action_history"])
    text = item.get("text") or ""
    fix, reasons, status = None, [], "ok"
    if not text:
        status, reasons = "parse_error", [str(item.get("finish_reason") or "no text")]
    else:
        try:
            fix = parse_fix(text, n_actions, max_turns)
        except FixParseError as exc:
            status, reasons = "parse_error", [str(exc)]
    if fix is not None:
        answers = [str(rec["ground_truth"])] + [str(a) for a in rec.get("aliases") or []]
        reasons = fix_leakage_errors(fix, rec["question"], answers, index)
        if reasons:
            status = "leakage"
    control_step = fix.step if fix is not None else min(n_actions, max_turns - 1)
    return {
        "rec": rec,
        "text": text,
        "fix": fix,
        "status": status,
        "reasons": reasons,
        "n_actions": n_actions,
        "control_step": control_step,
    }


def _control_only_entry(rec: dict, rescuing_row: dict) -> dict:
    return {
        "rec": rec,
        "text": "",
        "fix": None,
        "status": "rescued_earlier",
        "reasons": [],
        "n_actions": len(rec["action_history"]),
        "control_step": rescuing_row["step"],
    }


def _row(e: dict, state, control_state, attempt: int, seed: int) -> dict:
    rec, fix = e["rec"], e["fix"]
    prediction = (state.answer or "") if state is not None else ""
    control_prediction = control_state.answer or ""
    return {
        "question_id": rec["question_id"],
        "data_source": rec["data_source"],
        "domain": domain_of(rec["data_source"]),
        "question": rec["question"],
        "ground_truth": rec["ground_truth"],
        "aliases": list(rec.get("aliases") or []),
        "attempt": attempt,
        "seed": seed,
        "status": e["status"],
        "reasons": e["reasons"],
        "reflector_text": e["text"],
        "step": fix.step if fix else None,
        "kind": fix.kind(e["n_actions"]) if fix else None,
        "fix_text": fix.text if fix else None,
        "correct": _grade(prediction, rec) if state is not None else False,
        "prediction": prediction,
        "messages": trajectory_messages(state) if state is not None else [],
        "control_step": e["control_step"],
        "control_correct": _grade(control_prediction, rec),
        "control_prediction": control_prediction,
    }


def _run_chunk(
    chunk: List[dict],
    *,
    attempt: int,
    seed: int,
    rescuing: Dict[object, dict],
    previous_rows: Dict[object, dict],
    orchestrator,
    providers: Sequence,
    reflector,
    index: LeakageIndex,
    settings: RescueSettings,
    max_turns: int,
    count_tokens: Optional[Callable],
    show_gold: bool,
) -> List[dict]:
    """Reflector, fix continuations and controls for one chunk; returns its rows."""
    to_fix = [rec for rec in chunk if rec["question_id"] not in rescuing]
    fix_entries: Dict[object, dict] = {}
    if to_fix:
        messages_list = [
            build_fix_messages(
                rec,
                show_gold=show_gold,
                previous=_previous_for_prompt(previous_rows.get(rec["question_id"])),
                per_call_chars=settings.per_call_chars,
                count_tokens=count_tokens,
                budget_tokens=settings.input_budget_tokens,
            )
            for rec in to_fix
        ]
        items = reflector.sample_batch(messages_list, [seed] * len(to_fix))
        if len(items) != len(messages_list):
            raise RuntimeError(
                f"reflector returned {len(items)} items for {len(messages_list)} prompts"
            )
        if len(items) > 1 and all(is_error_item(item) for item in items):
            # A failed engine call, not bad fixes: writing parse_error rows would use up the
            # attempt for every question in the chunk on a rerun. A one-prompt chunk is exempt:
            # raising there would repeat on every resubmit and the job could never finish.
            raise RuntimeError(
                f"reflector failed on all {len(items)} prompts of attempt {attempt}: "
                f"{items[0].get('finish_reason')}"
            )
        if len(items) == 1 and is_error_item(items[0]):
            logger.warning(
                "reflector error on a single-prompt chunk (attempt %d, question %r): %s",
                attempt,
                to_fix[0]["question_id"],
                items[0].get("finish_reason"),
            )
        for rec, item in zip(to_fix, items):
            fix_entries[rec["question_id"]] = _fix_entry(rec, item, index, max_turns)

    entries = [
        (
            fix_entries[rec["question_id"]]
            if rec["question_id"] in fix_entries
            else _control_only_entry(rec, rescuing[rec["question_id"]])
        )
        for rec in chunk
    ]
    ok = [e for e in entries if e["status"] == "ok"]
    fix_states = [state_from_record(e["rec"], e["fix"].step) for e in ok]
    control_states = [state_from_record(e["rec"], e["control_step"]) for e in entries]
    if fix_states:
        set_sampling_seed(providers, seed)
        fix_states = orchestrator.resume_batch(fix_states, [e["fix"].text for e in ok])
    set_sampling_seed(providers, seed)
    control_states = orchestrator.resume_batch(control_states, [None] * len(control_states))

    fix_state_of = {id(e): s for e, s in zip(ok, fix_states)}
    return [
        _row(e, fix_state_of.get(id(e)), control_state, attempt, seed)
        for e, control_state in zip(entries, control_states)
    ]


def run_rescue(
    candidates: List[dict],
    *,
    orchestrator,
    providers: Sequence,
    reflector,
    index: LeakageIndex,
    settings: RescueSettings,
    out_path: Path,
    max_turns: int,
    count_tokens: Optional[Callable] = None,
    show_gold: bool = True,
) -> List[dict]:
    out_path = Path(out_path)
    for attempt in range(1, settings.max_attempts + 1):
        rows = _read_rows(out_path)
        # question_id -> the first fix that rescued it at an earlier attempt
        rescuing: Dict[object, dict] = {}
        for r in sorted(rows, key=lambda r: r.get("attempt") or 0):
            if r.get("status") == "ok" and r.get("correct") and r.get("attempt", 0) < attempt:
                rescuing.setdefault(r["question_id"], r)
        done = {r["question_id"] for r in rows if r.get("attempt") == attempt}
        previous_rows: Dict[object, dict] = {
            r["question_id"]: r for r in rows if r.get("attempt") == attempt - 1
        }
        pending = [c for c in candidates if c["question_id"] not in done]
        seed = settings.attempt_seed_base + attempt
        for start in range(0, len(pending), settings.chunk_size):
            new_rows = _run_chunk(
                pending[start : start + settings.chunk_size],
                attempt=attempt,
                seed=seed,
                rescuing=rescuing,
                previous_rows=previous_rows,
                orchestrator=orchestrator,
                providers=providers,
                reflector=reflector,
                index=index,
                settings=settings,
                max_turns=max_turns,
                count_tokens=count_tokens,
                show_gold=show_gold,
            )
            _append_rows(out_path, new_rows)
    return _read_rows(out_path)
