"""Rescue loop: one reflector fix per failed question, with a paired no-fix control.

For each attempt the orchestrator resumes from the fix's step with the fix injected, and, with
the same sampling seed, from the same step with nothing injected. The difference in rescue rate
between the two is the effect of the fix.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel

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

logger = logging.getLogger(__name__)


class RescueSettings(BaseModel):
    max_attempts: int = 2
    attempt_seed_base: int = 1000
    per_call_chars: int = 1500
    input_budget_tokens: int = 14000


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
) -> List[dict]:
    out_path = Path(out_path)
    for attempt in range(1, settings.max_attempts + 1):
        rows = _read_rows(out_path)
        rescued = {r["question_id"] for r in rows if r.get("correct")}
        done = {r["question_id"] for r in rows if r.get("attempt") == attempt}
        previous_rows: Dict[object, dict] = {
            r["question_id"]: r for r in rows if r.get("attempt") == attempt - 1
        }
        pending = [
            c
            for c in candidates
            if c["question_id"] not in rescued and c["question_id"] not in done
        ]
        if not pending:
            continue
        seed = settings.attempt_seed_base + attempt

        messages_list = []
        for rec in pending:
            prev_row = previous_rows.get(rec["question_id"])
            previous = None
            if prev_row and prev_row.get("fix_text"):
                previous = {
                    "fix_text": prev_row["fix_text"],
                    "prediction": prev_row.get("prediction") or "",
                    "step": prev_row.get("step"),
                }
            messages_list.append(
                build_fix_messages(
                    rec,
                    previous=previous,
                    per_call_chars=settings.per_call_chars,
                    count_tokens=count_tokens,
                    budget_tokens=settings.input_budget_tokens,
                )
            )
        items = reflector.sample_batch(messages_list, [seed] * len(pending))

        entries = []
        for rec, item in zip(pending, items):
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
            entries.append(
                {
                    "rec": rec,
                    "text": text,
                    "fix": fix,
                    "status": status,
                    "reasons": reasons,
                    "n_actions": n_actions,
                    "control_step": control_step,
                }
            )

        ok = [e for e in entries if e["status"] == "ok"]
        fix_states = [state_from_record(e["rec"], e["fix"].step) for e in ok]
        control_states = [state_from_record(e["rec"], e["control_step"]) for e in entries]
        if fix_states:
            set_sampling_seed(providers, seed)
            fix_states = orchestrator.resume_batch(fix_states, [e["fix"].text for e in ok])
        set_sampling_seed(providers, seed)
        control_states = orchestrator.resume_batch(control_states, [None] * len(control_states))

        fix_state_of = {id(e): s for e, s in zip(ok, fix_states)}
        new_rows = []
        for e, control_state in zip(entries, control_states):
            rec, fix = e["rec"], e["fix"]
            state = fix_state_of.get(id(e))
            prediction = (state.answer or "") if state is not None else ""
            control_prediction = control_state.answer or ""
            new_rows.append(
                {
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
            )
        _append_rows(out_path, new_rows)
    return _read_rows(out_path)
