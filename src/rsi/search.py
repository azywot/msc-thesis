"""The prompt search loop: elite of one, with every round written to disk so a job can resume."""

import copy
import difflib
import json
import os
import random
from pathlib import Path

from pydantic import BaseModel, Field

from rsi.bundle import (
    UNITS,
    editable_texts,
    load_bundle,
    planning_suffix,
    render_system_prompt,
    replace_unit,
    unit_text,
)
from rsi.data import PoolSizes, sample_minibatch
from rsi.executor import accuracy
from rsi.reflector import ProposalParseError, ReflectorSettings, parse_proposal
from rsi.reflector_input import HistoryEntry, ReflectorContext, build_reflector_messages
from rsi.validate import normalize_example, validate_unit

MAX_DIFF_CHARS = 2000
MAX_REJECTED_LINES = 5
REJECTED_REASON_CHARS = 200


class SearchConfig(BaseModel):
    data_path: Path
    pools: PoolSizes = PoolSizes()
    pool_seed: int = 0
    rounds: int = 10
    candidates_per_round: int = 3
    shown_minibatch: int = 16
    screen_minibatch: int = 16
    max_parse_retries: int = 2
    history_max: int = Field(8, ge=0)
    ngram: int = 8
    seed: int = 0
    output_root: Path = Path("experiments/results/rsi/phase1")
    reflector: ReflectorSettings


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _prompt_chars(bundle, tool_schemas, tool_call_format):
    system = render_system_prompt(bundle, tool_schemas, tool_call_format)
    subagents = sum(len(t) for t in bundle.subagents.values())
    return len(system) + len(planning_suffix(bundle)) + subagents


def _text_diff(old, new, old_name, new_name):
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), old_name, new_name, n=1, lineterm=""
    )
    return "\n".join(lines)


def _parts(value, path=""):
    """(path, text) for every leaf of an orchestrator dict, in document order."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _parts(item, f"{path}/{key}" if path else str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _parts(item, f"{path}/{i}")
    elif isinstance(value, str):
        yield path, value
    else:
        yield path, json.dumps(value, ensure_ascii=False)


def _unit_diff(old, new, unit):
    """Unified diff of one unit. Orchestrator edits are diffed per changed part, then capped."""
    if unit == "orchestrator":
        before, after = dict(_parts(old.orchestrator)), dict(_parts(new.orchestrator))
        paths = list(before) + [p for p in after if p not in before]
        chunks = [
            _text_diff(before.get(p, ""), after.get(p, ""), f"champion/{p}", f"candidate/{p}")
            for p in paths
            if before.get(p) != after.get(p)
        ]
        text = "\n".join(c for c in chunks if c)
    else:
        text = _text_diff(unit_text(old, unit), unit_text(new, unit), "champion", "candidate")
    if len(text) > MAX_DIFF_CHARS:
        text = text[:MAX_DIFF_CHARS] + "\n... (diff cut)"
    return text


def complete_tool_descriptions(payload, champion):
    """Fill the parts of a partial `tool_descriptions` payload from the champion's entries."""
    if not isinstance(payload, dict) or not isinstance(payload.get("tool_descriptions"), dict):
        return payload
    base = champion.orchestrator.get("tool_descriptions") or {}
    out = copy.deepcopy(payload)
    td = out["tool_descriptions"]
    for name, base_entry in base.items():
        if name not in td:
            td[name] = copy.deepcopy(base_entry)
            continue
        entry = td[name]
        if not isinstance(entry, dict):
            continue
        if "description" not in entry:
            entry["description"] = base_entry["description"]
        if "parameters" not in entry:
            entry["parameters"] = copy.deepcopy(base_entry["parameters"])
        elif isinstance(entry["parameters"], dict):
            for pname, text in base_entry["parameters"].items():
                entry["parameters"].setdefault(pname, text)
    return out


def _prepare_payload(unit, payload, champion):
    if unit == "orchestrator" and isinstance(payload, dict):
        payload = complete_tool_descriptions(payload, champion)
        if "example" in payload:
            payload = dict(payload)
            payload["example"] = normalize_example(payload["example"])
    return payload


def _failure(item):
    """Reason a sampled item must be re-sampled, or None."""
    finish = item.get("finish_reason") or ""
    if str(finish).startswith("error"):
        return f"reflector error: {finish}"
    if not (item.get("text") or "").strip():
        return "empty reflector output"
    if finish == "length":
        return "output truncated (finish_reason length)"
    return None


def _sample_proposals(cfg, reflector, messages, r):
    """Sample K slots, re-sampling failed slots. Returns one record per attempt."""
    base = cfg.seed * 100000 + r * 100
    records = []
    pending = list(range(cfg.candidates_per_round))
    for attempt in range(cfg.max_parse_retries + 1):
        if not pending:
            break
        seeds = [base + k + 10 * attempt for k in pending]
        items = list(reflector.sample(messages, seeds))
        failed = []
        for i, (k, seed) in enumerate(zip(pending, seeds)):
            item = items[i] if i < len(items) else {"text": "", "finish_reason": "error: missing"}
            rec = {
                "slot": k,
                "attempt": attempt,
                "seed": seed,
                "text": item.get("text") or "",
                "finish_reason": item.get("finish_reason"),
                "unit": None,
                "rationale": "",
                "expected_effect": "",
                "errors": [],
                "leak_errors": [],
                "screen": None,
            }
            reason = _failure(item)
            proposal = None
            if reason is None:
                try:
                    proposal = parse_proposal(rec["text"])
                except ProposalParseError as exc:
                    reason = f"parse error: {exc}"
            if proposal is not None and proposal.unit not in UNITS:
                reason = f"unknown unit {proposal.unit!r}"
            if proposal is not None:
                rec["unit"] = proposal.unit
                rec["rationale"] = proposal.rationale
                rec["expected_effect"] = proposal.expected_effect
            if reason is not None:
                rec["errors"] = [reason]
                failed.append(k)
            rec["_proposal"] = proposal
            records.append(rec)
        pending = failed
    return records


def shown_minibatch_ids(pool, size, seed, r):
    """Question ids shown in round `r` (1-based).

    One seeded permutation of the shown pool is cut into disjoint minibatches; round r takes
    the next one. When a pass over the pool is used up, the pool is permuted again (seeded by
    seed and the pass number), so the draw is the same on a resumed run.
    """
    pool = list(pool)
    if not pool:
        return []
    size = max(1, min(size, len(pool)))
    per_pass = len(pool) // size
    cycle, slot = divmod(r - 1, per_pass)
    perm = list(pool)
    random.Random(f"rsi-shown:{seed}:{cycle}").shuffle(perm)
    return perm[slot * size : (slot + 1) * size]


def _redact(reason):
    """Drop the quoted text of a leakage error, so no matched text is echoed back."""
    return reason.split(": '", 1)[0]


def _rejected_lines(out_dir, r):
    """Short "unit: first error" lines for the previous round's rejected proposals."""
    path = out_dir / "rounds" / f"round_{r - 1:02d}.json"
    if r <= 1 or not path.exists():
        return []
    lines = []
    for p in _read_json(path).get("proposals", []):
        if p.get("errors"):
            reason = str(p["errors"][0])
        elif p.get("leak_errors"):
            reason = "leakage: " + _redact(str(p["leak_errors"][0]))
        else:
            continue
        if len(reason) > REJECTED_REASON_CHARS:
            reason = reason[:REJECTED_REASON_CHARS] + "..."
        lines.append(f"{p.get('unit') or 'unparsed'}: {reason}")
        if len(lines) == MAX_REJECTED_LINES:
            break
    return lines


def _build_messages(cfg, ctx_args, history, count_tokens):
    """Reflector messages, halving the history on a budget failure. None means skipped.

    Returns (messages, stats, history_size).
    """
    n = min(cfg.history_max, len(history))
    sizes = []
    while n > 0:
        sizes.append(n)
        n //= 2
    sizes.append(0)
    for size in sizes:
        ctx = ReflectorContext(history=history[len(history) - size :] if size else [], **ctx_args)
        try:
            messages, stats = build_reflector_messages(
                ctx,
                count_tokens,
                cfg.reflector.input_budget_tokens,
                cfg.reflector.per_call_chars,
            )
            return messages, stats, size
        except ValueError:
            continue
    return None


def _run_round(
    r,
    cfg,
    domain,
    executor,
    reflector,
    examples,
    pools,
    seed,
    leakage,
    count_tokens,
    tool_schemas,
    tool_call_format,
    out_dir,
    state,
    shown_answers,
):
    rng = random.Random(cfg.seed * 1000 + r)
    champion_label = state["champion"]
    champion = load_bundle(out_dir / "bundles" / f"{champion_label}.json")
    record = {
        "round": r,
        "champion_before": champion_label,
        "champion_heldout_before": state["champion_heldout"],
        "skipped": None,
        "minibatch_question_ids": [],
        "shown_question_ids": [],
        "screen_question_ids": [],
        "reflector_stats": None,
        "proposals": [],
        "champion_screen": None,
        "chosen": None,
        "heldout": None,
        "accepted": False,
        "new_champion": None,
    }
    new_history, new_versions = [], []

    # 1. champion on a shown minibatch (disjoint from the earlier rounds' within a pass)
    ids = shown_minibatch_ids(pools.shown, cfg.shown_minibatch, cfg.seed, r)
    record["minibatch_question_ids"] = list(ids)
    trajectories = executor.run(champion, [examples[i] for i in ids])

    # 2. reflector input
    history = [HistoryEntry(**h) for h in state["history"]]
    ctx_args = dict(
        domain=domain,
        champion=champion,
        seed=seed,
        best=champion,
        trajectories=trajectories,
        rendered_system_prompt=render_system_prompt(champion, tool_schemas, tool_call_format),
        units=list(UNITS),
        rejected=_rejected_lines(out_dir, r),
    )
    built = _build_messages(cfg, ctx_args, history, count_tokens)
    if built is None:
        record["skipped"] = "prompt_over_budget"
        return record, new_history, new_versions, executor.rollouts
    messages, stats, history_size = built
    record["reflector_stats"] = stats
    # written before sampling, so a job killed mid-sample still leaves the exact input
    _write_json_atomic(
        out_dir / "rounds" / f"round_{r:02d}_reflector_input.json",
        {
            "messages": messages,
            "trajectories": [t.to_dict() for t in trajectories],
            "history_size": history_size,
        },
    )
    record["shown_question_ids"] = list(ids)
    for t in trajectories:
        shown_answers.add(t.gold)
        shown_answers.update(t.aliases)

    # 3. sample proposals
    proposals = _sample_proposals(cfg, reflector, messages, r)

    # 4. validate and leakage-check the final parsed proposal of every slot
    survivors = []
    for rec in proposals:
        proposal = rec.pop("_proposal")
        if rec["errors"] or proposal is None:
            continue
        payload = _prepare_payload(proposal.unit, copy.deepcopy(proposal.replacement), champion)
        errors = validate_unit(proposal.unit, payload, seed, tool_schemas)
        if errors:
            rec["errors"] = errors
            continue
        try:
            candidate = replace_unit(champion, proposal.unit, payload)
        except Exception as exc:
            rec["errors"] = [f"could not apply replacement: {exc}"]
            continue
        answers = sorted(shown_answers)
        cand_leaks = leakage.check(editable_texts(candidate), answers)
        base_leaks = set(leakage.check(editable_texts(champion), answers))
        rec["leak_errors"] = [e for e in cand_leaks if e not in base_leaks]
        if rec["leak_errors"]:
            continue
        survivors.append((rec, candidate, payload))
    record["proposals"] = proposals

    # 5. screen
    best = None
    if survivors:
        screen_ids = sample_minibatch(pools.screen, cfg.screen_minibatch, rng)
        record["screen_question_ids"] = list(screen_ids)
        screen_examples = [examples[i] for i in screen_ids]
        record["champion_screen"] = accuracy(executor.run(champion, screen_examples))
        for rec, candidate, _ in survivors:
            rec["screen"] = accuracy(executor.run(candidate, screen_examples))
        best = max(survivors, key=lambda s: (s[0]["screen"], -s[0]["slot"]))
        record["chosen"] = {
            "proposal_index": proposals.index(best[0]),
            "slot": best[0]["slot"],
            "unit": best[0]["unit"],
            "screen": best[0]["screen"],
        }

    # 6. held-out for the best screened candidate
    label = None
    if best is not None and best[0]["screen"] >= record["champion_screen"]:
        rec, candidate, _ = best
        label = f"v{len(state['versions'])}"
        held = executor.run(candidate, [examples[i] for i in pools.heldout])
        score = accuracy(held)
        _write_json_atomic(out_dir / "bundles" / f"{label}.json", candidate.to_dict())
        _write_json_atomic(out_dir / "heldout" / f"{label}.json", [h.to_dict() for h in held])
        record["heldout"] = score
        record["accepted"] = score > state["champion_heldout"]
        record["chosen"]["version"] = label
        if record["accepted"]:
            record["new_champion"] = label
        new_versions.append(
            {
                "label": label,
                "round": r,
                "unit": rec["unit"],
                "screen": rec["screen"],
                "heldout": score,
                "accepted": record["accepted"],
                "prompt_chars": _prompt_chars(candidate, tool_schemas, tool_call_format),
            }
        )

    for rec, candidate, _ in survivors:
        is_best = best is not None and rec is best[0]
        evaluated = is_best and label is not None
        new_history.append(
            HistoryEntry(
                label=label if evaluated else f"r{r}c{rec['slot']}",
                unit=rec["unit"],
                diff=_unit_diff(champion, candidate, rec["unit"]),
                rationale=rec["rationale"],
                screen_score=rec["screen"],
                heldout_score=record["heldout"] if evaluated else None,
                accepted=bool(evaluated and record["accepted"]),
            ).__dict__
        )
    return record, new_history, new_versions, executor.rollouts


def _initial_state(cfg, executor, seed, examples, pools, tool_schemas, tool_call_format, out_dir):
    _write_json_atomic(out_dir / "bundles" / "v0.json", seed.to_dict())
    held = executor.run(seed, [examples[i] for i in pools.heldout])
    score = accuracy(held)
    _write_json_atomic(out_dir / "heldout" / "v0.json", [h.to_dict() for h in held])
    state = {
        "rounds_done": 0,
        "champion": "v0",
        "champion_heldout": score,
        "seed_heldout": score,
        "history": [],
        "versions": [
            {
                "label": "v0",
                "round": 0,
                "unit": None,
                "screen": None,
                "heldout": score,
                "accepted": True,
                "prompt_chars": _prompt_chars(seed, tool_schemas, tool_call_format),
            }
        ],
        "rollouts": executor.rollouts,
        "beats_seed": False,
    }
    _write_json_atomic(out_dir / "state.json", state)
    return state


def _shown_answers_so_far(out_dir, examples, rounds_done):
    answers = set()
    for r in range(1, rounds_done + 1):
        path = out_dir / "rounds" / f"round_{r:02d}.json"
        if not path.exists():
            continue
        for qid in _read_json(path).get("shown_question_ids", []):
            ex = examples[qid]
            answers.add(ex.answer)
            answers.update(ex.metadata.get("answer_aliases") or [])
    return answers


def write_run_header(cfg, pools, out_dir):
    """Write config.json and pools.json, refusing to overwrite a run dir made with other settings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    cfg_dict = json.loads(cfg.model_dump_json())
    if config_path.exists():
        old = _read_json(config_path)
        ignore = {"rounds", "output_root"}
        keys = (set(old) | set(cfg_dict)) - ignore
        diff = sorted(k for k in keys if old.get(k) != cfg_dict.get(k))
        if diff:
            raise ValueError(f"{config_path} differs from the config given in: {diff}")
    _write_json_atomic(config_path, cfg_dict)
    pools_path = out_dir / "pools.json"
    if pools_path.exists() and _read_json(pools_path) != pools.to_dict():
        raise ValueError(f"{pools_path} differs from the pools given; use a fresh output dir")
    _write_json_atomic(pools_path, pools.to_dict())


def run_search(
    cfg,
    domain,
    executor,
    reflector,
    examples,
    pools,
    seed,
    leakage,
    count_tokens,
    tool_schemas,
    tool_call_format,
    out_dir,
):
    out_dir = Path(out_dir)
    write_run_header(cfg, pools, out_dir)

    state_path = out_dir / "state.json"
    if state_path.exists():
        state = _read_json(state_path)
        executor.rollouts = state["rollouts"]
        # the reflector's "Seed prompts" block must show the run's real v0
        seed = load_bundle(out_dir / "bundles" / "v0.json")
    else:
        state = _initial_state(
            cfg, executor, seed, examples, pools, tool_schemas, tool_call_format, out_dir
        )
    shown_answers = _shown_answers_so_far(out_dir, examples, state["rounds_done"])

    for r in range(state["rounds_done"] + 1, cfg.rounds + 1):
        record, new_history, new_versions, rollouts = _run_round(
            r,
            cfg,
            domain,
            executor,
            reflector,
            examples,
            pools,
            seed,
            leakage,
            count_tokens,
            tool_schemas,
            tool_call_format,
            out_dir,
            state,
            shown_answers,
        )
        _write_json_atomic(out_dir / "rounds" / f"round_{r:02d}.json", record)
        state["history"] = state["history"] + new_history
        state["versions"] = state["versions"] + new_versions
        if record["new_champion"]:
            state["champion"] = record["new_champion"]
            state["champion_heldout"] = record["heldout"]
        state["rounds_done"] = r
        state["rollouts"] = rollouts
        state["beats_seed"] = state["champion_heldout"] > state["seed_heldout"]
        _write_json_atomic(state_path, state)
    return state
