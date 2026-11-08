"""Markdown report of a prompt search run, built from the files on disk alone."""

import difflib
import json
from pathlib import Path

from agent_engine.prompts.templating import find_placeholders, render_placeholders
from rsi.bundle import (
    UNITS,
    acting_suffix,
    load_bundle,
    planning_suffix,
    render_system_prompt,
    unit_text,
)
from rsi.executor import TrajectoryRecord

MAX_PAIRED = 3
CELL_CHARS = 700


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fence(text, lang=""):
    """Fence `text`, using more backticks than any run inside it."""
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _cut(text, n=CELL_CHARS):
    text = str(text or "")
    return text if len(text) <= n else text[:n] + f"... (cut, {len(text)} chars)"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _rounds(run_dir, rounds_done):
    out = []
    for r in range(1, rounds_done + 1):
        path = run_dir / "rounds" / f"round_{r:02d}.json"
        if path.exists():
            out.append(_read(path))
    return out


def _round_rollouts(rec, n_heldout):
    """Rollouts one round used: shown batch, champion + survivors on screen, optional held-out."""
    n = len(rec.get("minibatch_question_ids") or [])
    screened = [p for p in rec.get("proposals", []) if p.get("screen") is not None]
    if screened:
        n += (len(screened) + 1) * len(rec.get("screen_question_ids") or [])
    if rec.get("heldout") is not None:
        n += n_heldout
    return n


def _classify(p):
    """Rejection reason of a proposal, or None when it survived validation."""
    errors = p.get("errors") or []
    if errors:
        first = errors[0]
        sampling = ("parse error", "reflector error", "empty reflector", "output truncated")
        if first.startswith(sampling) or first.startswith("unknown unit"):
            return "parse error"
        return "validation error"
    if p.get("leak_errors"):
        return "leakage"
    return None


def _summary(state, config, seed_bundle, rounds):
    reflector = (config.get("reflector") or {}).get("model", "unknown")
    skipped = sum(1 for r in rounds if r.get("skipped"))
    lines = [
        "## Summary",
        "",
        f"- Domain: {seed_bundle.domain}",
        f"- Reflector model: {reflector}",
        f"- Rounds done: {state['rounds_done']} (of {config.get('rounds', 'unknown')} configured;"
        f" {skipped} skipped)",
        f"- Rollouts: {state['rollouts']}",
        f"- Seed held-out accuracy: {_pct(state['seed_heldout'])}",
        f"- Champion: {state['champion']}, held-out accuracy {_pct(state['champion_heldout'])}",
        f"- Beats seed: {state['beats_seed']}",
        "",
    ]
    return lines


def _curve(state, rounds, n_heldout):
    chars = {v["label"]: v["prompt_chars"] for v in state["versions"]}
    lines = [
        "## Search curve",
        "",
        "Rollouts are derived from the round records (shown batch, screening, held-out).",
        "",
        "| round | rollouts so far | champion held-out | candidate held-out | accepted |"
        " champion prompt_chars |",
        "|---|---|---|---|---|---|",
        f"| 0 | {n_heldout} | {_pct(state['seed_heldout'])} | n/a | seed | {chars.get('v0', 'n/a')} |",
    ]
    total = n_heldout
    champ_held = state["seed_heldout"]
    champ = "v0"
    for rec in rounds:
        total += _round_rollouts(rec, n_heldout)
        if rec.get("accepted") and rec.get("new_champion"):
            champ, champ_held = rec["new_champion"], rec["heldout"]
        if rec.get("skipped"):
            accepted = f"skipped ({rec['skipped']})"
        else:
            accepted = "yes" if rec.get("accepted") else "no"
        lines.append(
            f"| {rec['round']} | {total} | {_pct(champ_held)} | {_pct(rec.get('heldout'))} |"
            f" {accepted} | {chars.get(champ, 'n/a')} |"
        )
    lines.append("")
    return lines


def _rejections(rounds):
    counts = {"parse error": 0, "validation error": 0, "leakage": 0, "screened out": 0}
    evaluated = 0
    for rec in rounds:
        chosen = rec.get("chosen") or {}
        for p in rec.get("proposals", []):
            reason = _classify(p)
            if reason:
                counts[reason] += 1
            elif p.get("screen") is not None:
                if (
                    chosen.get("version")
                    and chosen.get("slot") == p.get("slot")
                    and (p.get("screen") == chosen.get("screen"))
                ):
                    evaluated += 1
                else:
                    counts["screened out"] += 1
    lines = ["## Rejections", "", "| reason | proposals |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    lines += ["", f"Proposals that reached the held-out set: {evaluated}.", ""]
    return lines


def _units_edited(rounds):
    proposed = {u: 0 for u in UNITS}
    accepted = {u: 0 for u in UNITS}
    for rec in rounds:
        for p in rec.get("proposals", []):
            if p.get("unit") in proposed:
                proposed[p["unit"]] += 1
        if rec.get("accepted") and (rec.get("chosen") or {}).get("unit") in accepted:
            accepted[rec["chosen"]["unit"]] += 1
    lines = ["## Units edited", "", "| unit | proposals | accepted |", "|---|---|---|"]
    lines += [f"| {u} | {proposed[u]} | {accepted[u]} |" for u in UNITS]
    lines.append("")
    return lines


def _diffs(seed_bundle, champ_bundle, champion):
    lines = ["## Final prompt versus seed", "", f"Champion: {champion}.", ""]
    for unit in UNITS:
        old, new = unit_text(seed_bundle, unit), unit_text(champ_bundle, unit)
        lines += [f"### {unit}", ""]
        if old == new:
            lines += ["unchanged", ""]
            continue
        diff = difflib.unified_diff(
            old.splitlines(), new.splitlines(), "seed", champion, n=2, lineterm=""
        )
        lines += [_fence("\n".join(diff), "diff"), ""]
    return lines


def _rendered(champ_bundle, tool_schemas, tool_call_format):
    lines = ["## Rendered prompts", "", "All rendered under the champion bundle.", ""]
    system = render_system_prompt(champ_bundle, tool_schemas, tool_call_format)
    lines += ["### System prompt", "", _fence(system), ""]
    lines += [
        "### Planning user turn",
        "",
        _fence("<question>" + planning_suffix(champ_bundle)),
        "",
    ]
    memory = (
        "<question>\n\n**Query Analysis:**\n<query analysis>"
        "\n\n**Previous Steps:**\n<previous steps>\n" + acting_suffix(champ_bundle)
    )
    lines += ["### Acting user turn (placeholder memory block)", "", _fence(memory), ""]
    for name in sorted(champ_bundle.subagents):
        template = champ_bundle.subagents[name]
        values = {p: f"<{p}>" for p in find_placeholders(template)}
        lines += [
            f"### Sub-agent prompt: {name}",
            "",
            _fence(render_placeholders(template, values)),
        ]
        lines.append("")
    return lines


def _trajectory(label, t):
    lines = [
        f"**{label}**: prediction `{_cut(t.prediction, 200)}`, correct: {t.correct}"
        f" (failure label: {t.failure_label}), turns: {t.turns}",
        "",
        "Plan:",
        "",
        _fence(_cut(t.plan, 1500)),
        "",
    ]
    if t.action_history:
        lines.append("Actions:")
        lines.append("")
        for i, a in enumerate(t.action_history, 1):
            lines.append(
                f"{i}. {a.get('tool_name', '?')}: sub-goal {_cut(a.get('sub_goal'), 200)!r};"
                f" command {_cut(a.get('command'), 300)!r}; result {_cut(a.get('result'), 300)!r}"
            )
        lines.append("")
    else:
        lines += ["Actions: none", ""]
    return lines


def _paired(run_dir, champion):
    lines = ["## Paired trajectories", ""]
    seed_path, champ_path = (
        run_dir / "heldout" / "v0.json",
        run_dir / "heldout" / f"{champion}.json",
    )
    if champion == "v0":
        return lines + ["The champion is still the seed, so there are no differing questions.", ""]
    if not (seed_path.exists() and champ_path.exists()):
        return lines + ["Held-out trajectories are missing on disk.", ""]
    seed = {d["question_id"]: TrajectoryRecord.from_dict(d) for d in _read(seed_path)}
    champ = {d["question_id"]: TrajectoryRecord.from_dict(d) for d in _read(champ_path)}
    differ = [q for q in champ if q in seed and seed[q].correct != champ[q].correct]
    if not differ:
        return lines + ["No held-out question has a different outcome under seed and champion.", ""]
    lines += [
        f"{len(differ)} held-out questions differ; showing {min(MAX_PAIRED, len(differ))}.",
        "",
    ]
    for q in differ[:MAX_PAIRED]:
        s, c = seed[q], champ[q]
        lines += [f"### Question {q}", "", f"Question: {_cut(s.question, 1000)}", ""]
        lines += [f"Gold: `{s.gold}`", ""]
        lines += _trajectory("Seed (v0)", s) + _trajectory(f"Champion ({champion})", c)
    return lines


def write_report(run_dir, tool_schemas, tool_call_format):
    """Write run_dir/report.md from the files on disk and return its path."""
    run_dir = Path(run_dir)
    state = _read(run_dir / "state.json")
    config = _read(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    pools = _read(run_dir / "pools.json") if (run_dir / "pools.json").exists() else {}
    n_heldout = len(pools.get("heldout", []))
    seed_bundle = load_bundle(run_dir / "bundles" / "v0.json")
    champion = state["champion"]
    champ_bundle = load_bundle(run_dir / "bundles" / f"{champion}.json")
    rounds = _rounds(run_dir, state["rounds_done"])

    lines = [f"# Prompt search report: {run_dir.name}", ""]
    lines += _summary(state, config, seed_bundle, rounds)
    lines += _curve(state, rounds, n_heldout)
    lines += _rejections(rounds)
    lines += _units_edited(rounds)
    lines += _diffs(seed_bundle, champ_bundle, champion)
    lines += _rendered(champ_bundle, tool_schemas, tool_call_format)
    lines += _paired(run_dir, champion)
    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
