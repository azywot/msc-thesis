# Failure Mode Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write `scripts/failure_modes/analyze_failure_modes.py` — a self-contained script that re-classifies every failed MAS question-run into one of six failure modes and outputs `breakdown.json` + `breakdown.csv` + a console table.

**Architecture:** Priority-rule cascade applied to each `correct=False` record; results aggregated per benchmark; outputs written via stdlib only. Hard-coded 20-run inventory resolved from `--root` arg.

**Tech Stack:** Python 3.11+, stdlib only (`json`, `csv`, `pathlib`, `collections`, `argparse`). Tests with `pytest`.

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `scripts/failure_modes/__init__.py` | Create (empty) | Makes directory importable for tests |
| `scripts/failure_modes/analyze_failure_modes.py` | Create | All logic: classify, aggregate, output |
| `tests/unit/test_analyze_failure_modes.py` | Create | Unit tests for `classify_failure` |

---

## Task 1: Create directory skeleton

**Files:**
- Create: `scripts/failure_modes/__init__.py`
- Create: `scripts/failure_modes/analyze_failure_modes.py` (skeleton only)

- [ ] **Step 1: Create the package init**

```bash
mkdir -p scripts/failure_modes
touch scripts/failure_modes/__init__.py
```

- [ ] **Step 2: Write the skeleton script**

Create `scripts/failure_modes/analyze_failure_modes.py`:

```python
"""Analyze MAS failure modes from raw_results.json files.

Classifies every failed question-run from the 20 hard-coded MAS run inventory
into one of six mutually exclusive failure modes, then outputs:
  - breakdown.json  (machine-readable counts + question IDs)
  - breakdown.csv   (flat table)
  - console table   (thesis-style pretty print)
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS = ["aime", "gaia", "gpqa", "hle", "musique"]

FAILURE_MODES = [
    "modality_tool_gap",
    "tool_loop_or_empty_final",
    "direct_reasoning_no_action",
    "computational_subgoal_error",
    "retrieval_evidence_failure",
    "single_shot_tool_trust",
]

MODE_LABELS = {
    "modality_tool_gap":         "Modality / tool-coverage gap",
    "tool_loop_or_empty_final":  "Tool loop / empty final answer",
    "direct_reasoning_no_action":"Direct reasoning, no action",
    "computational_subgoal_error":"Computational sub-goal error",
    "retrieval_evidence_failure": "Retrieval/evidence failure",
    "single_shot_tool_trust":    "Single-shot tool trust",
}

VISUAL_TOOLS = {"video_analysis", "image_inspector"}
VISUAL_KEYWORDS = {
    "image", "photo", "picture", "figure", "diagram",
    "video", "screenshot", "visual", "illustration",
}

MAX_TURNS = 15
MIN_LOOP_REPEATS = 3

# Relative to repo root: experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/
_MAS_BASE = "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow"

_INVENTORY = [
    # (benchmark, variant, relative_path)
    ("aime", "all",          f"{_MAS_BASE}/aime/qwen8B_subagent_tools_all/train_2026-03-15-21-28-24_20752251/raw_results.json"),
    ("aime", "none",         f"{_MAS_BASE}/aime/qwen8B_subagent_tools_none/train_2026-03-15-21-29-20_20752253/raw_results.json"),
    ("aime", "orchestrator", f"{_MAS_BASE}/aime/qwen8B_subagent_tools_orchestrator/train_2026-03-15-21-30-27_20752258/raw_results.json"),
    ("aime", "subagents",    f"{_MAS_BASE}/aime/qwen8B_subagent_tools_subagents/train_2026-03-15-21-31-28_20752265/raw_results.json"),
    ("gaia", "all",          f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_all/all_validation_2026-03-15-20-53-22_20752029/raw_results.json"),
    ("gaia", "none",         f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_none/all_validation_2026-03-15-20-54-46_20752030/raw_results.json"),
    ("gaia", "orchestrator", f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json"),
    ("gaia", "subagents",    f"{_MAS_BASE}/gaia/qwen8B_subagent_tools_subagents/all_validation_2026-03-15-20-56-48_20752056/raw_results.json"),
    ("gpqa", "all",          f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_all/diamond_2026-03-15-21-17-57_20752192/raw_results.json"),
    ("gpqa", "none",         f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_none/diamond_2026-03-15-21-18-51_20752195/raw_results.json"),
    ("gpqa", "orchestrator", f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_orchestrator/diamond_2026-03-15-21-19-20_20752198/raw_results.json"),
    ("gpqa", "subagents",    f"{_MAS_BASE}/gpqa/qwen8B_subagent_tools_subagents/diamond_2026-03-15-21-20-26_20752204/raw_results.json"),
    ("hle",  "all",          f"{_MAS_BASE}/hle/qwen8B_subagent_tools_all/test_subset_200_2026-03-15-21-06-19_20752116/raw_results.json"),
    ("hle",  "none",         f"{_MAS_BASE}/hle/qwen8B_subagent_tools_none/test_subset_200_2026-03-15-21-06-56_20752118/raw_results.json"),
    ("hle",  "orchestrator", f"{_MAS_BASE}/hle/qwen8B_subagent_tools_orchestrator/test_subset_200_2026-03-15-21-07-48_20752122/raw_results.json"),
    ("hle",  "subagents",    f"{_MAS_BASE}/hle/qwen8B_subagent_tools_subagents/test_subset_200_2026-03-15-21-08-57_20752132/raw_results.json"),
    ("musique", "all",          f"{_MAS_BASE}/musique/qwen8B_subagent_tools_all/validation_2026-03-15-22-08-22_20752493/raw_results.json"),
    ("musique", "none",         f"{_MAS_BASE}/musique/qwen8B_subagent_tools_none/validation_2026-03-15-22-09-22_20752495/raw_results.json"),
    ("musique", "orchestrator", f"{_MAS_BASE}/musique/qwen8B_subagent_tools_orchestrator/validation_2026-03-15-22-09-55_20752496/raw_results.json"),
    ("musique", "subagents",    f"{_MAS_BASE}/musique/qwen8B_subagent_tools_subagents/validation_2026-03-15-22-10-51_20752501/raw_results.json"),
]

# ---------------------------------------------------------------------------
# Core functions (stubs — filled in subsequent tasks)
# ---------------------------------------------------------------------------

def classify_failure(record: dict) -> str:
    raise NotImplementedError


def load_run(path: Path) -> list:
    raise NotImplementedError


def run_inventory(root: Path) -> list:
    raise NotImplementedError


def analyze(root: Path, output_dir: Path) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit skeleton**

```bash
git add scripts/failure_modes/__init__.py scripts/failure_modes/analyze_failure_modes.py
git commit -m "chore: scaffold failure_modes analysis script"
```

---

## Task 2: Implement and test `classify_failure`

**Files:**
- Modify: `scripts/failure_modes/analyze_failure_modes.py` — implement `classify_failure`
- Create: `tests/unit/test_analyze_failure_modes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_analyze_failure_modes.py`:

```python
"""Unit tests for classify_failure cascade."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import pytest
from failure_modes.analyze_failure_modes import classify_failure


def _rec(action_history=None, prediction="wrong", turns=2, tool_counts=None):
    """Build a minimal failed record."""
    return {
        "correct": False,
        "prediction": prediction,
        "turns": turns,
        "action_history": action_history or [],
        "tool_counts": tool_counts or {},
    }


def _step(tool_name, sub_goal="", result="some result"):
    return {"tool_name": tool_name, "sub_goal": sub_goal, "command": "{}", "result": result}


# ── Priority 1: modality_tool_gap ───────────────────────────────────────────

class TestModalityToolGap:
    def test_video_analysis_tool(self):
        rec = _rec(
            action_history=[_step("video_analysis", result="")],
            tool_counts={"video_analysis": 1},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_image_inspector_tool(self):
        rec = _rec(
            action_history=[_step("image_inspector", sub_goal="inspect image", result="")],
            tool_counts={"image_inspector": 1},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_multiple_text_inspector_empty_with_image_keyword(self):
        rec = _rec(
            action_history=[
                _step("text_inspector", sub_goal="inspect the attached image file", result=""),
                _step("text_inspector", sub_goal="inspect the attached image file again", result=""),
            ],
            prediction="",
            tool_counts={"text_inspector": 2},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_multiple_text_inspector_empty_with_diagram_keyword(self):
        rec = _rec(
            action_history=[
                _step("text_inspector", sub_goal="read the diagram in the file", result=""),
                _step("text_inspector", sub_goal="read the diagram in the file", result=""),
            ],
            tool_counts={"text_inspector": 2},
        )
        assert classify_failure(rec) == "modality_tool_gap"

    def test_single_text_inspector_empty_no_modality_keyword_not_gap(self):
        # Only 1 text_inspector call — not a modality gap by signal B
        rec = _rec(
            action_history=[_step("text_inspector", sub_goal="read the file", result="")],
            tool_counts={"text_inspector": 1},
        )
        assert classify_failure(rec) != "modality_tool_gap"

    def test_multiple_text_inspector_with_results_not_gap(self):
        # All have non-empty results — not a modality gap
        rec = _rec(
            action_history=[
                _step("text_inspector", sub_goal="inspect image", result="some content"),
                _step("text_inspector", sub_goal="inspect image again", result="more content"),
            ],
            tool_counts={"text_inspector": 2},
        )
        assert classify_failure(rec) != "modality_tool_gap"


# ── Priority 2: tool_loop_or_empty_final ────────────────────────────────────

class TestToolLoop:
    def test_empty_prediction(self):
        rec = _rec(
            action_history=[_step("web_search", result="info")],
            prediction="",
            tool_counts={"web_search": 1},
        )
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_whitespace_only_prediction(self):
        rec = _rec(
            action_history=[_step("web_search", result="info")],
            prediction="   ",
            tool_counts={"web_search": 1},
        )
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_turns_at_max(self):
        steps = [_step("web_search", result="info") for _ in range(14)]
        rec = _rec(action_history=steps, prediction="wrong", turns=15, tool_counts={"web_search": 14})
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_turns_above_max(self):
        steps = [_step("web_search", result="info") for _ in range(15)]
        rec = _rec(action_history=steps, prediction="wrong", turns=16, tool_counts={"web_search": 15})
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_same_tool_repeated_3_times(self):
        steps = [_step("code_generator", result="") for _ in range(3)]
        rec = _rec(action_history=steps, prediction="5", turns=4, tool_counts={"code_generator": 3})
        assert classify_failure(rec) == "tool_loop_or_empty_final"

    def test_two_repeats_not_loop(self):
        steps = [_step("code_generator", result="5"), _step("code_generator", result="5")]
        rec = _rec(action_history=steps, prediction="5", turns=3, tool_counts={"code_generator": 2})
        assert classify_failure(rec) != "tool_loop_or_empty_final"


# ── Priority 3: direct_reasoning_no_action ──────────────────────────────────

class TestDirectReasoning:
    def test_empty_action_history(self):
        rec = _rec(action_history=[], prediction="42", tool_counts={})
        assert classify_failure(rec) == "direct_reasoning_no_action"

    def test_none_action_history(self):
        rec = {
            "correct": False, "prediction": "42", "turns": 1,
            "action_history": None, "tool_counts": {},
        }
        assert classify_failure(rec) == "direct_reasoning_no_action"


# ── Priority 4: computational_subgoal_error ─────────────────────────────────

class TestComputational:
    def test_two_code_generator_calls(self):
        steps = [
            _step("code_generator", sub_goal="compute step 1", result="10"),
            _step("code_generator", sub_goal="compute step 2", result="42"),
        ]
        rec = _rec(action_history=steps, prediction="42", tool_counts={"code_generator": 2})
        assert classify_failure(rec) == "computational_subgoal_error"

    def test_one_code_generator_one_web_search(self):
        steps = [
            _step("web_search", result="some info"),
            _step("code_generator", result="42"),
        ]
        rec = _rec(action_history=steps, prediction="42", tool_counts={"web_search": 1, "code_generator": 1})
        assert classify_failure(rec) == "computational_subgoal_error"

    def test_single_code_generator_not_computational(self):
        # Only 1 action → single-shot, not computational
        rec = _rec(
            action_history=[_step("code_generator", result="42")],
            prediction="42",
            tool_counts={"code_generator": 1},
        )
        assert classify_failure(rec) != "computational_subgoal_error"


# ── Priority 5: retrieval_evidence_failure ───────────────────────────────────

class TestRetrieval:
    def test_web_search_used(self):
        steps = [
            _step("web_search", result="some info"),
            _step("web_search", result="more info"),
        ]
        rec = _rec(action_history=steps, prediction="wrong", tool_counts={"web_search": 2})
        assert classify_failure(rec) == "retrieval_evidence_failure"

    def test_single_text_inspector_with_result_is_retrieval(self):
        # Single call with a non-empty result — not modality gap, not loop, not direct,
        # not computational → falls to retrieval since text_inspector counts as retrieval
        # when web_search >= 1 rule doesn't match... actually text_inspector alone
        # with count=1 and non-empty result → single_shot_tool_trust (1 action)
        # This tests the boundary — web_search is the retrieval signal
        rec = _rec(
            action_history=[_step("web_search", result="some useful info")],
            prediction="wrong",
            tool_counts={"web_search": 1},
        )
        # Single web_search call → exactly 1 action → single_shot wins over retrieval
        # because computational (no code_gen) and retrieval (web_search=1) both
        # match, but single-shot (1 action) is lower priority... wait, let's check:
        # Priority 4: code_generator >= 1? No. Priority 5: web_search >= 1? YES.
        assert classify_failure(rec) == "retrieval_evidence_failure"


# ── Priority 6: single_shot_tool_trust ──────────────────────────────────────

class TestSingleShot:
    def test_single_code_generator_call(self):
        rec = _rec(
            action_history=[_step("code_generator", result="wrong answer")],
            prediction="wrong answer",
            tool_counts={"code_generator": 1},
        )
        assert classify_failure(rec) == "single_shot_tool_trust"

    def test_single_mind_map_call(self):
        rec = _rec(
            action_history=[_step("mind_map", result="some map")],
            prediction="wrong",
            tool_counts={"mind_map": 1},
        )
        assert classify_failure(rec) == "single_shot_tool_trust"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /path/to/repo && python -m pytest tests/unit/test_analyze_failure_modes.py -v 2>&1 | head -30
```

Expected: `NotImplementedError` or `ImportError` — all tests fail.

- [ ] **Step 3: Implement `classify_failure`**

Replace the `classify_failure` stub in `scripts/failure_modes/analyze_failure_modes.py`:

```python
def classify_failure(record: dict) -> str:
    """Classify a single failed record into one of six failure modes.

    Rules are applied in priority order; the first match wins.
    """
    action_history = record.get("action_history") or []
    prediction = str(record.get("prediction") or "")
    turns = record.get("turns") or 0
    tool_counts = record.get("tool_counts") or {}

    tools_used = [s.get("tool_name", "") for s in action_history]
    tool_counter = Counter(tools_used)

    # ── Priority 1: modality / tool-coverage gap ────────────────────────────
    # Signal A: non-existent or image-specific tool was called
    if any(t in VISUAL_TOOLS for t in tools_used):
        return "modality_tool_gap"

    # Signal B: >= 2 text_inspector calls, at least one empty result,
    #           and at least one subgoal mentions an image/visual keyword
    ti_steps = [s for s in action_history if s.get("tool_name") == "text_inspector"]
    if len(ti_steps) >= 2:
        has_empty = any(not str(s.get("result") or "").strip() for s in ti_steps)
        has_keyword = any(
            kw in (s.get("sub_goal") or "").lower()
            for s in ti_steps
            for kw in VISUAL_KEYWORDS
        )
        if has_empty and has_keyword:
            return "modality_tool_gap"

    # ── Priority 2: tool loop / empty final answer ───────────────────────────
    if not prediction.strip():
        return "tool_loop_or_empty_final"
    if turns >= MAX_TURNS:
        return "tool_loop_or_empty_final"
    if tool_counter and max(tool_counter.values()) >= MIN_LOOP_REPEATS:
        return "tool_loop_or_empty_final"

    # ── Priority 3: direct reasoning without action ──────────────────────────
    if not action_history:
        return "direct_reasoning_no_action"

    # ── Priority 4: computational sub-goal error ─────────────────────────────
    if tool_counts.get("code_generator", 0) >= 1 and len(action_history) >= 2:
        return "computational_subgoal_error"

    # ── Priority 5: retrieval / evidence failure ─────────────────────────────
    if tool_counts.get("web_search", 0) >= 1:
        return "retrieval_evidence_failure"

    # ── Priority 6: single-shot tool trust (catch-all) ───────────────────────
    return "single_shot_tool_trust"
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/unit/test_analyze_failure_modes.py -v
```

Expected: all green, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add scripts/failure_modes/analyze_failure_modes.py tests/unit/test_analyze_failure_modes.py
git commit -m "feat: implement classify_failure cascade for 6 MAS failure modes"
```

---

## Task 3: Implement `load_run` and `run_inventory`

**Files:**
- Modify: `scripts/failure_modes/analyze_failure_modes.py`

- [ ] **Step 1: Implement `load_run` and `run_inventory`**

Replace the two stubs in `analyze_failure_modes.py`:

```python
def load_run(path: Path) -> list:
    """Load records from a raw_results.json. Returns [] and warns on error."""
    if not path.exists():
        print(f"  [WARN] missing: {path}", file=sys.stderr)
        return []
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON parse error in {path}: {exc}", file=sys.stderr)
            return []


def run_inventory(root: Path) -> list:
    """Return list of (benchmark, variant, resolved_path) from the hard-coded inventory."""
    return [(bench, variant, root / rel) for bench, variant, rel in _INVENTORY]
```

- [ ] **Step 2: Quick smoke test**

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from failure_modes.analyze_failure_modes import load_run, run_inventory

root = Path(".")
inv = run_inventory(root)
print(f"Inventory entries: {len(inv)}")
bench, variant, path = inv[0]
records = load_run(path)
print(f"First entry: bench={bench} variant={variant} records={len(records)}")
EOF
```

Expected output:
```
Inventory entries: 20
First entry: bench=aime variant=all records=60
```

- [ ] **Step 3: Commit**

```bash
git add scripts/failure_modes/analyze_failure_modes.py
git commit -m "feat: implement load_run and run_inventory"
```

---

## Task 4: Implement `analyze` — aggregation and output

**Files:**
- Modify: `scripts/failure_modes/analyze_failure_modes.py`

- [ ] **Step 1: Implement `analyze`**

Replace the `analyze` stub:

```python
def analyze(root: Path, output_dir: Path) -> None:
    """Run the full analysis: classify all failures, aggregate, write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # counts[mode][benchmark] = int
    counts: dict[str, dict[str, int]] = {m: {b: 0 for b in BENCHMARKS} for m in FAILURE_MODES}
    # unique_qids[mode][benchmark] = set of question_ids
    unique_qids: dict[str, dict[str, set]] = {m: {b: set() for b in BENCHMARKS} for m in FAILURE_MODES}
    # total failures per benchmark
    bench_totals: dict[str, int] = {b: 0 for b in BENCHMARKS}

    for benchmark, variant, path in run_inventory(root):
        records = load_run(path)
        for rec in records:
            if rec.get("correct"):
                continue
            bench_totals[benchmark] += 1
            mode = classify_failure(rec)
            counts[mode][benchmark] += 1
            unique_qids[mode][benchmark].add(rec.get("question_id"))

    overall_total = sum(bench_totals.values())

    # ── Build structured result ──────────────────────────────────────────────
    result = {"failure_modes": {}, "totals": {**bench_totals, "overall": overall_total}}
    for mode in FAILURE_MODES:
        total = sum(counts[mode].values())
        share = round(total / overall_total * 100, 1) if overall_total else 0.0
        unique = sum(len(unique_qids[mode][b]) for b in BENCHMARKS)
        result["failure_modes"][mode] = {
            "label": MODE_LABELS[mode],
            "benchmarks": {
                b: {
                    "count": counts[mode][b],
                    "question_ids": sorted(unique_qids[mode][b]),
                }
                for b in BENCHMARKS
            },
            "total": total,
            "share_pct": share,
            "unique_questions": unique,
        }

    # ── Write JSON ───────────────────────────────────────────────────────────
    json_path = output_dir / "breakdown.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Written: {json_path}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = output_dir / "breakdown.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_mode", "aime", "gaia", "gpqa", "hle", "musique",
                         "total", "share_pct", "unique_questions"])
        for mode in FAILURE_MODES:
            fm = result["failure_modes"][mode]
            writer.writerow([
                mode,
                fm["benchmarks"]["aime"]["count"],
                fm["benchmarks"]["gaia"]["count"],
                fm["benchmarks"]["gpqa"]["count"],
                fm["benchmarks"]["hle"]["count"],
                fm["benchmarks"]["musique"]["count"],
                fm["total"],
                fm["share_pct"],
                fm["unique_questions"],
            ])
    print(f"Written: {csv_path}")

    # ── Console table ────────────────────────────────────────────────────────
    _print_table(result)


def _print_table(result: dict) -> None:
    """Print a thesis-style breakdown table to stdout."""
    col_w = [38, 6, 6, 6, 6, 9, 7, 7, 6]
    header = ["Failure mode", "AIME", "GAIA", "GPQA", "HLE", "MuSiQue",
              "Total", "Share", "Uniq."]
    sep = "─" * sum(col_w + [2] * len(col_w))

    def row(*vals):
        parts = [str(v).ljust(col_w[i]) if i == 0 else str(v).rjust(col_w[i]) for i, v in enumerate(vals)]
        return "  ".join(parts)

    print()
    print(row(*header))
    print(sep)
    for mode in FAILURE_MODES:
        fm = result["failure_modes"][mode]
        print(row(
            fm["label"],
            fm["benchmarks"]["aime"]["count"],
            fm["benchmarks"]["gaia"]["count"],
            fm["benchmarks"]["gpqa"]["count"],
            fm["benchmarks"]["hle"]["count"],
            fm["benchmarks"]["musique"]["count"],
            fm["total"],
            f"{fm['share_pct']}%",
            fm["unique_questions"],
        ))
    print(sep)
    t = result["totals"]
    print(row(
        "Total failures",
        t["aime"], t["gaia"], t["gpqa"], t["hle"], t["musique"],
        t["overall"], "100%", "---",
    ))
    print()
```

- [ ] **Step 2: Smoke-test aggregation**

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from failure_modes.analyze_failure_modes import analyze

analyze(Path("."), Path("/tmp/fm_test"))
EOF
```

Expected: table printed to stdout, `/tmp/fm_test/breakdown.json` and `/tmp/fm_test/breakdown.csv` created. `Total failures` row overall should be close to 2534.

- [ ] **Step 3: Commit**

```bash
git add scripts/failure_modes/analyze_failure_modes.py
git commit -m "feat: implement analyze — aggregation, JSON/CSV output, console table"
```

---

## Task 5: Implement `main` (CLI entry point)

**Files:**
- Modify: `scripts/failure_modes/analyze_failure_modes.py`

- [ ] **Step 1: Implement `main`**

Replace the `main` stub:

```python
def main() -> None:
    default_root = Path(__file__).resolve().parent.parent.parent
    default_out = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Classify MAS failures into 6 failure modes and output breakdown."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Repo root directory (default: {default_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Directory for breakdown.json and breakdown.csv (default: {default_out})",
    )
    args = parser.parse_args()
    analyze(args.root, args.output_dir)
```

- [ ] **Step 2: Run end-to-end via CLI**

```bash
python scripts/failure_modes/analyze_failure_modes.py
```

Expected: table printed, `scripts/failure_modes/breakdown.json` and `scripts/failure_modes/breakdown.csv` created.

- [ ] **Step 3: Verify output files exist and JSON is valid**

```bash
python -c "
import json
data = json.load(open('scripts/failure_modes/breakdown.json'))
modes = list(data['failure_modes'].keys())
total = data['totals']['overall']
print(f'Modes: {modes}')
print(f'Overall total failures: {total}')
assert total > 2000, f'Expected ~2534, got {total}'
print('OK')
"
```

Expected:
```
Modes: ['modality_tool_gap', 'tool_loop_or_empty_final', 'direct_reasoning_no_action', 'computational_subgoal_error', 'retrieval_evidence_failure', 'single_shot_tool_trust']
Overall total failures: <number close to 2534>
OK
```

- [ ] **Step 4: Commit final script**

```bash
git add scripts/failure_modes/analyze_failure_modes.py scripts/failure_modes/breakdown.json scripts/failure_modes/breakdown.csv
git commit -m "feat: add CLI entry point and complete failure mode analyzer"
```

---

## Self-review

**Spec coverage:**
- ✅ Hard-coded 20-run inventory
- ✅ Re-classify from raw JSON (not from .md labels)
- ✅ 6-mode cascade in correct priority order
- ✅ Modality: signal A (video_analysis, image_inspector) + signal B (≥2 text_inspector, empty result, visual keyword)
- ✅ Output: `breakdown.json` + `breakdown.csv` + console table
- ✅ `--root` and `--output-dir` CLI args
- ✅ Missing-file warning, JSON parse error warning
- ✅ Unique question count per mode (deduplicated per benchmark)

**Placeholder scan:** None found — all steps have complete code.

**Type consistency:** `classify_failure` returns `str` (mode key); `load_run` returns `list`; `run_inventory` returns `list[tuple[str, str, Path]]`; `analyze` takes two `Path` args — consistent across all tasks.
