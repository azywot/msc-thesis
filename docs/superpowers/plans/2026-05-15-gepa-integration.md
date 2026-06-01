# GEPA Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate GEPA prompt optimisation into msc-thesis so that the orchestrator's system prompt and planning suffix are automatically evolved from execution traces, with failure-stratified training data and a held-out test set drawn from existing Qwen3-8B results.

**Architecture:** A new `src/gepa_integration/` module holds `seed.py` (candidate construction + split generation) and `adapter.py` (the `GEPAAdapter` implementation). A single `scripts/run_gepa.py` CLI drives four modes: `splits`, `optimize`, `evaluate`, `diff`. Two small changes to existing files: `ExecutionState` gains a `raw_query_analysis` field and `AgenticOrchestrator` gains a configurable `planning_suffix` parameter.

**Tech Stack:** Python 3.11, gepa (local path install), pydantic, pytest, difflib, existing agent_engine internals (vLLM, PromptBuilder, evaluate_answer, ToolRegistry).

---

## File Map

**Modified:**
- `src/agent_engine/core/state.py:59` — add `raw_query_analysis: Optional[str] = None`
- `src/agent_engine/core/orchestrator.py` — extract two planning-suffix constants, add `planning_suffix` param, store raw planning output

**Created:**
- `tests/gepa_integration/__init__.py`
- `tests/gepa_integration/test_state.py`
- `tests/gepa_integration/test_orchestrator.py`
- `tests/gepa_integration/test_seed.py`
- `tests/gepa_integration/test_adapter.py`
- `src/gepa_integration/__init__.py`
- `src/gepa_integration/seed.py` — `build_seed_candidate()`, `build_splits()`
- `src/gepa_integration/adapter.py` — `AgentGEPAAdapter`
- `scripts/run_gepa.py` — CLI: splits / optimize / evaluate / diff
- `experiments/configs/gepa/gaia.yaml`
- `experiments/configs/gepa/gpqa.yaml`

---

## Task 1: Install GEPA and create tests directory

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`, `tests/gepa_integration/__init__.py`

- [ ] **Step 1: Install gepa as editable path dependency**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
pip install -e ../gepa/
```

Expected output: `Successfully installed gepa-...`

- [ ] **Step 2: Verify gepa imports**

```bash
python -c "from gepa.api import optimize; from gepa.core.adapter import GEPAAdapter, EvaluationBatch; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Create test directories**

```bash
mkdir -p tests/gepa_integration
touch tests/__init__.py tests/gepa_integration/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/gepa_integration/__init__.py
git commit -m "chore: add tests/ directory and gepa dependency"
```

---

## Task 2: Add `raw_query_analysis` to `ExecutionState`

**Files:**
- Modify: `src/agent_engine/core/state.py`
- Create: `tests/gepa_integration/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gepa_integration/test_state.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.state import ExecutionState


def test_raw_query_analysis_defaults_to_none():
    state = ExecutionState(question_id=1, question="test question")
    assert state.raw_query_analysis is None


def test_raw_query_analysis_can_be_set():
    state = ExecutionState(question_id=1, question="test question")
    state.raw_query_analysis = "<think>internal reasoning</think>visible analysis"
    assert state.raw_query_analysis == "<think>internal reasoning</think>visible analysis"


def test_raw_query_analysis_independent_of_query_analysis():
    state = ExecutionState(question_id=1, question="test question")
    state.query_analysis = "stripped"
    state.raw_query_analysis = "<think>full</think>stripped"
    assert state.query_analysis == "stripped"
    assert state.raw_query_analysis == "<think>full</think>stripped"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/gepa_integration/test_state.py -v
```

Expected: `FAILED` — `ExecutionState` has no `raw_query_analysis` attribute.

- [ ] **Step 3: Add field to `ExecutionState`**

In `src/agent_engine/core/state.py`, after line 59 (`query_analysis: str = ""`):

```python
    # Structured memory (AgentFlow-inspired)
    query_analysis: str = ""
    raw_query_analysis: Optional[str] = None   # full planning output incl. <think> blocks
    action_history: List[Dict[str, str]] = []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/gepa_integration/test_state.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agent_engine/core/state.py tests/gepa_integration/test_state.py
git commit -m "feat: add raw_query_analysis field to ExecutionState"
```

---

## Task 3: Refactor orchestrator planning turn

**Files:**
- Modify: `src/agent_engine/core/orchestrator.py`
- Create: `tests/gepa_integration/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/gepa_integration/test_orchestrator.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.orchestrator import (
    AgenticOrchestrator,
    _DEFAULT_PLANNING_SUFFIX_NO_TOOLS,
    _DEFAULT_PLANNING_SUFFIX_TOOLS,
)
from agent_engine.core.tool import ToolRegistry


def _make_orchestrator(planning_suffix=None, with_tools=True):
    model = MagicMock()
    model.config = MagicMock()
    model.config.supports_thinking = False
    model.config.family = "qwen3"
    tools = ToolRegistry()
    if with_tools:
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        mock_tool.get_schema.return_value = {"function": {"name": "web_search"}}
        tools.register(mock_tool)
    return AgenticOrchestrator(
        model_provider=model,
        tool_registry=tools,
        planning_suffix=planning_suffix,
    )


def test_default_planning_suffix_constants_exist():
    assert isinstance(_DEFAULT_PLANNING_SUFFIX_NO_TOOLS, str)
    assert isinstance(_DEFAULT_PLANNING_SUFFIX_TOOLS, str)
    assert "tools" in _DEFAULT_PLANNING_SUFFIX_TOOLS.lower()
    assert len(_DEFAULT_PLANNING_SUFFIX_NO_TOOLS) > 20


def test_orchestrator_stores_planning_suffix():
    orch = _make_orchestrator(planning_suffix="custom suffix")
    assert orch.planning_suffix == "custom suffix"


def test_orchestrator_planning_suffix_defaults_none():
    orch = _make_orchestrator(planning_suffix=None)
    assert orch.planning_suffix is None


def test_raw_query_analysis_stored_on_state():
    """planning turn stores raw output (with thinking) before stripping."""
    orch = _make_orchestrator()

    raw_text = "<think>internal</think>analysis"
    gen_result = MagicMock()
    gen_result.text = raw_text
    gen_result.usage = {}
    orch.model.generate.return_value = [gen_result]
    orch.model.apply_chat_template.return_value = "prompt"

    from agent_engine.core.state import ExecutionState
    state = ExecutionState(
        question_id=1,
        question="Q",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "Q"}],
    )
    orch._run_planning_turn([state])

    assert state.raw_query_analysis == raw_text
    # stripped version should not contain <think>
    assert "<think>" not in state.query_analysis
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/gepa_integration/test_orchestrator.py -v
```

Expected: failures on `_DEFAULT_PLANNING_SUFFIX_NO_TOOLS` import and `planning_suffix` param.

- [ ] **Step 3: Extract constants and add param to orchestrator**

At the top of `src/agent_engine/core/orchestrator.py`, after the existing module-level constants (after `_TEXT_EXTS`), add:

```python
_DEFAULT_PLANNING_SUFFIX_NO_TOOLS = (
    "\n\nBefore answering, analyze this query to determine the approach needed.\n"
    "Instructions:\n"
    "1. Identify the main objectives in the query.\n"
    "2. Break down the problem into sub-tasks.\n"
    "3. Consider what knowledge or reasoning steps are required.\n"
    "Be brief and precise. Do NOT provide the final answer yet."
)

_DEFAULT_PLANNING_SUFFIX_TOOLS = (
    "\n\nBefore using any tools, analyze this query to determine the approach needed.\n"
    "Instructions:\n"
    "1. Identify the main objectives in the query.\n"
    "2. List the necessary skills and tools.\n"
    "3. For each tool, explain how it helps address the query.\n"
    "4. Note any additional considerations.\n\n"
    "Be brief and precise. Do NOT call any tools yet."
)
```

- [ ] **Step 4: Add `planning_suffix` to `__init__`**

In `AgenticOrchestrator.__init__`, add the parameter and store it (after `baseline: bool = False,`):

```python
    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        max_turns: int = 15,
        tool_limits: Optional[Dict[str, int]] = None,
        use_thinking: bool = False,
        cache_manager=None,
        baseline: bool = False,
        planning_suffix: Optional[str] = None,
    ):
```

Inside `__init__`, after `self.baseline = baseline`:

```python
        self.planning_suffix: Optional[str] = planning_suffix
```

- [ ] **Step 5: Replace hardcoded suffixes in `_run_planning_turn` and store raw output**

In `_run_planning_turn`, replace the two hardcoded `planning_suffix = (...)` blocks with one line:

```python
        suffix = self.planning_suffix if self.planning_suffix is not None else (
            _DEFAULT_PLANNING_SUFFIX_TOOLS if len(self.tools) > 0
            else _DEFAULT_PLANNING_SUFFIX_NO_TOOLS
        )
```

Then replace `planning_suffix` variable usages in the method with `suffix`. Also, in the per-state loop where `s.query_analysis` is set, add the raw store **before** stripping:

```python
        for s, gen_result in zip(states, gen_results):
            _accumulate_usage(s, gen_result.usage)
            text = gen_result.text

            tool_call = parse_tool_call(text)
            if tool_call:
                idx = text.find("<tool_call>")
                if idx == -1:
                    for marker in ('{"tool_call"', '{"name"'):
                        j = text.find(marker)
                        if j != -1:
                            idx = j
                            break
                before_tool = text[:idx].strip() if idx > 0 else ""
                analysis = strip_thinking_tags(before_tool) if before_tool else strip_thinking_tags(text)
                s.raw_query_analysis = text          # NEW
                s.query_analysis = analysis
                logger.info(...)
            elif "\\boxed{" in text or "\\boxed " in text:
                s.raw_query_analysis = text          # NEW
                s.query_analysis = strip_thinking_tags(text)
                s.finished = True
                s.answer = extract_answer(text)
                logger.info(...)
            else:
                s.raw_query_analysis = text          # NEW
                s.query_analysis = strip_thinking_tags(text)
                logger.info(...)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/gepa_integration/test_orchestrator.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Verify existing behaviour is unchanged**

```bash
pytest tests/ -v 2>/dev/null || echo "no other tests yet"
```

- [ ] **Step 8: Commit**

```bash
git add src/agent_engine/core/orchestrator.py tests/gepa_integration/test_orchestrator.py
git commit -m "feat: extract planning suffix constants and add configurable planning_suffix to orchestrator"
```

---

## Task 4: Create `src/gepa_integration/` and write `build_seed_candidate`

**Files:**
- Create: `src/gepa_integration/__init__.py`
- Create: `src/gepa_integration/seed.py` (partial — `build_seed_candidate` only)
- Create: `tests/gepa_integration/test_seed.py` (partial)

- [ ] **Step 1: Create module init**

```bash
mkdir -p src/gepa_integration
touch src/gepa_integration/__init__.py
```

- [ ] **Step 2: Write the failing test for `build_seed_candidate`**

Create `tests/gepa_integration/test_seed.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
from gepa_integration.seed import build_seed_candidate


def test_build_seed_candidate_returns_two_keys():
    # Use GAIA with no tool schemas (simplest case — only checks structure)
    candidate = build_seed_candidate(
        benchmark="gaia",
        tool_schemas=[],
        direct_tool_call=True,
    )
    assert set(candidate.keys()) == {"system_prompt", "planning_suffix"}


def test_build_seed_candidate_planning_suffix_matches_constant():
    candidate = build_seed_candidate(
        benchmark="gaia",
        tool_schemas=[],
        direct_tool_call=True,
    )
    assert candidate["planning_suffix"] == _DEFAULT_PLANNING_SUFFIX_TOOLS


def test_build_seed_candidate_system_prompt_is_string():
    candidate = build_seed_candidate(
        benchmark="gaia",
        tool_schemas=[],
        direct_tool_call=True,
    )
    assert isinstance(candidate["system_prompt"], str)
    assert len(candidate["system_prompt"]) > 0


def test_build_seed_candidate_system_prompt_contains_no_tool_schema_when_empty():
    candidate = build_seed_candidate(
        benchmark="gaia",
        tool_schemas=[],
        direct_tool_call=True,
    )
    assert "<tools>" not in candidate["system_prompt"]


def test_build_seed_candidate_system_prompt_contains_tool_schema_when_provided():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    candidate = build_seed_candidate(
        benchmark="gaia",
        tool_schemas=[tool_schema],
        direct_tool_call=True,
    )
    assert "<tools>" in candidate["system_prompt"]
    assert "web_search" in candidate["system_prompt"]
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/gepa_integration/test_seed.py -v
```

Expected: `FAILED` — `gepa_integration.seed` does not exist.

- [ ] **Step 4: Implement `build_seed_candidate` in `seed.py`**

Create `src/gepa_integration/seed.py`:

```python
"""GEPA integration: seed candidate construction and data split generation.

build_seed_candidate() renders the initial two-component candidate from
the PromptBuilder templates.

build_splits() partitions an existing raw_results.json into failure-stratified
train / random val / random test splits, saving them to a JSON file.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Ensure scripts/ is on path for classify_failure import
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
from agent_engine.models.base import ToolCallFormat
from agent_engine.prompts import PromptBuilder


def build_seed_candidate(
    benchmark: str,
    tool_schemas: list[dict],
    direct_tool_call: bool = True,
    tool_call_format: ToolCallFormat = ToolCallFormat.JSON,
    max_search_limit: int = 10,
) -> dict[str, str]:
    """Render the seed GEPA candidate from YAML templates.

    Args:
        benchmark: Dataset name passed to PromptBuilder (e.g. "gaia", "gpqa").
        tool_schemas: List of tool schema dicts (from ToolRegistry.get_all_schemas()).
        direct_tool_call: Whether tools are called directly (vs sub-agent mode).
        tool_call_format: Tool call syntax (JSON for Qwen3, PYTHONIC for OLMo).
        max_search_limit: Passed to PromptBuilder for the search-limit reminder.

    Returns:
        {"system_prompt": str, "planning_suffix": str}
    """
    builder = PromptBuilder()
    system_prompt = builder.build_system_prompt(
        dataset_name=benchmark,
        tool_schemas=tool_schemas,
        max_search_limit=max_search_limit,
        direct_tool_call=direct_tool_call,
        baseline=False,
        tool_call_format=tool_call_format,
    )
    return {
        "system_prompt": system_prompt,
        "planning_suffix": _DEFAULT_PLANNING_SUFFIX_TOOLS,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/gepa_integration/test_seed.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/gepa_integration/__init__.py src/gepa_integration/seed.py tests/gepa_integration/test_seed.py
git commit -m "feat: add gepa_integration module with build_seed_candidate"
```

---

## Task 5: Write `build_splits`

**Files:**
- Modify: `src/gepa_integration/seed.py` — add `build_splits()`
- Modify: `tests/gepa_integration/test_seed.py` — add split tests

- [ ] **Step 1: Write the failing tests** (append to `tests/gepa_integration/test_seed.py`)

```python
import json
import tempfile


def _make_raw_results(n_correct: int, failures_by_mode: dict) -> list:
    """Build a minimal raw_results.json-style list for testing."""
    records = []
    qid = 0

    for _ in range(n_correct):
        records.append({
            "question_id": qid,
            "question": f"q{qid}",
            "correct": True,
            "prediction": "right",
            "action_history": [],
            "turns": 1,
        })
        qid += 1

    # tool_loop_or_empty_final: prediction empty, turns < MAX
    for _ in range(failures_by_mode.get("tool_loop_or_empty_final", 0)):
        records.append({
            "question_id": qid,
            "question": f"q{qid}",
            "correct": False,
            "prediction": "",
            "action_history": [],
            "turns": 3,
        })
        qid += 1

    # retrieval_evidence_failure: 2+ web_search calls
    for _ in range(failures_by_mode.get("retrieval_evidence_failure", 0)):
        records.append({
            "question_id": qid,
            "question": f"q{qid}",
            "correct": False,
            "prediction": "wrong",
            "action_history": [
                {"tool_name": "web_search", "sub_goal": "search1", "result": "r1"},
                {"tool_name": "web_search", "sub_goal": "search2", "result": "r2"},
            ],
            "turns": 2,
        })
        qid += 1

    return records


def test_build_splits_returns_three_keys():
    from gepa_integration.seed import build_splits

    records = _make_raw_results(n_correct=60, failures_by_mode={
        "tool_loop_or_empty_final": 20,
        "retrieval_evidence_failure": 20,
    })
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)

    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    assert set(splits.keys()) == {"train", "val", "test"}


def test_build_splits_sizes_are_correct():
    from gepa_integration.seed import build_splits

    records = _make_raw_results(n_correct=60, failures_by_mode={
        "tool_loop_or_empty_final": 20,
        "retrieval_evidence_failure": 20,
    })
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)

    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    assert len(splits["train"]) == 50
    assert len(splits["val"]) == 20
    # test gets whatever remains
    assert len(splits["test"]) == 100 - 50 - 20


def test_build_splits_no_overlap():
    from gepa_integration.seed import build_splits

    records = _make_raw_results(n_correct=60, failures_by_mode={
        "tool_loop_or_empty_final": 20,
        "retrieval_evidence_failure": 20,
    })
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)

    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def test_build_splits_train_contains_mostly_failures():
    from gepa_integration.seed import build_splits

    records = _make_raw_results(n_correct=60, failures_by_mode={
        "tool_loop_or_empty_final": 20,
        "retrieval_evidence_failure": 20,
    })
    failed_qids = {r["question_id"] for r in records if not r["correct"]}

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        path = Path(f.name)

    splits = build_splits(raw_results_path=path, train_n=50, val_n=20, seed=42)
    train_failures = sum(1 for qid in splits["train"] if qid in failed_qids)
    # ~65% should be failures
    assert train_failures >= 25  # at least 50% failures (generous bound for small dataset)


def test_build_splits_saves_json(tmp_path):
    from gepa_integration.seed import build_splits

    records = _make_raw_results(n_correct=60, failures_by_mode={
        "tool_loop_or_empty_final": 20,
        "retrieval_evidence_failure": 20,
    })
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(records, f)
        raw_path = Path(f.name)

    out_path = tmp_path / "splits.json"
    build_splits(raw_results_path=raw_path, train_n=50, val_n=20, seed=42, output_path=out_path)

    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert set(loaded.keys()) == {"train", "val", "test"}
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/gepa_integration/test_seed.py::test_build_splits_returns_three_keys -v
```

Expected: `FAILED` — `build_splits` not defined.

- [ ] **Step 3: Implement `build_splits` in `seed.py`** (append to existing file)

```python
def build_splits(
    raw_results_path: Path,
    train_n: int,
    val_n: int,
    seed: int = 1,
    output_path: Optional[Path] = None,
) -> dict[str, list[int]]:
    """Build failure-stratified train / random val / random test splits.

    Reads an existing raw_results.json, classifies each failed record using
    the six-mode taxonomy from scripts/failure_modes/analyze_failure_modes.py,
    then samples:
      - train: ~65% failures (proportional across modes) + ~35% successes
      - val:   random sample from remaining questions
      - test:  all remaining questions after train + val

    Args:
        raw_results_path: Path to an existing raw_results.json.
        train_n: Number of examples in the train set.
        val_n:   Number of examples in the val (D_pareto) set.
        seed:    Random seed for reproducibility (default 1, distinct from
                 the existing experiment seed 0).
        output_path: If given, write splits dict as JSON to this path.

    Returns:
        {"train": [qid, ...], "val": [qid, ...], "test": [qid, ...]}
    """
    from failure_modes.analyze_failure_modes import FAILURE_MODES, classify_failure

    rng = random.Random(seed)

    with open(raw_results_path, encoding="utf-8") as f:
        records = json.load(f)

    correct_ids: list[int] = []
    failed_by_mode: dict[str, list[int]] = defaultdict(list)

    for rec in records:
        qid = rec["question_id"]
        if rec.get("correct"):
            correct_ids.append(qid)
        else:
            mode = classify_failure(rec)
            failed_by_mode[mode].append(qid)

    all_failed_ids = [qid for ids in failed_by_mode.values() for qid in ids]
    total_failed = len(all_failed_ids)

    # --- sample failures proportionally across modes ---
    n_failures_train = round(train_n * 0.65)
    train_failed: list[int] = []

    if total_failed > 0:
        for mode in FAILURE_MODES:
            mode_ids = list(failed_by_mode[mode])
            rng.shuffle(mode_ids)
            n_from_mode = round(n_failures_train * len(mode_ids) / total_failed)
            train_failed.extend(mode_ids[:n_from_mode])

        # Fix rounding: top up or trim to exactly n_failures_train
        used_failed = set(train_failed)
        remaining_failed = [qid for qid in all_failed_ids if qid not in used_failed]
        rng.shuffle(remaining_failed)
        if len(train_failed) < n_failures_train:
            train_failed.extend(remaining_failed[: n_failures_train - len(train_failed)])
        train_failed = train_failed[:n_failures_train]

    # --- sample successes ---
    n_success_train = train_n - len(train_failed)
    shuffled_correct = list(correct_ids)
    rng.shuffle(shuffled_correct)
    train_success = shuffled_correct[:n_success_train]

    train_ids = train_failed + train_success
    rng.shuffle(train_ids)

    # --- val + test from the remainder ---
    used = set(train_ids)
    remaining = [rec["question_id"] for rec in records if rec["question_id"] not in used]
    rng.shuffle(remaining)

    val_ids = remaining[:val_n]
    test_ids = remaining[val_n:]

    splits = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

    return splits
```

- [ ] **Step 4: Run all seed tests**

```bash
pytest tests/gepa_integration/test_seed.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/gepa_integration/seed.py tests/gepa_integration/test_seed.py
git commit -m "feat: add build_splits with failure-stratified train set construction"
```

---

## Task 6: Write `AgentGEPAAdapter.evaluate()`

**Files:**
- Create: `src/gepa_integration/adapter.py`
- Create: `tests/gepa_integration/test_adapter.py` (partial)

- [ ] **Step 1: Write the failing tests**

Create `tests/gepa_integration/test_adapter.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry
from agent_engine.datasets.base import DatasetExample
from gepa_integration.adapter import AgentGEPAAdapter, _extract_thinking


# ── _extract_thinking ────────────────────────────────────────────────────────

def test_extract_thinking_returns_content():
    text = "<think>internal reasoning here</think>visible output"
    assert _extract_thinking(text) == "internal reasoning here"


def test_extract_thinking_returns_empty_when_no_tags():
    assert _extract_thinking("no think tags here") == ""


def test_extract_thinking_handles_multiline():
    text = "<think>\nline1\nline2\n</think>answer"
    assert "line1" in _extract_thinking(text)
    assert "line2" in _extract_thinking(text)


# ── AgentGEPAAdapter.evaluate ────────────────────────────────────────────────

def _make_adapter():
    model = MagicMock()
    model.config = MagicMock()
    model.config.supports_thinking = True
    model.config.family = "qwen3"
    tools = ToolRegistry()
    return AgentGEPAAdapter(
        model_provider=model,
        tool_registry=tools,
        use_thinking=True,
        max_turns=3,
    )


def _make_example(qid: int, question: str, answer: str, choices=None) -> DatasetExample:
    meta = {}
    if choices is not None:
        meta["choices"] = choices
    return DatasetExample(question_id=qid, question=question, answer=answer, metadata=meta)


def test_evaluate_returns_evaluation_batch_with_correct_length():
    adapter = _make_adapter()
    candidate = {
        "system_prompt": "You are a helpful assistant.",
        "planning_suffix": "Plan your approach.",
    }
    examples = [_make_example(1, "What is 2+2?", "4")]

    # Mock orchestrator to return a finished state
    finished_state = ExecutionState(
        question_id=1, question="What is 2+2?",
        messages=[], answer="4", finished=True,
    )
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_batch.return_value = [finished_state]

        result = adapter.evaluate(examples, candidate, capture_traces=False)

    assert len(result.outputs) == 1
    assert len(result.scores) == 1
    assert result.trajectories is None


def test_evaluate_score_is_1_for_correct_answer():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "Paris")]

    state = ExecutionState(question_id=1, question="Q", messages=[], answer="Paris", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate)

    assert result.scores[0] == 1.0


def test_evaluate_score_is_0_for_wrong_answer():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "Paris")]

    state = ExecutionState(question_id=1, question="Q", messages=[], answer="Berlin", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate)

    assert result.scores[0] == 0.0


def test_evaluate_captures_trajectories_when_requested():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "A")]

    state = ExecutionState(question_id=1, question="Q", messages=[], answer="A", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        result = adapter.evaluate(examples, candidate, capture_traces=True)

    assert result.trajectories is not None
    assert len(result.trajectories) == 1
    assert result.trajectories[0] is state


def test_evaluate_passes_planning_suffix_to_orchestrator():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "MY_CUSTOM_SUFFIX"}
    examples = [_make_example(1, "Q", "A")]

    state = ExecutionState(question_id=1, question="Q", messages=[], answer="A", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        adapter.evaluate(examples, candidate)

    call_kwargs = MockOrch.call_args.kwargs
    assert call_kwargs["planning_suffix"] == "MY_CUSTOM_SUFFIX"


def test_evaluate_stores_ground_truth_in_state_metadata():
    adapter = _make_adapter()
    candidate = {"system_prompt": "sys", "planning_suffix": "plan"}
    examples = [_make_example(1, "Q", "correct answer")]

    state = ExecutionState(question_id=1, question="Q", messages=[], answer="wrong", finished=True)
    with patch("gepa_integration.adapter.AgenticOrchestrator") as MockOrch:
        MockOrch.return_value.run_batch.return_value = [state]
        adapter.evaluate(examples, candidate, capture_traces=True)

    assert state.metadata["ground_truth"] == "correct answer"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/gepa_integration/test_adapter.py -v
```

Expected: `FAILED` — module not found.

- [ ] **Step 3: Implement `adapter.py`**

Create `src/gepa_integration/adapter.py`:

```python
"""GEPA adapter for the AgenticOrchestrator.

AgentGEPAAdapter connects GEPA's optimization loop to the msc-thesis
orchestrator. It implements the GEPAAdapter protocol with two optimizable
components: "system_prompt" and "planning_suffix".

Thinking mode is fixed at ORCHESTRATOR_ONLY to match the main experimental
condition and provide rich <think> traces for the reflector.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry
from agent_engine.datasets.base import DatasetExample
from agent_engine.datasets.evaluators.metrics import evaluate_answer
from agent_engine.models.base import BaseModelProvider
from gepa.core.adapter import EvaluationBatch


def _extract_thinking(text: str) -> str:
    """Return the content of the first <think>…</think> block, or ''."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


class AgentGEPAAdapter:
    """GEPAAdapter implementation wrapping AgenticOrchestrator.

    Args:
        model_provider: Shared vLLM provider (not re-loaded between candidates).
        tool_registry:  Pre-built tool registry.
        use_thinking:   Whether to enable orchestrator thinking (default True —
                        ORCHESTRATOR_ONLY mode).
        max_turns:      Maximum reasoning turns per question.
        tool_limits:    Per-tool call limits dict.
    """

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        use_thinking: bool = True,
        max_turns: int = 15,
        tool_limits: Optional[dict[str, int]] = None,
    ) -> None:
        self.model_provider = model_provider
        self.tool_registry = tool_registry
        self.use_thinking = use_thinking
        self.max_turns = max_turns
        self.tool_limits = tool_limits or {"web_search": 10}

    # ------------------------------------------------------------------ #
    # GEPAAdapter protocol                                                 #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        batch: list[DatasetExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run the orchestrator on `batch` using `candidate`'s prompts.

        Stores ground_truth in each state's metadata so make_reflective_dataset
        can access it without needing the original examples.
        """
        orchestrator = AgenticOrchestrator(
            model_provider=self.model_provider,
            tool_registry=self.tool_registry,
            max_turns=self.max_turns,
            tool_limits=self.tool_limits,
            use_thinking=self.use_thinking,
            planning_suffix=candidate["planning_suffix"],
        )

        states: list[ExecutionState] = orchestrator.run_batch(
            questions=[ex.question for ex in batch],
            question_ids=[ex.question_id for ex in batch],
            system_prompts=[candidate["system_prompt"]] * len(batch),
            attachments=[ex.get_attachments() or None for ex in batch],
        )

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[ExecutionState] | None = [] if capture_traces else None

        for state, example in zip(states, batch):
            prediction = state.answer or ""
            choices = example.metadata.get("choices")
            result = evaluate_answer(prediction, example.answer, choices=choices)
            outputs.append(prediction)
            scores.append(float(result["accuracy"]))
            # Store ground truth for make_reflective_dataset
            state.metadata["ground_truth"] = example.answer
            if capture_traces:
                trajectories.append(state)  # type: ignore[union-attr]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build per-component reflective datasets from execution traces.

        Returns at most 12 records per component (6 correct, 6 wrong).
        """
        states: list[ExecutionState] = eval_batch.trajectories or []
        scores: list[float] = eval_batch.scores

        dataset: dict[str, list[dict]] = {}

        if "system_prompt" in components_to_update:
            dataset["system_prompt"] = self._system_prompt_records(states, scores)

        if "planning_suffix" in components_to_update:
            dataset["planning_suffix"] = self._planning_suffix_records(states, scores)

        return dataset

    # ------------------------------------------------------------------ #
    # Reflective dataset helpers                                           #
    # ------------------------------------------------------------------ #

    _MAX_RECORDS = 12          # 6 correct + 6 wrong per reflective call
    _RESULT_SNIPPET_LEN = 300  # chars per tool result

    def _balanced_sample(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[tuple[ExecutionState, float]]:
        """Return up to MAX_RECORDS pairs balanced between correct and wrong."""
        correct = [(s, sc) for s, sc in zip(states, scores) if sc > 0]
        wrong = [(s, sc) for s, sc in zip(states, scores) if sc == 0]
        half = self._MAX_RECORDS // 2
        return correct[:half] + wrong[:half]

    def _system_prompt_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            gt = state.metadata.get("ground_truth", "")
            first_thinking = (
                _extract_thinking(state.output_messages[0]["content"])
                if state.output_messages
                else ""
            )
            action_steps = [
                {
                    "tool": a["tool_name"],
                    "sub_goal": a.get("sub_goal", ""),
                    "result_snippet": str(a.get("result", ""))[: self._RESULT_SNIPPET_LEN],
                }
                for a in state.action_history
            ]
            if score > 0:
                feedback = "CORRECT"
            else:
                parts = [f"WRONG — ground truth: {gt}. Predicted: {state.answer or '(empty)'}."]
                if state.metadata.get("max_turns_reached"):
                    parts.append("Max turns reached without answer.")
                feedback = " ".join(parts)

            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "predicted_answer": state.answer or "",
                    "thinking_before_first_tool": first_thinking,
                    "action_steps": action_steps,
                },
                "Feedback": feedback,
            })
        return records

    def _planning_suffix_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            raw_plan = state.raw_query_analysis or state.query_analysis or ""
            tools_used = [tc["name"] for tc in state.tool_calls]
            if score > 0:
                feedback = "CORRECT — the planning analysis led to a successful solution."
            else:
                feedback = (
                    f"WRONG — the planning analysis was: '{state.query_analysis}'. "
                    "Consider whether the plan correctly identified the required steps and tools."
                )
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "raw_planning_output": raw_plan,
                    "tools_subsequently_used": tools_used,
                    "num_turns_taken": state.turn,
                },
                "Feedback": feedback,
            })
        return records
```

- [ ] **Step 4: Run adapter tests**

```bash
pytest tests/gepa_integration/test_adapter.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/gepa_integration/adapter.py tests/gepa_integration/test_adapter.py
git commit -m "feat: add AgentGEPAAdapter with evaluate() and make_reflective_dataset()"
```

---

## Task 7: Write `make_reflective_dataset` tests

**Files:**
- Modify: `tests/gepa_integration/test_adapter.py` — add reflective dataset tests

- [ ] **Step 1: Append tests to `tests/gepa_integration/test_adapter.py`**

```python
# ── make_reflective_dataset ──────────────────────────────────────────────────

from gepa.core.adapter import EvaluationBatch as GEPAEvaluationBatch


def _make_state(qid, question, answer, correct, tool_calls=None, raw_plan=None, action_history=None):
    state = ExecutionState(
        question_id=qid,
        question=question,
        messages=[],
        answer=answer,
        finished=True,
    )
    state.metadata["ground_truth"] = "correct_answer" if correct else "other"
    state.query_analysis = "plan summary"
    state.raw_query_analysis = raw_plan or "<think>think</think>plan summary"
    state.tool_calls = tool_calls or []
    state.action_history = action_history or []
    return state


def test_make_reflective_dataset_returns_system_prompt_key():
    adapter = _make_adapter()
    states = [_make_state(1, "Q", "correct_answer", correct=True)]
    scores = [1.0]
    batch = GEPAEvaluationBatch(outputs=["correct_answer"], scores=scores, trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert "system_prompt" in result


def test_make_reflective_dataset_returns_planning_suffix_key():
    adapter = _make_adapter()
    states = [_make_state(1, "Q", "correct_answer", correct=True)]
    scores = [1.0]
    batch = GEPAEvaluationBatch(outputs=["correct_answer"], scores=scores, trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    assert "planning_suffix" in result


def test_make_reflective_dataset_correct_feedback_label():
    adapter = _make_adapter()
    states = [_make_state(1, "Q", "correct_answer", correct=True)]
    scores = [1.0]
    batch = GEPAEvaluationBatch(outputs=["correct_answer"], scores=scores, trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert result["system_prompt"][0]["Feedback"] == "CORRECT"


def test_make_reflective_dataset_wrong_feedback_contains_ground_truth():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "wrong_answer", correct=False)
    state.metadata["ground_truth"] = "real_answer"
    scores = [0.0]
    batch = GEPAEvaluationBatch(outputs=["wrong_answer"], scores=scores, trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert "real_answer" in result["system_prompt"][0]["Feedback"]


def test_make_reflective_dataset_planning_uses_raw_plan():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "A", correct=True, raw_plan="<think>reasoning</think>plan")
    scores = [1.0]
    batch = GEPAEvaluationBatch(outputs=["A"], scores=scores, trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    raw = result["planning_suffix"][0]["Generated Outputs"]["raw_planning_output"]
    assert "<think>reasoning</think>" in raw


def test_make_reflective_dataset_capped_at_12_records():
    adapter = _make_adapter()
    # 20 correct states — should be capped at 6 correct
    states = [_make_state(i, "Q", "A", correct=True) for i in range(20)]
    scores = [1.0] * 20
    batch = GEPAEvaluationBatch(outputs=["A"] * 20, scores=scores, trajectories=states)
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    assert len(result["system_prompt"]) <= 12
```

- [ ] **Step 2: Run all adapter tests**

```bash
pytest tests/gepa_integration/test_adapter.py -v
```

Expected: all passed.

- [ ] **Step 3: Commit**

```bash
git add tests/gepa_integration/test_adapter.py
git commit -m "test: add make_reflective_dataset tests for AgentGEPAAdapter"
```

---

## Task 8: Write `run_gepa.py` — `splits` and `diff` modes

**Files:**
- Create: `scripts/run_gepa.py`

- [ ] **Step 1: Create the script with `splits` and `diff` modes**

Create `scripts/run_gepa.py`:

```python
"""GEPA prompt optimisation CLI.

Modes:
  splits   — generate train/val/test split files from existing raw_results.json
  optimize — run GEPA optimisation loop, save best candidate
  evaluate — evaluate best candidate on held-out test set
  diff     — print diff between seed and best candidate prompts

Usage:
  python scripts/run_gepa.py --mode splits   --config experiments/configs/gepa/gaia.yaml
  python scripts/run_gepa.py --mode optimize --config experiments/configs/gepa/gaia.yaml
  python scripts/run_gepa.py --mode evaluate --config experiments/configs/gepa/gaia.yaml
  python scripts/run_gepa.py --mode diff     --config experiments/configs/gepa/gaia.yaml
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add scripts/ to path (for failure_modes import inside seed.py)
sys.path.insert(0, str(Path(__file__).parent))

import yaml

from agent_engine.config import load_experiment_config
from agent_engine.models.base import ModelFamily, get_tool_call_format
from agent_engine.core import ToolRegistry, AgenticOrchestrator
from agent_engine.tools import WebSearchTool, CodeGeneratorTool, TextInspectorTool
from agent_engine.datasets import DatasetRegistry
from agent_engine.caching import CacheManager
from agent_engine.utils import setup_logging, set_seed
from gepa_integration.seed import build_seed_candidate, build_splits


def load_gepa_config(config_path: Path) -> dict:
    """Load the GEPA-specific YAML config (not an ExperimentConfig)."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_examples(cfg: dict, question_ids: list[int]) -> list:
    """Load DatasetExample objects for the given question IDs."""
    from agent_engine.config.schema import DatasetConfig
    from pathlib import Path as _Path

    ds_cfg = DatasetConfig(
        name=cfg["benchmark"],
        split=cfg.get("dataset_split", _get_default_split(cfg["benchmark"])),
        data_dir=_Path(cfg.get("data_dir", "./data")),
        subset_num=-1,
    )
    dataset = DatasetRegistry.get(ds_cfg)
    all_examples = list(dataset)
    id_set = set(question_ids)
    return [ex for ex in all_examples if ex.question_id in id_set]


def _get_default_split(benchmark: str) -> str:
    defaults = {
        "gaia": "all_validation",
        "gpqa": "diamond",
        "math500": "test",
        "aime": "train",
    }
    return defaults.get(benchmark, "validation")


def _build_tool_registry(cfg: dict, cache_manager: CacheManager) -> ToolRegistry:
    tools = ToolRegistry()
    enabled = cfg.get("tools", {}).get("enabled_tools", ["web_search", "code_generator"])
    direct = cfg.get("tools", {}).get("direct_tool_call", True)

    import os
    if "web_search" in enabled:
        provider = cfg.get("tools", {}).get("web_tool_provider", "serper")
        tools.register(WebSearchTool(
            api_key=os.environ.get("SERPER_API_KEY" if provider == "serper" else "TAVILY_API_KEY", ""),
            provider=provider,
            direct_mode=direct,
            cache_manager=cache_manager,
        ))
    if "code_generator" in enabled:
        tools.register(CodeGeneratorTool(direct_mode=direct))
    if "text_inspector" in enabled:
        tools.register(TextInspectorTool())

    return tools


# ─────────────────────────────────────────────── MODE: splits ──────────────

def run_splits(cfg: dict, config_path: Path) -> None:
    """Generate train/val/test split files from existing raw_results.json."""
    existing_results = Path(cfg["existing_results"])
    if not existing_results.is_absolute():
        existing_results = config_path.parent.parent.parent / existing_results

    gepa_cfg = cfg.get("gepa", {})
    splits_dir = config_path.parent / "splits"
    output_path = splits_dir / f"{cfg['benchmark']}_splits.json"

    split_cfg = cfg.get("splits", {})
    train_n = split_cfg.get("train_n", 80)
    val_n = split_cfg.get("val_n", 45)
    seed = cfg.get("seed", 1)

    print(f"Building splits for {cfg['benchmark']}...")
    print(f"  Source: {existing_results}")
    print(f"  Train: {train_n}, Val: {val_n}, seed: {seed}")

    splits = build_splits(
        raw_results_path=existing_results,
        train_n=train_n,
        val_n=val_n,
        seed=seed,
        output_path=output_path,
    )

    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val:   {len(splits['val'])} examples")
    print(f"  Test:  {len(splits['test'])} examples")
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────── MODE: diff ────────────────

def run_diff(cfg: dict, config_path: Path) -> None:
    """Print a readable diff between the seed candidate and best candidate."""
    import difflib

    run_dir = Path(cfg["gepa"]["run_dir"])
    best_path = run_dir / "best_candidate.json"
    if not best_path.exists():
        print(f"No best_candidate.json found at {best_path}. Run --mode optimize first.")
        sys.exit(1)

    with open(best_path) as f:
        best = json.load(f)

    # Load seed saved during optimize (avoids rebuilding tool schemas)
    seed_path = run_dir / "seed_candidate.json"
    if not seed_path.exists():
        print(f"ERROR: {seed_path} not found. Run --mode optimize first.")
        sys.exit(1)

    with open(seed_path) as f:
        seed = json.load(f)

    for component in ("system_prompt", "planning_suffix"):
        print(f"\n{'='*60}")
        print(f"DIFF: {component}")
        print("=" * 60)
        seed_lines = seed[component].splitlines(keepends=True)
        best_lines = best[component].splitlines(keepends=True)
        diff = list(difflib.unified_diff(seed_lines, best_lines, fromfile="seed", tofile="best", n=3))
        if diff:
            print("".join(diff))
        else:
            print("(no change)")


# ─────────────────────────────────────────────── main ──────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA prompt optimisation")
    parser.add_argument("--mode", choices=["splits", "optimize", "evaluate", "diff"], required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    cfg = load_gepa_config(config_path)
    setup_logging()
    set_seed(cfg.get("seed", 1))

    if args.mode == "splits":
        run_splits(cfg, config_path)
    elif args.mode == "diff":
        run_diff(cfg, config_path)
    elif args.mode in ("optimize", "evaluate"):
        # Implemented in Task 9 and Task 10
        print(f"Mode '{args.mode}' not yet implemented — see Task 9/10.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the diff mode guard**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/gaia.yaml 2>&1 | head -5
```

Expected output contains: `splits_file not found` or `No best_candidate.json` (any graceful error, not a Python traceback from a missing module).

Note: This will fail gracefully because the config doesn't exist yet — that's fine. The test here is that the script *imports* without error.

```bash
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_gepa
print('imports OK')
"
```

Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_gepa.py
git commit -m "feat: add run_gepa.py with splits and diff modes"
```

---

## Task 9: Write `run_gepa.py` — `optimize` mode

**Files:**
- Modify: `scripts/run_gepa.py` — implement `run_optimize()`

- [ ] **Step 1: Implement `run_optimize`**

Add the following function to `scripts/run_gepa.py` (before the `main()` function):

```python
# ─────────────────────────────────────────────── MODE: optimize ────────────

def run_optimize(cfg: dict, config_path: Path) -> None:
    """Run GEPA optimisation loop. Saves best candidate to run_dir/best_candidate.json."""
    from gepa.api import optimize

    gepa_cfg = cfg["gepa"]
    run_dir = Path(gepa_cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = config_path.parent.parent.parent / splits_file
    if not splits_file.exists():
        print(f"ERROR: splits file not found: {splits_file}")
        print("Run --mode splits first.")
        sys.exit(1)

    with open(splits_file) as f:
        splits = json.load(f)

    # Load examples
    train_examples = _load_examples(cfg, splits["train"])
    val_examples = _load_examples(cfg, splits["val"])
    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples.")

    # Build infrastructure
    cache_manager = CacheManager(
        cache_dir=Path(cfg.get("cache_dir", "./cache")),
        dataset_name=cfg["benchmark"],
        split=_get_default_split(cfg["benchmark"]),
    )
    tool_registry = _build_tool_registry(cfg, cache_manager)

    # Build model provider for agent
    import os
    from agent_engine.models.vllm_provider import VLLMProvider
    from agent_engine.models.base import ModelConfig, ModelFamily
    model_cfg_raw = cfg["model"]
    model_cfg = ModelConfig(
        name=model_cfg_raw["name"],
        path_or_id=model_cfg_raw["path_or_id"],
        family=ModelFamily.QWEN3,
        role="orchestrator",
        use_thinking=True,
    )
    model_provider = VLLMProvider(model_cfg)

    # Build adapter
    from gepa_integration.adapter import AgentGEPAAdapter
    adapter = AgentGEPAAdapter(
        model_provider=model_provider,
        tool_registry=tool_registry,
        use_thinking=True,
        max_turns=cfg.get("max_turns", 15),
    )

    # Build seed candidate
    tool_schemas = tool_registry.get_all_schemas()
    from agent_engine.models.base import get_tool_call_format
    tcf = get_tool_call_format(model_cfg.family)
    seed = build_seed_candidate(
        benchmark=cfg["benchmark"],
        tool_schemas=tool_schemas,
        direct_tool_call=cfg.get("tools", {}).get("direct_tool_call", True),
        tool_call_format=tcf,
    )
    print(f"Seed candidate built. system_prompt length: {len(seed['system_prompt'])} chars.")

    # Configure reflector
    reflector_cfg = cfg.get("reflector", {})
    reflector_model = reflector_cfg.get("path_or_id", "Qwen/Qwen3-32B")
    reflector_host = reflector_cfg.get("host", "localhost")
    reflector_port = reflector_cfg.get("port", 8001)
    reflection_lm_kwargs = {
        "base_url": f"http://{reflector_host}:{reflector_port}/v1",
        "api_key": "EMPTY",
    }

    print(f"Starting GEPA optimisation: budget={gepa_cfg['rollout_budget']}, "
          f"minibatch={gepa_cfg['minibatch_size']}")

    result = optimize(
        seed_candidate=seed,
        trainset=train_examples,
        valset=val_examples,
        adapter=adapter,
        reflection_lm=f"openai/{reflector_model}",
        reflection_lm_kwargs=reflection_lm_kwargs,
        max_metric_calls=gepa_cfg["rollout_budget"],
        reflection_minibatch_size=gepa_cfg["minibatch_size"],
        use_merge=gepa_cfg.get("merge_proposer", True),
        run_dir=str(run_dir),
        seed=cfg.get("seed", 1),
        raise_on_exception=False,
        display_progress_bar=True,
    )

    best = result.best_candidate
    best_path = run_dir / "best_candidate.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    # Also save the seed so diff mode can compare without rebuilding
    seed_path = run_dir / "seed_candidate.json"
    with open(seed_path, "w") as f:
        json.dump(seed, f, indent=2)

    print(f"\nOptimisation complete. Best candidate saved to {best_path}")
    print(f"Seed candidate saved to {seed_path}")
    print(f"system_prompt length: {len(best['system_prompt'])} chars")
    print(f"planning_suffix length: {len(best['planning_suffix'])} chars")
```

- [ ] **Step 2: Wire `run_optimize` into `main()`**

In `main()`, replace the placeholder `elif args.mode == "optimize":` block with:

```python
    elif args.mode == "optimize":
        run_optimize(cfg, config_path)
```

- [ ] **Step 3: Verify imports parse without error**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_gepa
print('imports OK')
"
```

Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_gepa.py
git commit -m "feat: add optimize mode to run_gepa.py"
```

---

## Task 10: Write `run_gepa.py` — `evaluate` mode

**Files:**
- Modify: `scripts/run_gepa.py` — implement `run_evaluate()`

- [ ] **Step 1: Implement `run_evaluate`**

Add `run_evaluate()` to `scripts/run_gepa.py` before `main()`:

```python
# ─────────────────────────────────────────────── MODE: evaluate ────────────

def run_evaluate(cfg: dict, config_path: Path) -> None:
    """Evaluate best GEPA candidate on held-out test set.

    Loads best_candidate.json from run_dir, runs the orchestrator on the
    test split, and writes gepa_results.json in the same format as
    raw_results.json so existing analysis scripts work on it.
    """
    gepa_cfg = cfg["gepa"]
    run_dir = Path(gepa_cfg["run_dir"])
    best_path = run_dir / "best_candidate.json"

    if not best_path.exists():
        print(f"ERROR: {best_path} not found. Run --mode optimize first.")
        sys.exit(1)

    with open(best_path) as f:
        best = json.load(f)

    # Load test split
    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = config_path.parent.parent.parent / splits_file

    with open(splits_file) as f:
        splits = json.load(f)

    test_examples = _load_examples(cfg, splits["test"])
    print(f"Evaluating on {len(test_examples)} held-out test examples...")

    # Build infrastructure (same as optimize)
    cache_manager = CacheManager(
        cache_dir=Path(cfg.get("cache_dir", "./cache")),
        dataset_name=cfg["benchmark"],
        split=_get_default_split(cfg["benchmark"]),
    )
    tool_registry = _build_tool_registry(cfg, cache_manager)

    from agent_engine.models.vllm_provider import VLLMProvider
    from agent_engine.models.base import ModelConfig, ModelFamily
    model_cfg_raw = cfg["model"]
    model_cfg = ModelConfig(
        name=model_cfg_raw["name"],
        path_or_id=model_cfg_raw["path_or_id"],
        family=ModelFamily.QWEN3,
        role="orchestrator",
        use_thinking=True,
    )
    model_provider = VLLMProvider(model_cfg)

    orchestrator = AgenticOrchestrator(
        model_provider=model_provider,
        tool_registry=tool_registry,
        max_turns=cfg.get("max_turns", 15),
        use_thinking=True,
        planning_suffix=best["planning_suffix"],
    )

    states = orchestrator.run_batch(
        questions=[ex.question for ex in test_examples],
        question_ids=[ex.question_id for ex in test_examples],
        system_prompts=[best["system_prompt"]] * len(test_examples),
        attachments=[ex.get_attachments() or None for ex in test_examples],
    )

    # Score and build results list matching raw_results.json schema
    from agent_engine.datasets.evaluators.metrics import evaluate_answer
    results = []
    n_correct = 0
    for state, ex in zip(states, test_examples):
        prediction = state.answer or ""
        choices = ex.metadata.get("choices")
        eval_result = evaluate_answer(prediction, ex.answer, choices=choices)
        correct = bool(eval_result["correct"])
        if correct:
            n_correct += 1
        results.append({
            "question_id": ex.question_id,
            "question": ex.question,
            "prediction": prediction,
            "answer": ex.answer,
            "correct": correct,
            "accuracy": eval_result["accuracy"],
            "turns": state.turn,
            "tool_counts": dict(state.tool_counts),
            "action_history": state.action_history,
            "metadata": state.metadata,
        })

    accuracy = n_correct / len(results) if results else 0.0
    print(f"\nTest accuracy: {n_correct}/{len(results)} = {accuracy:.1%}")

    out_path = run_dir / "gepa_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
```

- [ ] **Step 2: Wire into `main()`**

```python
    elif args.mode == "evaluate":
        run_evaluate(cfg, config_path)
```

- [ ] **Step 3: Verify imports**

```bash
python -c "
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_gepa
print('imports OK')
"
```

Expected: `imports OK`

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/gepa_integration/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_gepa.py
git commit -m "feat: add evaluate mode to run_gepa.py"
```

---

## Task 11: Write GEPA experiment configs

**Files:**
- Create: `experiments/configs/gepa/gaia.yaml`
- Create: `experiments/configs/gepa/gpqa.yaml`

- [ ] **Step 1: Create `experiments/configs/gepa/gaia.yaml`**

```bash
mkdir -p experiments/configs/gepa/splits
```

Create `experiments/configs/gepa/gaia.yaml`:

```yaml
name: "GEPA_gaia_qwen3_8b"
description: "GEPA two-component prompt optimisation on GAIA, Qwen3-8B, ORCHESTRATOR_ONLY thinking"

benchmark: "gaia"
dataset_split: "all_validation"
data_dir: "./data"
thinking_mode: "ORCHESTRATOR_ONLY"
seed: 1
max_turns: 15
cache_dir: "./cache"

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  role: "orchestrator"

# Reflector runs on a separate vLLM instance.
# Start it before running --mode optimize:
#   vllm serve Qwen/Qwen3-32B --port 8001 --enable-thinking
reflector:
  name: "Qwen3-32B"
  path_or_id: "Qwen/Qwen3-32B"
  host: "localhost"
  port: 8001

tools:
  enabled_tools:
    - web_search
    - code_generator
    - text_inspector
  direct_tool_call: true
  web_tool_provider: "serper"
  max_search_limit: 10

splits_file: "experiments/configs/gepa/splits/gaia_splits.json"
existing_results: "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gaia/qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json"

splits:
  train_n: 80
  val_n: 45
  # remaining ~40 become held-out test

gepa:
  rollout_budget: 150
  minibatch_size: 10
  merge_proposer: true
  run_dir: "experiments/results/gepa/gaia"

slurm:
  partition: "gpu_h100"
  num_gpus: 4
  ntasks: 1
  cpus_per_task: 16
  time: "12:00:00"
  conda_env: "agent_engine"
```

- [ ] **Step 2: Create `experiments/configs/gepa/gpqa.yaml`**

Create `experiments/configs/gepa/gpqa.yaml`:

```yaml
name: "GEPA_gpqa_qwen3_8b"
description: "GEPA two-component prompt optimisation on GPQA Diamond, Qwen3-8B, ORCHESTRATOR_ONLY thinking"

benchmark: "gpqa"
dataset_split: "diamond"
data_dir: "./data"
thinking_mode: "ORCHESTRATOR_ONLY"
seed: 1
max_turns: 15
cache_dir: "./cache"

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  role: "orchestrator"

reflector:
  name: "Qwen3-32B"
  path_or_id: "Qwen/Qwen3-32B"
  host: "localhost"
  port: 8001

tools:
  enabled_tools:
    - web_search
    - code_generator
    - text_inspector
  direct_tool_call: true
  web_tool_provider: "serper"
  max_search_limit: 10

splits_file: "experiments/configs/gepa/splits/gpqa_splits.json"
existing_results: "experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gpqa/qwen8B_subagent_tools_orchestrator/diamond_2026-03-15-21-19-20_20752198/raw_results.json"

splits:
  train_n: 100
  val_n: 48
  # remaining ~50 become held-out test

gepa:
  rollout_budget: 150
  minibatch_size: 10
  merge_proposer: true
  run_dir: "experiments/results/gepa/gpqa"

slurm:
  partition: "gpu_h100"
  num_gpus: 4
  ntasks: 1
  cpus_per_task: 16
  time: "12:00:00"
  conda_env: "agent_engine"
```

- [ ] **Step 3: Validate YAML parses**

```bash
python -c "
import yaml
for f in ['experiments/configs/gepa/gaia.yaml', 'experiments/configs/gepa/gpqa.yaml']:
    with open(f) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg['benchmark'] in ('gaia', 'gpqa'), f'bad benchmark in {f}'
    assert 'gepa' in cfg
    assert 'splits' in cfg
    print(f'{f}: OK')
"
```

Expected: both files print `OK`.

- [ ] **Step 4: Commit**

```bash
git add experiments/configs/gepa/
git commit -m "feat: add GEPA experiment configs for gaia and gpqa"
```

---

## Task 12: Generate splits

> **Note:** Requires the existing raw_results.json files on disk. Run from the repo root on the cluster or locally if results are available.

**Files:**
- Creates: `experiments/configs/gepa/splits/gaia_splits.json`
- Creates: `experiments/configs/gepa/splits/gpqa_splits.json`

- [ ] **Step 1: Generate GAIA splits**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
python scripts/run_gepa.py --mode splits --config experiments/configs/gepa/gaia.yaml
```

Expected output:
```
Building splits for gaia...
  Source: experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gaia/...
  Train: 80, Val: 45, seed: 1
  Train: 80 examples
  Val:   45 examples
  Test:  ~40 examples
  Saved: experiments/configs/gepa/splits/gaia_splits.json
```

- [ ] **Step 2: Generate GPQA splits**

```bash
python scripts/run_gepa.py --mode splits --config experiments/configs/gepa/gpqa.yaml
```

Expected: similar output for gpqa.

- [ ] **Step 3: Verify split files are valid JSON with correct keys**

```bash
python -c "
import json
for name in ['gaia', 'gpqa']:
    with open(f'experiments/configs/gepa/splits/{name}_splits.json') as f:
        s = json.load(f)
    assert set(s.keys()) == {'train', 'val', 'test'}
    assert len(set(s['train']) & set(s['val'])) == 0, 'train/val overlap'
    assert len(set(s['train']) & set(s['test'])) == 0, 'train/test overlap'
    print(f'{name}: train={len(s[\"train\"])}, val={len(s[\"val\"])}, test={len(s[\"test\"])} — OK')
"
```

Expected:
```
gaia: train=80, val=45, test=40 — OK
gpqa: train=100, val=48, test=50 — OK
```

- [ ] **Step 4: Commit splits**

```bash
git add experiments/configs/gepa/splits/
git commit -m "data: add gepa train/val/test splits for gaia and gpqa"
```

---

## Math Benchmark Note

The math config (`experiments/configs/gepa/math.yaml`) is intentionally deferred until the math benchmark and its result path are confirmed. Once known:

1. Copy `gpqa.yaml` → `math.yaml`
2. Update: `benchmark`, `dataset_split`, `existing_results`, `splits_file`, `splits.train_n/val_n`, `gepa.run_dir`
3. Run `python scripts/run_gepa.py --mode splits --config experiments/configs/gepa/math.yaml`
4. Follow Tasks 9–10 for optimize + evaluate

---

## Running the Full Pipeline (cluster)

After splits are generated, the end-to-end flow per benchmark is:

```bash
# 1. Start reflector vLLM on port 8001 (separate terminal / job)
vllm serve Qwen/Qwen3-32B --port 8001 --enable-thinking

# 2. Run GEPA optimisation (~3h on H100 per benchmark)
python scripts/run_gepa.py --mode optimize --config experiments/configs/gepa/gaia.yaml

# 3. Evaluate best candidate on held-out test
python scripts/run_gepa.py --mode evaluate --config experiments/configs/gepa/gaia.yaml

# 4. Inspect what changed
python scripts/run_gepa.py --mode diff --config experiments/configs/gepa/gaia.yaml

# 5. Filter existing raw_results.json to held-out test IDs for baseline comparison
python -c "
import json
with open('experiments/configs/gepa/splits/gaia_splits.json') as f:
    test_ids = set(json.load(f)['test'])
with open('experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/gaia/qwen8B_subagent_tools_orchestrator/all_validation_2026-03-15-20-55-53_20752049/raw_results.json') as f:
    all_results = json.load(f)
baseline_test = [r for r in all_results if r['question_id'] in test_ids]
n = len(baseline_test)
acc = sum(r['correct'] for r in baseline_test) / n
print(f'Baseline on held-out test: {acc:.1%} ({sum(r[\"correct\"] for r in baseline_test)}/{n})')
"
```
