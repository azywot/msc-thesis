# Repository Handover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CoSMAS repository clean, modular, testable and documented for handover to a new researcher, with behaviour byte-for-byte identical to today's implementation.

**Architecture:** A characterization-fixture harness is built first against unmodified code (Phase 0). Every later phase is a focused, independently revertible commit that must reproduce those fixtures exactly. Wiring logic is promoted out of `scripts/` into `src/agent_engine/runner/`, tool construction and per-dataset facts become registry-driven extension seams, and the orchestrator's two hardcoded batching paths collapse behind one protocol.

**Tech Stack:** Python 3.11, pydantic v2, pytest, setuptools (editable install), conda (`agent_engine` env), SLURM.

**Spec:** `docs/superpowers/specs/2026-08-15-repo-handover-design.md`

## Progress

| Phase | Tasks | Commit | Status |
|---|---|---|---|
| 0 — safety net | 1-7 | `3f8b206`..`b85c95e` | done |
| 1 — packaging, dead path inserts | 8 | `4472f6a` | done |
| 2 — promote runner into `src/` | 9-10 | `7625a9e` | done |
| 3 — tool factory registry | 11 | `caf6ad8` | done |
| 4 — `DatasetSpec` | 12 | `2cee5a2` | done |
| 5 — orchestrator batching collapse | 13-15 | | next |
| 6 — analysis move + shims | 16 | | |
| 7 — tests for untested modules | 17 | | |
| 8 — docs, archive, final verification | 18-20 | | |

Keep this table current when a phase lands. Checkboxes alone proved too easy to
misread after a context handoff: an all-unticked plan sitting on top of ten
landed commits invites redoing finished work.

## Global Constraints

- **Behaviour must be identical, not equivalent.** Where cleanliness and behaviour-identity conflict, behaviour-identity wins and the ugliness is documented instead of fixed.
- **Python interpreter for all commands:** `/home/xchen1/.conda/envs/agent_engine/bin/python`. There is no `conda` on PATH in non-interactive shells; do not try to `conda activate`.
- **Repo root:** `/gpfs/home3/xchen1/azywot/msc-thesis`. Branch: `chore/refactor-the-code`.
- **Branch point is `6a4671b`, not `main`.** The branch already carried unrelated
  commits (SFT support, job renames) before this work started, and those touched
  `experiments/configs/`. Every "did I change X?" check must diff against
  `6a4671b..HEAD`, never `main..HEAD`, or it reports someone else's changes as
  this refactor's.
- **Baseline to preserve: 503 tests pass** as of the Phase 0 gate, 0 failures,
  0 skips. (Was 496 before Phase 0; the 7 added are the characterization gates.)
  Every later phase must meet or exceed 503. `tests/unit/test_fine_tuning_rollout.py`
  is still uncollectable in the `agent_engine` env (needs `agentops`) and is
  skipped by `tests/conftest.py`.
- **B4/B5 skip when `experiments/results/` is absent.** On this machine they run;
  on a fresh clone they skip and the suite reports 501. That is expected, not a
  regression -- but it means a run showing 501 has *not* verified the metrics or
  the failure classifier.
- **Never run `scripts/generate_configs.py` against `experiments/configs/`.** It is not idempotent and rewrites 10 committed LoRA configs. All config tests write to a temp directory.
- **Never modify `src/fine_tuning/agentflow/`** except to add `VENDORED.md`. It is vendored third-party code.
- **`classify_failure` is frozen.** Its logic is moved byte-identical, never edited.
- **No new features, no dependency upgrades, no performance work, no reformatting commits.**
- **Pre-existing bugs found by new tests are reported, not fixed.**
- **Commit per task.** Do not squash phases together.

## Verification gates

Referenced by number throughout. Run from the repo root.

| ID | Command | Expectation |
|---|---|---|
| B1 | `python -m pytest tests/characterization/test_configs_unchanged.py -q` | pass |
| B2 | `python -m pytest tests/characterization/test_prompts_unchanged.py -q` | pass |
| B3 | `python -m pytest tests/characterization/test_orchestrator_trace.py -q` | pass |
| B4 | `python -m pytest tests/characterization/test_metrics_replay.py -q` | pass |
| B5 | `python -m pytest tests/characterization/test_failure_modes_replay.py -q` | pass |
| B6 | `python -m pytest -q` | 503+ pass, 0 fail (501 on a checkout without `experiments/results/`) |

`ALL` means run every gate B1-B6.

---

## File Structure

### Created

| Path | Responsibility |
|---|---|
| `tests/conftest.py` | Skip-on-missing-import marker so root `pytest` works in either env |
| `tests/characterization/__init__.py` | Package marker |
| `tests/characterization/conftest.py` | `FIXTURE_DIR`, `--update-fixtures` flag, `assert_matches_fixture` helper |
| `tests/characterization/scripted_provider.py` | `ScriptedProvider` — returns queued outputs in order, records prompts |
| `tests/characterization/test_configs_unchanged.py` | B1 |
| `tests/characterization/test_prompts_unchanged.py` | B2 |
| `tests/characterization/test_orchestrator_trace.py` | B3 |
| `tests/characterization/test_metrics_replay.py` | B4 |
| `tests/characterization/test_failure_modes_replay.py` | B5 |
| `tests/characterization/fixtures/` | Committed recorded baselines |
| `src/agent_engine/runner/__init__.py` | Re-exports `run_experiment` |
| `src/agent_engine/runner/providers.py` | `setup_model_provider` |
| `src/agent_engine/runner/tools.py` | `ToolDeps`, `build_tool_registry` |
| `src/agent_engine/runner/metrics.py` | `compute_metrics`, `level_key` |
| `src/agent_engine/runner/experiment.py` | `run_experiment` and run-dir helpers |
| `src/agent_engine/tools/registry.py` | `@register_tool` decorator + factory table |
| `src/agent_engine/datasets/spec.py` | `DatasetSpec` + `DATASET_SPECS` + `get_spec` |
| `src/agent_engine/core/batching.py` | `BatchJob`, `BatchedTool` protocol, `flush_batches` |
| `src/agent_engine/analysis/` | Moved failure-mode analysis package |
| `src/fine_tuning/agentflow/VENDORED.md` | Upstream provenance + local modifications |
| `CONTRIBUTING.md` | Dev setup, tests, style, how to add things |
| `docs/architecture.md`, `docs/configuration.md` | Reference docs |
| `docs/guides/*.md` | Four how-to guides |
| `docs/pipelines/*.md` | Four pipeline docs |
| `docs/archive/` | Superseded docs with HISTORICAL banners |

### Modified

| Path | Change |
|---|---|
| `pyproject.toml` | Entry points, python_version fixes, real URLs |
| `scripts/run_experiment.py` | Becomes an argparse shim |
| `scripts/failure_modes/*.py` | Become `main()` shims |
| `src/agent_engine/core/orchestrator.py` | Batching extracted |
| `src/agent_engine/prompts/builder.py` | Template dispatch reads `DatasetSpec` |
| `src/gepa_integration/seed.py` | Real import, `sys.path` hack removed |
| `README.md` | Rewritten |

---

# PHASE 0 — Characterization fixtures

No production code is touched in this phase. Every task records current behaviour.

### Task 1: Fixture harness and update flag

**Files:**
- Create: `tests/characterization/__init__.py`
- Create: `tests/characterization/conftest.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `FIXTURE_DIR: Path`, `assert_matches_fixture(name: str, actual: str, update: bool) -> None`, pytest option `--update-fixtures`.

- [x] **Step 1: Create the root conftest that unblocks `pytest`**

`tests/conftest.py`:

```python
"""Root pytest configuration.

`tests/unit/test_fine_tuning_rollout.py` imports `agentops`, which is installed
only in the `cosmas-train` environment. Without this hook, collecting it in the
`agent_engine` environment raises ModuleNotFoundError and aborts the entire run
before any test executes. Skipping the module keeps `pytest` from the repo root
working in both environments without changing any test outcome.
"""

import importlib.util

import pytest

# module path fragment -> import that must be available for it to be collected
_REQUIRES = {
    "test_fine_tuning_rollout": "agentops",
}


def collect_ignore_glob():
    return []


def pytest_collection_modifyitems(config, items):
    return


def pytest_ignore_collect(collection_path, config):
    name = collection_path.name
    for fragment, module in _REQUIRES.items():
        if fragment in name and importlib.util.find_spec(module) is None:
            return True
    return False
```

- [x] **Step 2: Create the fixture harness**

`tests/characterization/__init__.py` is empty. `tests/characterization/conftest.py`:

```python
"""Characterization-fixture harness.

These tests lock CURRENT behaviour. They exist to prove a refactor changed
nothing. A fixture is regenerated only with an explicit --update-fixtures run,
so refreshing a baseline is always a deliberate, reviewable act.
"""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--update-fixtures",
        action="store_true",
        default=False,
        help="Rewrite characterization fixtures from current behaviour.",
    )


@pytest.fixture
def update_fixtures(request):
    return request.config.getoption("--update-fixtures")


def assert_matches_fixture(name: str, actual: str, update: bool) -> None:
    """Compare `actual` against the recorded fixture `name`.

    With update=True the fixture is (re)written and the test passes.
    """
    path = FIXTURE_DIR / name
    if update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    assert path.exists(), (
        f"Missing fixture {path}. Record it with:\n"
        f"    pytest tests/characterization -q --update-fixtures"
    )
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, f"Behaviour changed against fixture {name}"
```

- [x] **Step 3: Verify root pytest now collects cleanly**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest -q 2>&1 | tail -5`
Expected: `496 passed` with 0 errors (previously: collection error).

- [x] **Step 4: Commit**

```bash
git add tests/conftest.py tests/characterization/
git commit -m "test: add characterization fixture harness and unblock root pytest"
```

---

### Task 2: B1 — config generator fixture

**Files:**
- Create: `tests/characterization/test_configs_unchanged.py`
- Create: `tests/characterization/fixtures/configs.manifest`

**Interfaces:**
- Consumes: `assert_matches_fixture`, `update_fixtures` from Task 1.

**Critical:** `generate_configs.py` writes to paths derived from its own constants. It must be invoked with an output root pointed at a temp directory. Read `scripts/generate_configs.py` to find how the output root is determined; if it is hardcoded, monkeypatch that constant rather than editing the script in this phase.

- [x] **Step 1: Write the test**

```python
"""B1: `generate_configs.py` output must not change.

This compares the generator against a snapshot of its OWN output, not against
the committed `experiments/configs/` tree. The two have drifted (the committed
LoRA configs were hand-edited); a generator-vs-committed diff would fail here
for reasons unrelated to any refactor. See the spec's Open Decisions.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

from .conftest import assert_matches_fixture

REPO = Path(__file__).resolve().parents[2]


def _manifest(root: Path) -> str:
    """path -> sha256, sorted, one per line. Content-addressed and order-stable."""
    lines = []
    for p in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{p.relative_to(root).as_posix()}  {digest}")
    return "\n".join(lines) + "\n"


def test_generated_configs_unchanged(tmp_path, update_fixtures, monkeypatch):
    out = tmp_path / "configs"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, "scripts/generate_configs.py", "--output-root", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert_matches_fixture("configs.manifest", _manifest(out), update_fixtures)
```

- [x] **Step 2: Check whether `--output-root` exists**

Run: `grep -n "output.root\|add_argument\|OUTPUT_ROOT\|CONFIG_ROOT" scripts/generate_configs.py | head -20`

If no such flag exists, add it in this task as a **strictly additive, default-preserving** argument: the default must be the existing hardcoded root, so invoking the script with no flags behaves exactly as before. This is the one production edit permitted in Phase 0, because the fixture cannot be recorded safely without it.

- [x] **Step 3: Record the fixture**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/characterization/test_configs_unchanged.py -q --update-fixtures`
Expected: PASS, and `tests/characterization/fixtures/configs.manifest` now exists with ~417 lines.

- [x] **Step 4: Verify the fixture is not vacuous**

Temporarily change any literal string in `scripts/generate_configs.py` (e.g. a `description` field), then run B1.
Expected: FAIL with "Behaviour changed against fixture configs.manifest".
Revert the change, re-run B1, expect PASS.

- [x] **Step 5: Confirm `experiments/configs/` was never touched**

Run: `git status --porcelain experiments/configs | wc -l`
Expected: `0`

- [x] **Step 6: Commit**

```bash
git add tests/characterization/test_configs_unchanged.py tests/characterization/fixtures/configs.manifest scripts/generate_configs.py
git commit -m "test: lock generate_configs.py output with a characterization fixture"
```

---

### Task 3: B2 — prompt export fixture

**Files:**
- Create: `tests/characterization/test_prompts_unchanged.py`
- Create: `tests/characterization/fixtures/prompts.json`

- [x] **Step 1: Write the test**

```python
"""B2: exported system prompts and tool schemas must not change."""

import json
import subprocess
import sys
from pathlib import Path

from .conftest import assert_matches_fixture

REPO = Path(__file__).resolve().parents[2]


def test_exported_prompts_unchanged(tmp_path, update_fixtures):
    out = tmp_path / "prompts.json"
    result = subprocess.run(
        [sys.executable, "scripts/export_prompts.py", "--output", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    canonical = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert_matches_fixture("prompts.json", canonical, update_fixtures)
```

- [x] **Step 2: Confirm the CLI flag name**

Run: `grep -n "add_argument" scripts/export_prompts.py`
If the flag is not `--output`, use the real one in the test. Do not change the script.

- [x] **Step 3: Record and verify non-vacuous**

Run: `pytest tests/characterization/test_prompts_unchanged.py -q --update-fixtures` → PASS.
Then edit one word in `src/agent_engine/prompts/templates/system/base.yaml`, run B2, expect FAIL. Revert, expect PASS.

- [x] **Step 4: Commit**

```bash
git add tests/characterization/test_prompts_unchanged.py tests/characterization/fixtures/prompts.json
git commit -m "test: lock exported prompts and tool schemas"
```

---

### Task 4: Scripted provider

**Files:**
- Create: `tests/characterization/scripted_provider.py`

**Interfaces:**
- Produces: `ScriptedProvider(outputs: list[str], usage: dict | None = None)` with `.prompts_seen: list[str]`, implementing `BaseModelProvider`.

**Why not reuse `_MockProvider`:** `tests/unit/test_smoke.py:_MockProvider` returns the *same* text for every prompt. B3 needs a different output per turn (a web_search call, then a code_generator call, then a final answer), so it needs a queue.

- [x] **Step 1: Write the provider**

```python
"""A model provider that replays a fixed script of outputs.

Returns queued outputs in order, one per prompt in each `generate()` batch, and
records every prompt it was given so the trace fixture can assert on prompt
construction as well as control flow.
"""

from typing import Dict, List, Optional

from agent_engine.models.base import (
    BaseModelProvider,
    GenerationResult,
    ModelConfig,
    ModelFamily,
)

_DEFAULT_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


class ScriptedProvider(BaseModelProvider):
    def __init__(
        self,
        outputs: List[str],
        usage: Optional[Dict[str, int]] = None,
        name: str = "scripted",
    ):
        config = ModelConfig(
            name=name,
            family=ModelFamily.QWEN3,
            path_or_id=name,
            role="orchestrator",
            seed=7,
        )
        super().__init__(config)
        self._queue = list(outputs)
        self._usage = dict(usage) if usage is not None else dict(_DEFAULT_USAGE)
        self.prompts_seen: List[str] = []

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        self.prompts_seen.extend(prompts)
        results = []
        for _ in prompts:
            assert self._queue, "ScriptedProvider ran out of scripted outputs"
            results.append(
                GenerationResult(
                    text=self._queue.pop(0),
                    finish_reason="stop",
                    usage=dict(self._usage),
                )
            )
        return results

    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False) -> str:
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "".join(parts) + f"|thinking={use_thinking}|force={force_tool_call}"

    def cleanup(self):
        pass
```

- [x] **Step 2: Sanity check it imports**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -c "import sys; sys.path.insert(0,'tests'); from characterization.scripted_provider import ScriptedProvider; print('ok')"`
Expected: `ok`

- [x] **Step 3: Commit**

```bash
git add tests/characterization/scripted_provider.py
git commit -m "test: add scripted model provider for trace fixtures"
```

---

### Task 5: B3 — orchestrator trace fixture

**Files:**
- Create: `tests/characterization/test_orchestrator_trace.py`
- Create: `tests/characterization/fixtures/orchestrator_trace.txt`

This is **the** gate for Phase 5. It must cover, in one batch: multiple states in one turn, web + code + immediate tool calls together, an `analysis_cache` hit, a missing-argument error, and an exception raised during prepare.

**Interfaces:**
- Consumes: `ScriptedProvider` (Task 4), `assert_matches_fixture` (Task 1).

- [x] **Step 1: Write fake tools that exercise both deferral paths**

```python
"""B3: the orchestrator's batching control flow must not change.

Locks: tool-call sequence, committed message text, structured-memory contents,
per-state token usage, and the ORDER of all of it. Phase 5 collapses the
hardcoded _WebJob/_CodeJob paths behind one protocol; if this fixture stays
green the collapse preserved behaviour.
"""

import json
from typing import Any, Dict, List

from agent_engine.core.tool import BaseTool, ToolResult

from .conftest import assert_matches_fixture
from .scripted_provider import ScriptedProvider


class FakeWebSearch(BaseTool):
    """Mimics WebSearchTool's sub-agent (deferred) contract."""

    direct_mode = False

    def __init__(self, model_provider):
        self.model_provider = model_provider
        self.url_cache: Dict[str, str] = {}
        self.use_jina = False
        self._analysis_cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "fake web search"

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "web_search", "parameters": {}}}

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="direct", metadata={})

    def search_and_format(self, query: str) -> Dict[str, Any]:
        if query == "boom":
            raise RuntimeError("search backend exploded")
        return {"results": [{"title": query}], "urls_to_fetch": [], "url_snippets": {}}

    def _format_results(self, results, query) -> str:
        return f"RESULTS({query})"

    def build_analysis_prompt(self, query: str, formatted: str) -> str:
        return f"ANALYSE {query} :: {formatted}"


class FakeCodeGenerator(BaseTool):
    """Mimics CodeGeneratorTool's sub-agent (deferred) contract."""

    direct_mode = False

    def __init__(self, model_provider):
        self.model_provider = model_provider

    @property
    def name(self) -> str:
        return "code_generator"

    @property
    def description(self) -> str:
        return "fake code generator"

    def get_schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "code_generator", "parameters": {}}}

    def build_task_prompt(self, task: str, context=None) -> str:
        return f"WRITE CODE FOR {task}"

    def extract_code_from_llm_response(self, text: str) -> str:
        return text.strip()

    def execute(self, code=None, task=None) -> ToolResult:
        return ToolResult(success=True, output=f"RAN[{code}]", metadata={})
```

- [x] **Step 2: Write the trace serialiser and the test**

The scenario below is chosen so each of the five required conditions is reachable.
**The `analysis_cache` hit must occur in a *later turn* than the miss**, because the
cache is only written during finalize: two jobs with the same query in the *same*
turn both miss and both defer.

Turn-by-turn script for five states (non-baseline, so turn 0 is the planning turn):

| Turn | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| 0 (planning) | analysis | analysis | analysis | analysis | analysis |
| 1 | `web("alpha")` deferred | `code("sum")` deferred | `web("boom")` prepare raises | `web()` no query, error | final answer |
| 2 | final answer | final answer | `web("alpha")` **cache hit** | final answer | done |
| 3 | done | done | final answer | done | done |

Generation counts: orchestrator provider 5 + 5 + 4 + 1 = **15**; web sub-agent provider
**1** (only s0's job in turn 1; s2's turn-2 call is a cache hit and never generates);
code sub-agent provider **1**.

```python
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry


def _call(name: str, **args) -> str:
    """Qwen3 tool-call format, which is what parse_tool_call matches first."""
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


def _serialise(states) -> str:
    """Deterministic, diff-friendly dump of everything B3 locks."""
    out = []
    for s in sorted(states, key=lambda x: x.question_id):
        out.append(f"=== state {s.question_id} ===")
        out.append(f"finished={s.finished} turn={s.turn} answer={s.answer!r}")
        out.append(f"tool_counts={json.dumps(s.tool_counts, sort_keys=True)}")
        out.append(f"token_usage={json.dumps(s.metadata.get('token_usage', {}), sort_keys=True)}")
        out.append(f"query_analysis={s.query_analysis!r}")
        out.append("action_history:")
        for i, step in enumerate(s.action_history):
            out.append(f"  [{i}] {json.dumps(step, sort_keys=True)}")
        out.append("output_messages:")
        for i, msg in enumerate(s.output_messages):
            out.append(f"  [{i}] {msg['role']}: {msg['content']}")
    return "\n".join(out) + "\n"


def test_orchestrator_trace_unchanged(update_fixtures):
    orch_outputs = (
        [f"PLAN for state {i}" for i in range(5)]
        + [
            _call("web_search", query="alpha"),
            _call("code_generator", task="sum"),
            _call("web_search", query="boom"),
            _call("web_search"),
            "Final Answer: four",
        ]
        + [
            "Final Answer: zero",
            "Final Answer: one",
            _call("web_search", query="alpha"),
            "Final Answer: three",
        ]
        + ["Final Answer: two"]
    )

    orch_provider = ScriptedProvider(orch_outputs, name="orchestrator")
    web_provider = ScriptedProvider(["ANALYSIS OF ALPHA"], name="web-subagent")
    code_provider = ScriptedProvider(["print(1 + 1)"], name="code-subagent")

    tools = ToolRegistry()
    tools.register(FakeWebSearch(web_provider))
    tools.register(FakeCodeGenerator(code_provider))

    orchestrator = AgenticOrchestrator(
        model_provider=orch_provider,
        tool_registry=tools,
        max_turns=15,
        use_thinking=False,
        baseline=False,
    )

    states = orchestrator.run_batch(
        questions=[f"question {i}" for i in range(5)],
        question_ids=list(range(5)),
        system_prompts=["SYSTEM"] * 5,
    )

    assert not orch_provider._queue, "scripted turns left unconsumed — scenario drifted"
    assert_matches_fixture("orchestrator_trace.txt", _serialise(states), update_fixtures)
```

**If the scenario drifts** (the leftover-queue assertion fires), print
`orch_provider.prompts_seen` to see how many generate calls actually happened and adjust
the queue. Do not silently pad the queue: a changed call count is itself a behaviour
change worth understanding.

- [x] **Step 3: Record the fixture**

Run: `pytest tests/characterization/test_orchestrator_trace.py -q --update-fixtures`
Expected: PASS, fixture written.

- [x] **Step 4: Prove the fixture is not vacuous — this step is mandatory**

Make each of these three mutations one at a time in `src/agent_engine/core/orchestrator.py`, run B3, confirm FAIL, then revert:

1. In `_process_batch_turn`, swap the flush order: call `self._flush_code_batch(code_jobs)` before `self._flush_web_batch(web_jobs)`.
2. In `_run_code_generation_batch`, also accumulate the finalize result's usage: add `_accumulate_usage(job.state, tr.usage)` after the `execute` call.
3. In `_apply_immediate_results`, drop the `strip_thinking_tags` call.

If any mutation does **not** turn B3 red, the fixture is too weak: extend it until it does. A fixture that never fails is worse than no fixture.

- [x] **Step 5: Commit**

```bash
git add tests/characterization/test_orchestrator_trace.py tests/characterization/fixtures/orchestrator_trace.txt
git commit -m "test: lock orchestrator batching trace (gate for the batching collapse)"
```

---

### Task 6: B4 and B5 — replay fixtures over real runs

**Files:**
- Create: `tests/characterization/test_metrics_replay.py`
- Create: `tests/characterization/test_failure_modes_replay.py`
- Create: `tests/characterization/fixtures/metrics_replay.json`
- Create: `tests/characterization/fixtures/failure_modes_replay.json`

**Critical:** `experiments/results/` is gitignored and 12 GB. Fixtures must **not** copy run data. Pick a small, fixed set of runs, record only the *derived output*, and skip the test with a clear reason if the runs are absent, so the suite still passes on a fresh clone.

- [x] **Step 1: Choose the replay corpus**

Run: `find experiments/results -name raw_results.json | sort | head -20`
Pick 3 runs spanning different datasets (at least one GAIA for `per_level`, one AIME). Record their repo-relative paths as a module constant `REPLAY_RUNS`.

- [x] **Step 2: Write the metrics replay test**

Verified shape: `raw_results.json` is a **bare JSON list** (not a dict with a `results`
key). Each row has `question_id`, `metadata` (e.g. `{"year": 2024, "problem_id": 60}` for
AIME), `evaluation` (`{"correct", "accuracy", "em", "f1"}`), `tool_counts`, `token_usage`,
`prediction`, `ground_truth`. `compute_metrics` reads `example.metadata` through
`level_key`, so the shim only needs a `.metadata` attribute.

```python
"""B4: metrics aggregation over recorded runs must not change."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .conftest import assert_matches_fixture

REPO = Path(__file__).resolve().parents[2]

# Repo-relative, chosen in Step 1 to span a stratified dataset (gaia -> per_level
# by "level"), a year-stratified one (aime -> per_level by "year"), and an
# unstratified one (gpqa -> no per_level block).
REPLAY_RUNS = [
    # "experiments/results/.../raw_results.json",
]


def _dataset_name_for(run: str) -> str:
    """Dataset name is a path component of every results run directory."""
    parts = Path(run).parts
    for known in ("gaia", "aime", "gpqa", "hle", "musique", "math500", "amc", "bigcodebench"):
        if known in parts:
            return known
    raise AssertionError(f"cannot infer dataset from run path: {run}")


def test_metrics_replay_unchanged(update_fixtures):
    missing = [r for r in REPLAY_RUNS if not (REPO / r).exists()]
    if missing:
        pytest.skip(f"replay corpus not present in this checkout: {missing}")

    from agent_engine.runner.metrics import compute_metrics

    payload = {}
    for run in REPLAY_RUNS:
        rows = json.loads((REPO / run).read_text(encoding="utf-8"))
        assert isinstance(rows, list), f"expected a list of rows in {run}"
        examples = [SimpleNamespace(metadata=r.get("metadata") or {}) for r in rows]
        payload[run] = compute_metrics(rows, examples, _dataset_name_for(run))

    assert_matches_fixture(
        "metrics_replay.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        update_fixtures,
    )
```

**Import note:** this imports `agent_engine.runner.metrics`, which does not exist until
Task 9. Until then, import from the script by file path:

```python
    import importlib.util
    spec = importlib.util.spec_from_file_location("_runexp", REPO / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    compute_metrics = mod._compute_metrics
```

Task 10 Step 3 switches it to the package import. The **fixture must not be regenerated**
at that switch.

- [x] **Step 3: Write the failure-mode replay test**

```python
"""B5: failure-mode classification over recorded runs must not change.

`classify_failure` is frozen: Chapter 6's taxonomy counts come from it. This
fixture is what makes "frozen" enforceable rather than aspirational.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from .conftest import assert_matches_fixture

REPO = Path(__file__).resolve().parents[2]

REPLAY_RUNS = [
    # same list as test_metrics_replay.py
]


def test_failure_modes_replay_unchanged(update_fixtures):
    missing = [r for r in REPLAY_RUNS if not (REPO / r).exists()]
    if missing:
        pytest.skip(f"replay corpus not present in this checkout: {missing}")

    from agent_engine.analysis.failure_modes import classify_failure

    payload = {}
    for run in REPLAY_RUNS:
        rows = json.loads((REPO / run).read_text(encoding="utf-8"))
        per_question = {}
        for row in rows:
            evaluation = row.get("evaluation") or {}
            if evaluation.get("correct"):
                continue  # classifier only runs on failures
            per_question[str(row["question_id"])] = classify_failure(row)
        payload[run] = {
            "per_question": dict(sorted(per_question.items())),
            "counts": dict(sorted(Counter(per_question.values()).items())),
        }

    assert_matches_fixture(
        "failure_modes_replay.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        update_fixtures,
    )
```

**Before recording:** confirm `classify_failure`'s real signature and return type with
`grep -n "def classify_failure" -A 15 scripts/failure_modes/analyze_failure_modes.py`. If
it takes more than a row (e.g. a dataset name or a config), pass the real arguments and
adjust. If it returns an object rather than a string, serialise its fields explicitly.
Until Task 16 the import is `from failure_modes.analyze_failure_modes import classify_failure`
with the existing `sys.path` insert, matching what `tests/unit/test_analyze_failure_modes.py`
already does.

- [x] **Step 4: Record both fixtures**

Run: `pytest tests/characterization/test_metrics_replay.py tests/characterization/test_failure_modes_replay.py -q --update-fixtures`
Expected: PASS (not skipped — if skipped, the corpus paths are wrong).

- [x] **Step 5: Prove non-vacuous**

Change `_STRATIFIED` in `run_experiment.py` to drop `"gaia"`, run B4, expect FAIL, revert.
Change one threshold in `classify_failure`, run B5, expect FAIL, revert.

- [x] **Step 6: Commit**

```bash
git add tests/characterization/test_metrics_replay.py tests/characterization/test_failure_modes_replay.py tests/characterization/fixtures/
git commit -m "test: lock metrics and failure-mode classification by replay over recorded runs"
```

---

### Task 7: Phase 0 gate

- [x] **Step 1: Run everything**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest -q`
Expected: all pass, 0 errors, 0 unexpected skips.

- [x] **Step 2: Confirm no production behaviour changed in Phase 0**

Run: `git diff main --stat -- src/ scripts/`
Expected: only `scripts/generate_configs.py` (the additive `--output-root` flag from Task 2). Nothing else.

- [x] **Step 3: Record the baseline count**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest -q 2>&1 | tail -1`
Write the number into the plan's Global Constraints as the running baseline. Every later phase must meet or exceed it.

---

# PHASE 1 — Packaging and imports

### Task 8: Entry points, metadata, dead path inserts

**Files:**
- Modify: `pyproject.toml`
- Modify: 14 files containing `sys.path.insert(..., "src")`

- [x] **Step 1: List the exact inserts to delete**

Run:
```bash
grep -rn 'sys.path.insert' src/ scripts/ tests/ examples/ --include='*.py' | grep '"src"\|/ "src"'
```
Delete only the ones resolving to `src`. **Keep** every insert resolving to `scripts` — those are load-bearing until Phase 6.

- [x] **Step 2: Delete them, plus any now-unused `import sys` / `from pathlib import Path`**

Only remove the imports if nothing else in the file uses them. Check each file.

- [x] **Step 3: Fix packaging metadata in `pyproject.toml`**

**No `[project.scripts]` block is added in this task.** An entry point whose target module
does not yet exist installs a console script that crashes on invocation, so each one is
added in the phase that creates its target: `cosmas-run` in Task 10, `cosmas-analyze` in
Task 16. Restrict this task's `pyproject.toml` changes to:

```toml
[tool.mypy]
python_version = "3.11"

[tool.black]
target-version = ["py311"]

[project.urls]
Homepage = "https://github.com/azywot/msc-thesis"
Repository = "https://github.com/azywot/msc-thesis"
```

- [x] **Step 4: Verify imports still resolve without the inserts**

Run:
```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -c "import agent_engine, gepa_integration, verl_ext; print('ok')"
/home/xchen1/.conda/envs/agent_engine/bin/python scripts/export_prompts.py --output /tmp/p.json && echo "script ok"
```
Expected: both succeed.

- [x] **Step 5: Run ALL gates**

Expected: B1-B6 all pass.

- [x] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: drop redundant sys.path inserts, correct packaging metadata"
```

---

# PHASE 2 — Promote the runner

### Task 9: Extract `setup_model_provider` and `compute_metrics`

**Files:**
- Create: `src/agent_engine/runner/__init__.py`, `providers.py`, `metrics.py`
- Modify: `scripts/run_experiment.py`

**Interfaces:**
- Produces: `agent_engine.runner.providers.setup_model_provider(model_config, api_keys, model_cache=None)`; `agent_engine.runner.metrics.compute_metrics(results, examples, dataset_name)` and `level_key(example, dataset_name)`.

- [x] **Step 1: Move the code verbatim**

Cut `setup_model_provider` (`scripts/run_experiment.py:42-99`) into `providers.py`, and `_level_key` + `_compute_metrics` (`:580-683`) into `metrics.py`, renamed to `level_key` and `compute_metrics`. Copy the bodies **character for character**, including comments and the `_STRATIFIED` set. Add the module imports each needs.

- [x] **Step 2: Re-export from the script for backwards compatibility**

In `scripts/run_experiment.py`:

```python
from agent_engine.runner.metrics import compute_metrics as _compute_metrics
from agent_engine.runner.metrics import level_key as _level_key
from agent_engine.runner.providers import setup_model_provider
```

The old private names stay bound so B4's dynamic import of the script keeps working unchanged.

- [x] **Step 3: Run ALL gates**

Expected: B1-B6 pass. B4 especially — it loads `run_experiment.py` and calls `_compute_metrics`.

- [x] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: promote model-provider setup and metrics into agent_engine.runner"
```

---

### Task 10: Extract `run_experiment` itself

**Files:**
- Create: `src/agent_engine/runner/experiment.py`
- Modify: `scripts/run_experiment.py`

- [x] **Step 1: Move `run_experiment`, `_make_run_dir`, `_write_json`, `_short_id`, `_config_to_dict`**

Move `scripts/run_experiment.py:201-579` verbatim into `experiment.py`. Move `main()` too. Keep every log message identical, including emoji and spacing — `experiment.log` is a run artefact.

- [x] **Step 2: Reduce the script to a shim**

```python
#!/usr/bin/env python
"""Thin CLI wrapper. Implementation lives in agent_engine.runner.experiment."""

from agent_engine.runner.experiment import main

if __name__ == "__main__":
    main()
```

- [x] **Step 3: Update B4's import**

`test_metrics_replay.py` currently loads the script by file path. Change it to `from agent_engine.runner.metrics import compute_metrics`. The recorded fixture must **not** be regenerated — it must still match.

- [x] **Step 4: Add the entry point**

```toml
[project.scripts]
cosmas-run = "agent_engine.runner.experiment:main"
```

Then reinstall: `/home/xchen1/.conda/envs/agent_engine/bin/pip install -e . --no-deps`

- [x] **Step 5: Verify the CLI still works both ways**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python scripts/run_experiment.py --help
/home/xchen1/.conda/envs/agent_engine/bin/cosmas-run --help
```
Expected: identical help text.

- [x] **Step 6: Run ALL gates, then commit**

```bash
git add -A
git commit -m "refactor: move run_experiment into agent_engine.runner, script becomes a shim"
```

---

# PHASE 3 — Tool factory registry

### Task 11: `@register_tool` and `build_tool_registry`

**Files:**
- Create: `src/agent_engine/tools/registry.py`
- Create: `src/agent_engine/runner/tools.py`
- Modify: each of the five tool modules
- Modify: `src/agent_engine/tools/__init__.py`

**Interfaces:**
- Produces: `ToolDeps` (frozen dataclass: `config`, `cache_manager`, `api_keys`, `model_providers`, `orchestrator_model`, `mind_map_storage_path`); `@register_tool(name)`; `build_tool_registry(deps) -> ToolRegistry`.

- [x] **Step 1: Write the registry**

```python
"""Tool construction registry.

`BaseTool`/`ToolRegistry` in core/tool.py handle tool *behaviour*. This module
handles tool *construction*: how a name in `config.tools.enabled_tools` becomes
a configured instance. Adding a sub-agent means writing a factory here, not
editing a dispatch chain in a script.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..core.tool import BaseTool

_FACTORIES: Dict[str, Callable[["ToolDeps"], Optional[BaseTool]]] = {}


@dataclass(frozen=True)
class ToolDeps:
    config: Any
    cache_manager: Any
    api_keys: Dict[str, Optional[str]]
    model_providers: Dict[str, Any]
    orchestrator_model: Any = None
    mind_map_storage_path: Optional[Path] = None

    @property
    def direct_mode(self) -> bool:
        return self.config.tools.direct_tool_call

    @property
    def use_subagent_thinking(self) -> bool:
        return self.config.use_subagent_thinking()

    def provider_for(self, tool_name: str):
        """Sub-agent model for `tool_name`, or None in direct mode."""
        if self.direct_mode or not self.model_providers:
            return None
        return self.model_providers.get(tool_name)


def register_tool(name: str):
    def wrapper(factory):
        if name in _FACTORIES:
            raise ValueError(f"Tool factory '{name}' is already registered")
        _FACTORIES[name] = factory
        return factory
    return wrapper


def build_tool(name: str, deps: ToolDeps) -> Optional[BaseTool]:
    """Construct one tool. Returns None if the factory declines (e.g. image
    inspector in direct mode), matching the old if/elif's skip behaviour."""
    factory = _FACTORIES.get(name)
    if factory is None:
        return None
    return factory(deps)


def registered_tools():
    return sorted(_FACTORIES)
```

- [x] **Step 2: Move each construction block verbatim into a decorated factory**

For each of the five tools, move the body from `setup_tools`'s `if/elif` into the tool's own module. Example for web_search, in `src/agent_engine/tools/web_search.py`:

```python
@register_tool("web_search")
def build_web_search(deps) -> "WebSearchTool":
    provider = deps.config.tools.web_tool_provider
    api_key = deps.api_keys.get(provider)
    if not api_key:
        raise RuntimeError(
            f"{provider.upper()}_API_KEY environment variable is required "
            f"for web_tool_provider='{provider}'"
        )
    return WebSearchTool(
        api_key=api_key,
        provider=provider,
        search_cache=deps.cache_manager.search_cache,
        url_cache=deps.cache_manager.url_cache,
        top_k=deps.config.tools.top_k_results,
        max_doc_len=deps.config.tools.max_doc_len,
        model_provider=deps.provider_for("web_search"),
        fetch_urls=True,
        use_thinking=deps.use_subagent_thinking,
        cache_manager=deps.cache_manager,
        max_search_content_chars=deps.config.tools.max_search_content_chars,
    )
```

**Preserve the quirks exactly:** the `mind_map` factory must keep the `mind_map_storage_path.mkdir(parents=True, exist_ok=True)` side effect and the `direct_mode` (not `provider_for`) provider selection; the `image_inspector` factory must return `None` in direct mode and log the same warning; `text_inspector` keeps `max_chars=50000`; `code_generator` keeps `timeout_seconds=60` and `temp_dir=str(config.cache_dir / "code_temp")`. Re-read `scripts/run_experiment.py:100-200` line by line while doing this; do not work from memory.

- [x] **Step 3: Write `build_tool_registry` in `runner/tools.py`**

```python
def build_tool_registry(deps: ToolDeps) -> ToolRegistry:
    tools = ToolRegistry()
    for tool_name in deps.config.tools.enabled_tools:
        tool = build_tool(tool_name, deps)
        if tool is not None:
            tools.register(tool)
    return tools
```

- [x] **Step 4: Delete `setup_tools` and call the new path from `experiment.py`**

- [x] **Step 5: Add seam-completeness tests**

Both seams get a test that fails loudly when someone adds a thing halfway. The
model-family one covers the spec's "model families" seam, which has no other task.

```python
# tests/unit/test_extension_seams.py
def test_every_default_tool_has_a_factory():
    assert set(registered_tools()) == {
        "web_search", "code_generator", "mind_map",
        "text_inspector", "image_inspector",
    }


@pytest.mark.parametrize("family", list(ModelFamily), ids=lambda f: f.name)
def test_every_model_family_resolves_to_a_tool_call_format(family):
    assert isinstance(get_tool_call_format(family), ToolCallFormat)


@pytest.mark.parametrize("name, members", [...])  # the six family tables
def test_family_tables_contain_only_real_families(name, members):
    stale = [m for m in members if not isinstance(m, ModelFamily)]
    assert not stale, f"{name} contains non-ModelFamily entries: {stale}"
```

**Corrected while executing (2026-08-16).** This step originally specified
`test_every_model_family_has_a_tool_call_format`, asserting that every
`ModelFamily` appears in `_TOOL_CALL_FORMAT`, and instructed that a failure be
`xfail`-ed as a pre-existing gap. That was wrong on both counts:
`_TOOL_CALL_FORMAT` is **sparse by design** — `models/base.py` says "Unlisted
families default to JSON" and `get_tool_call_format` resolves with
`.get(family, ToolCallFormat.JSON)` — so seven perfectly correct families
failed it. It was not a latent bug, so `xfail` would have parked a permanent
false accusation in the suite.

The replacement asserts what actually holds (every family *resolves* to a
format) plus the failure a sparse table can genuinely have: a stale or
misspelled entry left behind after a family is renamed, which is silently inert
because the lookup simply never matches. The same check covers all six
`_*_FAMILIES` tables.

**General lesson for the remaining tasks:** a seam test must assert the
invariant the code actually maintains. Before writing one from this plan, read
the lookup's default path — several of these tables are intentionally sparse.

- [x] **Step 6: Run ALL gates, then commit**

```bash
git add -A
git commit -m "refactor: tool construction moves to a factory registry"
```

---

# PHASE 4 — Dataset registry consolidation

### Task 12: `DatasetSpec`

**Files:**
- Create: `src/agent_engine/datasets/spec.py`
- Modify: `src/agent_engine/prompts/builder.py:82-96`
- Modify: `src/agent_engine/runner/metrics.py`

**Interfaces:**
- Produces: `DatasetSpec(template, stratified, level_field, level_fallback_field)`, `get_spec(name) -> DatasetSpec`.

**Critical constraint:** `builder.py` maps `"deepmath"` and `"math"` to the math template, but **neither is a registered dataset** — they are RL-prep names with no loader. The spec table must therefore be keyed **independently of `DatasetRegistry`**, and `get_spec` must return a safe default for unknown names so the existing `FileNotFoundError` → base-template fallback still works.

- [x] **Step 1: Write the spec table**

```python
"""Per-dataset facts that are not the loader's business.

Before this module these lived as string literals in three places:
prompts/builder.py's template dispatch, runner/metrics.py's _STRATIFIED set,
and runner/metrics.py's level_key(). Adding a benchmark meant finding all
three. Now it means adding one row here.

Keyed by dataset NAME, deliberately not by DatasetRegistry membership:
"math" and "deepmath" have templates but no loader (they are RL data-prep
names), and must keep resolving to the math template.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    template: str = "base"
    stratified: bool = False
    level_field: Optional[str] = None
    level_fallback_field: Optional[str] = None


_DEFAULT = DatasetSpec()

DATASET_SPECS = {
    "gaia":          DatasetSpec("gaia", True, "level"),
    "hle":           DatasetSpec("gaia", True, "category"),
    "musique":       DatasetSpec("gaia", False),
    "aime":          DatasetSpec("math", True, "year"),
    "math500":       DatasetSpec("math", True, "difficulty", "year"),
    "amc":           DatasetSpec("math", True, "difficulty", "year"),
    "deepmath":      DatasetSpec("math", False),
    "math":          DatasetSpec("math", False),
    "gpqa":          DatasetSpec("gpqa", False),
    "bigcodebench":  DatasetSpec("bigcodebench", False),
}


def get_spec(name: str) -> DatasetSpec:
    return DATASET_SPECS.get((name or "").lower(), _DEFAULT)
```

**Verify this table against the source before trusting it.** Cross-check every row against `prompts/builder.py:82-91`, `_STRATIFIED` in `run_experiment.py:614`, and `_level_key` at `:580-592`. Note `hle` is stratified by `category`; `musique` uses the gaia template but is **not** stratified; `math500`/`amc` fall back from `difficulty` to `year`.

- [x] **Step 2: Rewrite the builder dispatch**

Replace `builder.py:82-91` with:

```python
template_name = get_spec(dataset_name).template
if baseline:
    template_name = f"{template_name}_baseline"
template = self.load_template(template_name)
```

Confirm the `_baseline` suffix convention holds for every template in `src/agent_engine/prompts/templates/system/` before relying on it. Keep the surrounding `try/except FileNotFoundError` fallback untouched.

- [x] **Step 3: Rewrite `level_key` and `_STRATIFIED` in `runner/metrics.py`**

```python
def level_key(example, dataset_name: str) -> str:
    spec = get_spec(dataset_name)
    if not spec.level_field:
        return "all"
    value = example.metadata.get(spec.level_field)
    if value is None and spec.level_fallback_field:
        value = example.metadata.get(spec.level_fallback_field)
    return str(value if value is not None else "unknown")
```

and replace `dataset_name in _STRATIFIED` with `get_spec(dataset_name).stratified`.

- [x] **Step 4: Run B2 and B4 first — they are the gate for this task**

Run: `pytest tests/characterization/test_prompts_unchanged.py tests/characterization/test_metrics_replay.py -q`
Expected: PASS with fixtures **unmodified**. If either fails, the spec table is wrong; fix the table, never the fixture.

- [x] **Step 5: Run ALL gates, then commit**

```bash
git add -A
git commit -m "refactor: consolidate per-dataset facts into DatasetSpec"
```

**Corrected while executing (2026-08-16).** The mappings in this task's table
were right; three of its *mechanics* were not, and each would have changed
behaviour:

1. **`get_spec` must not lowercase.** `level_key` and `_STRATIFIED` compared the
   raw `dataset_name`, so `"GAIA"` was unstratified with no level field, while
   `builder.py` lowercased before dispatching. Folding case inside `get_spec`
   would have made `"GAIA"` stratified. The builder lowercases at its call site
   instead, exactly as before.
2. **`level_key` must keep `dict.get(key, default)`, not a `None` check.** The
   planned `value = md.get(f); if value is None and fallback:` differs from the
   original nested `.get` whenever a key is *present with value `None`*: the
   original yields `"None"`, the rewrite yields the fallback or `"unknown"`.
3. **`template` defaults to `None`, not `"base"`.** The old dispatch left
   `template_name = dataset_name` for unrecognised names, so the lookup failed
   and the `FileNotFoundError` handler logged `Template 'X' not found, using
   base template`. Resolving straight to `"base"` reaches the same prompt but
   skips that warning, which is a run artefact in `experiment.log`.

Verified by differential test against the pre-change logic: 42 template-dispatch
cases, 252 `level_key` cases (including present-but-`None` metadata), and 21
stratification cases, all with 0 mismatches.

---

# PHASE 5 — Orchestrator batching collapse

### Task 13: `BatchJob` and the `BatchedTool` protocol

**Files:**
- Create: `src/agent_engine/core/batching.py`

**Interfaces:**
- Produces: `BatchJob`, `BatchedTool` protocol, `flush_batches(jobs_by_tool, commit, accumulate_usage)`.

**This is the highest-risk task in the plan. B3 is the gate. Do not proceed to Task 14 until B3 is green.**

- [ ] **Step 1: Write the protocol**

```python
"""Tool-agnostic batching for sub-agent tools.

A sub-agent tool defers work: a cheap prepare step builds a prompt, all
prepared prompts are generated in one batched LLM call, then a finalize step
turns each generation into a ToolResult. Before this module the orchestrator
hardcoded two copies of this (web_search and code_generator) as _WebJob and
_CodeJob. A new batched sub-agent now implements this protocol instead of
requiring surgery on the core loop.

Behaviour notes preserved verbatim from the previous implementation:
  * Flush order is by `batch_priority` ASCENDING (web=10, code=20). This was
    hardcoded as "web then code" and IS observable: web analysis populates
    caches that later code jobs in the same turn can read.
  * The deferred path accumulates ONLY `generation.usage`, never the finalize
    ToolResult's usage. The immediate path does the opposite. Unifying these
    double-counts code-generator tokens.
  * Grouping is by `id(tool.model_provider)` — identity, not equality.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Protocol, runtime_checkable

from .tool import BaseTool, ToolResult


class BatchJob(NamedTuple):
    state: Any
    tool_call: Dict[str, Any]
    tool: BaseTool
    payload: Dict[str, Any]


@runtime_checkable
class BatchedTool(Protocol):
    batch_priority: int

    def prepare(self, state, tool_call, args) -> Any:
        """Return a BatchJob to defer, or a ToolResult to short-circuit."""

    def batch_prompt(self, job: BatchJob) -> str: ...

    def finalize(self, job: BatchJob, generation) -> ToolResult: ...

    def pre_batch(self, jobs: List[BatchJob]) -> None:
        """Optional hook run once over ALL jobs for this tool, before grouping."""
```

- [ ] **Step 2: Write `flush_batches`**

```python
def flush_batches(jobs_by_tool, commit, accumulate_usage) -> None:
    """Run every deferred job, in the order the old implementation used.

    `commit(state, tool_call, text)` and `accumulate_usage(state, usage)` are
    passed in so this module never imports the orchestrator.
    """
    ordered = sorted(
        jobs_by_tool.items(),
        key=lambda kv: getattr(kv[1][0].tool, "batch_priority", 100),
    )
    for _tool_name, jobs in ordered:
        if not jobs:
            continue
        tool = jobs[0].tool
        pre = getattr(tool, "pre_batch", None)
        if pre is not None:
            pre(jobs)

        groups: Dict[int, List[BatchJob]] = {}
        for job in jobs:
            groups.setdefault(id(getattr(job.tool, "model_provider", None)), []).append(job)

        for group in groups.values():
            provider = getattr(group[0].tool, "model_provider", None)
            prompts = [group[0].tool.batch_prompt(j) for j in group]
            outputs = provider.generate(prompts) if provider else []
            for job, out in zip(group, outputs):
                accumulate_usage(job.state, out.usage)
                result = job.tool.finalize(job, out)
                commit(job.state, job.tool_call, result.output or "")
```

**Check against the original before believing this is right.** In particular: the original computes prompts with `job.tool.<method>` per job, not `group[0].tool`; if a group can ever mix tool instances, use `job.tool`. Re-read `_run_web_analysis_batch` and `_run_code_generation_batch` and match exactly.

- [ ] **Step 3: Commit the module alone (not yet wired in)**

```bash
git add src/agent_engine/core/batching.py
git commit -m "feat: add tool-agnostic batching protocol (not yet wired)"
```

---

### Task 14: Implement the protocol on the two real tools

**Files:**
- Modify: `src/agent_engine/tools/web_search.py`
- Modify: `src/agent_engine/tools/code_generator.py`

- [ ] **Step 1: Implement on `WebSearchTool`**

`batch_priority = 10`. `prepare` carries over, in order: the missing-`query` guard returning a failed `ToolResult`, the `analysis_cache` hit returning a successful cached `ToolResult` with `metadata={"cached": True, "query": query, "mode": "sub-agent"}`, and the `search_and_format` call whose exception becomes a failed `ToolResult`. `pre_batch` performs the cross-job URL fetch (`fetch_page_content`, `url_cache.update`, `cache_manager.save_url_cache`). `batch_prompt` is `build_analysis_prompt(query, self._format_results(payload["results"], query))`. `finalize` strips thinking tags, writes `analysis_cache[query]`, returns `ToolResult(success=True, output=text, ...)`.

- [ ] **Step 2: Implement on `CodeGeneratorTool`**

`batch_priority = 20`. `prepare` carries over the missing-`task` guard and `build_task_prompt(task, context=get_attachment_context_for_code(state))`, with exceptions becoming failed `ToolResult`s. No `pre_batch`. `finalize` strips, calls `extract_code_from_llm_response`, logs the `Tool call:` line **at that point** (not earlier — the log order differs from web on purpose), calls `execute(code=code, task=None)`, and returns the result.

- [ ] **Step 3: Do not wire the orchestrator yet. Run ALL gates.**

Expected: B3 still green — nothing calls the new methods yet.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: implement BatchedTool on web_search and code_generator"
```

---

### Task 15: Wire the orchestrator and delete the old paths

**Files:**
- Modify: `src/agent_engine/core/orchestrator.py`

- [ ] **Step 1: Replace `_classify_tool_call`'s dispatch**

A tool is deferred if it satisfies the `BatchedTool` protocol **and** `not getattr(tool, "direct_mode", True)`. Otherwise it is immediate. Preserve the `mind_map` branch's `tool.set_current_question(state.question_id)` side effect and the `_index_reasoning_in_mind_map` call, which happens for **every** tool call before dispatch.

- [ ] **Step 2: Replace the flush block in `_process_batch_turn`**

```python
self._apply_immediate_results(immediate_results)
if jobs_by_tool:
    flush_batches(jobs_by_tool, self._commit_batched_result, _accumulate_usage)
```

where `_commit_batched_result(state, tool_call, text)` applies `strip_thinking_tags` then `self._commit_tool_result(...)`, matching the old commit path.

- [ ] **Step 3: Delete `_WebJob`, `_CodeJob`, `_schedule_web_job`, `_schedule_code_job`, `_flush_web_batch`, `_fetch_urls_for_web_jobs`, `_run_web_analysis_batch`, `_flush_code_batch`, `_run_code_generation_batch`, `_get_analysis_cache`**

- [ ] **Step 4: Run B3 — the moment of truth**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/characterization/test_orchestrator_trace.py -q`
Expected: PASS against the **unmodified** fixture.

If it fails, diff the actual trace against the fixture and fix the code, **never** the fixture. If it cannot be made green within reasonable effort, revert Tasks 13-15 (`git revert`) and report; every other phase still stands.

- [ ] **Step 5: Run ALL gates, then commit**

```bash
git add -A
git commit -m "refactor: collapse web/code batching behind the BatchedTool protocol"
```

---

# PHASE 6 — Analysis move

### Task 16: Move `scripts/failure_modes/` into `src/agent_engine/analysis/`

**Files:**
- Create: `src/agent_engine/analysis/` (mirroring the current tree)
- Modify: `scripts/failure_modes/*.py` → shims
- Modify: `src/gepa_integration/seed.py`, 4 test modules

- [ ] **Step 1: `git mv` the files, preserving history**

```bash
mkdir -p src/agent_engine/analysis
git mv scripts/failure_modes/analyze_failure_modes.py src/agent_engine/analysis/failure_modes.py
git mv scripts/failure_modes/eval_runs src/agent_engine/analysis/eval_runs
git mv scripts/failure_modes/fine_tuning src/agent_engine/analysis/fine_tuning
```

Do **not** edit `classify_failure`'s body. Fix only the relative imports that the move breaks.

- [ ] **Step 2: Recreate the script paths as shims**

`scripts/failure_modes/analyze_failure_modes.py`:

```python
#!/usr/bin/env python
"""Thin CLI wrapper. Implementation lives in agent_engine.analysis.failure_modes.

Kept at this path because thesis notes and job files invoke it here.
"""

from agent_engine.analysis.failure_modes import main

if __name__ == "__main__":
    main()
```

One shim per previously-invocable script. Same argv, same output paths.

- [ ] **Step 3: Replace the `sys.path` hacks with real imports**

In `src/gepa_integration/seed.py`, delete lines 20-22 and change line 98 to
`from agent_engine.analysis.failure_modes import classify_failure`.
In `tests/unit/test_analyze_failure_modes.py`, `test_rollout_groups.py`, `test_all_wrong_analysis.py`, `test_failure_modes_runs.py`: delete the inserts, import from `agent_engine.analysis.*`.

- [ ] **Step 4: Update B5's import and re-run without regenerating**

Expected: B5 PASS against the unmodified fixture.

- [ ] **Step 5: Verify the shim produces identical output**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python scripts/failure_modes/analyze_failure_modes.py --help
```
Expected: same help text as before the move (compare against `git show HEAD~1`).

- [ ] **Step 6: Confirm no `sys.path` hack survives**

Run: `grep -rn 'sys.path.insert' src/ tests/ scripts/ --include='*.py'`
Expected: no results referencing `scripts`.

- [ ] **Step 7: Add the entry point, run ALL gates, commit**

```toml
cosmas-analyze = "agent_engine.analysis.failure_modes:main"
```

```bash
git add -A
git commit -m "refactor: move failure-mode analysis into the package, scripts become shims"
```

---

# PHASE 7 — Tests for untested modules

### Task 17: Cover `caching`, the scorers, and `external/`

**Files:**
- Create: `tests/unit/test_cache_manager.py`, `test_gaia_scorer.py`, `test_bigcodebench_scorer.py`, `test_url_fetcher.py`, `test_serper.py`, `test_tavily.py`

These are new tests over **unchanged** code. Network must never be touched: stub `requests` at the module boundary.

- [ ] **Step 1: Read each module and list its real behaviours before writing a line of test**

For each, write down: what it returns on the happy path, what it does on malformed input, what it does on an HTTP error. Test what the code **does**, not what it should do — this is characterization, not specification.

- [ ] **Step 2: Write the tests**

Cover per module: cache round-trip and persistence for `CacheManager`; exact-match, numeric, and list answers for `gaia_scorer`; the `re.search` full-definition detection and the double-prepend guard for `bigcodebench_scorer` (documented in CLAUDE.md as a real subtlety); timeout and non-200 handling for `url_fetcher`, `serper`, `tavily`.

- [ ] **Step 3: If a test fails, STOP**

A failure here has found a pre-existing bug. Per the global constraints: **report it, do not fix it.** Record it in the commit message and in `docs/archive/known-issues.md`, and mark the test `xfail` with the reason so the suite stays green and the bug stays visible.

- [ ] **Step 4: Run ALL gates, then commit**

```bash
git add -A
git commit -m "test: cover cache manager, scorers, and external clients"
```

---

# PHASE 8 — Documentation

### Task 18: Archive superseded docs

- [ ] **Step 1: Move, don't delete**

```bash
mkdir -p docs/archive
git mv docs/sft_status.md docs/grpo_sft_walkthrough.md docs/DS_integration_plan.md \
       docs/ds_olmo_integration.md docs/failure_modes_fine_tuning_alignment.md docs/archive/
git mv docs/failure_mode_and_fine_tuning docs/fine_tuning_v2 docs/archive/
git mv docs/superpowers docs/archive/superpowers
```

Keep the current handover spec and plan out of the archive.

- [ ] **Step 2: Prepend a banner to every archived file**

```markdown
> **HISTORICAL — not maintained.** Archived 2026-08-15 during the repository
> handover. Kept for the reasoning it records; paths, commands, and numbers in
> it may be stale. For current documentation see `docs/`.
```

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs: archive superseded planning and status documents"
```

---

### Task 19: Write the reference docs and guides

**Files:**
- Rewrite: `README.md`
- Create: `CONTRIBUTING.md`, `docs/architecture.md`, `docs/configuration.md`, `docs/guides/*.md` (4), `docs/pipelines/*.md` (4), `src/fine_tuning/agentflow/VENDORED.md`

- [ ] **Step 1: Rewrite `README.md` to ~250 lines**

What it is; install (both conda envs, and which is for what); run one experiment end to end; a repo map that is **verified against `git ls-files`**, not written from memory; pointers into `docs/`. Delete the stale tree referencing `train/`, `experiments/configs/generate_configs.py`, `1_milestone_no_img_no_mindmap_AgentFlow/`, `jobs/008_prepare_fine_tuning_data.job`.

- [ ] **Step 2: Write the four guides against the seams built in Phases 3, 4 and 5**

`add-a-benchmark.md` (loader + `DatasetSpec` row + template), `add-a-tool-or-subagent.md` (tool class + `@register_tool` factory + `BatchedTool` if it needs batching), `add-a-model-family.md` (`ModelFamily` enum + the `_*_FAMILIES` frozensets + `_TOOL_CALL_FORMAT`), `add-an-adaptation-method.md` (where GEPA, RL and SFT each hook in).

- [ ] **Step 3: Walk each guide end to end and fix what is wrong**

Actually follow `add-a-tool-or-subagent.md`: add a throwaway `echo_tool`, run an experiment config that enables it, confirm it works, then delete it. A guide that has never been executed is a guess. Do the same for `add-a-benchmark.md` with a two-example toy dataset.

- [ ] **Step 4: Write `VENDORED.md`**

Record: upstream repository URL, the version/commit vendored, the date, the 8 fixed absolute imports, the removed `_agentflow_path.py`, and an explicit "do not restyle; re-vendor from upstream instead" instruction.

- [ ] **Step 5: Run ALL gates, then commit**

```bash
git add -A
git commit -m "docs: task-oriented README, contributing guide, reference and how-to docs"
```

---

### Task 20: Retire the scaffolding gates, then final verification

**Why this step exists.** The B-gates are not all the same kind of test, and
keeping them all would leave a maintenance tax on the person this refactor is
for.

*Scaffolding — delete here.* **B1** (`configs.manifest`) and **B2**
(`prompts.json`, `prompt_templates.manifest`) fail on every *intended* change:
adding a benchmark changes the generated configs, editing a prompt changes the
templates. Since the whole point of the handover is that a new researcher adds
benchmarks and edits prompts, these gates would fail on their first honest
commit. Worse, a test that fails on intended changes trains people to run
`--update-fixtures` reflexively, at which point it protects nothing. They did
their job the moment Phases 1-7 landed green.

*Keepers — leave in place.* **B3** (orchestrator trace) only goes red when
orchestrator *behaviour* changes; adding a dataset or a tool leaves it green,
so it is a genuine regression test. **B4/B5** replay metrics and failure
classification over frozen historical `raw_results.json`; adding a benchmark
cannot change what an old run scores, so they stay valid indefinitely and are
what protects the thesis numbers.

- [ ] **Step 0a: Delete B1 and B2**

```bash
git rm tests/characterization/test_configs_unchanged.py \
       tests/characterization/fixtures/configs.manifest \
       tests/characterization/test_prompts_unchanged.py \
       tests/characterization/fixtures/prompts.json \
       tests/characterization/fixtures/prompt_templates.manifest
```

Keep the `--output-root` flag added to `generate_configs.py` in Task 2: it is
useful on its own (dry-run a suite into a temp directory before overwriting the
committed tree) and removing it would be a behaviour change.

- [ ] **Step 0b: Replace them with property tests**

Create `tests/unit/test_wiring_invariants.py`. These assert *properties* rather
than snapshots, so they survive people adding things and only fail on real
breakage:

1. `test_every_dataset_spec_resolves_to_a_template` — for every key in
   `DATASET_SPECS` (Task 12), `PromptBuilder` loads a non-empty template in both
   AgentFlow and baseline mode. Catches a dataset wired up with no prompt.
2. `test_every_template_parses` — every `*.yaml` under `prompts/templates/`
   parses as YAML and carries the required keys. Catches a malformed edit.
3. `test_every_registered_tool_has_a_valid_schema` — for every tool in the
   registry (Task 11), `get_schema()` returns a dict with `name`, `description`,
   and `parameters`, and `name` matches its registry key. Catches schema drift
   and copy-paste registration bugs.

Run the suite; expect green.

- [ ] **Step 0c: Give B4/B5 a hermetic synthetic corpus**

**The gap this closes.** B4 and B5 replay over `experiments/results/`, which is
gitignored, multi-gigabyte, and full of ground-truth answers for gated datasets
(GAIA, GPQA). None of it can be committed, so both tests `pytest.skip` on a
fresh clone. They protect the thesis numbers on the machine that produced them
and offer a new researcher nothing.

Create `tests/unit/test_metrics_and_classifier.py` with a small hand-built
corpus — a few dozen synthetic rows, no real questions or answers — asserting
*properties* rather than a recorded snapshot:

1. Rows covering each of the six failure modes, asserting `classify_failure`
   returns the expected label for each. Derive the rows from the classifier's
   documented rules (visual tool called → `modality_tool_gap`, a tool repeated
   `MIN_LOOP_REPEATS` times → `tool_loop_or_empty_final`, and so on), so the
   test states the taxonomy rather than echoing whatever the code happens to do.
2. Priority-order coverage: a row matching two rules at once must get the
   higher-priority label. The classifier is explicitly first-match-wins, and
   that ordering is the part a careless edit breaks.
3. `compute_metrics` over a synthetic stratified dataset (`per_level` present,
   correct per-level accuracy and token sums) and an unstratified one
   (`per_level` absent).

Keep B4/B5 as they are: replay guards the real numbers, these guard the logic.
The two are complementary, and only these run on a fresh clone.

- [ ] **Step 1: Full suite from a clean shell**

Run: `cd /gpfs/home3/xchen1/azywot/msc-thesis && /home/xchen1/.conda/envs/agent_engine/bin/python -m pytest -q`
Expected: at or above the Phase 0 baseline count, 0 failures.

- [ ] **Step 2: Every fixture green without `--update-fixtures`**

Run: `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/characterization -q`
Expected: all pass. After Step 0a only the B3/B4/B5 fixtures remain.

- [ ] **Step 3: Confirm fixtures were never silently regenerated**

Run: `git log --oneline -- tests/characterization/fixtures/`
Expected: each fixture appears in its Phase 0 recording commit, then nowhere else until the Step 0a deletion commit. Any *other* commit touching a fixture means a refactor changed behaviour and the baseline was moved to hide it — investigate before proceeding. The one legitimate exception is a deliberate `--update-fixtures` commit for the LoRA-config decision, if that was resolved.

- [ ] **Step 4: Confirm `experiments/configs/` is untouched by this work**

Run: `git diff 6a4671b --stat -- experiments/configs/`
Expected: empty. Diff against the branch point, **not** `main`: five
`sft_inference` configs were already modified by the pre-existing commit
`0ba9338 rename 008 -> 007 jobs`, and a `main..HEAD` diff wrongly attributes
them to this refactor.

- [ ] **Step 5: Verify both CLIs and one real config load**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/cosmas-run --help
/home/xchen1/.conda/envs/agent_engine/bin/python -c "
from agent_engine.config.loader import load_experiment_config
from pathlib import Path
c = load_experiment_config(Path('experiments/configs/qwen3/agentflow/gaia/qwen8B_subagent_tools_all.yaml'))
print('loaded', c.name)
"
```

Confirm that config path exists first with `ls`; substitute a real one if not.

- [ ] **Step 6: Report**

Summarise: phases landed, phases reverted (if any), pre-existing bugs found and left unfixed, and the open LoRA-config decision if still unresolved.

---

## Open decision carried from the spec

`scripts/generate_configs.py` is not idempotent: it rewrites 10 committed
`qwen3/lora_inference/*` configs, reverting a hand-edited `_v2` run
(`global_step_40`) to `global_step_20`. **Blocked on Agata:** which is correct?
Not a blocker for any task here — B1 compares the generator against itself.
When answered, fix the generator and regenerate the B1 fixture with
`--update-fixtures` in a single dedicated commit.
