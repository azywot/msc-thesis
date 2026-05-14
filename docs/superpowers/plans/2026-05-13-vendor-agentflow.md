# Vendor AgentFlow into msc-thesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the external AgentFlow repository dependency into `src/fine_tuning/agentflow/` so the entire training pipeline lives in one repository with a single `pip install -e .`.

**Architecture:** Copy the ~30 relevant AgentFlow files into `src/fine_tuning/agentflow/`, fix the 8 absolute `agentflow` import references inside the vendored code, update all msc-thesis import sites from `agentflow.*` to `fine_tuning.agentflow.*`, delete the `_agentflow_path.py` path hack, and strip the AgentFlow clone/install logic from the SLURM job scripts. No existing AgentFlow behaviour changes — this is a pure relocation.

**Tech Stack:** Python 3.11, setuptools (auto-discovers `src/fine_tuning/agentflow/`), VERL, Hydra, agentops (pip dep, already in conda env)

---

## File Map

### Created
| Path | Role |
|---|---|
| `src/fine_tuning/agentflow/__init__.py` | Package root (path hack removed) |
| `src/fine_tuning/agentflow/client.py` | HTTP client — polls tasks, posts rollouts |
| `src/fine_tuning/agentflow/config.py` | CLI config helpers |
| `src/fine_tuning/agentflow/litagent.py` | `LitAgent` base class |
| `src/fine_tuning/agentflow/logging.py` | Logger factory |
| `src/fine_tuning/agentflow/reward.py` | `@reward` decorator |
| `src/fine_tuning/agentflow/runner.py` | `AgentRunner` — per-task rollout loop |
| `src/fine_tuning/agentflow/server.py` | FastAPI task-queue server |
| `src/fine_tuning/agentflow/trainer.py` | `Trainer` — multiprocessing orchestrator |
| `src/fine_tuning/agentflow/types.py` | Pydantic data models |
| `src/fine_tuning/agentflow/tracer/__init__.py` | Tracer re-exports |
| `src/fine_tuning/agentflow/tracer/agentops.py` | AgentOps tracer (import fixed) |
| `src/fine_tuning/agentflow/tracer/base.py` | `BaseTracer` ABC (import fixed) |
| `src/fine_tuning/agentflow/tracer/http.py` | HTTP tracer |
| `src/fine_tuning/agentflow/tracer/triplet.py` | `TripletExporter` (import fixed) |
| `src/fine_tuning/agentflow/instrumentation/__init__.py` | Instrumentation helpers |
| `src/fine_tuning/agentflow/instrumentation/agentops.py` | AgentOps hooks |
| `src/fine_tuning/agentflow/instrumentation/agentops_langchain.py` | LangChain hooks |
| `src/fine_tuning/agentflow/instrumentation/litellm.py` | LiteLLM hooks |
| `src/fine_tuning/agentflow/instrumentation/verl_chat_scheduler.py` | VERL scheduler hook |
| `src/fine_tuning/agentflow/instrumentation/vllm.py` | vLLM hooks |
| `src/fine_tuning/agentflow/verl/__init__.py` | VERL re-exports |
| `src/fine_tuning/agentflow/verl/__main__.py` | Entry point for `python -m fine_tuning.agentflow.verl` |
| `src/fine_tuning/agentflow/verl/async_server.py` | Async server helpers (import fixed) |
| `src/fine_tuning/agentflow/verl/config.yaml` | Hydra training config |
| `src/fine_tuning/agentflow/verl/daemon.py` | `AgentModeDaemon` (import fixed) |
| `src/fine_tuning/agentflow/verl/dataset.py` | `AgentDataset` |
| `src/fine_tuning/agentflow/verl/entrypoint.py` | Hydra entry point (config path fixed) |
| `src/fine_tuning/agentflow/verl/peft_vllm_weight_sync_patch.py` | PEFT/vLLM weight sync patch |
| `src/fine_tuning/agentflow/verl/trainer.py` | `AgentFlowTrainer` (VERL PPO wrapper) |

### Modified
| Path | Change |
|---|---|
| `src/fine_tuning/rollout.py` | Remove path-hack import/call; `agentflow.*` → `fine_tuning.agentflow.*` |
| `scripts/train_orchestrator.py` | Same import updates |
| `scripts/test_ft_smoke.py` | Same import updates |
| `scripts/launch_verl.py` | `agentflow.verl` → `fine_tuning.agentflow.verl` |
| `jobs/009_test_small_ft_example.job` | Remove AgentFlow clone/install block |
| `jobs/010_ft_orchestrator.job` | Remove AgentFlow clone/install block |
| `pyproject.toml` | Add `agentops` to `[training]` extras |

### Deleted
| Path | Reason |
|---|---|
| `src/fine_tuning/_agentflow_path.py` | Replaced by proper package install |

---

## Task 1: Copy vendored files

**Files:** Create all 30 files listed in the "Created" section above.

- [ ] **Step 1: Create the directory tree and copy root-level files**

```bash
AGENTFLOW=/gpfs/home3/xchen1/azywot/AgentFlow/agentflow
DEST=/home/xchen1/azywot/msc-thesis/src/fine_tuning/agentflow

mkdir -p "$DEST/tracer" "$DEST/instrumentation" "$DEST/verl"

for f in __init__.py client.py config.py litagent.py logging.py reward.py runner.py server.py trainer.py types.py; do
    cp "$AGENTFLOW/$f" "$DEST/$f"
done
```

- [ ] **Step 2: Copy tracer/, instrumentation/, and verl/**

```bash
AGENTFLOW=/gpfs/home3/xchen1/azywot/AgentFlow/agentflow
DEST=/home/xchen1/azywot/msc-thesis/src/fine_tuning/agentflow

for f in __init__.py agentops.py base.py http.py triplet.py; do
    cp "$AGENTFLOW/tracer/$f" "$DEST/tracer/$f"
done

for f in __init__.py agentops.py agentops_langchain.py litellm.py verl_chat_scheduler.py vllm.py; do
    cp "$AGENTFLOW/instrumentation/$f" "$DEST/instrumentation/$f"
done

for f in __init__.py __main__.py async_server.py config.yaml daemon.py dataset.py entrypoint.py peft_vllm_weight_sync_patch.py trainer.py; do
    cp "$AGENTFLOW/verl/$f" "$DEST/verl/$f"
done
```

- [ ] **Step 3: Verify all 30 files are present**

```bash
find /home/xchen1/azywot/msc-thesis/src/fine_tuning/agentflow -name "*.py" -o -name "*.yaml" | sort
```

Expected: 29 `.py` files + 1 `config.yaml` = 30 entries.

- [ ] **Step 4: Commit the raw copy**

```bash
cd /home/xchen1/azywot/msc-thesis
git add src/fine_tuning/agentflow/
git commit -m "chore: vendor AgentFlow into src/fine_tuning/agentflow (raw copy)"
```

---

## Task 2: Fix absolute imports inside vendored files

Eight locations reference `agentflow` by its old absolute package name. Each must become `fine_tuning.agentflow`.

**Files:** `src/fine_tuning/agentflow/__init__.py`, `tracer/base.py`, `tracer/agentops.py`, `tracer/triplet.py`, `verl/async_server.py`, `verl/daemon.py`, `verl/entrypoint.py`

- [ ] **Step 1: Remove the inner-package path hack from `__init__.py`**

Open `src/fine_tuning/agentflow/__init__.py`. Remove these lines (they exist only to resolve the nested `agentflow/agentflow/` layout we no longer have):

```python
# DELETE these 4 lines:
from pathlib import Path as _Path
_agf_inner = _Path(__file__).resolve().parent / "agentflow"
_agf_inner_s = str(_agf_inner)
if _agf_inner.is_dir() and _agf_inner_s not in __path__:
    __path__.append(_agf_inner_s)
```

The file should start with `__version__ = "0.1.2"` and then the `from .client import ...` block.

- [ ] **Step 2: Fix `tracer/base.py`**

In `src/fine_tuning/agentflow/tracer/base.py` line 5, change:

```python
# Before
from agentflow.types import ParallelWorkerBase

# After
from fine_tuning.agentflow.types import ParallelWorkerBase
```

- [ ] **Step 3: Fix `tracer/agentops.py`**

In `src/fine_tuning/agentflow/tracer/agentops.py` lines 14–15, change:

```python
# Before
from agentflow.instrumentation.agentops import AgentOpsServerManager
from agentflow.instrumentation import instrument_all, uninstrument_all

# After
from fine_tuning.agentflow.instrumentation.agentops import AgentOpsServerManager
from fine_tuning.agentflow.instrumentation import instrument_all, uninstrument_all
```

- [ ] **Step 4: Fix `tracer/triplet.py`**

In `src/fine_tuning/agentflow/tracer/triplet.py` line 9, change:

```python
# Before
from agentflow.types import Triplet

# After
from fine_tuning.agentflow.types import Triplet
```

- [ ] **Step 5: Fix `verl/async_server.py`**

In `src/fine_tuning/agentflow/verl/async_server.py` line 4, change:

```python
# Before
from agentflow.instrumentation.vllm import instrument_vllm, ChatCompletionResponsePatched

# After
from fine_tuning.agentflow.instrumentation.vllm import instrument_vllm, ChatCompletionResponsePatched
```

- [ ] **Step 6: Fix `verl/daemon.py`**

In `src/fine_tuning/agentflow/verl/daemon.py` line 14, change:

```python
# Before
from agentflow import LLM, AgentFlowServer, NamedResources, Rollout, configure_logger

# After
from fine_tuning.agentflow import LLM, AgentFlowServer, NamedResources, Rollout, configure_logger
```

- [ ] **Step 7: Fix `verl/entrypoint.py`**

In `src/fine_tuning/agentflow/verl/entrypoint.py` line 13, change:

```python
# Before
@hydra.main(config_path="pkg://agentflow/verl", config_name="config", version_base=None)

# After
@hydra.main(config_path="pkg://fine_tuning.agentflow/verl", config_name="config", version_base=None)
```

- [ ] **Step 8: Verify no remaining absolute `agentflow` references in vendored files**

```bash
grep -rn "from agentflow\|import agentflow\|pkg://agentflow" \
  /home/xchen1/azywot/msc-thesis/src/fine_tuning/agentflow/ \
  2>/dev/null | grep -v __pycache__
```

Expected: **zero lines** of output (all references now say `fine_tuning.agentflow`).

- [ ] **Step 9: Commit the import fixes**

```bash
cd /home/xchen1/azywot/msc-thesis
git add src/fine_tuning/agentflow/
git commit -m "chore: fix absolute imports in vendored AgentFlow (agentflow → fine_tuning.agentflow)"
```

---

## Task 3: Verify the vendored package imports cleanly

Before touching any msc-thesis files, confirm the vendored package is discoverable and importable.

**Files:** No file changes — import-only verification.

- [ ] **Step 1: Reinstall msc-thesis so setuptools discovers the new package**

```bash
cd /home/xchen1/azywot/msc-thesis
pip install -e . --quiet
```

Expected: exits 0. Setuptools auto-discovers `src/fine_tuning/agentflow/` via `[tool.setuptools.packages.find] where = ["src"]`.

- [ ] **Step 2: Verify the vendored package is importable under its new name**

```bash
python -c "
from fine_tuning.agentflow import LitAgent, Trainer, reward
from fine_tuning.agentflow.types import NamedResources
from fine_tuning.agentflow.tracer.base import BaseTracer
import fine_tuning.agentflow as af
print(f'OK: fine_tuning.agentflow={af.__file__}')
print(f'OK: LitAgent={LitAgent}, Trainer={Trainer}')
"
```

Expected output (path will differ):
```
OK: fine_tuning.agentflow=.../src/fine_tuning/agentflow/__init__.py
OK: LitAgent=<class 'fine_tuning.agentflow.litagent.LitAgent'>, Trainer=<class 'fine_tuning.agentflow.trainer.Trainer'>
```

- [ ] **Step 3: Verify the VERL entry point module is reachable**

```bash
python -c "import fine_tuning.agentflow.verl; print('verl subpackage OK')"
```

Expected: `verl subpackage OK`

---

## Task 4: Update `src/fine_tuning/rollout.py`

**Files:** Modify `src/fine_tuning/rollout.py`

- [ ] **Step 1: Write the failing import test**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
# Simulate importing rollout without the old path hack in place
import importlib, fine_tuning.rollout as r
print('rollout imported from:', r.__file__)
"
```

This will currently succeed (old code still in place). Note the output — after the edit it should still succeed but via the new import path.

- [ ] **Step 2: Remove the path-hack import and call**

In `src/fine_tuning/rollout.py`, delete these three lines (currently lines 19–21):

```python
from fine_tuning._agentflow_path import ensure_agentflow_litagent_importable

ensure_agentflow_litagent_importable()
```

- [ ] **Step 3: Update the agentflow imports**

In `src/fine_tuning/rollout.py`, change:

```python
# Before
from agentflow import LitAgent, reward
from agentflow.types import NamedResources

# After
from fine_tuning.agentflow import LitAgent, reward
from fine_tuning.agentflow.types import NamedResources
```

- [ ] **Step 4: Verify rollout imports cleanly**

```bash
python -c "from fine_tuning.rollout import OrchestratorRollout; print('OrchestratorRollout OK')"
```

Expected: `OrchestratorRollout OK`

- [ ] **Step 5: Commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add src/fine_tuning/rollout.py
git commit -m "refactor: update rollout.py to import from fine_tuning.agentflow"
```

---

## Task 5: Update `scripts/train_orchestrator.py`

**Files:** Modify `scripts/train_orchestrator.py`

- [ ] **Step 1: Remove the path-hack import**

Delete line 33 from `scripts/train_orchestrator.py`:

```python
# DELETE this line:
from fine_tuning._agentflow_path import ensure_agentflow_litagent_importable
```

- [ ] **Step 2: Remove the `ensure_agentflow_litagent_importable()` call**

Delete line 101 (currently inside `main()`, after `ensure_agentflow_litagent_importable()` is defined):

```python
# DELETE this line:
    ensure_agentflow_litagent_importable()
```

- [ ] **Step 3: Update the agentflow imports**

In `scripts/train_orchestrator.py`, change:

```python
# Before
    from agentflow.tracer.base import BaseTracer
    ...
    from agentflow import Trainer

# After
    from fine_tuning.agentflow.tracer.base import BaseTracer
    ...
    from fine_tuning.agentflow import Trainer
```

- [ ] **Step 4: Verify the script parses cleanly**

```bash
python -c "import scripts.train_orchestrator" 2>&1 || python scripts/train_orchestrator.py --help 2>&1 | head -5
```

Expected: argument parser help text appears (no ImportError).

- [ ] **Step 5: Commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add scripts/train_orchestrator.py
git commit -m "refactor: update train_orchestrator.py to import from fine_tuning.agentflow"
```

---

## Task 6: Update `scripts/test_ft_smoke.py` and `scripts/launch_verl.py`

**Files:** Modify `scripts/test_ft_smoke.py`, `scripts/launch_verl.py`

- [ ] **Step 1: Remove the path-hack bootstrap from `test_ft_smoke.py`**

Delete lines 21–23 from `scripts/test_ft_smoke.py`:

```python
# DELETE these 3 lines:
from fine_tuning._agentflow_path import ensure_agentflow_litagent_importable

ensure_agentflow_litagent_importable()
```

- [ ] **Step 2: Update the agentflow check function in `test_ft_smoke.py`**

In the `check_agentflow()` function (around line 48), change:

```python
# Before
@check("import agentflow")
def check_agentflow():
    import agentflow  # noqa: F401
    from agentflow import LitAgent, Trainer, reward  # noqa: F401
    return f"agentflow imported"

# After
@check("import agentflow")
def check_agentflow():
    import fine_tuning.agentflow as agentflow  # noqa: F401
    from fine_tuning.agentflow import LitAgent, Trainer, reward  # noqa: F401
    return f"agentflow imported"
```

- [ ] **Step 3: Update `launch_verl.py`**

In `scripts/launch_verl.py` line 50, change:

```python
# Before
    command = [sys.executable, "-u", "-m", "agentflow.verl"]

# After
    command = [sys.executable, "-u", "-m", "fine_tuning.agentflow.verl"]
```

Also update the comment on line 49:

```python
# Before
    # Build: python -u -m agentflow.verl key=value key=value ...  (-u: line-buffered logs under SLURM > redirect)

# After
    # Build: python -u -m fine_tuning.agentflow.verl key=value key=value ...  (-u: line-buffered logs under SLURM > redirect)
```

- [ ] **Step 4: Run the smoke pre-flight checks**

```bash
cd /home/xchen1/azywot/msc-thesis
python scripts/test_ft_smoke.py --data-dir data/training/smoke
```

Expected: all import checks PASS. The data parquet checks will PASS or SKIP depending on whether smoke data is present — both are acceptable.

- [ ] **Step 5: Commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add scripts/test_ft_smoke.py scripts/launch_verl.py
git commit -m "refactor: update smoke test and launch_verl to use fine_tuning.agentflow"
```

---

## Task 7: Delete `_agentflow_path.py`

**Files:** Delete `src/fine_tuning/_agentflow_path.py`

- [ ] **Step 1: Confirm no remaining references**

```bash
grep -rn "_agentflow_path\|ensure_agentflow_litagent_importable" \
  /home/xchen1/azywot/msc-thesis/src/ \
  /home/xchen1/azywot/msc-thesis/scripts/ \
  2>/dev/null | grep -v __pycache__
```

Expected: **zero lines** of output.

- [ ] **Step 2: Delete the file**

```bash
rm /home/xchen1/azywot/msc-thesis/src/fine_tuning/_agentflow_path.py
```

- [ ] **Step 3: Verify imports still work**

```bash
python -c "
from fine_tuning.agentflow import LitAgent, Trainer, reward
from fine_tuning.rollout import OrchestratorRollout
print('All imports OK — _agentflow_path.py successfully removed')
"
```

Expected: `All imports OK — _agentflow_path.py successfully removed`

- [ ] **Step 4: Commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add -u src/fine_tuning/_agentflow_path.py
git commit -m "chore: delete _agentflow_path.py (replaced by vendored package)"
```

---

## Task 8: Clean up SLURM job scripts

**Files:** Modify `jobs/009_test_small_ft_example.job`, `jobs/010_ft_orchestrator.job`, `jobs/cosmas_train_pip_reconcile.sh`

Both job scripts share the same pattern: a `normalize_agentflow_pythonpath()` bash function, `AGENTFLOW_ROOT`/`AGENTFLOW_REPO` variables, a clone-if-missing block, and a `pip install --no-deps -e "$AGENTFLOW_ROOT/agentflow"` line. All of this is replaced by `pip install -e .` which now covers the vendored package.

- [ ] **Step 1: Remove AgentFlow bootstrap from `009_test_small_ft_example.job`**

Remove the following blocks from `jobs/009_test_small_ft_example.job`:

```bash
# DELETE — variable declarations (near top of script):
AGENTFLOW_ROOT="${AGENTFLOW_ROOT:-$HOME/azywot/AgentFlow}"
AGENTFLOW_REPO="${AGENTFLOW_REPO:-https://github.com/shin-ee-chen/AgentFlow.git}"

# DELETE — the entire normalize_agentflow_pythonpath() function definition (lines 57–83)

# DELETE — the three normalize_agentflow_pythonpath calls (lines 90, 101, 123)

# DELETE — the clone-if-missing block (lines 103–112):
if [[ ! -d "$AGENTFLOW_ROOT/agentflow" ]]; then
    echo "AgentFlow clone not found at $AGENTFLOW_ROOT; cloning..."
    ...
fi

# DELETE — the pip install line (line 118):
python -m pip install --no-deps -e "$AGENTFLOW_ROOT/agentflow"
```

- [ ] **Step 2: Update the import check in `009_test_small_ft_example.job`**

Find the line (around line 124):
```bash
python -c "import agentflow, agent_engine, fine_tuning, verl, vllm; from agentflow import LitAgent, Trainer, reward; print(f'Training imports OK: agentflow={agentflow.__file__}')"
```

Replace with:
```bash
python -c "import fine_tuning.agentflow as agentflow, agent_engine, fine_tuning, verl, vllm; from fine_tuning.agentflow import LitAgent, Trainer, reward; print(f'Training imports OK: agentflow={agentflow.__file__}')"
```

- [ ] **Step 3: Apply the same changes to `010_ft_orchestrator.job`**

Repeat Steps 1–2 identically for `jobs/010_ft_orchestrator.job` (same blocks exist at the same relative positions).

- [ ] **Step 4: Clean up `cosmas_train_pip_reconcile.sh`**

Remove the comment referencing AgentFlow (lines 6–8):
```bash
# DELETE these comment lines:
#   - AgentFlow imports agentops / AgentOpsTracer at import time even if you only
#     use NullTracer — agentops itself must be pinned to a version that works
#     agentflow.instrumentation.agentops (cannot change upstream AgentFlow here).
```

Update the opening echo on line 12:
```bash
# Before
echo "Reconciling cosmas-train pins (openai ↔ vLLM; antlr ↔ Hydra/omegaconf; AgentFlow imports)..."

# After
echo "Reconciling cosmas-train pins (openai ↔ vLLM; antlr ↔ Hydra/omegaconf)..."
```

- [ ] **Step 5: Commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add jobs/009_test_small_ft_example.job jobs/010_ft_orchestrator.job jobs/cosmas_train_pip_reconcile.sh
git commit -m "chore: remove AgentFlow clone/install from SLURM job scripts"
```

---

## Task 9: Add `agentops` to `pyproject.toml` and reinstall

**Files:** Modify `pyproject.toml`

`agentops` is a runtime dependency of the vendored `tracer/agentops.py`. It was previously pulled in implicitly via the external AgentFlow pip install. Now it must be declared explicitly.

- [ ] **Step 1: Add `agentops` to training extras in `pyproject.toml`**

In `pyproject.toml`, find the `[project.optional-dependencies]` `training` section and add `agentops`:

```toml
# Before
training = [
    "verl==0.5.0",
    "filelock>=3.13.0",
    "omegaconf>=2.3.0",
    "codetiming>=1.4.0",
]

# After
training = [
    "verl==0.5.0",
    "filelock>=3.13.0",
    "omegaconf>=2.3.0",
    "codetiming>=1.4.0",
    "agentops>=0.3.0",
]
```

- [ ] **Step 2: Reinstall with training extras**

```bash
cd /home/xchen1/azywot/msc-thesis
pip install -e ".[training]" --quiet
```

Expected: exits 0. `agentops` is already present in the conda env so this is a no-op in practice, but it makes the dependency explicit.

- [ ] **Step 3: Verify `agentops` version satisfies the pin**

```bash
python -c "import agentops; print(agentops.__version__)"
```

Expected: a version string >= `0.3.0`. If the installed version is older, run:
```bash
pip install "agentops>=0.3.0"
```

- [ ] **Step 4: Commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add pyproject.toml
git commit -m "chore: declare agentops in training extras (previously implicit via AgentFlow install)"
```

---

## Task 10: Full end-to-end verification

**Files:** No changes — verification only.

- [ ] **Step 1: Confirm the old external repo is no longer referenced anywhere**

```bash
grep -rn "AgentFlow\|agentflow_path\|AGENTFLOW_ROOT\|normalize_agentflow" \
  /home/xchen1/azywot/msc-thesis/src/ \
  /home/xchen1/azywot/msc-thesis/scripts/ \
  /home/xchen1/azywot/msc-thesis/jobs/ \
  2>/dev/null | grep -v __pycache__ | grep -v ".pyc"
```

Expected: only `fine_tuning/agentflow/` internal references remain (the vendored package itself, which is correct). No references to the external `$HOME/azywot/AgentFlow` repo path.

- [ ] **Step 2: Run the full pre-flight smoke test**

```bash
cd /home/xchen1/azywot/msc-thesis
python scripts/test_ft_smoke.py --data-dir data/training/smoke
```

Expected:
```
ALL 9 checks passed — safe to submit training job.
```

- [ ] **Step 3: Verify the VERL module entry point resolves**

```bash
python -m fine_tuning.agentflow.verl --help 2>&1 | head -5
```

Expected: Hydra help output (not a `ModuleNotFoundError`). Hydra may print a config error if no full config is provided — that's acceptable; the import chain resolved correctly.

- [ ] **Step 4: Confirm `agentflow` is no longer installed as a separate package**

```bash
pip show agentflow 2>&1
```

Expected: `WARNING: Package(s) not found: agentflow` — the standalone package entry is gone; the code lives inside `fine_tuning` now.

- [ ] **Step 5: Final commit**

```bash
cd /home/xchen1/azywot/msc-thesis
git add -A
git status  # should show nothing uncommitted
git log --oneline -8
```

Expected: 8 clean commits covering tasks 1–9.
