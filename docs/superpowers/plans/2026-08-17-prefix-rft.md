# Prefix-RFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Prefix-RFT as a fourth adaptation method: GRPO in which one rollout per prompt replays the first `k` decisions of a Qwen3-32B teacher demonstration and the model writes the rest, with the replayed decisions supervised under the trajectory advantage and an entropy filter.

**Architecture:** Every extension is a subclass in our own trees (`src/verl_ext/prefix_rft/`, `src/fine_tuning/`). Nothing under `src/fine_tuning/agentflow/` is edited, and verl is neither forked nor patched. A demonstration store is built offline; the driver samples `k` per rollout and ships it in the task payload; the rollout worker replays teacher decisions instead of generating them; the daemon marks those tokens in a `prefix_mask`; the trainer rewrites the advantage on them; the actor zeroes all but the top 20% by entropy.

**Tech Stack:** Python 3.11, verl 0.7.1, vLLM 0.17.0, PyTorch/FSDP, Ray, Hydra/OmegaConf, pandas/pyarrow, pytest. Training runs in the `cosmas-train` conda env.

**Spec:** `docs/superpowers/specs/2026-08-17-prefix-rft-design.md`

## Status (revised 2026-08-17)

**Tasks 1-8 are built, tested and committed.** Their steps below are kept as the record of
what was planned; where the implementation departed from them, the code and the spec are
the truth, not this file. Every departure is listed under "Implementation record" in
`docs/superpowers/specs/2026-08-17-prefix-rft-design.md`. The headline ones:

- demonstrations are keyed on the **question text**, not `extra_info.idx`, which collides
  across data sources;
- coverage is **1358 of 1800** questions (1085 prefixable), not 700;
- logic lives in **verl-free modules** (`dispatch.py`, `masks.py`, `entropy.py`,
  `prefix_replay.py`) because verl is absent from the CPU test env and pytest is absent
  from the training env, so anything importing verl is untestable in both;
- the three copied bodies are **generated and drift-checked**
  (`scripts/check_prefix_rft_trainer_sync.py`), not hand-transcribed and commented.

**Task 9 is built and committed** (`5f4618b`): production and smoke configs, and the
`PREFIX_RFT` env-var switch in `launch_verl.py` / `train_orchestrator.py`.

**Task 10 is partially done, staged but not committed.** `008_build_prefix_demos.job`,
`009_run_tests_for_prefix_rft.job`, `010_smoke_prefix_rft.job`, and
`scripts/check_prefix_replay_tokenisation.py` exist and have been run: the demonstration
store is built (1358 questions, 4047 decisions), the CPU gate and the replay-tokenisation
check both pass against it, and the `cosmas-train` sync/import/hydra checks pass. The GPU
smoke run (`010`, Steps 4-6 of Task 10 below) has not been executed — it needs a real
SLURM allocation. Steps 7-9 (`docs/pipelines/prefix-rft.md`, the
`add-an-adaptation-method.md` row, README/CHANGELOG) are not started.

---

## Global Constraints

- **Never edit anything under `src/fine_tuning/agentflow/`.** It is vendored; `src/fine_tuning/agentflow/VENDORED.md` forbids patching, restyling and bug-fixing it. Extend by subclassing from our own modules.
- **Never edit the installed verl package.** Extend by subclassing and by verl's `register_adv_est` / `register_policy_loss` registries.
- Python 3.11+, `black` line-length 100, `isort` black profile. Type annotations are not required (`disallow_untyped_defs = false`).
- Run pytest from the repo root. `testpaths = ["tests"]` is set in `pyproject.toml`; do not pass a bare directory that walks `data/`.
- Every hyperparameter must trace to `papers/PrefixRFT_2507.01679v3.md` or `repos/prefix_rft`. Cite the source in a comment at the point of definition.
- Paper values, exact: prefix high `0.95`; prefix low init `0.95`, target `0.05`; cosine decay; `l ~ U(low_t, high)`; entropy keep ratio `0.2` (top 20%); one prefixed rollout of eight.
- Token IDs for replayed turns must be produced by exactly the two calls the vendored proxy uses (`src/fine_tuning/agentflow/verl/daemon.py:216-225`): `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)` and `tokenizer.encode(text, add_special_tokens=False)`. No `enable_thinking` kwarg. No appended EOS.
- Advantage grouping is per rollout, deduplicated by `rollout_id`, never per row.
- Tests that need the training stack (verl, torch) must be skipped with `pytest.importorskip` so the suite still runs in the `agent_engine` env.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/build_prefix_demos.py` | Build `prefix_demos.parquet` from the teacher JSONL |
| `scripts/check_prefix_demos.py` | Preflight gate over the store |
| `src/verl_ext/prefix_rft/__init__.py` | Package marker, lazy re-exports |
| `src/verl_ext/prefix_rft/schedule.py` | Controllers, Beta sampler, step discretisation |
| `src/verl_ext/prefix_rft/demos.py` | Runtime demonstration store, `idx -> steps` |
| `src/verl_ext/prefix_rft/advantage.py` | Prefix advantage correction |
| `src/verl_ext/prefix_rft/actor.py` | `PrefixRFTActor`, entropy clip |
| `src/verl_ext/prefix_rft/worker.py` | `PrefixRFTWorker`, installs the actor |
| `src/verl_ext/prefix_rft/daemon.py` | `PrefixRFTDaemon`, `k` dispatch and `prefix_mask` |
| `src/verl_ext/prefix_rft/trainer.py` | `PrefixRFTTrainer`, `_train_step` hooks |
| `src/verl_ext/prefix_rft/entrypoint.py` | Hydra entrypoint wiring our classes |
| `src/verl_ext/prefix_rft/__main__.py` | `python -m verl_ext.prefix_rft` |
| `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml` | Hydra config adding the `prefix_rft` block |
| `src/fine_tuning/prefix_rollout.py` | `PrefixOrchestratorRollout` and replay shims |
| `experiments/configs/fine_tuning/config_prefix_rft.yaml` | Production config |
| `experiments/configs/fine_tuning/config_prefix_rft_smoke8b.yaml` | Smoke config |
| `jobs/fine_tuning/008_build_prefix_demos.job` | CPU: build the store |
| `jobs/fine_tuning/009_run_tests_for_prefix_rft.job` | CPU: unit tests + gate |
| `jobs/fine_tuning/010_smoke_prefix_rft.job` | 2 GPU: 8B smoke |
| `docs/pipelines/prefix-rft.md` | Pipeline documentation |
| `tests/unit/test_prefix_rft_schedule.py` | Task 2 |
| `tests/unit/test_prefix_rft_demos.py` | Tasks 1, 3 |
| `tests/unit/test_prefix_rft_rollout.py` | Task 4 |
| `tests/unit/test_prefix_rft_daemon.py` | Task 5 |
| `tests/unit/test_prefix_rft_advantage.py` | Task 6 |
| `tests/unit/test_prefix_rft_actor.py` | Task 7 |

---

## Task 1: Demonstration store

**Files:**
- Create: `scripts/build_prefix_demos.py`
- Create: `scripts/check_prefix_demos.py`
- Create: `tests/unit/test_prefix_rft_demos.py`

**Interfaces:**
- Consumes: `scripts/build_sft_parquet.py`'s `_strip_thinking` (line 106) and `_classify_turns` (line 189, returns `(plan, [(action_content, tool_name, tool_result)], answer)`).
- Produces: `data/training/prefix_rft/prefix_demos.parquet` with columns `idx: int`, `data_source: str`, `question: str`, `n_steps: int`, `steps: list[dict]` where each dict has keys `response: str`, `tool_name: str | None`, `tool_result: str | None`. The last step always has `tool_name is None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_demos.py`:

```python
"""Tests for the Prefix-RFT demonstration store."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_prefix_demos import records_to_demo_rows


def _record(idx, correct=True):
    return {
        "question_id": idx,
        "question": f"q{idx}",
        "data_source": "deepmath",
        "correct": correct,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"q{idx}"},
            {"role": "assistant", "content": "<think>hidden</think>plan text"},
            {"role": "assistant", "content": "call the tool"},
            {"role": "tool", "tool_name": "code_generator", "content": "42"},
            {"role": "assistant", "content": "the answer is 42"},
        ],
    }


def test_rows_have_one_entry_per_decision_in_order():
    rows = records_to_demo_rows([_record(7)])
    assert len(rows) == 1
    row = rows[0]
    assert row["idx"] == 7
    assert row["n_steps"] == 3
    assert [s["response"] for s in row["steps"]] == [
        "plan text",
        "call the tool",
        "the answer is 42",
    ]


def test_thinking_is_stripped():
    rows = records_to_demo_rows([_record(7)])
    assert "<think>" not in rows[0]["steps"][0]["response"]


def test_tool_results_are_attached_to_the_calling_step():
    steps = records_to_demo_rows([_record(7)])[0]["steps"]
    assert steps[1]["tool_name"] == "code_generator"
    assert steps[1]["tool_result"] == "42"
    assert steps[2]["tool_name"] is None
    assert steps[2]["tool_result"] is None


def test_incorrect_trajectories_are_dropped():
    assert records_to_demo_rows([_record(7, correct=False)]) == []


def test_one_row_per_question_when_duplicates_exist():
    rows = records_to_demo_rows([_record(7), _record(7)])
    assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_demos.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_prefix_demos'`

- [ ] **Step 3: Write the builder**

Create `scripts/build_prefix_demos.py`:

```python
"""Build the Prefix-RFT demonstration store from collected teacher trajectories.

One row per question, holding the teacher's decisions in order. Prefix-RFT
replays the first ``k`` of them and lets the policy continue from there, so each
decision needs both the assistant response and, where the decision was a tool
call, the tool result that came back.

Reuses ``build_sft_parquet``'s helpers so the two pipelines cannot drift: the
same correctness filter, the same thinking strip, the same positional split of a
stored trajectory into (plan, actions, answer).

Usage:
    python scripts/build_prefix_demos.py \
        data/training/sft/collected_20260605_214650.jsonl \
        --output data/training/prefix_rft/prefix_demos.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from build_sft_parquet import _classify_turns, _strip_thinking

logger = logging.getLogger(__name__)


def record_to_steps(record: dict) -> list[dict]:
    """Return the teacher's decisions in order, or [] if the record is unusable."""
    messages = _strip_thinking(record["messages"])
    plan, actions, answer = _classify_turns(messages)

    steps: list[dict] = []
    if plan is not None:
        steps.append({"response": plan, "tool_name": None, "tool_result": None})
    for content, tool_name, tool_result in actions:
        steps.append(
            {"response": content, "tool_name": tool_name, "tool_result": tool_result}
        )
    if answer is not None:
        steps.append({"response": answer, "tool_name": None, "tool_result": None})

    if any(not s["response"] or not s["response"].strip() for s in steps):
        return []
    return steps


def records_to_demo_rows(records: list[dict]) -> list[dict]:
    """Filter to correct trajectories, one per question, and build store rows."""
    rows: dict[int, dict] = {}
    for record in records:
        if not record.get("correct") or not record.get("messages"):
            continue
        idx = int(record["question_id"])
        if idx in rows:
            continue
        steps = record_to_steps(record)
        if not steps:
            continue
        rows[idx] = {
            "idx": idx,
            "data_source": str(record.get("data_source", "")),
            "question": str(record.get("question", "")),
            "n_steps": len(steps),
            "steps": steps,
        }
    return [rows[k] for k in sorted(rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="collected_<ts>.jsonl from 006")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training/prefix_rft/prefix_demos.parquet"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = []
    with args.jsonl.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Read %d records from %s", len(records), args.jsonl)

    rows = records_to_demo_rows(records)
    if not rows:
        raise SystemExit("No usable demonstrations found; store not written.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)

    n_single = sum(1 for r in rows if r["n_steps"] == 1)
    logger.info("Wrote %d demonstrations to %s", len(rows), args.output)
    logger.info(
        "Decisions: total %d, mean %.2f, single-decision questions %d "
        "(these can never carry a prefix, see the spec)",
        sum(r["n_steps"] for r in rows),
        sum(r["n_steps"] for r in rows) / len(rows),
        n_single,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_demos.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the gate**

Create `scripts/check_prefix_demos.py`:

```python
"""Preflight gate for the Prefix-RFT demonstration store.

Run before training. Exits non-zero on the first class of defect that would
make replay silently wrong, in the spirit of check_sft_folded_format.py.

Usage:
    python scripts/check_prefix_demos.py \
        --demos data/training/prefix_rft/prefix_demos.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agent_engine.utils.parsing import parse_tool_call


def check_row(row) -> list[str]:
    problems: list[str] = []
    steps = list(row["steps"])
    idx = row["idx"]

    if row["n_steps"] != len(steps):
        problems.append(f"idx={idx}: n_steps={row['n_steps']} but {len(steps)} steps")

    for i, step in enumerate(steps):
        where = f"idx={idx} step={i}"
        if not str(step["response"]).strip():
            problems.append(f"{where}: empty response")
        if "<think>" in str(step["response"]):
            problems.append(f"{where}: surviving <think> block")

        is_last = i == len(steps) - 1
        if is_last:
            if step["tool_name"] is not None:
                problems.append(f"{where}: final step must not be a tool call")
        else:
            if step["tool_name"] is None:
                problems.append(f"{where}: non-final step has no tool_name")
            if step["tool_result"] is None:
                problems.append(f"{where}: non-final step has no stored tool_result")
            if parse_tool_call(str(step["response"])) is None:
                problems.append(f"{where}: tool call does not parse")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demos",
        type=Path,
        default=Path("data/training/prefix_rft/prefix_demos.parquet"),
    )
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_parquet(args.demos)
    problems: list[str] = []
    for _, row in frame.iterrows():
        problems.extend(check_row(row))

    n_single = int((frame["n_steps"] == 1).sum())
    print(f"Checked {len(frame)} demonstrations from {args.demos}")
    print(f"  decisions: {int(frame['n_steps'].sum())}, mean {frame['n_steps'].mean():.2f}")
    print(f"  single-decision (never prefixed): {n_single}")

    if problems:
        print(f"\nFAILED with {len(problems)} problems, first {args.max_report}:")
        for problem in problems[: args.max_report]:
            print(f"  {problem}")
        return 1

    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Add a gate test**

Append to `tests/unit/test_prefix_rft_demos.py`:

```python
from check_prefix_demos import check_row


class _Row(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)


def _good_row():
    return _Row(
        idx=1,
        n_steps=2,
        steps=[
            {
                "response": '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>',
                "tool_name": "web_search",
                "tool_result": "result text",
            },
            {"response": "final answer", "tool_name": None, "tool_result": None},
        ],
    )


def test_gate_accepts_a_well_formed_row():
    assert check_row(_good_row()) == []


def test_gate_rejects_a_missing_tool_result():
    row = _good_row()
    row["steps"][0]["tool_result"] = None
    assert any("stored tool_result" in p for p in check_row(row))


def test_gate_rejects_surviving_thinking():
    row = _good_row()
    row["steps"][1]["response"] = "<think>oops</think>final answer"
    assert any("<think>" in p for p in check_row(row))
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/unit/test_prefix_rft_demos.py -v`
Expected: 8 passed

- [ ] **Step 8: Build the real store and run the gate**

```bash
python scripts/build_prefix_demos.py \
    data/training/sft/collected_20260605_214650.jsonl \
    --output data/training/prefix_rft/prefix_demos.parquet
python scripts/check_prefix_demos.py
```

Expected: about 700 demonstrations, mean around 4.3 decisions, roughly 71 single-decision questions, and `PASSED`. If the gate fails on tool-call parsing for a handful of rows, drop those rows in `records_to_demo_rows` rather than loosening the gate, and log how many were dropped.

- [ ] **Step 9: Commit**

```bash
git add scripts/build_prefix_demos.py scripts/check_prefix_demos.py tests/unit/test_prefix_rft_demos.py
git commit -m "feat(prefix-rft): build and gate the demonstration store"
```

---

## Task 2: Prefix length schedule

**Files:**
- Create: `src/verl_ext/prefix_rft/__init__.py`
- Create: `src/verl_ext/prefix_rft/schedule.py`
- Create: `tests/unit/test_prefix_rft_schedule.py`

**Interfaces:**
- Produces: `PrefixStepSchedule(low_init=0.95, low_target=0.05, high=0.95, n_steps=500, alpha=1.0, beta=1.0, seed=None)` with `sample_l(global_step) -> (l, low, high)` and `sample_k(n_demo_steps, global_step) -> int`. Also `CosineDecayController`, `ConstController`, `CTRL_MAPPING`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_schedule.py`:

```python
"""Tests for the Prefix-RFT prefix length schedule."""

import pytest

from verl_ext.prefix_rft.schedule import (
    ConstController,
    CosineDecayController,
    PrefixStepSchedule,
)


def test_cosine_decay_endpoints_match_the_paper():
    ctrl = CosineDecayController(init=0.95, target=0.05, n_steps=500)
    assert ctrl.value(global_step=0) == pytest.approx(0.95)
    assert ctrl.value(global_step=500) == pytest.approx(0.05)
    assert ctrl.value(global_step=250) == pytest.approx(0.5)


def test_cosine_decay_is_monotone_and_clamps_past_the_end():
    ctrl = CosineDecayController(init=0.95, target=0.05, n_steps=500)
    values = [ctrl.value(global_step=s) for s in range(0, 501, 25)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert ctrl.value(global_step=900) == pytest.approx(0.05)


def test_const_controller_ignores_the_step():
    ctrl = ConstController(init=0.8)
    assert ctrl.value(global_step=0) == 0.8
    assert ctrl.value(global_step=999) == 0.8


def test_sampled_l_stays_inside_the_scheduled_window():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    for step in (0, 100, 250, 499):
        for _ in range(50):
            l, low, high = sched.sample_l(global_step=step)
            assert low <= l <= high
            assert high == pytest.approx(0.95)


def test_k_is_clamped_to_leave_one_on_policy_decision():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    for step in (0, 250, 499):
        for m in (1, 2, 3, 4, 8, 13):
            for _ in range(20):
                k = sched.sample_k(m, global_step=step)
                assert 0 <= k <= m - 1


def test_single_decision_demonstrations_never_carry_a_prefix():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    assert all(sched.sample_k(1, global_step=s) == 0 for s in range(0, 500, 10))


def test_k_shrinks_as_training_advances():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    early = sum(sched.sample_k(8, global_step=0) for _ in range(200))
    late = sum(sched.sample_k(8, global_step=490) for _ in range(200))
    assert early > late


def test_zero_demonstration_steps_gives_zero():
    sched = PrefixStepSchedule(n_steps=500, seed=0)
    assert sched.sample_k(0, global_step=0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl_ext.prefix_rft'`

- [ ] **Step 3: Write the implementation**

Create `src/verl_ext/prefix_rft/__init__.py`:

```python
"""Prefix-RFT: blending demonstration and exploration in the RL pipeline.

Implements arXiv:2507.01679v3 on top of the existing GRPO pipeline. Heavy
imports (torch, verl) are deferred to the modules that need them so that the
schedule and the demonstration store can be exercised without the training
stack installed.
"""
```

Create `src/verl_ext/prefix_rft/schedule.py`:

```python
"""Prefix length schedule.

Ports the controllers from the reference implementation at
``repos/prefix_rft/recipe/prefix_rft/scheduler/global_step.py`` (CosineDecayController
lines 148-186, BetaSampler lines 256-275) and adds the step-level discretisation this
project uses in place of the paper's token ratio.

Paper Appendix A.2: "At each time step t, we sample l uniformly from [low_t, 0.95] to
decide the prefix length as l times the total demonstration length. And low_t follows a
cosine decay scheduler, starting from 0.95 and decaying to 0.05 at the 500th step."
"""

from __future__ import annotations

import math

import numpy as np

# Paper A.2. Named so the config and the tests read against one source of truth.
PAPER_HIGH = 0.95
PAPER_LOW_INIT = 0.95
PAPER_LOW_TARGET = 0.05


class ConstController:
    """Constant value. Reference: global_step.py:5-17."""

    def __init__(self, init: float = 0.0, **kwargs):
        self.c = init

    def value(self, global_step: int = 0, **kwargs) -> float:
        return self.c

    def __str__(self) -> str:
        return f"Constant({self.c})"


class CosineDecayController:
    """Cosine interpolation from ``init`` to ``target``. Reference: global_step.py:148-186."""

    def __init__(self, init=PAPER_LOW_INIT, target=PAPER_LOW_TARGET, n_steps=500,
                 warmup_ratio=0.0, **kwargs):
        if init == target:
            raise ValueError("init and target must differ")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.init = init
        self.target = target
        self.n_steps = n_steps
        self.warmup_steps = int(warmup_ratio * n_steps)
        self.mode = "decay" if init > target else "rise"

    def value(self, global_step: int = 0, **kwargs) -> float:
        if self.warmup_steps and global_step < self.warmup_steps:
            return (global_step / self.warmup_steps) * self.init
        step = global_step - self.warmup_steps
        if step > self.n_steps:
            return self.target
        decay_ratio = 0.5 * (1 + math.cos(math.pi * step / self.n_steps))
        if self.mode == "decay":
            return self.target + decay_ratio * (self.init - self.target)
        return self.init + (1 - decay_ratio) * (self.target - self.init)

    def __str__(self) -> str:
        return (f"CosineDecay(init={self.init}, target={self.target}, "
                f"n_steps={self.n_steps}, warmup={self.warmup_steps})")


CTRL_MAPPING = {"cosine_decay": CosineDecayController, "const": ConstController}


class PrefixStepSchedule:
    """Draw the number of teacher decisions to replay.

    ``l`` follows the paper exactly: a Beta(alpha, beta) draw rescaled onto
    ``[low_t, high]``, which at alpha = beta = 1 is the uniform draw A.2 specifies.
    Discretisation to whole decisions is ours (see the spec's "Adaptation" section):
    ``k = clamp(floor(l * m), 0, m - 1)``. The upper clamp is the step-level analogue of
    the reference's ``prefix_len >= demo_len -> demo_len - 1`` guard
    (rl_dataset.py:300-301) and guarantees at least one on-policy decision.
    """

    def __init__(self, low_init=PAPER_LOW_INIT, low_target=PAPER_LOW_TARGET,
                 high=PAPER_HIGH, n_steps=500, alpha=1.0, beta=1.0, seed=None):
        self.low_ctrl = CosineDecayController(init=low_init, target=low_target,
                                              n_steps=n_steps)
        self.high_ctrl = ConstController(init=high)
        self.alpha = alpha
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def sample_l(self, global_step: int) -> tuple[float, float, float]:
        """Return (l, low_t, high). Reference: BetaSampler.value, global_step.py:268-275."""
        low = self.low_ctrl.value(global_step=global_step)
        high = self.high_ctrl.value(global_step=global_step)
        lower, higher = min(low, high), max(low, high)
        u = float(self._rng.beta(self.alpha, self.beta))
        return lower + (higher - lower) * u, lower, higher

    def sample_k(self, n_demo_steps: int, global_step: int) -> int:
        """Number of teacher decisions to replay for one rollout."""
        if n_demo_steps <= 1:
            return 0
        l, _, _ = self.sample_l(global_step)
        return max(0, min(int(math.floor(l * n_demo_steps)), n_demo_steps - 1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_schedule.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/verl_ext/prefix_rft/__init__.py src/verl_ext/prefix_rft/schedule.py tests/unit/test_prefix_rft_schedule.py
git commit -m "feat(prefix-rft): port the cosine prefix length schedule"
```

---

## Task 3: Runtime demonstration store

**Files:**
- Create: `src/verl_ext/prefix_rft/demos.py`
- Modify: `tests/unit/test_prefix_rft_demos.py`

**Interfaces:**
- Consumes: the parquet from Task 1.
- Produces: `DemoStore.from_parquet(path) -> DemoStore`; `DemoStore.n_steps(idx) -> int` (0 when absent); `DemoStore.steps(idx) -> list[dict]` (empty when absent); `DemoStore.coverage() -> tuple[int, int]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_prefix_rft_demos.py`:

```python
import pandas as pd

from verl_ext.prefix_rft.demos import DemoStore


def _write_store(tmp_path):
    path = tmp_path / "demos.parquet"
    pd.DataFrame(
        [
            {
                "idx": 3,
                "data_source": "deepmath",
                "question": "q3",
                "n_steps": 2,
                "steps": [
                    {"response": "a", "tool_name": "web_search", "tool_result": "r"},
                    {"response": "b", "tool_name": None, "tool_result": None},
                ],
            }
        ]
    ).to_parquet(path, index=False)
    return path


def test_store_returns_steps_for_a_known_question(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.n_steps(3) == 2
    assert store.steps(3)[0]["tool_result"] == "r"


def test_store_misses_return_zero_rather_than_raising(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.n_steps(999) == 0
    assert store.steps(999) == []


def test_store_reports_coverage(tmp_path):
    store = DemoStore.from_parquet(_write_store(tmp_path))
    assert store.coverage() == (1, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_demos.py -k store -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl_ext.prefix_rft.demos'`

- [ ] **Step 3: Write the implementation**

Create `src/verl_ext/prefix_rft/demos.py`:

```python
"""Runtime access to the Prefix-RFT demonstration store.

Coverage is partial by design: 700 of the 1800 RL training questions carry a
teacher demonstration. A miss is not an error, it means that question trains as
ordinary GRPO, which is what the reference implementation's ``demo_ratio``
mechanism does (rl_dataset.py:194-203) and what the paper's Table 2 validates
down to 1% coverage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DemoStore:
    """Maps a question index to the teacher's ordered decisions."""

    def __init__(self, by_idx: dict[int, list[dict]]):
        self._by_idx = by_idx

    @classmethod
    def from_parquet(cls, path) -> "DemoStore":
        frame = pd.read_parquet(Path(path))
        by_idx: dict[int, list[dict]] = {}
        for _, row in frame.iterrows():
            by_idx[int(row["idx"])] = [dict(step) for step in row["steps"]]
        return cls(by_idx)

    def n_steps(self, idx: int) -> int:
        return len(self._by_idx.get(int(idx), ()))

    def steps(self, idx: int) -> list[dict]:
        return list(self._by_idx.get(int(idx), ()))

    def coverage(self) -> tuple[int, int]:
        """Return (questions with a demonstration, total decisions)."""
        return len(self._by_idx), sum(len(v) for v in self._by_idx.values())

    def __len__(self) -> int:
        return len(self._by_idx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_demos.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/verl_ext/prefix_rft/demos.py tests/unit/test_prefix_rft_demos.py
git commit -m "feat(prefix-rft): add the runtime demonstration store"
```

---

## Task 4: Replay rollout

**Files:**
- Create: `src/fine_tuning/prefix_rollout.py`
- Create: `tests/unit/test_prefix_rft_rollout.py`

**Interfaces:**
- Consumes: `fine_tuning.rollout.OrchestratorRollout`, `_CapturingProvider`, `_get_task_metadata`; `agent_engine.core.tool.ToolRegistry` and `ToolResult`; `fine_tuning.agentflow.types.Triplet`.
- Produces: `ReplayController(steps, k, tokenizer)` with `.exhausted -> bool`, `.next_response() -> dict | None`, `.next_tool_result() -> str | None`, `.replayed_turns -> int`; `PrefixOrchestratorRollout(..., demos_path=None, base_model=None)`.

Read `src/fine_tuning/rollout.py:186-354` before starting: the new class changes only how the first `k` decisions are produced, and reuses `_run_episode`'s reward, saving and triplet construction.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_rollout.py`:

```python
"""Tests for Prefix-RFT replay in the rollout worker."""

import json

import pytest

from fine_tuning.prefix_rollout import ReplayController, ReplayToolRegistry


class _FakeTokenizer:
    """Stands in for the HF tokenizer, mirroring the proxy's two calls."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert add_generation_prompt is True
        assert tokenize is True
        return [len(m["content"]) for m in messages]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def _steps():
    return [
        {"response": "plan", "tool_name": None, "tool_result": None},
        {"response": "call", "tool_name": "web_search", "tool_result": "hits"},
        {"response": "final", "tool_name": None, "tool_result": None},
    ]


def _payload(messages):
    return json.dumps({"messages": messages, "use_thinking": False})


def test_controller_serves_k_responses_then_stops():
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    first = ctrl.next_response(_payload([{"role": "user", "content": "ab"}]))
    second = ctrl.next_response(_payload([{"role": "user", "content": "abc"}]))
    assert first["text"] == "plan"
    assert second["text"] == "call"
    assert ctrl.exhausted is True
    assert ctrl.next_response(_payload([{"role": "user", "content": "x"}])) is None
    assert ctrl.replayed_turns == 2


def test_controller_tokenises_exactly_as_the_proxy_does():
    ctrl = ReplayController(_steps(), k=1, tokenizer=_FakeTokenizer())
    out = ctrl.next_response(_payload([{"role": "user", "content": "ab"}]))
    assert out["prompt_token_ids"] == [2]
    assert out["response_token_ids"] == [ord(c) for c in "plan"]


def test_controller_maps_tool_role_to_user_before_templating():
    ctrl = ReplayController(_steps(), k=1, tokenizer=_FakeTokenizer())
    messages = [{"role": "tool", "content": "xyz"}]
    out = ctrl.next_response(_payload(messages))
    assert out["prompt_token_ids"] == [3]


def test_k_zero_replays_nothing():
    ctrl = ReplayController(_steps(), k=0, tokenizer=_FakeTokenizer())
    assert ctrl.exhausted is True
    assert ctrl.next_response(_payload([{"role": "user", "content": "a"}])) is None


def test_tool_results_are_served_in_order_for_replayed_steps():
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    assert ctrl.next_tool_result() is None
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    assert ctrl.next_tool_result() == "hits"


def test_replay_registry_delegates_once_exhausted():
    class _RealTool:
        name = "web_search"

        def execute(self, **kwargs):
            from agent_engine.core.tool import ToolResult

            return ToolResult(success=True, output="live", metadata={})

    class _RealRegistry:
        def get(self, name):
            return _RealTool()

        def list_tools(self):
            return ["web_search"]

        def get_all_schemas(self):
            return []

    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    registry = ReplayToolRegistry(_RealRegistry(), ctrl)

    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    ctrl.next_response(_payload([{"role": "user", "content": "a"}]))
    replayed = registry.get("web_search").execute(query="q")
    assert replayed.output == "hits"
    assert replayed.metadata["replayed"] is True

    assert registry.get("web_search").execute(query="q").output == "live"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_rollout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fine_tuning.prefix_rollout'`

- [ ] **Step 3: Write the implementation**

Create `src/fine_tuning/prefix_rollout.py`:

```python
"""Prefix-RFT rollout: replay the first k teacher decisions, then go on-policy.

The orchestrator is not modified. Two shims sit between it and its resources:

- ``ReplayController`` holds the teacher's decisions and a counter. While the
  counter is below ``k`` it answers generation requests with the teacher's stored
  response and tool requests with the teacher's stored result.
- ``ReplayToolRegistry`` consults the controller before delegating.

Because the orchestrator still builds every prompt itself, a replayed turn is
conditioned on exactly the prompt inference would have built. Token IDs are
produced with the same two calls the vendored proxy uses
(fine_tuning/agentflow/verl/daemon.py:216-225), so replayed and generated turns
are tokenised identically.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agent_engine.core.tool import ToolResult
from agent_engine.models.base import GenerationResult

from .rollout import OrchestratorRollout, _CapturingProvider, _get_task_metadata


class ReplayController:
    """Serves the first ``k`` teacher decisions in place of live generation."""

    def __init__(self, steps: list[dict], k: int, tokenizer):
        self.steps = list(steps)
        self.k = max(0, min(int(k), len(self.steps)))
        self.tokenizer = tokenizer
        self._served = 0

    @property
    def exhausted(self) -> bool:
        return self._served >= self.k

    @property
    def replayed_turns(self) -> int:
        return self._served

    def next_response(self, prompt_payload: str) -> Optional[dict]:
        """Return the next teacher decision with proxy-identical token IDs."""
        if self.exhausted:
            return None
        step = self.steps[self._served]
        self._served += 1

        messages = self._decode_messages(prompt_payload)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        text = str(step["response"])
        response_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return {
            "text": text,
            "prompt_token_ids": list(prompt_ids),
            "response_token_ids": list(response_ids),
        }

    def next_tool_result(self) -> Optional[str]:
        """The stored result for the decision just served, if it was a tool call."""
        if self._served == 0:
            return None
        step = self.steps[self._served - 1]
        return step["tool_result"] if step["tool_name"] else None

    @staticmethod
    def _decode_messages(prompt_payload: str) -> list[dict]:
        """Mirror OpenAIProvider._generate_one's decoding (api_provider.py:82-100).

        The tool -> user remap happens before the request leaves the provider, so
        the proxy tokenises the remapped list and we must too.
        """
        raw = None
        try:
            payload = json.loads(prompt_payload)
            if isinstance(payload, dict) and "messages" in payload:
                raw = payload["messages"]
            elif isinstance(payload, list):
                raw = payload
        except (json.JSONDecodeError, TypeError):
            raw = None
        if raw is None:
            raw = [{"role": "user", "content": prompt_payload}]
        return [{**m, "role": "user"} if m.get("role") == "tool" else m for m in raw]


class _ReplayedTool:
    """Returns a stored tool result. Accepts **kwargs so argument sanitising is a no-op."""

    def __init__(self, name: str, output: str):
        self.name = name
        self._output = output

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=self._output, metadata={"replayed": True})


class ReplayToolRegistry:
    """Wraps a ToolRegistry, serving stored results while the controller is replaying."""

    def __init__(self, registry, controller: ReplayController):
        self._registry = registry
        self._controller = controller

    def get(self, name: str):
        # next_tool_result() returns None once the last served decision was not a
        # tool call, and the controller stops serving past k, so no extra guard
        # is needed here.
        pending = self._controller.next_tool_result()
        if pending is not None:
            return _ReplayedTool(name, pending)
        return self._registry.get(name)

    def __getattr__(self, item):
        return getattr(self._registry, item)


class ReplayProvider:
    """Serves replayed turns, then delegates to the capturing provider."""

    def __init__(self, capturing: _CapturingProvider, controller: ReplayController):
        object.__setattr__(self, "_capturing", capturing)
        object.__setattr__(self, "_controller", controller)
        object.__setattr__(self, "prefix_turns", [])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_capturing"), name)

    def generate(self, prompts: list) -> list:
        controller = object.__getattribute__(self, "_controller")
        capturing = object.__getattribute__(self, "_capturing")
        results = []
        for prompt in prompts:
            replayed = controller.next_response(prompt)
            if replayed is None:
                results.extend(capturing.generate([prompt]))
                continue
            object.__getattribute__(self, "prefix_turns").append(replayed)
            capturing.captured_turns.append(
                {
                    "prompt_ids": replayed["prompt_token_ids"],
                    "response_ids": replayed["response_token_ids"],
                    "response_text": replayed["text"],
                    "is_prefix": True,
                }
            )
            results.append(
                GenerationResult(
                    text=replayed["text"],
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": len(replayed["prompt_token_ids"]),
                        "completion_tokens": len(replayed["response_token_ids"]),
                        "total_tokens": len(replayed["prompt_token_ids"])
                        + len(replayed["response_token_ids"]),
                    },
                    metadata={"replayed": True},
                    prompt_token_ids=replayed["prompt_token_ids"],
                    response_token_ids=replayed["response_token_ids"],
                )
            )
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_rollout.py -v`
Expected: 6 passed

- [ ] **Step 5: Add PrefixOrchestratorRollout**

Append to `src/fine_tuning/prefix_rollout.py`:

```python
class PrefixOrchestratorRollout(OrchestratorRollout):
    """OrchestratorRollout that can start from a teacher-demonstration prefix.

    ``k`` arrives per rollout in the task payload under ``prefix_k``; the driver
    owns the schedule because it owns ``global_step``. ``k = 0`` reproduces the
    base class exactly.
    """

    def __init__(self, *args, demos_path=None, base_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.demos_path = demos_path
        self.base_model = base_model
        self._store = None
        self._tokenizer = None

    def _get_store(self):
        if self._store is None and self.demos_path:
            from verl_ext.prefix_rft.demos import DemoStore

            self._store = DemoStore.from_parquet(self.demos_path)
        return self._store

    def _get_tokenizer(self):
        if self._tokenizer is None and self.base_model:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer

    def _make_controller(self, task: Any) -> Optional[ReplayController]:
        k = int(task.get("prefix_k", 0) or 0)
        if k <= 0:
            return None
        store = self._get_store()
        tokenizer = self._get_tokenizer()
        if store is None or tokenizer is None:
            return None
        _, _, _, idx = _get_task_metadata(task)
        steps = store.steps(idx)
        if not steps:
            return None
        return ReplayController(steps, k, tokenizer)
```

Then modify `_run_episode`. Rather than duplicating the base method, override the two construction points by extracting them in the base class first:

- In `src/fine_tuning/rollout.py`, replace the line `capturing_provider = _CapturingProvider(base_provider)` (line 285) with `capturing_provider = self._wrap_provider(_CapturingProvider(base_provider), task)`, and replace `tool_registry = self._get_or_build_tools()` (line 286) with `tool_registry = self._wrap_tools(self._get_or_build_tools(), task)`.
- Add two no-op seams to `OrchestratorRollout`:

```python
    def _wrap_provider(self, provider, task):
        """Seam for subclasses. Base returns the provider unchanged."""
        return provider

    def _wrap_tools(self, registry, task):
        """Seam for subclasses. Base returns the registry unchanged."""
        return registry
```

- In `PrefixOrchestratorRollout`, implement them:

```python
    def _wrap_provider(self, provider, task):
        controller = self._make_controller(task)
        if controller is None:
            return provider
        self._active_controller = controller
        return ReplayProvider(provider, controller)

    def _wrap_tools(self, registry, task):
        controller = getattr(self, "_active_controller", None)
        if controller is None:
            return registry
        return ReplayToolRegistry(registry, controller)
```

Finally, mark the triplets. In `src/fine_tuning/rollout.py`, change the `Triplet(...)` construction (lines 341-348) to pass the flag the capturing provider already recorded:

```python
                triplets = [
                    Triplet(
                        prompt={"token_ids": t["prompt_ids"]},
                        response={"token_ids": t["response_ids"], "text": t.get("response_text", "")},
                        reward=reward_value,
                        metadata={"prefix": bool(t.get("is_prefix", False))},
                    )
                    for t in turns
                ]
```

These are the only edits to `src/fine_tuning/rollout.py`, and they are behaviour-preserving for the base class: the seams return their inputs and `is_prefix` is absent, so `metadata["prefix"]` is `False`.

- [ ] **Step 6: Test the seams**

Append to `tests/unit/test_prefix_rft_rollout.py`:

```python
def test_base_rollout_seams_are_identity():
    from fine_tuning.rollout import OrchestratorRollout

    agent = OrchestratorRollout(subagent_endpoint="http://x/v1")
    sentinel = object()
    assert agent._wrap_provider(sentinel, {}) is sentinel
    assert agent._wrap_tools(sentinel, {}) is sentinel


def test_prefix_rollout_without_prefix_k_is_identity():
    from fine_tuning.prefix_rollout import PrefixOrchestratorRollout

    agent = PrefixOrchestratorRollout(subagent_endpoint="http://x/v1")
    sentinel = object()
    assert agent._wrap_provider(sentinel, {"prefix_k": 0}) is sentinel
    assert agent._wrap_tools(sentinel, {"prefix_k": 0}) is sentinel
```

- [ ] **Step 7: Run the tests, including the existing rollout suite**

Run: `pytest tests/unit/test_prefix_rft_rollout.py tests/unit/test_fine_tuning_rollout.py -v`
Expected: all pass. The existing suite must be untouched; if it fails, the seams changed behaviour and the edit is wrong.

- [ ] **Step 8: Commit**

```bash
git add src/fine_tuning/prefix_rollout.py src/fine_tuning/rollout.py tests/unit/test_prefix_rft_rollout.py
git commit -m "feat(prefix-rft): replay teacher decisions in the rollout worker"
```

---

## Task 5: Daemon, k dispatch and prefix_mask

**Files:**
- Create: `src/verl_ext/prefix_rft/daemon.py`
- Create: `tests/unit/test_prefix_rft_daemon.py`

**Interfaces:**
- Consumes: `fine_tuning.agentflow.verl.daemon.AgentModeDaemon`; `PrefixStepSchedule`; `DemoStore`.
- Produces: `PrefixRFTDaemon(..., schedule=None, demo_store=None, n_prefixed_rollouts=1)` with `set_global_step(step)`; `build_prefix_mask(trace_list, max_response_length) -> list[list[int]]`; `get_train_data_batch` returning a `DataProto` whose `batch` also holds `prefix_mask` and whose `non_tensor_batch` also holds `is_prefix_rollout_list`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_daemon.py`:

```python
"""Tests for the Prefix-RFT daemon."""

import pytest

pytest.importorskip("verl")

from verl_ext.prefix_rft.daemon import build_prefix_mask


def test_mask_is_one_over_prefix_turn_tokens_only():
    traces = [
        {"response_ids": [1, 2, 3], "is_prefix": True},
        {"response_ids": [4, 5], "is_prefix": False},
    ]
    assert build_prefix_mask(traces, max_response_length=5) == [
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_mask_truncates_with_the_response():
    traces = [{"response_ids": [1, 2, 3, 4, 5, 6], "is_prefix": True}]
    assert build_prefix_mask(traces, max_response_length=4) == [[1, 1, 1, 1]]


def test_mask_is_all_zero_when_nothing_was_replayed():
    traces = [{"response_ids": [1, 2], "is_prefix": False}]
    assert build_prefix_mask(traces, max_response_length=3) == [[0, 0, 0]]


def test_empty_turns_are_skipped_like_the_base_daemon_does():
    traces = [
        {"response_ids": [], "is_prefix": True},
        {"response_ids": [9], "is_prefix": True},
    ]
    assert build_prefix_mask(traces, max_response_length=2) == [[1, 0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_daemon.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl_ext.prefix_rft.daemon'`

- [ ] **Step 3: Write the implementation**

Create `src/verl_ext/prefix_rft/daemon.py`:

```python
"""Prefix-RFT daemon: dispatch k per rollout, and mark replayed tokens.

Subclasses the vendored AgentModeDaemon. Two vendored methods are overridden:

- ``_async_set_up`` (daemon.py:264-315) queues one task per rollout from a shared
  sample dict. We write a per-rollout copy carrying ``prefix_k`` so rollout 0 of
  each question is the hybrid one.
- ``get_train_data_batch`` (daemon.py:668-844) builds the training tensors. We add
  ``prefix_mask`` and ``is_prefix_rollout_list`` alongside them.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import torch

from fine_tuning.agentflow.types import LLM, NamedResources
from fine_tuning.agentflow.verl.daemon import AgentModeDaemon

logger = logging.getLogger(__name__)


def build_prefix_mask(trace_list, max_response_length):
    """One right-padded 0/1 row per non-empty turn, 1 on replayed tokens.

    Mirrors the truncation and skip rules the base daemon applies to responses
    (daemon.py:740-772) so the mask stays aligned with ``responses`` row for row.
    """
    rows = []
    for trace in trace_list:
        response_ids = trace.get("response_ids", [])
        prompt_ids = trace.get("prompt_ids", [])
        if len(prompt_ids) == 0 and len(response_ids) == 0:
            continue
        length = min(len(response_ids), max_response_length)
        fill = 1 if trace.get("is_prefix", False) else 0
        rows.append([fill] * length + [0] * (max_response_length - length))
    return rows


class PrefixRFTDaemon(AgentModeDaemon):
    """AgentModeDaemon that dispatches prefix lengths and reports a prefix mask."""

    def __init__(self, *args, schedule=None, demo_store=None,
                 n_prefixed_rollouts=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.schedule = schedule
        self.demo_store = demo_store
        self.n_prefixed_rollouts = n_prefixed_rollouts
        self._global_step = 0
        self.last_prefix_metrics = {}

    def set_global_step(self, step: int) -> None:
        self._global_step = int(step)

    def _prefix_k_for(self, sample, rollout_index, is_train):
        """Paper A.2: one rollout of N starts from the prefix; validation never does."""
        if not is_train or self.schedule is None or self.demo_store is None:
            return 0
        if rollout_index >= self.n_prefixed_rollouts:
            return 0
        extra = sample.get("extra_info") or {}
        if not isinstance(extra, dict):
            return 0
        idx = int(extra.get("idx", -1))
        m = self.demo_store.n_steps(idx)
        if m <= 1:
            return 0
        return self.schedule.sample_k(m, global_step=self._global_step)

    async def _async_set_up(self, data, server_addresses, is_train=True):
        self.clear_data_and_server()
        try:
            orphaned = await self.server.retrieve_completed_rollouts()
            if orphaned:
                logger.info(f"Cleared {len(orphaned)} orphaned rollouts from previous runs")
        except Exception as exc:
            logger.warning(f"Failed to clear orphaned rollouts: {exc}")

        self.backend_llm_server_addresses = server_addresses
        self.is_train = is_train

        llm_resource = LLM(
            endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
            model=self.train_information.get("model", "default-model"),
            sampling_parameters={"temperature": self.train_information.get("temperature", 0.7)},
        )
        resources: NamedResources = {"main_llm": llm_resource}
        resources_id = await self.server.update_resources(resources)
        self._current_resources_id = resources_id

        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        ks = []
        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            base_sample = {key: data[key][i] for key in keys}
            base_sample["data_id"] = data_id

            for j in range(rollouts_per_sample):
                sample = dict(base_sample)
                sample["prefix_k"] = self._prefix_k_for(sample, j, is_train)
                ks.append(sample["prefix_k"])
                rollout_id = await self.server.queue_task(
                    sample=sample,
                    mode="train" if is_train else "val",
                    resources_id=resources_id,
                    metadata={"data_id": data_id, "is_train": is_train},
                )
                self._task_id_to_original_sample[rollout_id] = sample
                self._total_tasks_queued += 1

        prefixed = [k for k in ks if k > 0]
        _, low, high = (
            self.schedule.sample_l(global_step=self._global_step)
            if self.schedule is not None
            else (0.0, 0.0, 0.0)
        )
        self.last_prefix_metrics = {
            "actor/n_prefixed_rollouts": len(prefixed),
            "actor/prefix_steps": float(np.mean(prefixed)) if prefixed else 0.0,
            "actor/prefix_low": float(low),
            "actor/prefix_high": float(high),
        }
        logger.info(
            "Queued %d tasks, %d with a prefix, mean k=%.2f",
            self._total_tasks_queued,
            len(prefixed),
            self.last_prefix_metrics["actor/prefix_steps"],
        )

    def get_train_data_batch(self, max_prompt_length, max_response_length, device):
        data_proto, metrics = super().get_train_data_batch(
            max_prompt_length, max_response_length, device
        )
        if data_proto is None:
            return data_proto, metrics

        mask_rows, is_prefix_rollout = self._rebuild_prefix_rows(max_response_length)
        n_rows = data_proto.batch["responses"].shape[0]
        if len(mask_rows) != n_rows:
            raise ValueError(
                f"prefix_mask has {len(mask_rows)} rows but the batch has {n_rows}; "
                "the base daemon's row filter and build_prefix_mask have diverged"
            )

        data_proto.batch["prefix_mask"] = torch.LongTensor(mask_rows).to(device)
        data_proto.non_tensor_batch["is_prefix_rollout_list"] = np.array(is_prefix_rollout)
        # actor/num_prefix_tokens and actor/off_ratio are logged by the trainer,
        # which has the response_mask to divide by; do not duplicate them here.
        metrics.update(self.last_prefix_metrics)
        return data_proto, metrics

    def _rebuild_prefix_rows(self, max_response_length):
        """Walk completed rollouts in the same order the base daemon does."""
        mask_rows, is_prefix_rollout = [], []
        for rollout_id, rollout in self._completed_rollouts.items():
            if rollout_id not in self._task_id_to_original_sample:
                continue
            if not rollout.triplets:
                continue
            traces = [
                {
                    "prompt_ids": t.prompt.get("token_ids", []),
                    "response_ids": t.response.get("token_ids", []),
                    "is_prefix": bool((t.metadata or {}).get("prefix", False)),
                }
                for t in rollout.triplets
            ]
            rows = build_prefix_mask(traces, max_response_length)
            mask_rows.extend(rows)
            rollout_is_prefixed = any(t["is_prefix"] for t in traces)
            is_prefix_rollout.extend([rollout_is_prefixed] * len(rows))
        return mask_rows, is_prefix_rollout
```

`_rebuild_prefix_rows` walks `self._completed_rollouts` in the same insertion order the base method does and applies the same skip rules, which is why the row-count check above is a hard error rather than a warning: if the two ever diverge, `prefix_mask` would be silently misaligned with `responses` and the wrong tokens would be treated as demonstrations.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_daemon.py -v`
Expected: 4 passed (or 4 skipped if verl is not importable in the active env; run it in `cosmas-train`)

- [ ] **Step 5: Add a dispatch test**

Append to `tests/unit/test_prefix_rft_daemon.py`:

```python
def test_only_the_first_rollout_of_each_question_gets_a_prefix():
    from verl_ext.prefix_rft.daemon import PrefixRFTDaemon

    class _Store:
        def n_steps(self, idx):
            return 4

    class _Schedule:
        def sample_k(self, m, global_step):
            return 2

    daemon = PrefixRFTDaemon.__new__(PrefixRFTDaemon)
    daemon.schedule = _Schedule()
    daemon.demo_store = _Store()
    daemon.n_prefixed_rollouts = 1
    daemon._global_step = 0

    sample = {"extra_info": {"idx": 5}}
    assert daemon._prefix_k_for(sample, 0, True) == 2
    assert daemon._prefix_k_for(sample, 1, True) == 0
    assert daemon._prefix_k_for(sample, 0, False) == 0


def test_single_decision_questions_get_no_prefix():
    from verl_ext.prefix_rft.daemon import PrefixRFTDaemon

    class _Store:
        def n_steps(self, idx):
            return 1

    daemon = PrefixRFTDaemon.__new__(PrefixRFTDaemon)
    daemon.schedule = object()
    daemon.demo_store = _Store()
    daemon.n_prefixed_rollouts = 1
    daemon._global_step = 0
    assert daemon._prefix_k_for({"extra_info": {"idx": 5}}, 0, True) == 0


def test_questions_without_a_demonstration_get_no_prefix():
    from verl_ext.prefix_rft.daemon import PrefixRFTDaemon

    class _Store:
        def n_steps(self, idx):
            return 0

    daemon = PrefixRFTDaemon.__new__(PrefixRFTDaemon)
    daemon.schedule = object()
    daemon.demo_store = _Store()
    daemon.n_prefixed_rollouts = 1
    daemon._global_step = 0
    assert daemon._prefix_k_for({"extra_info": {"idx": 999}}, 0, True) == 0
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/test_prefix_rft_daemon.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add src/verl_ext/prefix_rft/daemon.py tests/unit/test_prefix_rft_daemon.py
git commit -m "feat(prefix-rft): dispatch prefix lengths and emit prefix_mask"
```

---

## Task 6: Prefix advantage correction

**Files:**
- Create: `src/verl_ext/prefix_rft/advantage.py`
- Create: `tests/unit/test_prefix_rft_advantage.py`

**Interfaces:**
- Produces: `apply_prefix_advantage(advantages, token_level_rewards, response_mask, prefix_mask, uid, rollout_id, is_prefix_rollout, num_rollouts_per_prefix=1, epsilon=1e-6) -> torch.Tensor`.

Port of `repos/prefix_rft/recipe/prefix_rft/core_algos.py:162-217`, lifted from row level to rollout level because Flow GRPO emits one row per turn.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_advantage.py`:

```python
"""Tests for the Prefix-RFT advantage correction."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verl_ext.prefix_rft.advantage import apply_prefix_advantage


def _batch(scores, is_prefix_rollout, rows_per_rollout=1, prefix_len=2, resp_len=4):
    """Build a flat batch of rows, rows_per_rollout rows per rollout."""
    n_rollouts = len(scores)
    n_rows = n_rollouts * rows_per_rollout
    token_level_rewards = torch.zeros(n_rows, resp_len)
    response_mask = torch.ones(n_rows, resp_len)
    prefix_mask = torch.zeros(n_rows, resp_len, dtype=torch.long)
    uid, rollout_id, flags = [], [], []
    row = 0
    for r in range(n_rollouts):
        for _ in range(rows_per_rollout):
            token_level_rewards[row, -1] = scores[r]
            if is_prefix_rollout[r]:
                prefix_mask[row, :prefix_len] = 1
            uid.append("q0")
            rollout_id.append(f"r{r}")
            flags.append(is_prefix_rollout[r])
            row += 1
    return (
        torch.zeros(n_rows, resp_len),
        token_level_rewards,
        response_mask,
        prefix_mask,
        np.array(uid),
        np.array(rollout_id),
        np.array(flags),
    )


def test_prefix_tokens_get_score_minus_the_unprefixed_mean():
    # 4 unprefixed rollouts scoring 1, 0, 0, 0 -> mean 0.25; hybrid scores 1.
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    hybrid_row = 4
    assert out[hybrid_row, 0].item() == pytest.approx(0.75)
    assert out[hybrid_row, 1].item() == pytest.approx(0.75)


def test_non_prefix_tokens_of_the_hybrid_rollout_pass_through_uncentred():
    # Reference behaviour: a singleton prefix group takes mean 0 and std 1.
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    assert out[4, 2].item() == pytest.approx(1.0)
    assert out[4, 3].item() == pytest.approx(1.0)


def test_unprefixed_rollouts_are_centred_over_the_unprefixed_group_only():
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args)
    # mean 0.25, population-consistent std over [1, 0, 0, 0]
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0])
    expected = (1.0 - scores.mean()) / (scores.std() + 1e-6)
    assert out[0, 0].item() == pytest.approx(expected.item(), rel=1e-4)


def test_grouping_is_per_rollout_not_per_row():
    # The hybrid rollout has 3 turns. A row-level port would centre them against
    # themselves and produce exactly zero prefix advantage.
    args = _batch(
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [False, False, False, False, True],
        rows_per_rollout=3,
    )
    out = apply_prefix_advantage(*args)
    hybrid_first_row = 4 * 3
    assert out[hybrid_first_row, 0].item() == pytest.approx(0.75)
    assert out[hybrid_first_row, 0].item() != 0.0


def test_questions_without_a_prefixed_rollout_are_left_untouched():
    args = _batch([1.0, 0.0], [False, False])
    before = args[0].clone()
    before[:] = 42.0
    args = (before,) + args[1:]
    out = apply_prefix_advantage(*args)
    assert torch.equal(out, before)


def test_padding_positions_stay_zero():
    args = list(_batch([1.0, 0.0, 1.0], [False, False, True]))
    args[2][:, -1] = 0  # response_mask marks the last position as padding
    out = apply_prefix_advantage(*args)
    assert out[:, -1].abs().sum().item() == pytest.approx(0.0)


def test_group_baseline_recentres_only_the_continuation():
    args = _batch([1.0, 0.0, 0.0, 0.0, 1.0], [False, False, False, False, True])
    out = apply_prefix_advantage(*args, singleton_baseline="group")
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0])
    expected = ((1.0 - scores.mean()) / (scores.std() + 1e-6)).item()
    # continuation recentred, prefix tokens unchanged from the reference value
    assert out[4, 2].item() == pytest.approx(expected, rel=1e-4)
    assert out[4, 0].item() == pytest.approx(0.75)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_advantage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl_ext.prefix_rft.advantage'`

- [ ] **Step 3: Write the implementation**

Create `src/verl_ext/prefix_rft/advantage.py`:

```python
"""Prefix-aware GRPO advantage.

Port of ``compute_grpo_prefix_outcome_advantage``
(repos/prefix_rft/recipe/prefix_rft/core_algos.py:162-217), lifted from the
reference's one-row-per-rollout layout to Flow GRPO's one-row-per-turn layout.

Three properties of the reference are preserved:

1. Groups are ``(question, prefix identity)``. The unprefixed rollouts share one
   group; the hybrid rollout is alone in its own. Excluding the hybrid from the
   on-policy baseline matters: at one of eight with a hybrid reward near 1.0,
   including it would lift the group mean and bias every on-policy advantage down.
2. A singleton group takes mean 0 and std 1, so the hybrid rollout's continuation
   tokens pass through uncentred. The reference authors flag this at
   core_algos.py:294-296; it is kept for fidelity and watched via metrics.
3. Prefix tokens are overwritten with ``score - mean(unprefixed)``, divided by the
   rollouts-per-prefix count. This is the quantity the paper's Figure 4 plots as
   the gap between reward-with-prefix and overall training reward.

Grouping is per rollout: scores are deduplicated by ``rollout_id`` before any
statistic is taken. A row-level port would put the hybrid rollout's several turns
in a group of their own and yield a prefix advantage of exactly zero.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

NO_PREFIX = "__no_prefix__"


def apply_prefix_advantage(
    advantages: torch.Tensor,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prefix_mask: torch.Tensor,
    uid: np.ndarray,
    rollout_id: np.ndarray,
    is_prefix_rollout: np.ndarray,
    num_rollouts_per_prefix: int = 1,
    epsilon: float = 1e-6,
    singleton_baseline: str = "none",
) -> torch.Tensor:
    """Rewrite advantages for questions that have a prefixed rollout.

    ``singleton_baseline`` is "none" for the reference behaviour, or "group" to
    recentre the hybrid rollout's continuation tokens against the unprefixed
    rollouts. See the spec's risk note; flip it only if training destabilises.
    """
    out = advantages.clone()
    row_scores = token_level_rewards.sum(dim=-1)

    # One score per rollout, and the rows that belong to it.
    rollout_rows: dict[str, list[int]] = defaultdict(list)
    for i, rid in enumerate(rollout_id):
        rollout_rows[str(rid)].append(i)

    # Question -> its rollouts, split by prefix identity.
    question_rollouts: dict[str, list[str]] = defaultdict(list)
    for rid, rows in rollout_rows.items():
        question_rollouts[str(uid[rows[0]])].append(rid)

    with torch.no_grad():
        for question, rids in question_rollouts.items():
            prefixed = [r for r in rids if bool(is_prefix_rollout[rollout_rows[r][0]])]
            if not prefixed:
                continue  # plain GRPO; verl's advantage stands

            unprefixed = [r for r in rids if r not in set(prefixed)]
            if unprefixed:
                base = torch.stack([row_scores[rollout_rows[r][0]] for r in unprefixed])
                mean_np, std_np = base.mean(), base.std()
            else:
                mean_np = torch.tensor(0.0, device=row_scores.device)
                std_np = torch.tensor(1.0, device=row_scores.device)

            # Unprefixed rollouts: standard GRPO over their own group.
            for rid in unprefixed:
                centred = (row_scores[rollout_rows[rid][0]] - mean_np) / (std_np + epsilon)
                for row in rollout_rows[rid]:
                    out[row] = centred * response_mask[row]

            # Prefixed rollouts: singleton group means mean 0, std 1, so the score
            # passes through on continuation tokens; prefix tokens take the
            # advantage-over-baseline the paper defines.
            for rid in prefixed:
                rows = rollout_rows[rid]
                score = row_scores[rows[0]]
                group = torch.stack([row_scores[rollout_rows[r][0]] for r in prefixed])
                if len(prefixed) == 1:
                    own_mean = torch.tensor(0.0, device=row_scores.device)
                    own_std = torch.tensor(1.0, device=row_scores.device)
                else:
                    own_mean, own_std = group.mean(), group.std()
                passthrough = (score - own_mean) / (own_std + epsilon)
                prefix_value = (passthrough - mean_np) / num_rollouts_per_prefix
                if singleton_baseline == "group":
                    # Opt-in mitigation for the uncentred continuation the reference
                    # authors flag at core_algos.py:294-296. Prefix tokens keep the
                    # reference value; only the continuation is recentred.
                    continuation = (score - mean_np) / (std_np + epsilon)
                else:
                    continuation = passthrough
                for row in rows:
                    filled = torch.where(
                        prefix_mask[row].bool(),
                        prefix_value.expand_as(out[row]),
                        continuation.expand_as(out[row]),
                    )
                    out[row] = filled * response_mask[row]

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_advantage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/verl_ext/prefix_rft/advantage.py tests/unit/test_prefix_rft_advantage.py
git commit -m "feat(prefix-rft): port the prefix-aware GRPO advantage"
```

---

## Task 7: Entropy clipping actor

**Files:**
- Create: `src/verl_ext/prefix_rft/actor.py`
- Create: `src/verl_ext/prefix_rft/worker.py`
- Create: `tests/unit/test_prefix_rft_actor.py`

**Interfaces:**
- Produces: `clip_prefix_advantage_by_entropy(advantages, prefix_mask, entropy, keep_ratio) -> (torch.Tensor, int)` returning the reshaped advantages and the number of prefix tokens zeroed; `PrefixRFTActor(DataParallelPPOActor)`; `PrefixRFTWorker(AsyncActorRolloutRefWorker)`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_actor.py`:

```python
"""Tests for Prefix-RFT entropy clipping."""

import pytest

torch = pytest.importorskip("torch")

from verl_ext.prefix_rft.actor import clip_prefix_advantage_by_entropy


def test_keeps_exactly_the_top_20_percent_of_prefix_tokens():
    adv = torch.ones(1, 10)
    prefix_mask = torch.ones(1, 10, dtype=torch.long)
    entropy = torch.arange(10, dtype=torch.float).unsqueeze(0)
    out, n_zeroed = clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=0.2)
    assert n_zeroed == 8
    assert out[0, :8].abs().sum().item() == 0.0
    assert out[0, 8:].tolist() == [1.0, 1.0]


def test_non_prefix_tokens_are_never_touched():
    adv = torch.ones(1, 6)
    prefix_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    entropy = torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0, 0.0]])
    out, _ = clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=0.5)
    assert out[0, 3:].tolist() == [1.0, 1.0, 1.0]


def test_selection_is_global_across_the_micro_batch():
    # Reference sorts the flattened prefix tokens, not per row (dp_actor.py:138-139).
    adv = torch.ones(2, 4)
    prefix_mask = torch.ones(2, 4, dtype=torch.long)
    entropy = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    out, n_zeroed = clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=0.25)
    assert n_zeroed == 6
    assert out[0].abs().sum().item() == 0.0
    assert out[1].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_no_prefix_tokens_is_a_no_op():
    adv = torch.ones(1, 4)
    prefix_mask = torch.zeros(1, 4, dtype=torch.long)
    entropy = torch.zeros(1, 4)
    out, n_zeroed = clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=0.2)
    assert n_zeroed == 0
    assert torch.equal(out, adv)


def test_keep_ratio_of_one_keeps_everything():
    adv = torch.ones(1, 4)
    prefix_mask = torch.ones(1, 4, dtype=torch.long)
    entropy = torch.arange(4, dtype=torch.float).unsqueeze(0)
    out, n_zeroed = clip_prefix_advantage_by_entropy(adv, prefix_mask, entropy, keep_ratio=1.0)
    assert n_zeroed == 0
    assert torch.equal(out, adv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prefix_rft_actor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verl_ext.prefix_rft.actor'`

- [ ] **Step 3: Write the clipping function and the actor**

Create `src/verl_ext/prefix_rft/actor.py`:

```python
"""Prefix-RFT actor: entropy clipping on demonstration tokens.

Paper section 3: "we propose an entropy-based clipping approach, i.e. that
involves only the top-k% high-entropy demonstration tokens. Regarding
implementation, we directly set the corresponding advantages of all other tokens
to zero." Appendix A.2 fixes k at 20.

Port of ``reshape_func``'s ``entropy`` branch
(repos/prefix_rft/recipe/prefix_rft/dp_actor.py:132-158). The reference sorts the
flattened prefix tokens of the whole micro-batch and zeroes the lowest
``1 - keep_ratio`` of them, so selection is global across the micro-batch rather
than per row; that is reproduced here.

Entropy is the current policy's, recomputed per micro-batch. Doing this at the
driver on old-policy entropy is not equivalent: ppo_mini_batch_size is 8 against a
train_batch_size of 32, so the policy moves between mini-batches.
"""

from __future__ import annotations

import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor


def clip_prefix_advantage_by_entropy(advantages, prefix_mask, entropy, keep_ratio=0.2):
    """Zero the advantage of all but the highest-entropy ``keep_ratio`` prefix tokens."""
    mask = prefix_mask.bool()
    n_prefix = int(mask.sum().item())
    if n_prefix == 0 or keep_ratio >= 1.0:
        return advantages, 0

    prefix_entropy = entropy[mask]
    order = torch.argsort(prefix_entropy)
    n_drop = int(len(order) * (1.0 - keep_ratio))
    if n_drop == 0:
        return advantages, 0

    drop_positions = order[:n_drop]
    keep_flat = torch.ones(n_prefix, dtype=torch.bool, device=advantages.device)
    keep_flat[drop_positions] = False

    out = advantages.clone()
    prefix_values = out[mask]
    prefix_values = torch.where(keep_flat, prefix_values, torch.zeros_like(prefix_values))
    out[mask] = prefix_values
    return out, n_drop


class PrefixRFTActor(DataParallelPPOActor):
    """DataParallelPPOActor that entropy-clips prefix-token advantages.

    Installed by reassigning ``__class__`` on an already-constructed actor (see
    worker.py), so it must not define ``__init__`` or require new instance state.
    Configuration is read lazily from ``self.config``.
    """

    @property
    def prefix_keep_ratio(self):
        return float(self.config.get("prefix_entropy_keep_ratio", 0.2))
```

`update_policy` must be overridden too, because verl drops `prefix_mask` during
`data.select(...)`. Copy verl 0.7.1's `DataParallelPPOActor.update_policy` body into
`PrefixRFTActor`, from the file printed by

```bash
conda run -n cosmas-train python -c "import verl, os; print(os.path.dirname(verl.__file__) + '/workers/actor/dp_actor.py')"
```

starting at `def update_policy(self, data):` (line 509 in 0.7.1) and running to the end of
the method, then make exactly these three changes:

1. After the `select_keys` list is built, add:

```python
        if "prefix_mask" in data.batch.keys():
            select_keys.append("prefix_mask")
```

2. Force entropy on, since the clip needs it:

```python
        calculate_entropy = True
```

replacing the existing `calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)`.

3. Immediately before the `policy_loss_fn(...)` call, insert:

```python
                    if "prefix_mask" in model_inputs:
                        advantages, n_zeroed = clip_prefix_advantage_by_entropy(
                            advantages,
                            model_inputs["prefix_mask"],
                            entropy,
                            keep_ratio=self.prefix_keep_ratio,
                        )
                        micro_batch_metrics["actor/prefix_tokens_zeroed"] = n_zeroed
                        micro_batch_metrics["actor/prefix_tokens_total"] = int(
                            model_inputs["prefix_mask"].sum().item()
                        )
```

Add a docstring at the top of the method recording that it is a copy of verl 0.7.1's `DataParallelPPOActor.update_policy` with those three changes, so a verl upgrade knows to re-sync it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prefix_rft_actor.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the worker**

Create `src/verl_ext/prefix_rft/worker.py`:

```python
"""Worker that installs the Prefix-RFT actor.

verl's fsdp_workers hardcodes ``DataParallelPPOActor`` (fsdp_workers.py:923) with
no configuration hook. Rather than fork verl or copy its several-hundred-line
``init_model``, the actor's class is reassigned after construction. PrefixRFTActor
adds no instance state, so this is safe.
"""

from __future__ import annotations

import logging

from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .actor import PrefixRFTActor

logger = logging.getLogger(__name__)


class PrefixRFTWorker(AsyncActorRolloutRefWorker):
    def init_model(self):
        out = super().init_model()
        actor = getattr(self, "actor", None)
        if actor is not None and self._is_actor:
            actor.__class__ = PrefixRFTActor
            logger.info("Installed PrefixRFTActor (entropy clipping on prefix tokens)")
        return out
```

- [ ] **Step 6: Add a worker test**

Append to `tests/unit/test_prefix_rft_actor.py`:

```python
def test_worker_swaps_the_actor_class():
    pytest.importorskip("verl")
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    from verl_ext.prefix_rft.actor import PrefixRFTActor
    from verl_ext.prefix_rft.worker import PrefixRFTWorker

    class _Stub(DataParallelPPOActor):
        def __init__(self):
            pass

    worker = PrefixRFTWorker.__new__(PrefixRFTWorker)
    worker.actor = _Stub()
    worker._is_actor = True
    # Bypass the real init_model, which needs GPUs.
    actor = worker.actor
    actor.__class__ = PrefixRFTActor
    assert isinstance(worker.actor, PrefixRFTActor)
    assert isinstance(worker.actor, DataParallelPPOActor)
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/unit/test_prefix_rft_actor.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add src/verl_ext/prefix_rft/actor.py src/verl_ext/prefix_rft/worker.py tests/unit/test_prefix_rft_actor.py
git commit -m "feat(prefix-rft): entropy-clip prefix advantages in the actor"
```

---

## Task 8: Trainer and entrypoint

**Files:**
- Create: `src/verl_ext/prefix_rft/trainer.py`
- Create: `src/verl_ext/prefix_rft/entrypoint.py`
- Create: `src/verl_ext/prefix_rft/__main__.py`
- Create: `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`

**Interfaces:**
- Consumes: `fine_tuning.agentflow.verl.trainer.AgentFlowTrainer`; `PrefixRFTDaemon`; `apply_prefix_advantage`; `PrefixStepSchedule`; `DemoStore`; `PrefixRFTWorker`.
- Produces: `PrefixRFTTrainer(AgentFlowTrainer)`; `main()` for `python -m verl_ext.prefix_rft`.

- [ ] **Step 1: Write the hydra config**

Create `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`:

```yaml
# Extends the vendored AgentFlow config with the prefix_rft block, so the keys
# exist in the schema and Hydra overrides need no + prefix.
hydra:
  searchpath:
    - pkg://verl/trainer/config
    - pkg://fine_tuning.agentflow/verl

defaults:
  - config
  - _self_

prefix_rft:
  enable: true
  demos_path: data/training/prefix_rft/prefix_demos.parquet
  # Paper A.2: 8 rollouts per prompt, one of them starts with the sampled prefix.
  n_prefixed_rollouts: 1
  # Paper A.2: l ~ U(low_t, 0.95); low_t cosine-decays 0.95 -> 0.05.
  high: 0.95
  low_init: 0.95
  low_target: 0.05
  # BetaSampler defaults; alpha = beta = 1 is the uniform draw A.2 specifies.
  sampler_alpha: 1.0
  sampler_beta: 1.0
  # Paper A.2 and section 6: keep the top 20% highest-entropy prefix tokens.
  entropy_keep_ratio: 0.2
  # "none" reproduces the reference exactly. "group" recentres the hybrid rollout's
  # continuation tokens; see the spec's risk note. Do not change without recording it.
  singleton_baseline: none
  seed: 42
```

- [ ] **Step 2: Write the trainer**

Create `src/verl_ext/prefix_rft/trainer.py`:

```python
"""Prefix-RFT trainer.

Subclasses the vendored AgentFlowTrainer. ``_train_step`` is a copy of the
vendored method with two insertions, because it is a single long method with no
smaller seam and VENDORED.md forbids editing it in place. The alternative,
rebinding the vendored module's ``compute_advantage`` attribute, was rejected as
an invisible patch.

COPIED FROM: src/fine_tuning/agentflow/verl/trainer.py, method _train_step,
vendored revision 8ed8f41 (see VENDORED.md). Re-sync on re-vendor.
"""

from __future__ import annotations

import logging

import numpy as np

from fine_tuning.agentflow.verl.trainer import AgentFlowTrainer

from .advantage import apply_prefix_advantage
from .daemon import PrefixRFTDaemon
from .demos import DemoStore
from .schedule import PrefixStepSchedule

logger = logging.getLogger(__name__)


class PrefixRFTTrainer(AgentFlowTrainer):
    """AgentFlowTrainer with prefix dispatch and prefix-aware advantages."""

    def _build_prefix_components(self):
        cfg = self.config.prefix_rft
        total_steps = getattr(self, "total_training_steps", None) or int(
            self.config.trainer.get("total_training_steps") or 500
        )
        schedule = PrefixStepSchedule(
            low_init=cfg.low_init,
            low_target=cfg.low_target,
            high=cfg.high,
            n_steps=total_steps,
            alpha=cfg.sampler_alpha,
            beta=cfg.sampler_beta,
            seed=cfg.seed,
        )
        store = DemoStore.from_parquet(cfg.demos_path)
        n_questions, n_decisions = store.coverage()
        logger.info(
            "Prefix-RFT: %d demonstrated questions, %d decisions, schedule over %d steps",
            n_questions,
            n_decisions,
            total_steps,
        )
        return schedule, store
```

Then add `_train_step`. Copy the whole method from `src/fine_tuning/agentflow/verl/trainer.py:161-359` into this class verbatim, then make these two insertions:

1. Immediately after the `batch = compute_advantage(...)` call (vendored lines 288-296), insert:

```python
                if self.config.prefix_rft.enable and "prefix_mask" in batch.batch.keys():
                    batch.batch["advantages"] = apply_prefix_advantage(
                        advantages=batch.batch["advantages"],
                        token_level_rewards=batch.batch["token_level_rewards"],
                        response_mask=batch.batch["response_mask"],
                        prefix_mask=batch.batch["prefix_mask"],
                        uid=batch.non_tensor_batch["uid"],
                        rollout_id=batch.non_tensor_batch["rollout_id_list"],
                        is_prefix_rollout=batch.non_tensor_batch["is_prefix_rollout_list"],
                        num_rollouts_per_prefix=int(
                            self.config.prefix_rft.n_prefixed_rollouts
                        ),
                        singleton_baseline=str(self.config.prefix_rft.singleton_baseline),
                    )
                    prefix_mask = batch.batch["prefix_mask"]
                    response_mask = batch.batch["response_mask"]
                    metrics["actor/num_prefix_tokens"] = int(prefix_mask.sum().item())
                    metrics["actor/off_ratio"] = float(
                        prefix_mask.sum().item() / max(1, int(response_mask.sum().item()))
                    )
                    is_prefixed = np.asarray(
                        batch.non_tensor_batch["is_prefix_rollout_list"], dtype=bool
                    )
                    scores = batch.batch["token_level_scores"].sum(dim=-1).float().cpu().numpy()
                    if is_prefixed.any():
                        metrics["actor/reward_with_prefix"] = float(scores[is_prefixed].mean())
                    if (~is_prefixed).any():
                        metrics["actor/reward_without_prefix"] = float(
                            scores[~is_prefixed].mean()
                        )
```

2. The daemon is built once inside the vendored `fit()` at
   `src/fine_tuning/agentflow/verl/trainer.py:427` as `self.agent_mode_daemon = AgentModeDaemon(...)`,
   not inside `_train_step`, so copying `fit` as well would double the copied surface.
   Promote the existing instance in place instead, the same trick the worker uses for the
   actor. Add this method to `PrefixRFTTrainer`:

```python
    def _ensure_prefix_daemon(self):
        """Promote the vendored daemon in place.

        AgentModeDaemon is constructed inside the vendored fit() (trainer.py:427).
        PrefixRFTDaemon adds only attributes, so reassigning __class__ and setting
        them explicitly is equivalent to having constructed it directly, and avoids
        copying fit() as well.
        """
        daemon = self.agent_mode_daemon
        if isinstance(daemon, PrefixRFTDaemon):
            return
        schedule, store = self._build_prefix_components()
        daemon.__class__ = PrefixRFTDaemon
        daemon.schedule = schedule
        daemon.demo_store = store
        daemon.n_prefixed_rollouts = int(self.config.prefix_rft.n_prefixed_rollouts)
        daemon._global_step = 0
        daemon.last_prefix_metrics = {}
        logger.info("Promoted AgentModeDaemon to PrefixRFTDaemon")
```

   Then, at the very top of the copied `_train_step`, before anything else, add:

```python
        if self.config.prefix_rft.enable:
            self._ensure_prefix_daemon()
            self.agent_mode_daemon.set_global_step(self.global_steps)
```

   Validation is unaffected: `_validate` (trainer.py:118) calls the same daemon with
   `is_train=False`, and `_prefix_k_for` returns 0 for validation whether or not the
   promotion has happened yet. With `val_before_train: true` the first validation runs on
   the unpromoted daemon, whose tasks carry no `prefix_k` key; the rollout reads
   `task.get("prefix_k", 0)` and so behaves identically.

- [ ] **Step 3: Write the entrypoint**

Create `src/verl_ext/prefix_rft/entrypoint.py` as a copy of `src/fine_tuning/agentflow/verl/entrypoint.py` with four changes:

```python
# 1. the hydra decorator
@hydra.main(config_path="pkg://verl_ext.prefix_rft/config",
            config_name="prefix_rft_trainer", version_base=None)

# 2. the worker class
from verl_ext.prefix_rft.worker import PrefixRFTWorker
actor_rollout_cls = PrefixRFTWorker  # async mode only; Prefix-RFT requires it

# 3. the trainer class
from verl_ext.prefix_rft.trainer import PrefixRFTTrainer
trainer = PrefixRFTTrainer(...)  # same arguments as AgentFlowTrainer

# 4. force calculate_entropy on, since the clip needs it
with open_dict(config):
    config.actor_rollout_ref.actor.calculate_entropy = True
```

Everything else, including `AgentDataset`, `create_rl_sampler`, the Ray init block and the role mapping, is unchanged from the vendored entrypoint.

Create `src/verl_ext/prefix_rft/__main__.py`:

```python
from .entrypoint import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the config composes**

Run:

```bash
conda run -n cosmas-train python -c "
from hydra import compose, initialize_config_dir
import os, verl_ext.prefix_rft as p
d = os.path.join(os.path.dirname(p.__file__), 'config')
with initialize_config_dir(config_dir=d, version_base=None):
    cfg = compose(config_name='prefix_rft_trainer')
print(cfg.prefix_rft.entropy_keep_ratio, cfg.prefix_rft.low_target, cfg.agentflow.port)
"
```

Expected: `0.2 0.05 9999`. A failure here means the `defaults`/`searchpath` composition is wrong, not the trainer.

- [ ] **Step 5: Commit**

```bash
git add src/verl_ext/prefix_rft/trainer.py src/verl_ext/prefix_rft/entrypoint.py src/verl_ext/prefix_rft/__main__.py src/verl_ext/prefix_rft/config/
git commit -m "feat(prefix-rft): trainer, entrypoint and hydra config"
```

---

## Task 9: Experiment configs and launcher wiring

**Files:**
- Create: `experiments/configs/fine_tuning/config_prefix_rft.yaml`
- Create: `experiments/configs/fine_tuning/config_prefix_rft_smoke8b.yaml`
- Modify: `scripts/launch_verl.py:206`
- Modify: `scripts/train_orchestrator.py:140-154`

**Interfaces:**
- Consumes: the `env:` block convention both scripts already read.
- Produces: `PREFIX_RFT` env key selecting the Prefix-RFT module and rollout class.

- [x] **Step 1: Write the production config**

Create `experiments/configs/fine_tuning/config_prefix_rft.yaml` as a copy of `experiments/configs/fine_tuning/config.yaml` with these changes only:

```yaml
env:
  # ... every existing key unchanged, plus:
  PREFIX_RFT: "true"
  PREFIX_DEMOS_PATH: "data/training/prefix_rft/prefix_demos.parquet"
  EXPERIMENT_NAME: qwen3-8b-prefix-rft-search-math   # was qwen3-8b-grpo-search-math

python_args:
  # ... every existing key unchanged, plus:
  prefix_rft.enable: true
  prefix_rft.demos_path: data/training/prefix_rft/prefix_demos.parquet
  prefix_rft.n_prefixed_rollouts: 1
  prefix_rft.high: 0.95
  prefix_rft.low_init: 0.95
  prefix_rft.low_target: 0.05
  prefix_rft.sampler_alpha: 1.0
  prefix_rft.sampler_beta: 1.0
  prefix_rft.entropy_keep_ratio: 0.2
  prefix_rft.singleton_baseline: none
  prefix_rft.seed: 42
  actor_rollout_ref.actor.calculate_entropy: true
  +actor_rollout_ref.actor.prefix_entropy_keep_ratio: 0.2

These keys already exist in `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`, so
they need no `+` prefix. `prefix_entropy_keep_ratio` is the exception: it is read off the
actor config, which is verl's schema, so it does need `+`.
```

`EXPERIMENT_NAME` must differ from the GRPO run so checkpoints and W&B rows stay distinct.

- [x] **Step 2: Write the smoke config**

Create `experiments/configs/fine_tuning/config_prefix_rft_smoke8b.yaml` as a copy of `config_smoke8b.yaml` with the same additions, plus `EXPERIMENT_NAME: qwen3-8b-prefix-rft-smoke`. With `rollout.n: 2` the hybrid rollout is 1 of 2, which exercises every path while distorting the imitation balance; the smoke test asserts machinery, not quality.

The store is keyed on question text, so no separate smoke store is needed: the same
`prefix_demos.parquet` serves both, and a smoke question is covered if and only if its
text appears in the store. **Check that before running the smoke job**, because if none of
the eight smoke questions is covered then `k` is always 0 and the run proves nothing:

```bash
python - <<'PYCHECK'
import pandas as pd
from verl_ext.prefix_rft.demos import DemoStore
store = DemoStore.from_parquet("data/training/prefix_rft/prefix_demos.parquet")
smoke = pd.read_parquet("data/training/smoke/train/combined_train.parquet")
covered = [(q, store.n_steps(q)) for q in smoke["question"]]
print(f"{sum(1 for _, n in covered if n > 1)} of {len(covered)} smoke questions are prefixable")
for q, n in covered:
    print(f"  n_steps={n}  {q[:70]}")
PYCHECK
```

If too few are covered, rebuild `data/training/smoke` from questions the store does cover
rather than weakening the assertion in the smoke job.

- [x] **Step 3: Wire launch_verl.py**

In `scripts/launch_verl.py`, replace line 206:

```python
    command = [sys.executable, "-u", "-m", "fine_tuning.agentflow.verl"]
```

with:

```python
    # Prefix-RFT runs its own entrypoint, which substitutes the trainer, daemon,
    # worker and actor. Everything else about the launch is identical.
    prefix_rft = os.environ.get("PREFIX_RFT", "").strip().lower() in ("1", "true", "yes", "on")
    module = "verl_ext.prefix_rft" if prefix_rft else "fine_tuning.agentflow.verl"
    if prefix_rft:
        print("  Prefix-RFT enabled: launching verl_ext.prefix_rft")
    command = [sys.executable, "-u", "-m", module]
```

- [x] **Step 4: Wire train_orchestrator.py**

In `scripts/train_orchestrator.py`, replace the rollout construction at lines 140-154:

```python
    from fine_tuning.rollout import OrchestratorRollout
    from fine_tuning.agentflow import Trainer

    rollout_dir = str(output_dir / "rollout_data")
    prefix_rft = os.environ.get("PREFIX_RFT", "").strip().lower() in ("1", "true", "yes", "on")
    common = dict(
        rollout_dir=rollout_dir,
        rollout_n=rollout_n,
        train_temperature=train_temperature,
        test_temperature=test_temperature,
        max_turns=max_turns,
        max_tokens=max_tokens,
        use_thinking=use_thinking,
        subagent_endpoint=subagent_endpoint,
        subagent_model=subagent_model,
    )
    if prefix_rft:
        from fine_tuning.prefix_rollout import PrefixOrchestratorRollout

        agent = PrefixOrchestratorRollout(
            demos_path=os.environ.get(
                "PREFIX_DEMOS_PATH", "data/training/prefix_rft/prefix_demos.parquet"
            ),
            base_model=env.get("BASE_MODEL"),
            **common,
        )
        print("  Prefix-RFT enabled: using PrefixOrchestratorRollout")
    else:
        agent = OrchestratorRollout(**common)
```

- [x] **Step 5: Verify both launchers still build the base command**

Run:

```bash
python - <<'PY'
import os, subprocess, sys
env = dict(os.environ); env.pop("PREFIX_RFT", None)
out = subprocess.run(
    [sys.executable, "scripts/launch_verl.py", "--config",
     "experiments/configs/fine_tuning/config.yaml", "--help"],
    capture_output=True, text=True, env=env)
print(out.returncode)
PY
```

Expected: exit 0, and no `verl_ext.prefix_rft` in the printed command. Then set `PREFIX_RFT=true` and confirm the module switches.

- [x] **Step 6: Commit**

```bash
git add experiments/configs/fine_tuning/config_prefix_rft.yaml experiments/configs/fine_tuning/config_prefix_rft_smoke8b.yaml scripts/launch_verl.py scripts/train_orchestrator.py
git commit -m "feat(prefix-rft): configs and launcher wiring"
```

---

## Task 10: Jobs, smoke run and documentation

**Files:**
- Create: `jobs/fine_tuning/008_build_prefix_demos.job`
- Create: `jobs/fine_tuning/009_run_tests_for_prefix_rft.job`
- Create: `jobs/fine_tuning/010_smoke_prefix_rft.job`
- Create: `docs/pipelines/prefix-rft.md`
- Modify: `docs/guides/add-an-adaptation-method.md:8-13`
- Modify: `README.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the build job**

Copy the preamble of `jobs/fine_tuning/007_run_tests_for_sft_folded.job` (the `#SBATCH`
block, `module load`, and the `_is_repo` checkout-location loop) into
`jobs/fine_tuning/008_build_prefix_demos.job`, changing `--job-name=PrefixRFTDemos`,
`--time=00:30:00`, and the log paths to `out/fine_tuning/prefix_rft/build_%A.{log,err}`.
Then the body:

```bash
set -euo pipefail
mkdir -p out/fine_tuning/prefix_rft data/training/prefix_rft
source activate agent_engine

JSONL=$(ls -t data/training/sft/collected_*.jsonl | head -1)
echo "Using teacher collection: $JSONL"

python scripts/build_prefix_demos.py "$JSONL" \
    --output data/training/prefix_rft/prefix_demos.parquet

python scripts/check_prefix_demos.py \
    --demos data/training/prefix_rft/prefix_demos.parquet
```

`set -e` is correct here, unlike in the test job: there is nothing to summarise, and a
failed gate must stop the pipeline.

- [ ] **Step 2: Write the test job**

Copy `jobs/fine_tuning/007_run_tests_for_sft_folded.job` to
`jobs/fine_tuning/009_run_tests_for_prefix_rft.job`. Keep its whole preamble unchanged:
the `#SBATCH` block (change only `--job-name=PrefixRFTTests` and the two log paths to
`out/fine_tuning/tests/prefix_rft_tests_%A.{log,err}`), the `module load` lines, the
`_is_repo` checkout-location loop, and the `note_pass` / `note_fail` / `FAILURES` summary
machinery. That preamble solves problems (sbatch spooling the script away from the repo,
collection walking `data/`) that this job has too.

Replace the five stages with these four:

```bash
DEMOS=data/training/prefix_rft/prefix_demos.parquet

if [[ ! -f "$DEMOS" ]]; then
    echo "ERROR: missing $DEMOS — run 008_build_prefix_demos.job first"
    exit 1
fi

# ── Stage 1/4: unit tests under agent_engine ─────────────────────────────────────
echo ""
echo "── Stage 1/4: Prefix-RFT unit tests (env: agent_engine) ─────────────────────"
source activate agent_engine
python -m pytest tests/unit/test_prefix_rft_schedule.py \
                 tests/unit/test_prefix_rft_demos.py \
                 tests/unit/test_prefix_rft_rollout.py \
                 tests/unit/test_prefix_rft_daemon.py \
                 tests/unit/test_prefix_rft_advantage.py \
                 tests/unit/test_prefix_rft_actor.py -v --no-header
if [[ $? -eq 0 ]]; then note_pass "unit tests (agent_engine)"; else note_fail "unit tests (agent_engine)"; fi

# ── Stage 2/4: the gate on the real store ────────────────────────────────────────
echo ""
echo "── Stage 2/4: pre-flight gate on the demonstration store ────────────────────"
python scripts/check_prefix_demos.py --demos "$DEMOS"
if [[ $? -eq 0 ]]; then note_pass "gate on the demonstration store"; else note_fail "gate on the demonstration store"; fi

# ── Stage 3/4: trip-wire — the gate must reject a corrupted store ────────────────
# A gate that passes everything proves nothing.
echo ""
echo "── Stage 3/4: trip-wire — the gate must REJECT a store with a dropped result ─"
python - "$DEMOS" <<'PY'
import sys, pandas as pd
frame = pd.read_parquet(sys.argv[1]).head(20).copy()
steps = list(frame.iloc[0]["steps"])
if len(steps) < 2:
    for i in range(len(frame)):
        steps = list(frame.iloc[i]["steps"])
        if len(steps) >= 2:
            break
steps[0] = {**steps[0], "tool_result": None}
frame.at[frame.index[0], "steps"] = steps
frame.to_parquet("/tmp/prefix_demos_corrupt.parquet", index=False)
PY
if python scripts/check_prefix_demos.py --demos /tmp/prefix_demos_corrupt.parquet > /dev/null 2>&1; then
    note_fail "trip-wire — gate ACCEPTED a store with a missing tool_result"
else
    note_pass "trip-wire — gate rejected the corrupted store"
fi

# ── Stage 4/4: verl-dependent tests under cosmas-train ───────────────────────────
# The daemon and actor import verl, which only exists in cosmas-train.
echo ""
echo "── Stage 4/4: daemon and actor tests (env: cosmas-train) ────────────────────"
conda deactivate 2>/dev/null || true
source activate cosmas-train
# pytest is not installed in cosmas-train, so the verl-dependent checks run as
# scripts. These are the same assertions the skipped pytest cases make.
python scripts/check_prefix_rft_trainer_sync.py
if [[ $? -eq 0 ]]; then note_pass "copied methods and config in sync"; else note_fail "copied methods and config in sync"; fi

python -c "
import sys; sys.path.insert(0, 'src')
from verl_ext.prefix_rft.daemon import PrefixRFTDaemon
from verl_ext.prefix_rft.actor import PrefixRFTActor
from verl_ext.prefix_rft.worker import PrefixRFTWorker
from verl_ext.prefix_rft.trainer import PrefixRFTTrainer
from verl_ext.prefix_rft import entrypoint
from fine_tuning.prefix_rollout import PrefixOrchestratorRollout
print('  all Prefix-RFT modules import under cosmas-train')
"
if [[ \$? -eq 0 ]]; then note_pass "modules import under cosmas-train"; else note_fail "modules import under cosmas-train"; fi
```

Change the final summary line from `safe to submit 007_train_sft_folded.job` to
`safe to submit 010_smoke_prefix_rft.job`.

- [ ] **Step 3: Write the smoke job**

Create `jobs/fine_tuning/010_smoke_prefix_rft.job` as a copy of `jobs/fine_tuning/004_smoke_8b.job` pointing at `config_prefix_rft_smoke8b.yaml`, with one addition: after the run, assert the prefix machinery was active.

```bash
LOG="out/fine_tuning/prefix_rft/smoke_${SLURM_JOB_ID}_verl.log"
python - "$LOG" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
tokens = [int(m) for m in re.findall(r"actor/num_prefix_tokens[\"']?[:=]\s*(\d+)", text)]
if not tokens or max(tokens) == 0:
    sys.exit("FAIL: no prefix tokens entered the loss; k was 0 for every rollout")
if "Installed PrefixRFTActor" not in text:
    sys.exit("FAIL: PrefixRFTActor was never installed")
print(f"PASS: prefix machinery active, max num_prefix_tokens={max(tokens)}")
PY
```

- [ ] **Step 4: Run the CPU jobs**

```bash
sbatch jobs/fine_tuning/008_build_prefix_demos.job
sbatch jobs/fine_tuning/009_run_tests_for_prefix_rft.job
```

Expected: the build reports about 700 demonstrations and `PASSED`; the test job reports all unit tests passing.

- [ ] **Step 5: Run the smoke test**

```bash
sbatch jobs/fine_tuning/010_smoke_prefix_rft.job
```

Expected: two gradient steps, a checkpoint saved, and `PASS: prefix machinery active`. Also check by hand in the log that `actor/reward_with_prefix` exceeds `actor/reward_without_prefix`, which is the paper's Figure 4 signature. On a two-step smoke run this is weak evidence, so record it rather than gating on it.

- [ ] **Step 6: Verify the tokenisation assumption against the live proxy**

The replay path tokenises locally and assumes it matches the proxy. Confirm once, from the smoke run's rollout JSONs:

```bash
python - <<'PY'
import json, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
path = sorted(glob.glob("experiments/results/fine_tuning/qwen3-8b-prefix-rft-smoke/*/rollout_data/train/*/*.json"))[0]
record = json.load(open(path))
print("inspect", path)
print("messages roles:", [m["role"] for m in record["output_messages"]])
PY
```

Then compare one replayed turn's `prompt_token_ids` against
`tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)` for the
same messages, after applying the `tool -> user` remap. They must be identical.

This is the highest-risk unverified assumption in the whole implementation. The replay
path tokenises locally, mirroring the proxy's two calls exactly
(`daemon.py:216-225`: `apply_chat_template` with `add_generation_prompt=True` and no
`enable_thinking` kwarg, then `encode` with `add_special_tokens=False` and no appended
EOS), and `ReplayController._decode_messages` mirrors the provider's `tool -> user` remap
(`api_provider.py:94-100`). All of that is unit-tested against a fake tokenizer, so the
*shape* is right; what is untested is that the real tokenizer and the real proxy agree.

If they do not, every prefix triplet is misaligned: the prefix_mask would mark tokens that
are not the teacher's, the advantage would be applied to the wrong positions, and training
would proceed normally and report success. No metric in the run would reveal it. Do not
skip this step, and do not launch the production run until it has passed.

- [ ] **Step 7: Write the pipeline documentation**

Create `docs/pipelines/prefix-rft.md` covering, in the style of `docs/pipelines/sft.md`: what the method is in three sentences; the step-prefix adaptation and its two accepted consequences; the run sequence (008, 009, 010); the config keys and where each value comes from; the metrics to watch and what they mean; and the seven divergences from the paper. Link to the spec for the reasoning and to `docs/pipelines/rl.md` for the shared RL machinery.

- [ ] **Step 8: Register the method in the guide**

In `docs/guides/add-an-adaptation-method.md`, add a row to the table at lines 8-13:

```markdown
| **Prefix-RFT** | the weights, from its own rollouts seeded by demonstrations | `PrefixOrchestratorRollout` replays teacher decisions inside verl |
```

and a sentence in the Level 3 section noting that Prefix-RFT is the worked example of extending Level 3 without touching vendored code.

- [ ] **Step 9: Update README and CHANGELOG**

Add Prefix-RFT to the README's list of adaptation methods and a CHANGELOG entry describing the addition, the spec path, and the fact that only the smoke path has been run.

- [ ] **Step 10: Run the full test suite**

Run: `pytest`
Expected: every previously passing test still passes, plus the new ones.

- [ ] **Step 11: Commit**

```bash
git add jobs/fine_tuning/008_build_prefix_demos.job jobs/fine_tuning/009_run_tests_for_prefix_rft.job jobs/fine_tuning/010_smoke_prefix_rft.job docs/pipelines/prefix-rft.md docs/guides/add-an-adaptation-method.md README.md CHANGELOG.md
git commit -m "feat(prefix-rft): jobs, smoke verification and documentation"
```

---

## Notes for the executor

- Tasks 1, 2 and 3 need no GPU and no verl; run them in `agent_engine` if that is faster.
- Tasks 5 to 8 need `conda activate cosmas-train`.
- Tasks 4, 5, 7 and 8 each copy a method from vendored or third-party code. Every copy carries a docstring naming the source file and revision. Do not skip that: it is the only thing that makes the copy re-syncable.
- If a task reveals that a vendored method cannot be extended by subclassing, stop and report it rather than editing the vendored file.
