# GEPA Reflective Feedback — Iteration 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the reflective records that `AgentGEPAAdapter` produces — unify thinking-snippet caps, add last-turn thinking to `system_prompt` records, symmetrize `planning_suffix` records, and replace head-of-list slicing in `_balanced_sample` with a seeded shuffle.

**Architecture:** Three small refinements to `src/gepa_integration/adapter.py` plus tests and docs. No orchestrator or state-schema changes — all data is already on `ExecutionState`. Iteration 1's `_diagnose` is unchanged. Spec: `docs/superpowers/specs/2026-05-15-gepa-integration-design.md`, "Addendum 2026-05-18 (Iteration 2)".

**Tech stack:** Python 3.11+, pytest, `gepa==0.0.22`. Test env on Snellius: `conda activate agent_engine` (pydantic, gepa installed). Login-node Python lacks pydantic, so tests must run on a compute node or the user's local `.venv`.

---

## File map

| File | What changes |
|---|---|
| `src/gepa_integration/adapter.py` | Constant `_THINKING_SNIPPET_LEN` 1500 → 800; constructor adds `sample_seed: int = 0`; `_balanced_sample` shuffles deterministically; `_system_prompt_records` adds conditional `thinking_at_last_turn`; `_planning_suffix_records` replaces `raw_planning_output` with `plan` + `thinking_in_plan` |
| `tests/gepa_integration/test_adapter.py` | Update `test_make_reflective_dataset_planning_uses_raw` (field rename); add ~6 new tests covering thinking cap, sample shuffle, last-turn thinking inclusion + skip, plan-split fields |
| `src/gepa_integration/README.md` | Update "Feedback design (μ_f for CoSMAS)" section: note unified 800-char cap, list the new fields (`thinking_at_last_turn`, `plan`, `thinking_in_plan`), mention seeded sample shuffle. Bump test count |
| `CHANGELOG.md` | Append an Iteration 2 bullet under the existing **Changed** subsection of `[Unreleased]` |

---

## Task 1: Unify thinking-snippet cap at 800 chars

**Files:**
- Modify: `src/gepa_integration/adapter.py` (constant `_THINKING_SNIPPET_LEN`, around line 156)
- Test: `tests/gepa_integration/test_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/gepa_integration/test_adapter.py` (new section at end of file):

```python
# ── thinking-snippet caps ────────────────────────────────────────────────────

def _state_with_thinking(qid, thinking_text, correct=True):
    """ExecutionState whose first assistant message wraps `thinking_text` in <think>."""
    state = _make_state(qid, "Q?", "answer", correct=correct)
    state.output_messages = [
        {"role": "assistant", "content": f"<think>{thinking_text}</think>some output"},
    ]
    return state


def test_first_turn_thinking_capped_at_800_chars():
    adapter = _make_adapter()
    long_think = "x" * 2000
    state = _state_with_thinking(1, long_think, correct=True)
    batch = GEPAEvaluationBatch(outputs=["answer"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    snippet = result["system_prompt"][0]["Generated Outputs"]["thinking_before_first_tool"]
    # 800 chars + "…[truncated]" suffix
    assert len(snippet) <= 800 + len("…[truncated]")
    assert snippet.endswith("…[truncated]")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gepa_integration/test_adapter.py::test_first_turn_thinking_capped_at_800_chars -v`
Expected: FAIL — current cap is 1500, so the snippet length will exceed `800 + len("…[truncated]")`.

- [ ] **Step 3: Change the constant**

Edit `src/gepa_integration/adapter.py`. Find:

```python
    _THINKING_SNIPPET_LEN = 1500  # chars per <think> trace (truncated)
```

Replace with:

```python
    _THINKING_SNIPPET_LEN = 800   # chars per <think> trace (truncated, unified across record types)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gepa_integration/test_adapter.py::test_first_turn_thinking_capped_at_800_chars -v`
Expected: PASS.

- [ ] **Step 5: Run the full file to make sure nothing else broke**

Run: `pytest tests/gepa_integration/test_adapter.py -v`
Expected: all previously-passing tests still pass; one new test passes.

- [ ] **Step 6: Commit**

```bash
git add src/gepa_integration/adapter.py tests/gepa_integration/test_adapter.py
git commit -m "refactor(gepa): unify thinking-snippet cap to 800 chars

Iteration 2 of the reflective feedback design (see spec addendum
2026-05-18) uses a single 800-char cap across all thinking fields,
in preparation for the new last-turn and plan thinking snippets."
```

---

## Task 2: Seeded shuffle in `_balanced_sample`

**Files:**
- Modify: `src/gepa_integration/adapter.py` (`__init__` signature, around line 60; `_balanced_sample` body, around line 152; add `import random` at the top)
- Test: `tests/gepa_integration/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/gepa_integration/test_adapter.py` (new section):

```python
# ── _balanced_sample shuffle ─────────────────────────────────────────────────

def _wrong_states_with_distinguishable_ids(n):
    """n wrong states, each with question_id i so we can verify ordering."""
    return [_make_state(i, "Q", "x", correct=False) for i in range(n)]


def test_balanced_sample_is_deterministic_given_seed():
    a = AgentGEPAAdapter(model_provider=MagicMock(), tool_registry=ToolRegistry(),
                         use_thinking=True, max_turns=3, sample_seed=42)
    states = _wrong_states_with_distinguishable_ids(20)
    scores = [0.0] * 20
    first = [s.question_id for s, _ in a._balanced_sample(states, scores)]
    second = [s.question_id for s, _ in a._balanced_sample(states, scores)]
    assert first == second  # same seed → same selection


def test_balanced_sample_shuffles_off_head_of_list():
    a = AgentGEPAAdapter(model_provider=MagicMock(), tool_registry=ToolRegistry(),
                         use_thinking=True, max_turns=3, sample_seed=0)
    states = _wrong_states_with_distinguishable_ids(20)
    scores = [0.0] * 20
    picked = [s.question_id for s, _ in a._balanced_sample(states, scores)]
    head = list(range(4))  # what head-of-list slicing would return for half=4
    assert picked != head, (
        "Expected shuffled selection to differ from the first 4 IDs; "
        "if this collides, change sample_seed in the test."
    )


def test_balanced_sample_different_seeds_give_different_order():
    states = _wrong_states_with_distinguishable_ids(20)
    scores = [0.0] * 20
    a0 = AgentGEPAAdapter(model_provider=MagicMock(), tool_registry=ToolRegistry(),
                          use_thinking=True, max_turns=3, sample_seed=0)
    a1 = AgentGEPAAdapter(model_provider=MagicMock(), tool_registry=ToolRegistry(),
                          use_thinking=True, max_turns=3, sample_seed=1)
    p0 = [s.question_id for s, _ in a0._balanced_sample(states, scores)]
    p1 = [s.question_id for s, _ in a1._balanced_sample(states, scores)]
    assert p0 != p1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gepa_integration/test_adapter.py -v -k balanced_sample`
Expected: FAIL on `sample_seed` kwarg not accepted by `__init__` (TypeError).

- [ ] **Step 3: Add `import random` to adapter.py**

Edit `src/gepa_integration/adapter.py`. Find the imports block at the top:

```python
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional
```

Replace with:

```python
from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional
```

- [ ] **Step 4: Add `sample_seed` to constructor**

Edit `src/gepa_integration/adapter.py`. Find:

```python
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
```

Replace with:

```python
    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        use_thinking: bool = True,
        max_turns: int = 15,
        tool_limits: Optional[dict[str, int]] = None,
        sample_seed: int = 0,
    ) -> None:
        self.model_provider = model_provider
        self.tool_registry = tool_registry
        self.use_thinking = use_thinking
        self.max_turns = max_turns
        self.tool_limits = tool_limits or {"web_search": 10}
        self._sample_seed = sample_seed
```

- [ ] **Step 5: Implement seeded shuffle in `_balanced_sample`**

Edit `src/gepa_integration/adapter.py`. Find:

```python
    def _balanced_sample(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[tuple[ExecutionState, float]]:
        """Return up to MAX_RECORDS pairs balanced between correct and wrong."""
        correct = [(s, sc) for s, sc in zip(states, scores) if sc > 0]
        wrong = [(s, sc) for s, sc in zip(states, scores) if sc == 0]
        half = self._MAX_RECORDS // 2
        return correct[:half] + wrong[:half]
```

Replace with:

```python
    def _balanced_sample(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[tuple[ExecutionState, float]]:
        """Return up to MAX_RECORDS pairs balanced between correct and wrong.

        Shuffled with a fixed seed before slicing so the records the reflector
        sees are not biased by the minibatch's arrival order, while keeping
        repeated GEPA runs reproducible.
        """
        correct = [(s, sc) for s, sc in zip(states, scores) if sc > 0]
        wrong = [(s, sc) for s, sc in zip(states, scores) if sc == 0]
        rng = random.Random(self._sample_seed)
        rng.shuffle(correct)
        rng.shuffle(wrong)
        half = self._MAX_RECORDS // 2
        return correct[:half] + wrong[:half]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/gepa_integration/test_adapter.py -v -k balanced_sample`
Expected: all three new tests PASS.

- [ ] **Step 7: Run the full file**

Run: `pytest tests/gepa_integration/test_adapter.py -v`
Expected: all tests pass (no existing tests reference `sample_seed`; the default of `0` preserves any tests that constructed `AgentGEPAAdapter` without it).

- [ ] **Step 8: Commit**

```bash
git add src/gepa_integration/adapter.py tests/gepa_integration/test_adapter.py
git commit -m "feat(gepa): seeded shuffle in _balanced_sample

Replaces head-of-list slicing with a deterministic random.Random(seed)
shuffle so the reflector's records are not biased by the minibatch's
arrival order. New constructor param sample_seed (default 0) keeps
re-runs reproducible. See spec addendum 2026-05-18 (Iteration 2)."
```

---

## Task 3: Add `thinking_at_last_turn` to `system_prompt` records

**Files:**
- Modify: `src/gepa_integration/adapter.py` (`_system_prompt_records`, around line 168)
- Test: `tests/gepa_integration/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/gepa_integration/test_adapter.py` (new section):

```python
# ── last-turn thinking ───────────────────────────────────────────────────────

def _multi_turn_state(qid, first_think, last_think, correct=True):
    """ExecutionState with two assistant turns (first tool call + final answer)."""
    state = _make_state(qid, "Q?", "answer", correct=correct)
    state.output_messages = [
        {"role": "assistant", "content": f"<think>{first_think}</think><tool_call>{{}}</tool_call>"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": f"<think>{last_think}</think>The answer is X."},
    ]
    return state


def test_last_turn_thinking_included_when_multi_turn():
    adapter = _make_adapter()
    state = _multi_turn_state(1, "first reasoning here", "last reasoning here")
    batch = GEPAEvaluationBatch(outputs=["answer"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    outputs = result["system_prompt"][0]["Generated Outputs"]
    assert "thinking_at_last_turn" in outputs
    assert "last reasoning" in outputs["thinking_at_last_turn"]
    # First-turn field still populated and distinct
    assert "first reasoning" in outputs["thinking_before_first_tool"]


def test_last_turn_thinking_truncated_at_800():
    adapter = _make_adapter()
    long = "y" * 2000
    state = _multi_turn_state(1, "first", long)
    batch = GEPAEvaluationBatch(outputs=["answer"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    snippet = result["system_prompt"][0]["Generated Outputs"]["thinking_at_last_turn"]
    assert len(snippet) <= 800 + len("…[truncated]")
    assert snippet.endswith("…[truncated]")


def test_last_turn_thinking_omitted_when_single_assistant_turn():
    adapter = _make_adapter()
    state = _make_state(1, "Q?", "answer", correct=True)
    state.output_messages = [
        {"role": "assistant", "content": "<think>only turn</think>final"},
    ]
    batch = GEPAEvaluationBatch(outputs=["answer"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    outputs = result["system_prompt"][0]["Generated Outputs"]
    assert "thinking_at_last_turn" not in outputs


def test_last_turn_thinking_omitted_when_output_messages_empty():
    adapter = _make_adapter()
    state = _make_state(1, "Q?", "answer", correct=True)
    state.output_messages = []  # explicit
    batch = GEPAEvaluationBatch(outputs=["answer"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["system_prompt"])
    outputs = result["system_prompt"][0]["Generated Outputs"]
    assert "thinking_at_last_turn" not in outputs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gepa_integration/test_adapter.py -v -k last_turn_thinking`
Expected: FAIL — `thinking_at_last_turn` key absent from record.

- [ ] **Step 3: Implement last-turn thinking in `_system_prompt_records`**

Edit `src/gepa_integration/adapter.py`. Find the current `_system_prompt_records` body:

```python
    def _system_prompt_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            first_thinking = (
                _extract_thinking(state.output_messages[0]["content"])
                if state.output_messages
                else ""
            )
            if len(first_thinking) > self._THINKING_SNIPPET_LEN:
                first_thinking = first_thinking[: self._THINKING_SNIPPET_LEN] + "…[truncated]"
            action_steps = [
                {
                    "tool": a["tool_name"],
                    "sub_goal": a.get("sub_goal", ""),
                    "result_snippet": str(a.get("result", ""))[: self._RESULT_SNIPPET_LEN],
                }
                for a in state.action_history
            ]
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "predicted_answer": state.answer or "",
                    "thinking_before_first_tool": first_thinking,
                    "action_steps": action_steps,
                },
                "Feedback": self._diagnose(state, score),
            })
        return records
```

Replace with:

```python
    def _system_prompt_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            assistant_msgs = [
                m for m in state.output_messages if m.get("role") == "assistant"
            ]
            first_thinking = (
                _extract_thinking(assistant_msgs[0]["content"]) if assistant_msgs else ""
            )
            if len(first_thinking) > self._THINKING_SNIPPET_LEN:
                first_thinking = first_thinking[: self._THINKING_SNIPPET_LEN] + "…[truncated]"
            action_steps = [
                {
                    "tool": a["tool_name"],
                    "sub_goal": a.get("sub_goal", ""),
                    "result_snippet": str(a.get("result", ""))[: self._RESULT_SNIPPET_LEN],
                }
                for a in state.action_history
            ]
            generated_outputs: dict[str, Any] = {
                "predicted_answer": state.answer or "",
                "thinking_before_first_tool": first_thinking,
                "action_steps": action_steps,
            }
            # Include last-turn thinking only when there's a distinct last
            # assistant turn — otherwise the field would just duplicate
            # thinking_before_first_tool under a misleading name.
            if len(assistant_msgs) >= 2:
                last_thinking = _extract_thinking(assistant_msgs[-1]["content"])
                if len(last_thinking) > self._THINKING_SNIPPET_LEN:
                    last_thinking = (
                        last_thinking[: self._THINKING_SNIPPET_LEN] + "…[truncated]"
                    )
                generated_outputs["thinking_at_last_turn"] = last_thinking
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": generated_outputs,
                "Feedback": self._diagnose(state, score),
            })
        return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gepa_integration/test_adapter.py -v -k last_turn_thinking`
Expected: all four new tests PASS.

- [ ] **Step 5: Run the full file**

Run: `pytest tests/gepa_integration/test_adapter.py -v`
Expected: all tests pass. The change from `state.output_messages[0]` to `assistant_msgs[0]` is benign — existing tests use states whose first `output_messages` entry is an assistant message (or no messages at all).

- [ ] **Step 6: Commit**

```bash
git add src/gepa_integration/adapter.py tests/gepa_integration/test_adapter.py
git commit -m "feat(gepa): add thinking_at_last_turn to system_prompt records

Captures the orchestrator's reasoning at the stopping/answering turn,
where late-stage failure modes (single-shot tool trust, retrieval
evidence failure) live. Capped at 800 chars; omitted when there is
only one distinct assistant turn (would duplicate first-turn field).
See spec addendum 2026-05-18 (Iteration 2)."
```

---

## Task 4: Replace `raw_planning_output` with `plan` + `thinking_in_plan`

**Files:**
- Modify: `src/gepa_integration/adapter.py` (`_planning_suffix_records`, around line 200)
- Modify: `tests/gepa_integration/test_adapter.py` — existing test `test_make_reflective_dataset_planning_uses_raw` references the removed field; rename + rewrite it.

- [ ] **Step 1: Write the new failing tests**

Append to `tests/gepa_integration/test_adapter.py` (new section):

```python
# ── planning thinking extraction ─────────────────────────────────────────────

def test_planning_records_have_plan_and_thinking_fields():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "A", correct=True,
                        raw_plan="<think>plan reasoning</think>stripped plan text")
    state.query_analysis = "stripped plan text"
    batch = GEPAEvaluationBatch(outputs=["A"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    outputs = result["planning_suffix"][0]["Generated Outputs"]
    assert "plan" in outputs
    assert "thinking_in_plan" in outputs
    assert "raw_planning_output" not in outputs
    assert outputs["plan"] == "stripped plan text"
    assert "plan reasoning" in outputs["thinking_in_plan"]
    assert "<think>" not in outputs["plan"]  # plan field is strictly stripped


def test_planning_thinking_truncated_at_800():
    adapter = _make_adapter()
    long = "z" * 2000
    state = _make_state(1, "Q", "A", correct=True,
                        raw_plan=f"<think>{long}</think>plan")
    state.query_analysis = "plan"
    batch = GEPAEvaluationBatch(outputs=["A"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    snippet = result["planning_suffix"][0]["Generated Outputs"]["thinking_in_plan"]
    assert len(snippet) <= 800 + len("…[truncated]")
    assert snippet.endswith("…[truncated]")


def test_planning_thinking_empty_when_no_think_tags():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "A", correct=True,
                        raw_plan="no thinking tags, just a plan")
    state.query_analysis = "no thinking tags, just a plan"
    batch = GEPAEvaluationBatch(outputs=["A"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    outputs = result["planning_suffix"][0]["Generated Outputs"]
    assert outputs["thinking_in_plan"] == ""
    assert outputs["plan"] == "no thinking tags, just a plan"
```

- [ ] **Step 2: Update the existing test that references `raw_planning_output`**

Edit `tests/gepa_integration/test_adapter.py`. Find:

```python
def test_make_reflective_dataset_planning_uses_raw():
    adapter = _make_adapter()
    state = _make_state(1, "Q", "A", correct=True, raw_plan="<think>reasoning</think>plan")
    batch = GEPAEvaluationBatch(outputs=["A"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    raw = result["planning_suffix"][0]["Generated Outputs"]["raw_planning_output"]
    assert "<think>reasoning</think>" in raw
```

Replace with:

```python
def test_make_reflective_dataset_planning_extracts_thinking():
    """Planning records expose the <think> block as its own field, not buried
    inside a raw blob (Iteration 2)."""
    adapter = _make_adapter()
    state = _make_state(1, "Q", "A", correct=True, raw_plan="<think>reasoning</think>plan")
    state.query_analysis = "plan"
    batch = GEPAEvaluationBatch(outputs=["A"], scores=[1.0], trajectories=[state])
    result = adapter.make_reflective_dataset({}, batch, ["planning_suffix"])
    outputs = result["planning_suffix"][0]["Generated Outputs"]
    assert outputs["thinking_in_plan"] == "reasoning"
    assert outputs["plan"] == "plan"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/gepa_integration/test_adapter.py -v -k "planning"`
Expected: FAIL — record still has `raw_planning_output`, missing `plan` and `thinking_in_plan`.

- [ ] **Step 4: Implement the field split in `_planning_suffix_records`**

Edit `src/gepa_integration/adapter.py`. Find:

```python
    def _planning_suffix_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            raw_plan = state.raw_query_analysis or state.query_analysis or ""
            tools_used = [tc["name"] for tc in state.tool_calls]
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "raw_planning_output": raw_plan,
                    "tools_subsequently_used": tools_used,
                    "num_turns_taken": state.turn,
                },
                "Feedback": self._diagnose(state, score),
            })
        return records
```

Replace with:

```python
    def _planning_suffix_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            thinking_in_plan = _extract_thinking(state.raw_query_analysis or "")
            if len(thinking_in_plan) > self._THINKING_SNIPPET_LEN:
                thinking_in_plan = (
                    thinking_in_plan[: self._THINKING_SNIPPET_LEN] + "…[truncated]"
                )
            tools_used = [tc["name"] for tc in state.tool_calls]
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "plan": state.query_analysis or "",
                    "thinking_in_plan": thinking_in_plan,
                    "tools_subsequently_used": tools_used,
                    "num_turns_taken": state.turn,
                },
                "Feedback": self._diagnose(state, score),
            })
        return records
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/gepa_integration/test_adapter.py -v -k "planning"`
Expected: all planning-related tests PASS (the rewritten one + the three new ones).

- [ ] **Step 6: Run the full file**

Run: `pytest tests/gepa_integration/test_adapter.py -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gepa_integration/adapter.py tests/gepa_integration/test_adapter.py
git commit -m "feat(gepa): split planning record into plan + thinking_in_plan

Replaces the single raw_planning_output blob with two extracted fields
mirroring the system_prompt record shape. The reflector now sees the
planning <think> block as its own (capped) field instead of having to
parse XML out of a raw blob. See spec addendum 2026-05-18 (Iteration 2)."
```

---

## Task 5: Update README and CHANGELOG

**Files:**
- Modify: `src/gepa_integration/README.md` ("Feedback design (μ_f for CoSMAS)" section, around lines 178–270 depending on prior edits; and the "Tests" line near the end)
- Modify: `CHANGELOG.md` (append to existing **Changed** subsection of `[Unreleased] — feat/gepa-integration`)

- [ ] **Step 1: Update the README's record schema description**

Edit `src/gepa_integration/README.md`. Find:

```markdown
**`system_prompt` records** include:
- The question
- The orchestrator's predicted answer
- The first `<think>` block before the first tool call
- All action steps (tool name, sub-goal, result snippet)
- The enriched `Feedback` string (see [§ Feedback design](#feedback-design-μf-for-cosmas) below)

**`planning_suffix` records** include:
- The question
- The raw planning output (including `<think>` tags via `raw_query_analysis`)
- The list of tools subsequently used
- Number of turns taken
- The enriched `Feedback` string — same `_diagnose()` output as the system-prompt records
```

Replace with:

```markdown
**`system_prompt` records** include:
- The question
- The orchestrator's predicted answer
- `thinking_before_first_tool` — the `<think>` block from the first assistant turn (capped at 800 chars)
- `thinking_at_last_turn` — the `<think>` block from the last assistant turn (capped at 800 chars); **omitted when there is only one distinct assistant turn** to avoid duplicating the first-turn field
- All action steps (tool name, sub-goal, result snippet)
- The enriched `Feedback` string (see [§ Feedback design](#feedback-design-μf-for-cosmas) below)

**`planning_suffix` records** include:
- The question
- `plan` — the planning turn output with `<think>` blocks stripped (i.e. `state.query_analysis`)
- `thinking_in_plan` — the `<think>` block extracted from the raw planning output (capped at 800 chars)
- The list of tools subsequently used
- Number of turns taken
- The enriched `Feedback` string — same `_diagnose()` output as the system-prompt records
```

- [ ] **Step 2: Append a note about Iteration 2 to the Feedback design section**

Edit `src/gepa_integration/README.md`. Find (immediately above the `#### Why not an LLM judge?` subsection):

```markdown
Both record types use the same `_diagnose` output, by design. The
`system_prompt` and `planning_suffix` records differ in their
`Generated Outputs` payload (one shows the `<think>` block + action steps,
the other the raw planning analysis + downstream tool list) — the reflector
already has the per-component context it needs, so the *feedback* itself can
be component-agnostic.
```

Replace with:

```markdown
Both record types use the same `_diagnose` output, by design. The
`system_prompt` and `planning_suffix` records differ in their
`Generated Outputs` payload (one shows first + last assistant `<think>`
blocks + action steps, the other shows the planning `<think>` block +
the stripped plan + the downstream tool list) — the reflector already
has the per-component context it needs, so the *feedback* itself can be
component-agnostic.

**Iteration 2 (2026-05-18)** unified the thinking-snippet cap at 800 chars
across every field, added `thinking_at_last_turn` to `system_prompt`
records so the stopping decision is visible to the reflector, and split
the planning record's raw blob into `plan` + `thinking_in_plan` for
parity with the system-prompt record shape. `_balanced_sample` now
shuffles each bucket with `random.Random(self._sample_seed)` (default
seed `0`) before slicing, so the reflector's records are not biased by
the minibatch's arrival order while remaining reproducible across runs.
See `docs/superpowers/specs/2026-05-15-gepa-integration-design.md`
Iteration 2 addendum for the full rationale.
```

- [ ] **Step 3: Update the Tests count line**

Edit `src/gepa_integration/README.md`. Find:

```markdown
58 unit tests covering: `ExecutionState.raw_query_analysis`,
`_DEFAULT_PLANNING_SUFFIX_TOOLS` constant, `build_seed_candidate` (structure,
planning suffix match, tool schema embedding), `build_splits` (sizes,
no-overlap, failure ratio, JSON output), `_extract_thinking`, all
`AgentGEPAAdapter` methods (`evaluate`, `make_reflective_dataset`, balanced
sampling cap), and the `_diagnose` feedback function (score breakdown,
normalised-form line, empty-prediction, format-mismatch, verbosity, high-f1,
parametric-memory, tool-error counting, max-turns, multiple-choice skip).
```

Replace with:

```markdown
~68 unit tests covering: `ExecutionState.raw_query_analysis`,
`_DEFAULT_PLANNING_SUFFIX_TOOLS` constant, `build_seed_candidate` (structure,
planning suffix match, tool schema embedding), `build_splits` (sizes,
no-overlap, failure ratio, JSON output), `_extract_thinking`, all
`AgentGEPAAdapter` methods (`evaluate`, `make_reflective_dataset`, balanced
sampling cap), the `_diagnose` feedback function (score breakdown,
normalised-form line, empty-prediction, format-mismatch, verbosity, high-f1,
parametric-memory, tool-error counting, max-turns, multiple-choice skip),
and Iteration 2 additions (unified 800-char thinking cap, last-turn
thinking inclusion/skip, planning record `plan`/`thinking_in_plan` split,
seeded `_balanced_sample` shuffle determinism).
```

(Run `grep -c "^def test_" tests/gepa_integration/*.py` after Task 4 to confirm the exact count and replace `~68` with the real number.)

- [ ] **Step 4: Append the Iteration 2 entry to CHANGELOG**

Edit `CHANGELOG.md`. Find the first line under `### Changed`:

```markdown
### Changed
- **GEPA reflective feedback enriched** (`src/gepa_integration/adapter.py`) — the `Feedback` string passed to the Qwen3-32B reflector now exposes the deterministic environment-derived signals GEPA's μ_f calls for, instead of the previous one-line `WRONG — ground truth: X. Predicted: Y.` placeholder
```

Insert a new bullet directly above it (so the Iteration 2 entry sits on top of the Iteration 1 entry, both under the same **Changed** subsection):

```markdown
- **GEPA reflective records iteration 2** (`src/gepa_integration/adapter.py`) — record-shape and sample-selection refinements layered on top of the Iteration 1 enriched feedback (see spec addendum 2026-05-18)
  - Unified `_THINKING_SNIPPET_LEN = 800` (was 1500); the same cap now applies to every thinking field across both record types so per-call budget is predictable
  - `system_prompt` records gain `thinking_at_last_turn` — the `<think>` block from the last assistant turn, capped at 800 chars; omitted when there is only one distinct assistant turn so the field never duplicates `thinking_before_first_tool`. The first-turn extraction also moves off `output_messages[0]` to "first assistant message" to handle interleaved tool messages correctly
  - `planning_suffix` records replace the unbounded `raw_planning_output` blob with two parallel fields: `plan` (= `state.query_analysis`, stripped of `<think>`) and `thinking_in_plan` (= `_extract_thinking(state.raw_query_analysis)`, capped at 800 chars). Mirrors the system_prompt record shape so the reflector sees the same structure on both components
  - `AgentGEPAAdapter.__init__` gains `sample_seed: int = 0`; `_balanced_sample` now shuffles each bucket with `random.Random(self._sample_seed)` before slicing, replacing head-of-list selection. Re-runs remain reproducible (default seed `0`), but the records the reflector sees are no longer biased by minibatch arrival order
  - Token-budget impact: roughly *neutral* — the first-turn snippet shrink (1500→800 across 8 records ≈ −1.4 K tokens) offsets the new `thinking_at_last_turn` field (0–1.6 K tokens at 8 records, often less when single-turn rollouts are present). The unified cap also removes the previous Iteration 1 worst-case where `raw_planning_output` could run arbitrarily long
```

- [ ] **Step 5: Run all GEPA tests one more time end-to-end**

Run: `pytest tests/gepa_integration/ -v`
Expected: every test passes; observed count matches the README's Tests line.

- [ ] **Step 6: Commit**

```bash
git add src/gepa_integration/README.md CHANGELOG.md
git commit -m "docs(gepa): document Iteration 2 record + sample changes

Updates the gepa_integration README to describe the new
thinking_at_last_turn / plan / thinking_in_plan record fields, the
unified 800-char thinking cap, and the seeded shuffle in
_balanced_sample. Adds an Iteration 2 entry to CHANGELOG."
```

---

## Self-review (run after writing the plan)

**1. Spec coverage**

Spec sections (Iteration 2 addendum) → tasks:

| Spec section | Task |
|---|---|
| Unified thinking cap (800) | Task 1 |
| Change 1 — last-turn thinking | Task 3 |
| Change 2 — extract planning thinking | Task 4 |
| Change 3 — seeded shuffle | Task 2 |
| Tests list | Tasks 1–4 (each test in the spec maps to a step in the corresponding task) |
| Out of scope (LLM judge, cross-record correlation, adaptive length) | Intentionally absent from tasks |

No gaps.

**2. Placeholder scan**

No "TBD", "TODO", "implement later", or "similar to Task N" phrases. The single `~68` test-count placeholder in Task 5 Step 3 has an explicit instruction to verify the exact count via `grep` and replace.

**3. Type / signature consistency**

- `sample_seed: int = 0` introduced in Task 2 Step 4, used in Task 2 Step 5 — consistent.
- `self._sample_seed` attribute set in Task 2 Step 4, read in Task 2 Step 5 — consistent.
- `_extract_thinking` (existing helper) — used in Tasks 3, 4 with the same single-argument signature.
- `assistant_msgs` filter (Task 3 Step 3) defined once and reused for both `assistant_msgs[0]` and `assistant_msgs[-1]` — no naming drift.
- `_THINKING_SNIPPET_LEN` referenced consistently in Tasks 1, 3, 4 — same name throughout.

**4. Ordering**

Task 1 (cap) must precede Tasks 3 and 4 because both new tests rely on the 800-char cap to assert truncation length. Plan order respects this.

Task 2 (sample_seed param) is independent of the rest. Placed second because adding a constructor param is a low-risk standalone change; later tasks don't depend on it but won't conflict either.

Tasks 3 and 4 are independent of each other but both build on Task 1's smaller cap. Either order works.

Task 5 (docs) must be last since it summarises everything that landed.
