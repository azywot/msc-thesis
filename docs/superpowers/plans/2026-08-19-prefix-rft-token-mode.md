# Prefix-RFT Token Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the paper's token-fraction prefix as a second Prefix-RFT mode alongside the existing decision-count prefix, so a rollout can be handed the first `r` tokens of a teacher's turn and made to finish that turn, selectable with `--prefix-mode {steps,tokens}`.

**Architecture:** The driver keeps owning the schedule and dispatches either `prefix_k` (steps) or `prefix_l` (tokens) in the task payload; the worker infers the mode from which key arrives, so the two processes cannot disagree. The token budget is computed worker-side from the demonstration's own token lengths. A split turn is served through vLLM's `continue_final_message`, and `ReplayProvider` rewrites the captured turn afterwards so the batch sees one whole turn whose first `r` tokens are the teacher's. `prefix_mask` becomes partially filled, which the advantage and entropy code already accept.

**Tech Stack:** Python 3.11, verl 0.7.1, vLLM 0.10.1.1, Ray, Hydra/OmegaConf, pandas, pytest. Unit tests run in the `agent_engine` conda env; the training stack lives in `cosmas-train`.

**Spec:** `docs/superpowers/specs/2026-08-19-prefix-rft-token-mode-design.md`

---

## Global Constraints

- **Never edit anything under `src/fine_tuning/agentflow/`.** It is vendored; `VENDORED.md` forbids patching it. Extend by subclassing from our own modules.
- **Never edit the installed verl package.**
- **Step mode must not change behaviour.** Every existing test in `tests/unit/test_prefix_rft_*.py` must still pass without being edited. If a change to step mode's output looks necessary, stop and raise it rather than updating the test.
- Modules under `src/verl_ext/prefix_rft/` that hold logic must import **no verl**, and `src/fine_tuning/prefix_replay.py` must import **only `agent_engine`**. verl is absent from the `agent_engine` env and pytest is absent from `cosmas-train`, so anything importing verl cannot be tested in either.
- Python 3.11+, `black` line-length 100, `isort` black profile. Type annotations are not required.
- Run pytest from the repo root with `/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest`.
- Paper values, unchanged by this work: prefix high `0.95`; low init `0.95`, target `0.05`; cosine decay; `l ~ U(low_t, high)`; entropy keep ratio `0.2`; one prefixed rollout of eight.
- Token IDs for teacher tokens come from `tokenizer.encode(text, add_special_tokens=False)`, and prompt IDs from `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)`, matching the proxy (`src/fine_tuning/agentflow/verl/daemon.py:216-225`). No `enable_thinking` kwarg, no appended EOS.
- The token budget clamp is `B <= T - 1`, the paper's `prefix_len <= demo_len - 1` guard applied to the concatenated demonstration.
- `papers/` and `repos/` are not in this checkout. Do not invent citations to them; the spec's source note explains why the existing ones are second-hand.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/verl_ext/prefix_rft/budget.py` | **new.** `split_for_budget()`: pure token-budget arithmetic, no tokenizer, no verl |
| `src/verl_ext/prefix_rft/masks.py` | `build_prefix_mask()` gains partial rows via `prefix_len` |
| `src/verl_ext/prefix_rft/dispatch.py` | gains `prefix_l_for()` and `prefix_spec_for()`; `prefix_k_for()` untouched |
| `src/verl_ext/prefix_rft/daemon.py` | dispatches the mode's payload key; mode-aware metrics |
| `src/verl_ext/prefix_rft/trainer.py` | passes `prefix_rft.mode` to the daemon; adds `actor/prefix_split_fraction` |
| `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml` | `prefix_rft.mode: steps` |
| `src/fine_tuning/prefix_replay.py` | `ReplayController.from_token_fraction()`, `next_partial()`, `_safe_prefix()`; `ReplayProvider` split path |
| `src/fine_tuning/prefix_rollout.py` | picks the controller from the payload key |
| `src/fine_tuning/rollout.py` | `Triplet` metadata carries `prefix_len` |
| `src/agent_engine/models/api_provider.py` | optional `continue_final_message` in the JSON payload |
| `scripts/launch_verl.py` | `--prefix-mode {steps,tokens}` |
| `scripts/check_prefix_replay_tokenisation.py` | `--mode tokens` preflight case |
| `experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml` | **new.** production config, one line different |
| `tests/unit/test_prefix_rft_budget.py` | **new.** budget arithmetic |
| `tests/unit/test_prefix_rft_daemon.py` | partial mask rows, token dispatch |
| `tests/unit/test_prefix_rft_rollout.py` | token-mode controller and split serving |

---

### Task 1: Token budget arithmetic

The only genuinely new maths. Kept in its own module with no tokenizer and no verl so it is trivially testable and can be read against the paper's guard on its own.

**Files:**
- Create: `src/verl_ext/prefix_rft/budget.py`
- Test: `tests/unit/test_prefix_rft_budget.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `split_for_budget(lengths: list[int], l: float) -> tuple[int, int]` returning `(n_full, r)`: decisions replayed whole, then `r` tokens of decision `n_full + 1`. `r == 0` means no split.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_prefix_rft_budget.py`:

```python
"""Tests for the Prefix-RFT token budget.

Pure arithmetic over decision token lengths: no tokenizer, no verl. The clamp
under test is the paper's ``prefix_len <= demo_len - 1`` guard applied to the
concatenated demonstration, so there is always a token left to generate.
"""

import pytest

from verl_ext.prefix_rft.budget import split_for_budget


def test_budget_splits_the_decision_that_straddles_it():
    # total 30, l = 0.8 -> budget 24: two whole decisions (20) then 4 tokens.
    assert split_for_budget([10, 10, 10], 0.8) == (2, 4)


def test_exact_boundary_gives_no_split():
    # total 20, l = 0.5 -> budget 10, which is exactly one whole decision.
    assert split_for_budget([10, 10], 0.5) == (1, 0)


def test_zero_fraction_replays_nothing():
    assert split_for_budget([10, 10, 10], 0.0) == (0, 0)


def test_the_top_of_the_range_still_leaves_a_token_to_generate():
    n_full, r = split_for_budget([10, 10, 10], 0.95)
    assert (n_full, r) == (2, 8)
    assert n_full * 10 + r == 28  # 29 tokens is the cap; 28 is floor(0.95 * 30)


def test_a_fraction_of_one_never_exceeds_the_cap():
    n_full, r = split_for_budget([10, 10, 10], 1.0)
    assert n_full * 10 + r == 29


def test_a_single_decision_demonstration_is_prefixable():
    """Step mode forces k <= m - 1 = 0 here. The token guard does not, which is
    what takes prefixable coverage from 1085 questions to all 1358."""
    assert split_for_budget([5], 0.9) == (0, 4)


def test_a_single_token_demonstration_is_not_prefixable():
    assert split_for_budget([1], 0.9) == (0, 0)


def test_an_empty_demonstration_is_not_prefixable():
    assert split_for_budget([], 0.9) == (0, 0)


@pytest.mark.parametrize("l", [0.0, 0.05, 0.3, 0.5, 0.75, 0.95])
def test_at_least_one_token_is_always_generated(l):
    lengths = [3, 11, 2, 7]
    n_full, r = split_for_budget(lengths, l)
    assert sum(lengths[:n_full]) + r <= sum(lengths) - 1
    assert n_full < len(lengths) or r == 0


@pytest.mark.parametrize("l", [0.1, 0.4, 0.6, 0.9])
def test_the_split_decision_is_never_replayed_whole(l):
    lengths = [4, 4, 4, 4]
    n_full, r = split_for_budget(lengths, l)
    if r > 0:
        assert r < lengths[n_full]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_budget.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'verl_ext.prefix_rft.budget'`.

- [ ] **Step 3: Write the implementation**

Create `src/verl_ext/prefix_rft/budget.py`:

```python
"""Turn a prefix fraction into a token split over a multi-decision demonstration.

The paper's prefix is a token fraction of one response (A.2). A demonstration here
is ``m`` decisions, so the fraction is taken over their concatenation: whole
decisions are replayed while they fit the budget, and the one that straddles it is
split.

``budget <= total - 1`` is the paper's ``prefix_len >= demo_len -> demo_len - 1``
guard (recorded in the 2026-08-17 spec against ``recipe/prefix_rft/rl_dataset.py:300-301``)
applied to the concatenation. It guarantees at least one generated token, and with it
``n_full < len(lengths)`` whenever ``r > 0``, so the split decision always exists.

No tokenizer and no verl: this is arithmetic, and keeping it separate is what lets it
be tested in the agent_engine env.
"""

from __future__ import annotations

import math


def split_for_budget(lengths, l):
    """Return ``(n_full, r)`` for prefix fraction ``l`` over decision token ``lengths``.

    ``n_full`` decisions are replayed whole, then ``r`` tokens of decision
    ``n_full + 1``. ``r == 0`` means the budget landed on a decision boundary and
    nothing is split, which is exactly step mode at ``k = n_full``.
    """
    total = sum(lengths)
    if total <= 1:
        # Nothing can be replayed without consuming the only token there is.
        return 0, 0

    budget = int(math.floor(l * total))
    budget = max(0, min(budget, total - 1))

    n_full = 0
    used = 0
    for n in lengths:
        if used + n > budget:
            break
        used += n
        n_full += 1
    return n_full, budget - used
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_budget.py -q
```

Expected: 18 passed (the two parametrized cases expand to six).

- [ ] **Step 5: Commit**

```bash
git add src/verl_ext/prefix_rft/budget.py tests/unit/test_prefix_rft_budget.py
git commit -m "feat(prefix-rft): token budget arithmetic for a mid-turn prefix"
```

---

### Task 2: Partial prefix rows in the mask

Changes the contract three components read, so all three move in one commit. A turn stops being wholly teacher or wholly policy and carries a token count instead.

**Files:**
- Modify: `src/verl_ext/prefix_rft/masks.py:11-29`
- Modify: `src/fine_tuning/rollout.py:343-351`
- Modify: `src/verl_ext/prefix_rft/daemon.py:207-215`
- Test: `tests/unit/test_prefix_rft_daemon.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: trace dicts and `Triplet` metadata carry `prefix_len: int`. `build_prefix_mask(trace_list, max_response_length)` keeps its signature; a trace with `is_prefix=True` and no `prefix_len` still fills the whole row, so step mode is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_prefix_rft_daemon.py`, immediately after `test_a_turn_with_a_prompt_but_no_response_is_kept`:

```python
def test_mask_marks_only_the_replayed_head_of_a_split_turn():
    """Token mode hands the model the first r tokens of a turn and it writes the
    rest, so the row is 1 up to r and 0 after."""
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3, 4], "is_prefix": True, "prefix_len": 2}]
    assert build_prefix_mask(traces, max_response_length=6) == [[1, 1, 0, 0, 0, 0]]


def test_prefix_len_is_clamped_to_the_truncated_response():
    traces = [
        {"prompt_ids": [9], "response_ids": [1, 2, 3, 4, 5], "is_prefix": True, "prefix_len": 5}
    ]
    assert build_prefix_mask(traces, max_response_length=3) == [[1, 1, 1]]


def test_a_fully_replayed_turn_needs_no_prefix_len():
    """Step mode writes is_prefix only. It must keep meaning 'the whole turn'."""
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3], "is_prefix": True}]
    assert build_prefix_mask(traces, max_response_length=4) == [[1, 1, 1, 0]]


def test_prefix_len_without_is_prefix_still_marks_tokens():
    traces = [{"prompt_ids": [9], "response_ids": [1, 2, 3], "prefix_len": 1}]
    assert build_prefix_mask(traces, max_response_length=3) == [[1, 0, 0]]


def test_a_zero_prefix_len_on_a_prefixed_turn_is_read_as_the_whole_turn():
    """0 is the absent-value default, so it cannot also mean 'no tokens'. A turn
    that genuinely replayed nothing carries is_prefix=False."""
    traces = [{"prompt_ids": [9], "response_ids": [1, 2], "is_prefix": True, "prefix_len": 0}]
    assert build_prefix_mask(traces, max_response_length=2) == [[1, 1]]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_daemon.py -q
```

Expected: `test_mask_marks_only_the_replayed_head_of_a_split_turn`, `test_prefix_len_is_clamped_to_the_truncated_response` and `test_prefix_len_without_is_prefix_still_marks_tokens` FAIL. The other two already pass, which is the point: they pin the old behaviour.

- [ ] **Step 3: Rewrite the mask builder**

Replace the body of `build_prefix_mask` in `src/verl_ext/prefix_rft/masks.py` (keep the module docstring):

```python
def build_prefix_mask(trace_list, max_response_length):
    """One right-padded 0/1 row per kept turn, 1 on replayed teacher tokens.

    ``prefix_len`` is how many leading response tokens were the teacher's. Token mode
    splits a turn, so a row can be 1 on its head and 0 after. Step mode replays whole
    turns and writes ``is_prefix`` only; a prefixed turn with no ``prefix_len`` is read
    as wholly replayed, which reproduces the mask exactly as it was.

    ``prefix_len = 0`` is indistinguishable from the key being absent, so it cannot
    also mean "no tokens". A turn that replayed nothing carries ``is_prefix = False``.

    Mirrors the truncation and skip rules the vendored daemon applies to responses
    (``fine_tuning/agentflow/verl/daemon.py:740-772``) so the mask stays aligned with
    ``responses`` row for row. In particular a turn is dropped only when prompt *and*
    response are both empty, which is the base daemon's rule; dropping on a different
    condition would shift every later row's mask onto the wrong response.
    """
    rows = []
    for trace in trace_list:
        response_ids = trace.get("response_ids", [])
        prompt_ids = trace.get("prompt_ids", [])
        if len(prompt_ids) == 0 and len(response_ids) == 0:
            continue
        length = min(len(response_ids), max_response_length)
        n_prefix = int(trace.get("prefix_len", 0) or 0)
        if n_prefix == 0 and trace.get("is_prefix", False):
            n_prefix = length
        n_prefix = max(0, min(n_prefix, length))
        rows.append([1] * n_prefix + [0] * (max_response_length - n_prefix))
    return rows
```

- [ ] **Step 4: Carry `prefix_len` through the Triplet**

In `src/fine_tuning/rollout.py`, replace the `metadata=` line and its comment inside the `Triplet(...)` construction:

```python
                        # Prefix-RFT marks replayed teacher turns here; absent for
                        # ordinary GRPO rollouts, where it is always False.
                        # prefix_len is how many leading response tokens were the
                        # teacher's: the whole turn in step mode, a head of it when
                        # token mode split the turn.
                        metadata={
                            "prefix": bool(t.get("is_prefix", False)),
                            "prefix_len": int(t.get("prefix_len", 0) or 0),
                        },
```

- [ ] **Step 5: Read it back in the daemon**

In `src/verl_ext/prefix_rft/daemon.py`, inside `_rebuild_prefix_rows`, replace the `traces = [...]` comprehension:

```python
            traces = [
                {
                    "prompt_ids": t.prompt.get("token_ids", []),
                    "response_ids": t.response.get("token_ids", []),
                    "is_prefix": bool((t.metadata or {}).get("prefix", False)),
                    "prefix_len": int((t.metadata or {}).get("prefix_len", 0) or 0),
                }
                for t in rollout.triplets
            ]
```

`rollout_is_prefixed = any(t["is_prefix"] for t in traces)` on the following line is left alone: a split turn sets `is_prefix` true, so rollout-level grouping is unaffected.

- [ ] **Step 6: Run the full prefix suite**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/ -k prefix_rft -q
```

Expected: all pass, including the five new mask cases and every pre-existing one unedited.

- [ ] **Step 7: Commit**

```bash
git add src/verl_ext/prefix_rft/masks.py src/verl_ext/prefix_rft/daemon.py \
        src/fine_tuning/rollout.py tests/unit/test_prefix_rft_daemon.py
git commit -m "feat(prefix-rft): carry prefix_len so a turn can be partly teacher"
```

---

### Task 3: A controller that splits a turn

**Files:**
- Modify: `src/fine_tuning/prefix_replay.py:32-101`
- Test: `tests/unit/test_prefix_rft_rollout.py`

**Interfaces:**
- Consumes: `split_for_budget(lengths, l) -> (n_full, r)` from Task 1.
- Produces:
  - `ReplayController.from_token_fraction(steps, l, tokenizer) -> ReplayController`
  - `ReplayController.split_index: int | None`, `.split_tokens: int`
  - `ReplayController.next_partial(prompt_payload) -> dict | None` with keys `messages`, `prefix_text`, `prefix_ids`, `prompt_token_ids`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_prefix_rft_rollout.py`. `_FakeTokenizer` encodes one id per character, so token counts equal string lengths and the arithmetic is readable. It needs a `decode`, added here as a subclass so the existing fake is untouched:

```python
class _RoundTripTokenizer(_FakeTokenizer):
    """_FakeTokenizer plus a decode that round-trips, since ord/chr is a bijection."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _LossyTokenizer(_RoundTripTokenizer):
    """A prefix ending on "n" does not survive the round trip.

    Stands in for a real tokenizer splitting a mergeable pair: the decoded text
    re-encodes to more ids than it came from, which would shift every position in the
    mask. Only ``decode`` lies, so the decision token lengths ``from_token_fraction``
    measures are unaffected and the budget arithmetic stays readable.
    """

    def decode(self, ids):
        text = super().decode(ids)
        if text.endswith("n"):
            return text + "?"
        return text


def test_token_fraction_replays_whole_decisions_then_splits_one():
    # responses are "plan" (4), "call" (4), "final" (5); total 13.
    # l = 0.8 -> budget 10: two whole decisions (8) then 2 tokens of "final".
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    assert ctrl.k == 2
    assert ctrl.split_index == 2
    assert ctrl.split_tokens == 2


def test_the_split_turn_is_served_after_the_whole_ones():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    assert ctrl.next_response(payload)["text"] == "plan"
    assert ctrl.next_response(payload)["text"] == "call"
    assert ctrl.next_response(payload) is None
    partial = ctrl.next_partial(payload)
    assert partial["prefix_text"] == "fi"
    assert partial["prefix_ids"] == [ord("f"), ord("i")]
    assert partial["prompt_token_ids"] == [2]


def test_the_split_is_served_only_once():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    ctrl.next_response(payload)
    ctrl.next_response(payload)
    assert ctrl.next_partial(payload) is not None
    assert ctrl.next_partial(payload) is None


def test_no_partial_is_offered_before_the_whole_decisions_are_done():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    assert ctrl.next_partial(_payload([{"role": "user", "content": "ab"}])) is None


def test_a_boundary_landing_on_a_decision_edge_offers_no_partial():
    # budget 8 is exactly "plan" + "call".
    ctrl = ReplayController.from_token_fraction(
        _steps(), l=8 / 13, tokenizer=_RoundTripTokenizer()
    )
    payload = _payload([{"role": "user", "content": "ab"}])
    assert ctrl.split_index is None
    ctrl.next_response(payload)
    ctrl.next_response(payload)
    assert ctrl.next_partial(payload) is None


def test_a_split_decision_does_not_arm_the_teacher_tool_result():
    """The model writes its own tool call after the prefill, so the teacher's stored
    result must not be served for it."""
    steps = [
        {"response": "plan", "tool_name": None, "tool_result": None},
        {"response": "call", "tool_name": "web_search", "tool_result": "hits"},
    ]
    # total 8, l = 0.75 -> budget 6: "plan" whole, then 2 tokens of "call".
    ctrl = ReplayController.from_token_fraction(steps, l=0.75, tokenizer=_RoundTripTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    ctrl.next_response(payload)
    assert ctrl.next_tool_result() is None
    ctrl.next_partial(payload)
    assert ctrl.next_tool_result() is None


def test_a_fully_replayed_decision_still_arms_its_tool_result():
    steps = [
        {"response": "call", "tool_name": "web_search", "tool_result": "hits"},
        {"response": "final", "tool_name": None, "tool_result": None},
    ]
    # total 9, l = 0.7 -> budget 6: "call" whole, then 2 tokens of "final".
    ctrl = ReplayController.from_token_fraction(steps, l=0.7, tokenizer=_RoundTripTokenizer())
    ctrl.next_response(_payload([{"role": "user", "content": "ab"}]))
    assert ctrl.next_tool_result() == "hits"


def test_the_boundary_backs_off_until_the_text_round_trips():
    """A prefix whose text does not re-encode to itself is shortened, not sent.

    l = 0.85 over 13 tokens gives a budget of 11: "plan" and "call" whole, then 3
    tokens of "final". "fin" does not round-trip, so the boundary drops to "fi".
    """
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.85, tokenizer=_LossyTokenizer())
    payload = _payload([{"role": "user", "content": "ab"}])
    ctrl.next_response(payload)
    ctrl.next_response(payload)
    assert ctrl.split_tokens == 3
    partial = ctrl.next_partial(payload)
    assert partial["prefix_ids"] == [ord("f"), ord("i")]
    assert partial["prefix_text"] == "fi"


def test_a_single_decision_demonstration_can_be_split():
    steps = [{"response": "final", "tool_name": None, "tool_result": None}]
    ctrl = ReplayController.from_token_fraction(steps, l=0.5, tokenizer=_RoundTripTokenizer())
    assert ctrl.k == 0
    assert ctrl.split_index == 0
    partial = ctrl.next_partial(_payload([{"role": "user", "content": "ab"}]))
    assert partial["prefix_text"] == "fi"


def test_step_mode_controllers_have_no_split():
    ctrl = ReplayController(_steps(), k=2, tokenizer=_FakeTokenizer())
    assert ctrl.split_index is None
    assert ctrl.split_tokens == 0
    assert ctrl.next_partial(_payload([{"role": "user", "content": "ab"}])) is None
```

Add the import of `ReplayProvider` is not needed yet; keep the existing import line as it is.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_rollout.py -q
```

Expected: FAIL with `AttributeError: type object 'ReplayController' has no attribute 'from_token_fraction'`.

- [ ] **Step 3: Add the split state to the controller**

In `src/fine_tuning/prefix_replay.py`, add the import at the top of the module, after the `agent_engine` imports:

```python
from verl_ext.prefix_rft.budget import split_for_budget
```

`verl_ext.prefix_rft.budget` imports only `math`, so this keeps the module free of verl and of `fine_tuning.rollout`.

Extend `ReplayController.__init__` (keep the existing lines, add the last three):

```python
    def __init__(self, steps: list[dict], k: int, tokenizer):
        self.steps = list(steps)
        self.k = max(0, min(int(k), len(self.steps)))
        self.tokenizer = tokenizer
        self._served = 0
        # Armed by a replayed tool-call decision, consumed by the next tool lookup.
        self._pending_tool_result: Optional[str] = None
        # Token mode only: the decision that straddles the budget, and how much of
        # it the teacher supplies. None in step mode, where turns are never split.
        self.split_index: Optional[int] = None
        self.split_tokens = 0
        self._split_served = False
```

- [ ] **Step 4: Add the constructor and the partial turn**

Add to `ReplayController`, after `next_tool_result`:

```python
    @classmethod
    def from_token_fraction(cls, steps: list[dict], l: float, tokenizer) -> "ReplayController":
        """Build a controller whose prefix is a token fraction of the whole demonstration.

        The paper measures the prefix in tokens (A.2). Whole decisions are replayed
        while they fit the budget, and the decision that straddles it is split, so the
        model finishes a turn the teacher started.
        """
        lengths = [
            len(tokenizer.encode(str(s["response"]), add_special_tokens=False)) for s in steps
        ]
        n_full, r = split_for_budget(lengths, l)
        ctrl = cls(steps, k=n_full, tokenizer=tokenizer)
        if r > 0:
            # split_for_budget guarantees n_full < len(steps) whenever r > 0.
            ctrl.split_index = n_full
            ctrl.split_tokens = r
        return ctrl

    def next_partial(self, prompt_payload: str) -> Optional[dict]:
        """Return the prefill for the split decision, or None.

        Offered once, and only after every whole decision has been served. Unlike
        ``next_response`` this deliberately does not arm ``_pending_tool_result``:
        the model writes its own tool call after the prefill, so the teacher's stored
        result does not apply to it.
        """
        if self.split_index is None or self._split_served:
            return None
        if self._served != self.split_index:
            return None
        self._split_served = True

        step = self.steps[self.split_index]
        ids = self.tokenizer.encode(str(step["response"]), add_special_tokens=False)
        prefix_ids, prefix_text = self._safe_prefix(ids[: self.split_tokens])
        if not prefix_ids:
            return None

        messages = self._decode_messages(prompt_payload)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        return {
            "messages": messages,
            "prefix_text": prefix_text,
            "prefix_ids": list(prefix_ids),
            "prompt_token_ids": list(prompt_ids),
        }

    def _safe_prefix(self, ids) -> tuple[list, str]:
        """Longest head of ``ids`` whose decoded text re-encodes to itself.

        The prefill travels to vLLM as text, so a boundary inside a mergeable token
        pair comes back as different ids and shifts every prefix position in the mask.
        Nothing downstream would notice: the run would train and report success. So
        the boundary is backed off a token at a time until the round trip holds.
        """
        candidate = list(ids)
        while candidate:
            text = self.tokenizer.decode(candidate)
            if list(self.tokenizer.encode(text, add_special_tokens=False)) == candidate:
                return candidate, text
            candidate = candidate[:-1]
        return [], ""
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_rollout.py -q
```

Expected: all pass, including the ten new cases and the five pre-existing ones unedited.

- [ ] **Step 6: Commit**

```bash
git add src/fine_tuning/prefix_replay.py tests/unit/test_prefix_rft_rollout.py
git commit -m "feat(prefix-rft): controller that splits a teacher turn on a token budget"
```

---

### Task 4: Serve the split turn through vLLM

A whole replayed turn never reaches vLLM. A split one must, because the model has to finish it. This is the one place the design touches shared `agent_engine` code.

**Files:**
- Modify: `src/agent_engine/models/api_provider.py:78-117`
- Modify: `src/fine_tuning/prefix_replay.py` (`ReplayProvider`)
- Test: `tests/unit/test_prefix_rft_rollout.py`

**Interfaces:**
- Consumes: `next_partial()` and `_safe_prefix()` from Task 3; `prefix_len` in `captured_turns` from Task 2.
- Produces: `ReplayProvider.generate(prompts)` handles three cases per prompt (whole replay, split prefill, plain delegation). A split turn lands in `capturing.captured_turns` with `is_prefix=True` and `prefix_len=len(prefix_ids)`. The JSON payload protocol gains an optional `continue_final_message` key.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_prefix_rft_rollout.py`, and extend the import at the top of the file to `from fine_tuning.prefix_replay import (ReplayController, ReplayProvider, ReplayToolRegistry)`:

```python
class _FakeCapturing:
    """Stands in for _CapturingProvider: records one turn per generated result."""

    def __init__(self, reply="nal"):
        self.captured_turns = []
        self.calls = []
        self._reply = reply

    def generate(self, prompts):
        from agent_engine.models.base import GenerationResult

        self.calls.append(prompts[0])
        self.captured_turns.append(
            {"prompt_ids": [0], "response_ids": [1], "response_text": self._reply}
        )
        return [
            GenerationResult(
                text=self._reply,
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                metadata={},
                prompt_token_ids=[0],
                response_token_ids=[1],
            )
        ]


def test_provider_stitches_a_split_turn_back_into_one_turn():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing(reply="nal")
    provider = ReplayProvider(capturing, ctrl)
    payload = _payload([{"role": "user", "content": "ab"}])

    provider.generate([payload])  # "plan"
    provider.generate([payload])  # "call"
    out = provider.generate([payload])  # split: "fi" + "nal"

    turn = capturing.captured_turns[-1]
    assert turn["response_text"] == "final"
    assert turn["response_ids"] == [ord(c) for c in "final"]
    assert turn["prefix_len"] == 2
    assert turn["is_prefix"] is True
    assert turn["prompt_ids"] == [2]
    assert out[0].text == "final"


def test_the_split_request_asks_vllm_to_continue_the_assistant_message():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing()
    provider = ReplayProvider(capturing, ctrl)
    payload = _payload([{"role": "user", "content": "ab"}])

    provider.generate([payload])
    provider.generate([payload])
    provider.generate([payload])

    sent = json.loads(capturing.calls[-1])
    assert sent["continue_final_message"] is True
    assert sent["messages"][-1] == {"role": "assistant", "content": "fi"}


def test_whole_replays_record_their_full_length_as_the_prefix():
    ctrl = ReplayController(_steps(), k=1, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing()
    provider = ReplayProvider(capturing, ctrl)
    provider.generate([_payload([{"role": "user", "content": "ab"}])])
    turn = capturing.captured_turns[-1]
    assert turn["is_prefix"] is True
    assert turn["prefix_len"] == len("plan")
    assert capturing.calls == []  # never reached vLLM


def test_after_the_split_the_provider_delegates_normally():
    ctrl = ReplayController.from_token_fraction(_steps(), l=0.8, tokenizer=_RoundTripTokenizer())
    capturing = _FakeCapturing()
    provider = ReplayProvider(capturing, ctrl)
    payload = _payload([{"role": "user", "content": "ab"}])
    for _ in range(4):
        provider.generate([payload])
    last = capturing.captured_turns[-1]
    assert "prefix_len" not in last
    assert json.loads(capturing.calls[-1]).get("continue_final_message") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_rollout.py -q -k "split or whole_replays"
```

Expected: FAIL, `KeyError: 'prefix_len'` on the stitched turn.

- [ ] **Step 3: Let the provider request a continuation**

In `src/agent_engine/models/api_provider.py`, inside `_generate_single`, add `continue_final` beside `use_thinking`. Replace:

```python
        raw_messages = None
        use_thinking = False
        try:
            payload = json.loads(prompt)
            if isinstance(payload, dict) and "messages" in payload:
                raw_messages = payload["messages"]
                use_thinking = bool(payload.get("use_thinking", False))
```

with:

```python
        raw_messages = None
        use_thinking = False
        continue_final = False
        try:
            payload = json.loads(prompt)
            if isinstance(payload, dict) and "messages" in payload:
                raw_messages = payload["messages"]
                use_thinking = bool(payload.get("use_thinking", False))
                continue_final = bool(payload.get("continue_final_message", False))
```

and replace the `extra_body` block:

```python
        extra_body = {}
        if self.config.family in _ENABLE_THINKING_KWARG_FAMILIES:
            extra_body["chat_template_kwargs"] = {"enable_thinking": use_thinking}
        if continue_final:
            # Prefill: the final assistant message is left open-ended and the model
            # continues it rather than starting a new turn. Prefix-RFT token mode uses
            # this to have the model finish a half-replayed teacher turn. vLLM rejects
            # this together with add_generation_prompt
            # (vllm/entrypoints/openai/protocol.py:918-928), so that is turned off here.
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
```

- [ ] **Step 4: Stitch the split turn back together**

In `src/fine_tuning/prefix_replay.py`, add `import json` if it is not already imported (it is, at the top). Inside `ReplayProvider`, record the prefix length on the whole-replay path by replacing the `capturing.captured_turns.append({...})` call:

```python
            capturing.captured_turns.append(
                {
                    "prompt_ids": replayed["prompt_token_ids"],
                    "response_ids": replayed["response_token_ids"],
                    "response_text": replayed["text"],
                    "is_prefix": True,
                    "prefix_len": len(replayed["response_token_ids"]),
                }
            )
```

Then replace the `generate` loop body so a split prefill is tried once the whole decisions run out. Replace:

```python
        for prompt in prompts:
            replayed = controller.next_response(prompt)
            if replayed is None:
                results.extend(capturing.generate([prompt]))
                continue
```

with:

```python
        for prompt in prompts:
            replayed = controller.next_response(prompt)
            if replayed is None:
                partial = controller.next_partial(prompt)
                if partial is not None:
                    results.extend(self._generate_from_prefill(capturing, controller, partial))
                else:
                    results.extend(capturing.generate([prompt]))
                continue
```

and add the method to `ReplayProvider`:

```python
    def _generate_from_prefill(self, capturing, controller, partial) -> list:
        """Have the model finish a teacher turn, then record it as one whole turn.

        vLLM formats the final assistant message open-ended under
        ``continue_final_message`` and returns only the continuation. The daemon's proxy
        will inject token IDs for the request it saw, whose prompt contains the partial
        assistant message; those are wrong for training, so the captured turn is
        overwritten here with the prompt the turn would have had and a response that is
        teacher tokens followed by generated ones. This is what keeps the vendored daemon
        out of the change.
        """
        payload = json.dumps(
            {
                "messages": partial["messages"]
                + [{"role": "assistant", "content": partial["prefix_text"]}],
                "use_thinking": False,
                "continue_final_message": True,
            }
        )
        before = len(capturing.captured_turns)
        results = capturing.generate([payload])
        if len(capturing.captured_turns) != before + 1:
            raise RuntimeError(
                "ReplayProvider expected exactly one captured turn for a split prefill, "
                f"got {len(capturing.captured_turns) - before}. The turn cannot be "
                "corrected, so prefix_mask would mark tokens that are not the teacher's."
            )

        continuation = results[0].text or ""
        prefix_ids = list(partial["prefix_ids"])
        cont_ids = list(controller.tokenizer.encode(continuation, add_special_tokens=False))

        turn = capturing.captured_turns[-1]
        turn["prompt_ids"] = list(partial["prompt_token_ids"])
        turn["response_ids"] = prefix_ids + cont_ids
        turn["response_text"] = partial["prefix_text"] + continuation
        turn["is_prefix"] = True
        turn["prefix_len"] = len(prefix_ids)

        # Same reason as the whole-replay print: a run where the split never happened
        # is otherwise indistinguishable from one where it did.
        print(
            f"[ReplayProvider] served split turn: {len(prefix_ids)} teacher tokens + "
            f"{len(cont_ids)} generated"
        )

        # Return the whole turn, not just the continuation, so the orchestrator parses
        # the complete decision.
        results[0] = GenerationResult(
            text=turn["response_text"],
            finish_reason=results[0].finish_reason,
            usage=results[0].usage,
            metadata={"replayed": "partial"},
            prompt_token_ids=turn["prompt_ids"],
            response_token_ids=turn["response_ids"],
        )
        return results
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/ -k "prefix_rft or api_provider" -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/agent_engine/models/api_provider.py src/fine_tuning/prefix_replay.py \
        tests/unit/test_prefix_rft_rollout.py
git commit -m "feat(prefix-rft): finish a split teacher turn via continue_final_message"
```

---

### Task 5: Dispatch the mode from the driver

**Files:**
- Modify: `src/verl_ext/prefix_rft/dispatch.py`
- Modify: `src/verl_ext/prefix_rft/daemon.py:44-73, 118-122, 152-163`
- Test: `tests/unit/test_prefix_rft_daemon.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-4.
- Produces:
  - `prefix_l_for(sample, rollout_index, is_train, schedule, demo_store, n_prefixed_rollouts, global_step) -> float`
  - `prefix_spec_for(..., mode="steps") -> dict` returning `{"prefix_k": int}` or `{"prefix_l": float}`
  - `PrefixRFTDaemon.prefix_mode: str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_prefix_rft_daemon.py`. The existing `_Store` and `_Schedule` fakes are reused; `_Schedule` needs a `sample_l`, added as a subclass:

```python
from verl_ext.prefix_rft.dispatch import prefix_l_for, prefix_spec_for


class _LSchedule(_Schedule):
    def __init__(self, l=0.6):
        super().__init__()
        self._l = l
        self.l_calls = []

    def sample_l(self, global_step):
        self.l_calls.append(global_step)
        return self._l, 0.1, 0.95


def test_token_mode_dispatches_the_raw_fraction():
    sched = _LSchedule(l=0.6)
    l = prefix_l_for(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        schedule=sched,
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=7,
    )
    assert l == pytest.approx(0.6)
    assert sched.l_calls == [7]


def test_token_mode_prefixes_a_single_decision_demonstration():
    """Step mode cannot: k <= m - 1 = 0. The token guard is applied to the token
    total instead, so these 273 questions become prefixable."""
    l = prefix_l_for(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        schedule=_LSchedule(l=0.6),
        demo_store=_Store(1),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert l == pytest.approx(0.6)


def test_token_mode_skips_questions_with_no_demonstration():
    assert prefix_l_for(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        schedule=_LSchedule(),
        demo_store=_Store(0),
        n_prefixed_rollouts=1,
        global_step=0,
    ) == 0.0


def test_token_mode_never_prefixes_validation_or_extra_rollouts():
    kwargs = dict(
        sample={"question": "q"},
        schedule=_LSchedule(),
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_l_for(rollout_index=0, is_train=False, **kwargs) == 0.0
    assert prefix_l_for(rollout_index=1, is_train=True, **kwargs) == 0.0


def test_the_spec_carries_exactly_one_key_per_mode():
    kwargs = dict(
        sample={"question": "q"},
        rollout_index=0,
        is_train=True,
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_spec_for(schedule=_Schedule(k=2), mode="steps", **kwargs) == {"prefix_k": 2}
    assert prefix_spec_for(schedule=_LSchedule(l=0.6), mode="tokens", **kwargs) == pytest.approx(
        {"prefix_l": 0.6}
    )


def test_the_spec_still_carries_its_key_when_nothing_is_prefixed():
    """The worker warns when its key is missing entirely, because that means dispatch
    broke. A non-prefixed rollout must carry the key with a zero, not drop it."""
    kwargs = dict(
        sample={"question": "q"},
        rollout_index=5,
        is_train=True,
        demo_store=_Store(3),
        n_prefixed_rollouts=1,
        global_step=0,
    )
    assert prefix_spec_for(schedule=_Schedule(), mode="steps", **kwargs) == {"prefix_k": 0}
    assert prefix_spec_for(schedule=_LSchedule(), mode="tokens", **kwargs) == {"prefix_l": 0.0}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown prefix mode"):
        prefix_spec_for(
            sample={"question": "q"},
            rollout_index=0,
            is_train=True,
            schedule=_Schedule(),
            demo_store=_Store(3),
            n_prefixed_rollouts=1,
            global_step=0,
            mode="fractions",
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_daemon.py -q
```

Expected: `ImportError: cannot import name 'prefix_l_for'`.

- [ ] **Step 3: Add the token dispatch**

Append to `src/verl_ext/prefix_rft/dispatch.py`, leaving `prefix_k_for` exactly as it is:

```python
def prefix_l_for(
    sample,
    rollout_index,
    is_train,
    schedule,
    demo_store,
    n_prefixed_rollouts,
    global_step,
):
    """Prefix fraction for rollout ``rollout_index``, or 0.0 for no prefix.

    Token mode's analogue of ``prefix_k_for``. The driver ships the raw ``l`` and the
    worker turns it into a token budget, because the worker is the process with a
    tokenizer and it would otherwise need the per-decision token counts precomputed
    into the demonstration store.

    There is deliberately no ``m > 1`` guard. Step mode needs one because ``k <= m - 1``
    leaves a single-decision demonstration unprefixable. The token budget's own
    ``B <= T - 1`` clamp already leaves a token to generate, so those 273 questions are
    prefixable here.
    """
    if not is_train:
        return 0.0
    if schedule is None or demo_store is None:
        return 0.0
    if rollout_index >= n_prefixed_rollouts:
        return 0.0

    question = str((sample or {}).get("question", ""))
    if not question:
        return 0.0
    if demo_store.n_steps(question) < 1:
        return 0.0

    l, _, _ = schedule.sample_l(global_step=global_step)
    return float(l)


def prefix_spec_for(
    sample,
    rollout_index,
    is_train,
    schedule,
    demo_store,
    n_prefixed_rollouts,
    global_step,
    mode="steps",
):
    """The payload keys that tell the rollout worker what prefix to replay.

    One key per mode, and the worker reads the mode off whichever arrives. Keeping the
    choice in one process is why there is no PREFIX_MODE environment variable: two
    settings that must agree are two settings that can disagree, which is the failure
    PREFIX_DEMOS_PATH is already exposed to.

    The key is always present, with a zero value when this rollout is not prefixed.
    ``_make_controller`` warns when its key is missing entirely, because that means
    dispatch itself broke, and dropping the key would fire that warning constantly.
    """
    kwargs = dict(
        sample=sample,
        rollout_index=rollout_index,
        is_train=is_train,
        schedule=schedule,
        demo_store=demo_store,
        n_prefixed_rollouts=n_prefixed_rollouts,
        global_step=global_step,
    )
    if mode == "tokens":
        return {"prefix_l": prefix_l_for(**kwargs)}
    if mode == "steps":
        return {"prefix_k": prefix_k_for(**kwargs)}
    raise ValueError(f"unknown prefix mode {mode!r}; expected 'steps' or 'tokens'")
```

- [ ] **Step 4: Wire it into the daemon**

In `src/verl_ext/prefix_rft/daemon.py`, change the import:

```python
from .dispatch import prefix_spec_for
```

Add `prefix_mode` to `__init__` (keep the other parameters and assignments):

```python
    def __init__(
        self,
        *args,
        schedule=None,
        demo_store=None,
        n_prefixed_rollouts=1,
        prefix_mode="steps",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedule = schedule
        self.demo_store = demo_store
        self.n_prefixed_rollouts = n_prefixed_rollouts
        self.prefix_mode = prefix_mode
        self._global_step = 0
        self.last_prefix_metrics = {}
```

Replace `_prefix_k_for` with:

```python
    def _prefix_spec_for(self, sample, rollout_index, is_train):
        return prefix_spec_for(
            sample=sample,
            rollout_index=rollout_index,
            is_train=is_train,
            schedule=self.schedule,
            demo_store=self.demo_store,
            n_prefixed_rollouts=self.n_prefixed_rollouts,
            global_step=self._global_step,
            mode=self.prefix_mode,
        )
```

In `_async_set_up`, replace the two lines that stamp `prefix_k`:

```python
                sample["prefix_k"] = self._prefix_k_for(sample, j, is_train)
                ks.append(sample["prefix_k"])
```

with:

```python
                spec = self._prefix_spec_for(sample, j, is_train)
                sample.update(spec)
                ks.append(next(iter(spec.values())))
```

Replace `_summarise_prefix_dispatch` with a mode-aware version:

```python
    def _summarise_prefix_dispatch(self, values):
        """Metrics for what was dispatched this step.

        ``sample_l`` is called once more here to report the window. That draw already
        happened before token mode existed, so it is left as it is; adding a second one
        would change the curriculum the run actually sees.
        """
        prefixed = [v for v in values if v > 0]
        if self.schedule is not None:
            _, low, high = self.schedule.sample_l(global_step=self._global_step)
        else:
            low, high = 0.0, 0.0
        mean = float(np.mean(prefixed)) if prefixed else 0.0
        out = {
            "actor/n_prefixed_rollouts": len(prefixed),
            "actor/prefix_low": float(low),
            "actor/prefix_high": float(high),
        }
        # Different names on purpose: mean k and mean l are not comparable quantities
        # and must not land in the same W&B series.
        if self.prefix_mode == "tokens":
            out["actor/prefix_l"] = mean
        else:
            out["actor/prefix_steps"] = mean
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/ -k prefix_rft -q
```

Expected: all pass. The existing `prefix_k_for` tests are untouched and still pass.

- [ ] **Step 6: Commit**

```bash
git add src/verl_ext/prefix_rft/dispatch.py src/verl_ext/prefix_rft/daemon.py \
        tests/unit/test_prefix_rft_daemon.py
git commit -m "feat(prefix-rft): dispatch prefix_l in token mode, prefix_k in step mode"
```

---

### Task 6: Pick the controller from the payload

**Files:**
- Modify: `src/verl_ext/prefix_rft/dispatch.py`
- Modify: `src/fine_tuning/prefix_rollout.py:59-105`
- Test: `tests/unit/test_prefix_rft_daemon.py`

**Why the decision lives in `dispatch.py`:** `fine_tuning.prefix_rollout` imports
`fine_tuning.rollout`, which pulls in agentflow and therefore agentops, absent from the
`agent_engine` env. Verify for yourself before starting:

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -c "import fine_tuning.prefix_rollout"
```

Expected: `ModuleNotFoundError: No module named 'agentops'`. So the reading of the
payload goes in `dispatch.py`, which imports nothing, and `_make_controller` becomes a
thin dispatcher over it. The dispatcher itself is covered by the import check in
`009_run_tests_for_prefix_rft.job`, which runs in `cosmas-train`.

**Interfaces:**
- Consumes: `ReplayController.from_token_fraction` (Task 3), the payload keys (Task 5).
- Produces: `read_prefix_spec(task) -> tuple[str | None, float | int]` returning
  `("tokens", l)`, `("steps", k)` or `(None, 0)`;
  `PrefixOrchestratorRollout._make_step_controller(task, k)` and
  `._make_token_controller(task, l)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_prefix_rft_daemon.py`, and extend its dispatch import to
`from verl_ext.prefix_rft.dispatch import prefix_k_for, prefix_l_for, prefix_spec_for, read_prefix_spec`:

```python
def test_a_prefix_l_payload_reads_as_token_mode():
    assert read_prefix_spec({"question": "q", "prefix_l": 0.8}) == ("tokens", 0.8)


def test_a_prefix_k_payload_reads_as_step_mode():
    assert read_prefix_spec({"question": "q", "prefix_k": 2}) == ("steps", 2)


def test_both_keys_at_once_is_refused():
    """One key per mode is the whole reason the worker cannot disagree with the
    driver. Both arriving means that invariant is already broken."""
    with pytest.raises(RuntimeError, match="disagree about prefix_rft.mode"):
        read_prefix_spec({"prefix_k": 1, "prefix_l": 0.5})


def test_neither_key_reads_as_no_dispatch():
    assert read_prefix_spec({"question": "q"}) == (None, 0)
    assert read_prefix_spec(None) == (None, 0)


def test_a_null_value_reads_as_a_zero_of_the_right_type():
    assert read_prefix_spec({"prefix_l": None}) == ("tokens", 0.0)
    assert read_prefix_spec({"prefix_k": None}) == ("steps", 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_daemon.py -q
```

Expected: `ImportError: cannot import name 'read_prefix_spec'`.

- [ ] **Step 3: Add the reader**

Append to `src/verl_ext/prefix_rft/dispatch.py`:

```python
def read_prefix_spec(task):
    """Read the dispatched prefix off a task payload.

    Returns ``("tokens", l)``, ``("steps", k)``, or ``(None, 0)`` when neither key is
    present, which means dispatch itself broke rather than that this rollout is
    unprefixed (an unprefixed one carries its key with a zero).

    Kept here rather than in the rollout worker so it can be tested: the worker module
    imports agentflow, which needs agentops, absent from the agent_engine env.
    """
    task = task or {}
    has_k = "prefix_k" in task
    has_l = "prefix_l" in task
    if has_k and has_l:
        raise RuntimeError(
            "task payload carries both prefix_k and prefix_l; the driver and the "
            "rollout worker disagree about prefix_rft.mode."
        )
    if has_l:
        return "tokens", float(task["prefix_l"] or 0.0)
    if has_k:
        return "steps", int(task["prefix_k"] or 0)
    return None, 0
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_prefix_rft_daemon.py -q
```

Expected: all pass.

- [ ] **Step 5: Split the controller factory by mode**

In `src/fine_tuning/prefix_rollout.py`, add the top-level import beside the existing ones
(`dispatch` imports nothing, so this keeps the module's dependencies as they were):

```python
from verl_ext.prefix_rft.dispatch import read_prefix_spec
```

Then replace `_make_controller` with a dispatcher and two builders:

```python
    def _make_controller(self, task: Any) -> Optional[ReplayController]:
        """Build a replay controller for this task, or None to run plain on-policy.

        The mode comes from the payload, so the driver and this worker cannot disagree
        about which experiment is running. The reading itself is in
        ``verl_ext.prefix_rft.dispatch.read_prefix_spec`` because this module is not
        importable in the env that has pytest.
        """
        mode, value = read_prefix_spec(task)
        if mode is None:
            # Only worth reporting when neither key is there, which means dispatch broke.
            print(
                "[PrefixOrchestratorRollout] neither prefix_k nor prefix_l in the task "
                f"payload — the daemon did not dispatch one. "
                f"keys={sorted((task or {}).keys())}"
            )
            return None
        if mode == "tokens":
            return self._make_token_controller(task, value)
        return self._make_step_controller(task, value)

    def _make_step_controller(self, task, k):
        """Step mode: replay k whole teacher decisions.

        Every ``return None`` here is a silent downgrade to GRPO, so each one says why.
        print(), not logging: INFO from this package does not reach the SLURM log, and a
        run where replay never happened is indistinguishable from a working one without
        these lines (job 25753032).
        """
        if k <= 0:
            return None
        store = self._get_store()
        tokenizer = self._get_tokenizer()
        if store is None or tokenizer is None:
            print(
                "[PrefixOrchestratorRollout] prefix_k=%s but no store/tokenizer; "
                "running on-policy" % k
            )
            return None
        question_text, _, _, _ = _get_task_metadata(task)
        steps = store.steps(question_text)
        if not steps:
            print(
                f"[PrefixOrchestratorRollout] prefix_k={k} but the store has no "
                f"demonstration for this question; running on-policy. "
                f"question={question_text[:60]!r}"
            )
            return None
        print(
            f"[PrefixOrchestratorRollout] replaying {k} of {len(steps)} teacher "
            f"decisions for {question_text[:50]!r}"
        )
        return ReplayController(steps, k, tokenizer)

    def _make_token_controller(self, task, l):
        """Token mode: replay a token fraction of the whole demonstration."""
        if l <= 0.0:
            return None
        store = self._get_store()
        tokenizer = self._get_tokenizer()
        if store is None or tokenizer is None:
            raise RuntimeError(
                f"prefix_l={l} was dispatched but this worker has no demonstration "
                "store or tokenizer; token mode cannot downgrade silently or the run "
                "is not the experiment it claims to be."
            )
        question_text, _, _, _ = _get_task_metadata(task)
        steps = store.steps(question_text)
        if not steps:
            # Coverage is partial by design (1358 of 1800), so this one is legitimate.
            print(
                f"[PrefixOrchestratorRollout] prefix_l={l:.3f} but the store has no "
                f"demonstration for this question; running on-policy. "
                f"question={question_text[:60]!r}"
            )
            return None
        ctrl = ReplayController.from_token_fraction(steps, l, tokenizer)
        print(
            f"[PrefixOrchestratorRollout] token prefix l={l:.3f}: {ctrl.k} of "
            f"{len(steps)} decisions replayed whole"
            + (
                f", then {ctrl.split_tokens} tokens of decision {ctrl.k + 1}"
                if ctrl.split_index is not None
                else " (no split)"
            )
        )
        return ctrl
```

- [ ] **Step 6: Check the worker module still imports in the training env**

```bash
/home/xchen1/.conda/envs/cosmas-train/bin/python -c "import fine_tuning.prefix_rollout; print('prefix_rollout OK')"
```

Expected: `prefix_rollout OK`. This is the only place the dispatcher itself is exercised.

- [ ] **Step 7: Commit**

```bash
git add src/verl_ext/prefix_rft/dispatch.py src/fine_tuning/prefix_rollout.py \
        tests/unit/test_prefix_rft_daemon.py
git commit -m "feat(prefix-rft): choose the replay controller from the dispatched key"
```

---


### Task 7: The mode switch

**Files:**
- Modify: `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`
- Modify: `src/verl_ext/prefix_rft/trainer.py:82-101, 126-140`
- Modify: `scripts/launch_verl.py:19-33, 205-215`
- Create: `experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml`

**Interfaces:**
- Consumes: `PrefixRFTDaemon.prefix_mode` (Task 5), `prefix_mask` rows (Task 2).
- Produces: `prefix_rft.mode` config key; `--prefix-mode` on `launch_verl.py`; `actor/prefix_split_fraction` metric.

- [ ] **Step 1: Add the config key**

In `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`, inside the `prefix_rft:` block, immediately after `n_prefixed_rollouts: 1`:

```yaml
  # How prefix length is measured. "steps" replays whole teacher decisions,
  # k = clamp(floor(l * m), 0, m - 1), which is this project's adaptation.
  # "tokens" is the paper's own measure (A.2): a token fraction of the whole
  # demonstration, which can split a decision in half. Default "steps" so every
  # existing config keeps its meaning.
  mode: steps
```

- [ ] **Step 2: Pass it to the daemon and add the split metric**

In `src/verl_ext/prefix_rft/trainer.py`, inside `_ensure_prefix_daemon`, after `daemon.n_prefixed_rollouts = ...`:

```python
        daemon.prefix_mode = str(self.config.prefix_rft.get("mode", "steps"))
```

and extend the print on the following lines to name the mode:

```python
        print(f"Promoted AgentModeDaemon to PrefixRFTDaemon (mode={daemon.prefix_mode})")
```

In `_apply_prefix_advantage`, after the two existing `metrics[...]` assignments for `num_prefix_tokens` and `off_ratio`:

```python
        # How many prefixed turns were split mid-turn rather than replayed whole.
        # Derived from the mask so the driver needs nothing back from the worker;
        # it is near 0 in step mode and near 1 in token mode.
        per_row_prefix = prefix_mask.sum(dim=-1)
        per_row_response = response_mask.sum(dim=-1)
        n_prefixed_rows = int((per_row_prefix > 0).sum().item())
        n_split_rows = int(((per_row_prefix > 0) & (per_row_prefix < per_row_response)).sum().item())
        metrics["actor/prefix_split_fraction"] = n_split_rows / max(1, n_prefixed_rows)
```

- [ ] **Step 3: Add the flag**

In `scripts/launch_verl.py`, add the argument after `--dry-run`:

```python
    parser.add_argument(
        "--prefix-mode",
        choices=["steps", "tokens"],
        default=None,
        help=(
            "Prefix-RFT only: how prefix length is measured. 'steps' replays whole "
            "teacher decisions; 'tokens' is the paper's token fraction, which can split "
            "a decision in half. Overrides prefix_rft.mode in the config. This flag is "
            "on the driver alone: the rollout workers read the mode off the key the "
            "driver dispatches, so they cannot get out of step with it."
        ),
    )
```

and, right after the `prefix_rft` env-var block that selects the module:

```python
    if args.prefix_mode is not None:
        if not prefix_rft:
            print(
                "  WARNING: --prefix-mode was given but PREFIX_RFT is not set; "
                "the flag has no effect on a plain GRPO launch."
            )
        else:
            python_args["prefix_rft.mode"] = args.prefix_mode
            print(f"  prefix_rft.mode={args.prefix_mode} (from --prefix-mode)")
```

This must come after `python_args` is built and before the command is assembled, so place it directly below the existing `if prefix_rft: print(...)` line.

- [ ] **Step 4: Add the token-mode experiment config**

```bash
sed -e "s/^  EXPERIMENT_NAME: .*/  EXPERIMENT_NAME: 'qwen3-8b-prefix-rft-tokens'/" \
    experiments/configs/fine_tuning/config_prefix_rft.yaml \
    > experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml
```

Then edit the new file by hand: add `prefix_rft.mode: 'tokens'` to the `overrides:` block directly under `prefix_rft.n_prefixed_rollouts: 1`, and replace the header comment block with:

```yaml
# experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml
#
# config_prefix_rft.yaml with ONE change: prefix_rft.mode steps -> tokens.
# The prefix becomes a token fraction of the whole demonstration (the paper's own
# measure, A.2) instead of a count of whole teacher decisions, so a decision can be
# split in half and the model finishes the turn the teacher started.
#
# Two consequences to read the results against:
#   - the 273 single-decision demonstrations become prefixable, so prefixed coverage
#     rises from 1085 questions to all 1358;
#   - a tool call split mid-JSON may come back malformed, which depresses
#     actor/reward_with_prefix early when l is near 0.95.
#
# Compare against config_prefix_rft.yaml. Everything else is identical by design.
```

- [ ] **Step 5: Verify the config composes**

```bash
grep -n "prefix_rft.mode" experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml
diff <(grep -v "EXPERIMENT_NAME\|prefix_rft.mode\|^#" experiments/configs/fine_tuning/config_prefix_rft.yaml) \
     <(grep -v "EXPERIMENT_NAME\|prefix_rft.mode\|^#" experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml)
```

Expected: the grep finds `prefix_rft.mode: 'tokens'`, and the diff is empty, proving the configs differ only in the name and the mode.

- [ ] **Step 6: Dry-run both modes in the training env**

```bash
/home/xchen1/.conda/envs/cosmas-train/bin/python scripts/launch_verl.py \
  --config experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml --dry-run
/home/xchen1/.conda/envs/cosmas-train/bin/python scripts/launch_verl.py \
  --config experiments/configs/fine_tuning/config_prefix_rft.yaml \
  --prefix-mode tokens --dry-run
```

Expected: both print `DRY RUN OK` and the second prints `prefix_rft.mode=tokens (from --prefix-mode)`. If the env lacks a GPU-free path to Hydra composition, run these under `jobs/fine_tuning/009_run_tests_for_prefix_rft.job` instead, which already does the `cosmas-train` half of the gate.

- [ ] **Step 7: Commit**

```bash
git add src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml \
        src/verl_ext/prefix_rft/trainer.py scripts/launch_verl.py \
        experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml
git commit -m "feat(prefix-rft): --prefix-mode switch and a token-mode config"
```

---

### Task 8: Extend the tokenisation preflight

The standing risk of this whole method is that a wrong mask trains normally and reports success. Token mode widens the surface, so the existing gate has to cover it before any GPU time is spent.

**Files:**
- Modify: `scripts/check_prefix_replay_tokenisation.py`
- Modify: `jobs/fine_tuning/009_run_tests_for_prefix_rft.job`

**Interfaces:**
- Consumes: `ReplayController.from_token_fraction`, `next_partial` (Task 3).
- Produces: `--mode {steps,tokens,both}` on the check script, default `both`.

- [ ] **Step 1: Add the token-mode case**

In `scripts/check_prefix_replay_tokenisation.py`, add the argument:

```python
    parser.add_argument(
        "--mode",
        choices=["steps", "tokens", "both"],
        default="both",
        help="which replay mode's tokenisation to check",
    )
```

and add this function above `main`:

```python
def _check_token_mode(store, frame, tokenizer, n):
    """Check the split turn the way the training batch will see it.

    Two properties, both of which fail silently in a real run:

    1. The teacher tokens sent as a prefill must survive the round trip through text.
       ``_safe_prefix`` backs the boundary off until they do, so what it returns must
       re-encode to itself exactly.
    2. The response row the batch sees is ``prefix_ids + encode(continuation)``, and
       ``prefix_mask`` marks its first ``len(prefix_ids)`` positions. Those ids must be
       a true prefix of the teacher's own encoding of that decision, or the mask marks
       tokens the teacher never wrote.
    """
    checked = 0
    for i in range(min(n, len(frame))):
        row = frame.iloc[i]
        steps = store.steps(row["question"])
        if not steps:
            return _fail(f"row {i}: the store does not resolve its own question text")

        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": row["question"]},
        ]
        payload = json.dumps({"messages": messages, "use_thinking": False})

        ctrl = ReplayController.from_token_fraction(steps, l=0.8, tokenizer=tokenizer)
        for _ in range(ctrl.k):
            ctrl.next_response(payload)
        partial = ctrl.next_partial(payload)
        if partial is None:
            # A budget that landed on a decision boundary. Legal, nothing to check.
            continue

        prefix_ids = partial["prefix_ids"]
        if tokenizer.encode(partial["prefix_text"], add_special_tokens=False) != prefix_ids:
            return _fail(
                f"row {i}: the prefill text does not re-encode to the ids it came from; "
                "_safe_prefix should have backed the boundary off further"
            )

        full = tokenizer.encode(str(steps[ctrl.split_index]["response"]), add_special_tokens=False)
        if list(full[: len(prefix_ids)]) != list(prefix_ids):
            return _fail(
                f"row {i}: the prefill ids are not a prefix of the teacher's own encoding "
                "of that decision; prefix_mask would mark tokens the teacher never wrote"
            )

        proxy_prompt = list(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )
        if partial["prompt_token_ids"] != proxy_prompt:
            return _fail(f"row {i}: the split turn's prompt ids differ from the proxy's")

        checked += 1

    print(f"PASSED (token mode): {checked} split turns tokenise as the batch will see them")
    return 0
```

Then split `main` so each half can run on its own. Extract everything in `main` from
`checked = 0` down to its `return 0` into a new function, replacing `args.n` with `n` and
`args.model` with `model`:

```python
def _check_steps_mode(store, frame, tokenizer, n, model):
    """The existing check, unchanged: whole replayed turns tokenise as the proxy would."""
    checked = 0
    ...  # the extracted body, verbatim
```

and replace the tail of `main` (from `tokenizer = AutoTokenizer...` onwards) with:

```python
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    frame = pd.read_parquet(args.demos)
    store = DemoStore.from_parquet(args.demos)

    # _fail returns 1, so OR-ing accumulates a non-zero exit from either half while
    # still running both: one failing mode should not hide the other's result.
    rc = 0
    if args.mode in ("steps", "both"):
        rc |= _check_steps_mode(store, frame, tokenizer, args.n, args.model)
    if args.mode in ("tokens", "both"):
        rc |= _check_token_mode(store, frame, tokenizer, args.n)
    return rc
```

- [ ] **Step 2: Run the check against the real store**

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python scripts/check_prefix_replay_tokenisation.py --mode both
```

Expected: `PASSED` for both modes. Do **not** set `HF_HUB_OFFLINE=1`; it makes the tokenizer load fail with `OfflineModeIsEnabled`.

If the token-mode half fails on property 1 for every row, the tokenizer is not round-tripping at any boundary and the design assumption is wrong. Stop and report it rather than loosening the check.

- [ ] **Step 3: Add it to the CPU gate**

In `jobs/fine_tuning/009_run_tests_for_prefix_rft.job`, add `tests/unit/test_prefix_rft_budget.py` to the pytest invocation at line 103, and add the check-script call with `--mode both` next to the existing tokenisation check.

- [ ] **Step 4: Commit**

```bash
git add scripts/check_prefix_replay_tokenisation.py jobs/fine_tuning/009_run_tests_for_prefix_rft.job
git commit -m "test(prefix-rft): preflight the split turn's tokenisation"
```

---

### Task 9: Document the mode

**Files:**
- Modify: `docs/pipelines/prefix-rft.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a mode section to the pipeline doc**

Add a section directly after the Flow GRPO callout, titled **"Two ways to measure a prefix"**, covering:
- the two modes and the one-line difference between them, with the `split_for_budget` arithmetic stated as `B = clamp(floor(l * T), 0, T - 1)`;
- that step mode is the default and unchanged;
- that token mode makes the 273 single-decision demonstrations prefixable, taking prefixed coverage from 1085 to 1358 questions, and that this is a confound to report when comparing the two runs;
- that a tool call split mid-JSON may come back malformed early in training, which is a finding rather than a defect;
- how to run each: `--prefix-mode tokens` or `config_prefix_rft_tokens.yaml`;
- the metrics table from the spec, naming `actor/prefix_steps`, `actor/prefix_l`, `actor/prefix_split_fraction` and `actor/off_ratio`.

Update the existing divergence list: step mode's entry gains a note that it is now one of two modes, and token mode is recorded as the paper-faithful one whose only remaining divergence is that turn boundaries still exist.

- [ ] **Step 2: Update the changelog**

Add under the current unreleased heading:

```markdown
- **Prefix-RFT token mode.** The prefix can now be measured in tokens, the paper's own
  measure, instead of whole teacher decisions. A decision that straddles the budget is
  split and the model finishes the turn, served through vLLM's `continue_final_message`.
  Select with `--prefix-mode tokens` on `scripts/launch_verl.py` or `prefix_rft.mode` in
  the config; the default stays `steps`, so existing configs are unchanged. Token mode
  makes the 273 single-decision demonstrations prefixable, raising prefixed coverage
  from 1085 to 1358 questions.
```

- [ ] **Step 3: Commit**

```bash
git add docs/pipelines/prefix-rft.md CHANGELOG.md
git commit -m "docs(prefix-rft): document the steps/tokens prefix modes"
```

---

## Verification before any GPU time

Run all of these from the repo root. Every one must pass before `013` or a token-mode run is launched.

```bash
/home/xchen1/.conda/envs/agent_engine/bin/python -m pytest tests/unit/ -k prefix_rft -q
/home/xchen1/.conda/envs/agent_engine/bin/python scripts/check_prefix_replay_tokenisation.py --mode both
/home/xchen1/.conda/envs/cosmas-train/bin/python scripts/check_prefix_rft_trainer_sync.py
/home/xchen1/.conda/envs/cosmas-train/bin/python scripts/launch_verl.py \
  --config experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml --dry-run
```

Then a tiny GPU run before anything longer, using the existing tiny split with `--prefix-mode tokens`. It is working when `actor/off_ratio` is non-zero, `actor/prefix_split_fraction` is close to 1, and the SLURM log carries `[ReplayProvider] served split turn` lines.
