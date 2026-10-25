# IDC Week-1 Go / No-Go Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the four week-1 measurements (test 0, test 1, test 2, coverage) and the decision-point check, and turn them into the branch decision of the research plan's Section 8.1.

**Architecture:** CPU analyses live in a new package `agent_engine.analysis.idc` and read rollout records in the existing on-disk format. GPU work reuses the training rollout worker outside verl: a generator samples fresh 8-rollout groups from the warm-start checkpoint, and a counterfactual runner replays each rollout's own prefix with the existing Prefix-RFT replay shims, then forces an option, resamples a turn, or swaps in a strong coder. A final report script applies the decision table.

**Tech Stack:** Python 3.11, pytest, pandas, numpy, vLLM OpenAI-compatible servers, the repo's `fine_tuning` rollout and `prefix_replay` modules, Serper web search.

**Spec:** `docs/superpowers/specs/2026-10-25-idc-research-plan.md` (Section 8 and tasks 1.1 to 1.8). Read Section 8 before starting.

All paths below are relative to the repository root.

## Global Constraints

- Thinking off everywhere (`use_thinking=False`). Orchestrator sampling temperature 0.7; sub-agents greedy (temperature 0.0), `max_tokens=2048`.
- Episode limits match training: `max_turns=5`, orchestrator `max_tokens=2048`, `MAX_SEARCH_CONTENT_CHARS=14000`, search `top_k=5`.
- Group size 8. Fresh question sample: 200 DeepMath, 200 HotpotQA, 100 NQ from `data/training/train/combined_train.parquet`, `seed=0`.
- Weak executor: `Qwen/Qwen3-1.7B`. Strong coder: `Qwen/Qwen3-32B` (decision D11: 32B for week 1).
- Every random choice uses `seed=0` and is logged.
- Results go to `data/results/idc_week1/`. Fresh rollouts go to `experiments/results/idc_week1/`.
- Web cache for new runs: `WEB_CACHE_DATASET=idc_week1` (keeps the cache small; the existing `cache/serper/fine_tuning/url_cache.json` is 6.3 GB and must not be loaded by these jobs).
- CPU tests run in the `agent_engine` env: `~/.conda/envs/agent_engine/bin/python -m pytest <path> -q`. Anything importing `fine_tuning.rollout` runs in `cosmas-train`.
- Code style: black, line length 100, isort black profile, match the docstring density of `src/agent_engine/analysis/fine_tuning/`.
- **Do not run `git commit`.** End each task by staging (`git add`); the user reviews and commits.
- Any prose written into reports: no em dashes.

## Review Focus

1. **Short or yes/no gold answers** match page text by coincidence. Expected: these dispatches are excluded from the search adequacy check and counted separately. Test: `test_ineligible_gold_is_excluded` (Task 5).
2. **Pages differ from what the sub-agent saw.** Adequacy must use the same top-5 formatting and the 14,000-character truncation as `WebSearchTool._analyze_with_llm`. Test: `test_seen_pages_truncates_like_subagent` (Task 5).
3. **Recorded steps do not line up with the episode** (crashed episode, synthesis turn after `max_turns`). Expected: `episode_steps` raises, the runner skips that rollout and counts it, and nothing is replayed from a misaligned record. Test: `test_episode_steps_rejects_mismatch` (Task 6).
4. **The code swap hits the wrong call.** Only the first live `code_generator` call after replay goes to the strong coder; replayed calls, search calls and later code calls use the recorded result or the weak coder. Tests: `test_swap_only_first_live_code_call`, `test_swap_leaves_search_on_weak` (Task 7).
5. **Group chunking is wrong for fresh groups.** Each question directory must hold exactly 8 rollouts from one sampling pass. Test: `test_generator_writes_exact_groups` (Task 8).

---

### Task 1: Locate checkpoints and runs (manual, research plan task 1.1)

**Files:**
- Create: `experiments/configs/idc/week1.yaml`

**Interfaces:**
- Produces: `week1.yaml` with keys `warm_start_ckpt` (str, HF directory), `tokenizer` (str, `Qwen/Qwen3-8B`), `grpo_rollout_runs` (list of `{name, path, group_size}`), `eval_pairs` (list of `{dataset, start, final}` folder paths, each holding timestamped runs with `raw_results.json`). Every later job reads this file.

- [ ] **Step 1: Find the warm-start checkpoint** (SFT on Qwen3-30B-A3B-Instruct traces, step 1200, +6.70 pp). Search the Snellius home and scratch of both users and the W&B run of that SFT job. Record the HF directory.
- [ ] **Step 2: Find the Flow-GRPO run directories** `.../28-07-2026_09-51-310076/rollout_data/train` (n = 8) and `.../05-08-2026_16-30-335104/rollout_data/train` (n = 16). They are not under this repo's `experiments/results/fine_tuning/qwen3-8b-grpo-search-math-v2/`, which only holds `29-05-2026_11-36-23210365`. Include that May run too.
- [ ] **Step 3: Write `week1.yaml`.** `eval_pairs` uses the existing thinking-off runs: `start` = `experiments/results/orchestrator_capacity/<ds>/orch8b_sub1_7b_none`, `final` = `experiments/results/lora_inference/<ds>/qwen8B_sub1_7b_none` (and a second entry with `_none_v2`) for `ds` in `aime, gpqa, gaia, hle, musique`.
- [ ] **Step 4: Verify** every path exists: `python -c "import yaml,os;c=yaml.safe_load(open('experiments/configs/idc/week1.yaml'));print([p for p in [c['warm_start_ckpt']]+[r['path'] for r in c['grpo_rollout_runs']] if not os.path.exists(p)])"` prints `[]`.
- [ ] **Step 5: Stage** `git add experiments/configs/idc/week1.yaml`.

If the Aug runs cannot be found by end of day 1, continue with the May run and say so in the report.

---

### Task 2: Trajectory parsing and gold matching helpers

**Files:**
- Create: `src/agent_engine/analysis/idc/__init__.py`, `src/agent_engine/analysis/idc/decisions.py`
- Test: `tests/unit/test_idc_decisions.py`

**Interfaces:**
- Produces, in `agent_engine.analysis.idc.decisions`:
  - `SEARCH, CODE, ANSWER = "SEARCH", "CODE", "ANSWER"`; `OPTIONS = (SEARCH, CODE, ANSWER)`
  - `option_of(tool_name: str | None) -> str` (`web_search` to SEARCH, `code_generator` to CODE, `None` to ANSWER, other tool names raise `ValueError`)
  - `@dataclass(frozen=True) Dispatch: turn: int; option: str; argument: str; result: str`
  - `decision_options(output_messages: list[dict]) -> list[str]`: one option per assistant message, in order. An assistant message followed by a `role == "tool"` message takes that message's `tool_name`.
  - `dispatches(output_messages: list[dict]) -> list[Dispatch]`: argument is `query` for search, `task` for code, parsed with `agent_engine.utils.parsing.parse_tool_call`.
  - `gold_answers(record: dict) -> list[str]`: `groundtruth` plus `extra_info.golden_answers` when present, deduplicated.
  - `eligible_gold(gold: str) -> bool`: false for normalised `yes`/`no` and for normalised length below 4 characters.
  - `answer_in_text(golds: list[str], text: str) -> bool`: any eligible gold, after `metrics.normalize_answer`, occurs in the normalised text on word boundaries.
  - `pretext(response: str) -> tuple[str, str]`: `(sub_goal_text, other_text)` before the first `<tool_call>`; other text excludes the `<sub_goal>` block.

- [ ] **Step 1: Write the failing tests**

```python
def test_decision_options_tool_then_answer():
    msgs = [
        {"role": "assistant", "content": '<tool_call>\n{"name": "web_search", "arguments": {"query": "q"}}\n</tool_call>'},
        {"role": "tool", "tool_name": "web_search", "content": "x"},
        {"role": "assistant", "content": "Final Answer: Paris"},
    ]
    assert decision_options(msgs) == [SEARCH, ANSWER]

def test_dispatches_parse_arguments():
    d = dispatches(msgs_with_code_task("count to 3"))  # helper builds a code call + tool result "3"
    assert d == [Dispatch(turn=0, option=CODE, argument="count to 3", result="3")]

def test_option_of_unknown_tool_raises():
    with pytest.raises(ValueError):
        option_of("text_inspector_v9")

def test_ineligible_gold():
    assert not eligible_gold("no") and not eligible_gold("Yes") and not eligible_gold("12")
    assert eligible_gold("1984") and eligible_gold("Glen of Imaal")

def test_answer_in_text_word_boundary():
    assert answer_in_text(["Paris"], "The capital is Paris, France.")
    assert not answer_in_text(["Paris"], "Parisian cafes")
    assert not answer_in_text(["no"], "no no no")

def test_pretext_splits_sub_goal():
    r = '<sub_goal>find X</sub_goal>\n\nLet me search.\n<tool_call>\n{"name": "web_search", "arguments": {"query": "X"}}\n</tool_call>'
    assert pretext(r) == ("find X", "Let me search.")
```

- [ ] **Step 2: Run** `~/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_idc_decisions.py -q`. Expected: FAIL, `ModuleNotFoundError: agent_engine.analysis.idc`.
- [ ] **Step 3: Implement** the interfaces above in `decisions.py`. `__init__.py` is a one-line docstring.
- [ ] **Step 4: Run the tests again.** Expected: all pass.
- [ ] **Step 5: Stage** the new files.

---

### Task 3: Test 0, tool avoidance under GRPO (research plan task 1.2)

**Files:**
- Create: `src/agent_engine/analysis/idc/stats.py`, `src/agent_engine/analysis/idc/tool_avoidance.py`
- Test: `tests/unit/test_idc_stats.py`, `tests/unit/test_idc_tool_avoidance.py`

**Interfaces:**
- Consumes: `decisions.dispatches`, `decisions.gold_answers`, `decisions.answer_in_text`, `decisions.eligible_gold`; `analysis.fine_tuning.runs.latest_run`; `analysis.fine_tuning.rollout_groups.resolve_rollout_dir`, `iter_question_dirs`.
- Produces:
  - `stats.mean_diff_permutation_p(x: list[float], y: list[float], n_perm=10000, seed=0) -> float`: one-sided p for `mean(x) < mean(y)` by permuting labels.
  - `stats.sign_flip_p(d: list[float], n_perm=10000, seed=0) -> float`: one-sided p for `mean(d) < 0`.
  - `tool_avoidance.dispatch_failed(d: Dispatch, golds: list[str]) -> bool`. Search: the extract contains `No helpful information found`, or an eligible gold exists and `answer_in_text` is false. Code: result is empty or matches `r"Traceback|Error|error:|timed out|Timeout"`.
  - `tool_avoidance.QuestionDelta(question: str, domain: str, exec_fail: float, start_tools: float, final_tools: float)`
  - `tool_avoidance.from_eval_pair(start_dir, final_dir, dataset) -> list[QuestionDelta]`: per question, `exec_fail` from the start run's dispatches, tool counts as `len(action_history)`. Questions with no dispatch in the start run are dropped and counted.
  - `tool_avoidance.from_training_run(rollout_dir) -> list[QuestionDelta]`: rollouts binned into thirds by the record `timestamp` (quantiles over the whole run). A question counts only if it has rollouts in both the first and the last third. `exec_fail` from first-third dispatches; tool counts are the share of tool-using rollouts per third.
  - `tool_avoidance.summarise(deltas) -> dict` per domain: `n`, `n_high`, `n_low` (high means `exec_fail >= 0.5`), `delta_high`, `delta_low`, `p_a` (`mean_diff_permutation_p(high deltas, low deltas)`), `p_b` (`sign_flip_p(low deltas)`), `pass_a = p_a < 0.05`.
  - CLI: `python -m agent_engine.analysis.idc.tool_avoidance --config experiments/configs/idc/week1.yaml --out data/results/idc_week1/test0.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_permutation_detects_shift():
    assert mean_diff_permutation_p([-1.0] * 20, [0.0] * 20) < 0.01
    assert mean_diff_permutation_p([0.0] * 20, [0.0] * 20) > 0.5

def test_sign_flip():
    assert sign_flip_p([-1.0] * 15) < 0.01

def test_dispatch_failed_search_and_code():
    s = Dispatch(0, SEARCH, "q", "**Final Information**\n\nNo helpful information found.")
    c = Dispatch(0, CODE, "t", "Traceback (most recent call last): ...")
    assert dispatch_failed(s, ["Paris"]) and dispatch_failed(c, ["42"])
    assert not dispatch_failed(Dispatch(0, CODE, "t", "42"), ["42"])

def test_training_thirds(tmp_path):
    # write 3 rollouts for one question at t=0,5,10 (tool, tool, no tool) plus a filler
    # question spanning the same window; expect start_tools=1.0, final_tools=0.0
    ...
    deltas = from_training_run(tmp_path)
    q = next(d for d in deltas if d.question == "Q1")
    assert (q.start_tools, q.final_tools) == (1.0, 0.0)

def test_summarise_pass_a():
    deltas = [QuestionDelta(f"h{i}", "math", 1.0, 1.0, 0.0) for i in range(20)] + \
             [QuestionDelta(f"l{i}", "math", 0.0, 1.0, 1.0) for i in range(20)]
    s = summarise(deltas)["math"]
    assert s["pass_a"] and s["delta_high"] == -1.0 and s["delta_low"] == 0.0
```

- [ ] **Step 2: Run** both test files. Expected: FAIL on import.
- [ ] **Step 3: Implement** `stats.py` with numpy (`default_rng(seed)`), then `tool_avoidance.py`. The CLI loops over `eval_pairs` and `grpo_rollout_runs` from the YAML and writes `{"eval": {dataset: summary}, "training": {run_name: summary}, "dropped": {...}}`.
- [ ] **Step 4: Run tests.** Expected: pass.
- [ ] **Step 5: Run on real data** with the CLI. Expected: `data/results/idc_week1/test0.json` written; printed table has one row per dataset or domain with `delta_high`, `delta_low`, `p_a`, `p_b`. Sanity check: AIME mean tool calls for base vs v1 should be close to 1.32 and 1.05.
- [ ] **Step 6: Stage** code, tests and `test0.json`.

---

### Task 4: Coverage count (research plan task 1.3)

**Files:**
- Create: `src/agent_engine/analysis/idc/coverage.py`
- Test: `tests/unit/test_idc_coverage.py`

**Interfaces:**
- Consumes: `decisions.decision_options`; `rollout_groups.iter_question_dirs`, `infer_group_size`.
- Produces: `coverage.group_coverage(option_lists: list[list[str]]) -> dict` returning, for `first` and `last`, the set of options sampled across the group. `coverage.collect(rollout_dir, group_size) -> dict` per domain, over all-wrong groups only: `{"n_all_wrong": int, "first": {"missing": {option: share}, "any_missing": share}, "last": {...same}}`, where `missing[option]` is the share of all-wrong groups in which that option was never sampled at that decision, and `any_missing` the share in which at least one of the three was never sampled. CLI: `python -m agent_engine.analysis.idc.coverage --config experiments/configs/idc/week1.yaml --out data/results/idc_week1/coverage.json`.

The existing `iter_groups` returns `Rollout(correct, tool)` only, so `collect` loads the record JSONs itself, chunking each directory's files exactly as `iter_groups` does.

- [ ] **Step 1: Write the failing tests**

```python
def test_group_coverage_first_last():
    cov = group_coverage([[SEARCH, ANSWER], [ANSWER], [SEARCH, SEARCH, ANSWER]])
    assert cov["first"] == {SEARCH, ANSWER} and cov["last"] == {ANSWER}

def test_collect_only_all_wrong(tmp_path):
    # one all-wrong group of 2 (both ANSWER-only) and one mixed group
    ...
    out = collect(tmp_path, group_size=2)
    assert out["hotpotqa"]["n_all_wrong"] == 1
    assert out["hotpotqa"]["first"]["missing"][SEARCH] == 1.0
```

- [ ] **Step 2: Run.** Expected: FAIL on import.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: pass. Then run the CLI on every run in `grpo_rollout_runs`.
- [ ] **Step 5: Stage.**

---

### Task 5: Search adequacy check (CPU half of test 1, research plan task 1.5)

**Files:**
- Create: `src/agent_engine/analysis/idc/search_adequacy.py`
- Test: `tests/unit/test_idc_search_adequacy.py`

**Interfaces:**
- Consumes: `decisions.*`; `agent_engine.tools.WebSearchTool`; `agent_engine.caching.CacheManager`.
- Produces:
  - `seen_pages(tool: WebSearchTool, query: str, max_chars: int = 14000) -> str | None`: `tool._format_results(tool.search_cache[query], query)` cut to `max_chars`, exactly as `_analyze_with_llm` cuts it. `None` when the query is not in the cache.
  - `offline_search_tool(cache_dir: str, dataset: str) -> WebSearchTool`: built with `api_key="offline"`, `provider="serper"`, the cache's `search_cache` and `url_cache`, `model_provider=None`, `top_k=5`. It must never call the network; `_format_results` reads only the caches.
  - `label_rollout(record: dict, tool) -> list[dict]`: for each SEARCH dispatch of a failed rollout, `{"rollout_id", "turn", "query", "status"}` where `status` is `adequate`, `inadequate`, `ineligible_gold` or `cache_miss`.
  - CLI: `python -m agent_engine.analysis.idc.search_adequacy --groups experiments/results/idc_week1/<run>/train --cache-dir cache --cache-dataset idc_week1 --out data/results/idc_week1/search_adequacy.json`. Writes per-domain counts and the label list. Covers failed rollouts in both mixed and all-wrong groups, and records the group type of each.

- [ ] **Step 1: Write the failing tests**

```python
def test_seen_pages_truncates_like_subagent():
    tool = offline_search_tool_from_dicts(
        search_cache={"q": [{"url": "u", "title": "t", "content": "hint"}]},
        url_cache={"u": "hint " + "x" * 50000},
    )
    pages = seen_pages(tool, "q")
    assert len(pages) <= 14000 and "hint" in pages

def test_label_adequate_and_inadequate():
    ...  # gold "Glen of Imaal" in cached page for query A, absent for query B
    labels = label_rollout(record_with_two_searches, tool)
    assert [l["status"] for l in labels] == ["adequate", "inadequate"]

def test_ineligible_gold_is_excluded():
    labels = label_rollout(failed_record(gold="no"), tool)
    assert {l["status"] for l in labels} == {"ineligible_gold"}

def test_correct_rollouts_are_skipped():
    assert label_rollout(correct_record(), tool) == []
```

`offline_search_tool_from_dicts` is a test helper that calls the same constructor as `offline_search_tool` with in-memory caches. If `WebSearchTool.__init__` contacts the network, construct it with `unittest.mock.patch` on the search client in the helper and say so in a comment.

- [ ] **Step 2: Run.** Expected: FAIL on import.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests.** Expected: pass.
- [ ] **Step 5: Stage.** The CLI runs after Task 8 produces fresh groups.

---

### Task 6: Record each episode's own steps

**Files:**
- Create: `src/fine_tuning/idc/__init__.py`, `src/fine_tuning/idc/steps.py`
- Modify: `src/fine_tuning/rollout.py` (`_save_rollout` gets a `steps` argument; `_run_episode` passes it; the orchestrator is built with `max_turns=self._max_turns(task)`, a new seam returning `self.max_turns`)
- Test: `tests/unit/test_idc_steps.py`

**Interfaces:**
- Produces:
  - `fine_tuning.idc.steps.episode_steps(captured_turns: list[dict], output_messages: list[dict]) -> list[dict]`: the policy's own decisions in the `ReplayController` step format `{"response", "tool_name", "tool_result"}`. `captured_turns[0]` is the planning turn (`tool_name=None`). Each later turn whose `response_text` parses to a tool call takes `tool_name` and `tool_result` from the next unused `role == "tool"` message of `output_messages`. Raises `ValueError` if tool calls and tool messages do not pair one-to-one, or if `captured_turns` is empty.
  - Rollout records gain `"steps": [...]` (or `"steps": null` plus `"steps_error": str` when `episode_steps` raised). Existing fields are unchanged, so `rollout_groups` keeps working.
  - `OrchestratorRollout._max_turns(task) -> int`, base returns `self.max_turns`.

`steps.py` imports only `agent_engine.utils.parsing`, so it is testable in the `agent_engine` env.

- [ ] **Step 1: Write the failing tests**

```python
def test_episode_steps_plan_tool_answer():
    turns = [{"response_text": "plan"}, {"response_text": CALL_SEARCH_Q}, {"response_text": "Final Answer: X"}]
    msgs = [{"role": "assistant", "content": CALL_SEARCH_Q},
            {"role": "tool", "tool_name": "web_search", "content": "hits"},
            {"role": "assistant", "content": "Final Answer: X"}]
    assert episode_steps(turns, msgs) == [
        {"response": "plan", "tool_name": None, "tool_result": None},
        {"response": CALL_SEARCH_Q, "tool_name": "web_search", "tool_result": "hits"},
        {"response": "Final Answer: X", "tool_name": None, "tool_result": None},
    ]

def test_episode_steps_rejects_mismatch():
    with pytest.raises(ValueError):
        episode_steps([{"response_text": "plan"}, {"response_text": CALL_SEARCH_Q}], [])

def test_steps_replay_reproduces_prompts():
    # Feed episode_steps output to ReplayController with k=len(steps) and the
    # _FakeTokenizer from test_prefix_rft_rollout; every next_response returns the
    # recorded text in order and every tool lookup returns the recorded result.
    ...
```

- [ ] **Step 2: Run** `~/.conda/envs/agent_engine/bin/python -m pytest tests/unit/test_idc_steps.py -q`. Expected: FAIL on import.
- [ ] **Step 3: Implement `episode_steps`**, then the two-line changes in `rollout.py`.
- [ ] **Step 4: Run** the new tests in `agent_engine`, then `tests/unit/test_fine_tuning_rollout.py` and `tests/unit/test_prefix_rft_rollout.py` in `cosmas-train`. Expected: all pass, no existing test changes.
- [ ] **Step 5: Stage.**

---

### Task 7: Counterfactual shims (force, resample, swap)

**Files:**
- Create: `src/fine_tuning/idc/counterfactual.py`
- Test: `tests/unit/test_idc_counterfactual.py`

**Interfaces:**
- Consumes: `fine_tuning.prefix_replay.ReplayController`, `ReplayProvider`, `ReplayToolRegistry`, `_ReplayedTool`; `decisions.SEARCH/CODE/ANSWER`.
- Produces:
  - `FORCE_PREFILL = {SEARCH: '<tool_call>\n{"name": "web_search", "arguments": {"query": "', CODE: '<tool_call>\n{"name": "code_generator", "arguments": {"task": "'}`. These prefills skip the `<sub_goal>` line; the orchestrator records an empty sub-goal, which `_extract_sub_goal` already allows. ANSWER has no prefill (see `answer_max_turns`).
  - `ForcedReplayController(ReplayController)`: `__init__(steps, k, tokenizer, prefill: str)`. Replays `steps[:k]`, then `next_partial` returns the `ReplayProvider` prefill dict with `prefix_text=prefill` once, at decision `k`. It reuses `ReplayProvider._generate_from_prefill` unchanged.
  - `answer_max_turns(k: int) -> int`: `k - 1`. Forcing ANSWER at step `k` replays `steps[:k]` with a plain `ReplayController(steps, k)` and caps the orchestrator at `k - 1` action turns. The orchestrator then runs its own synthesis turn with the existing "provide your final answer now" nudge.
  - Resampling: a plain `ReplayController(steps, k)`; the turn at `k` is generated on-policy.
  - `SwapReplayController(ReplayController)`: `__init__(steps, k, tokenizer)` replays `steps[:k+1]` but does not arm the recorded tool result of step `k`.
  - `SwapToolRegistry(ReplayToolRegistry)`: `__init__(registry, controller, strong_code_tool)`. A pending replayed result is served first. The first live `code_generator` lookup returns `strong_code_tool` (stored as `self._strong`) and sets `self.swapped = True`. Every other lookup goes to the weak registry.

- [ ] **Step 1: Write the failing tests** (fakes from `test_prefix_rft_rollout.py`: `_RoundTripTokenizer`, a capturing provider that returns canned text)

```python
def test_forced_prefill_served_at_k():
    ctrl = ForcedReplayController(STEPS, k=1, tokenizer=_RoundTripTokenizer(), prefill=FORCE_PREFILL[CODE])
    provider = ReplayProvider(_Capturing(['x"}}\n</tool_call>']), ctrl)
    provider.generate([json.dumps({"messages": [{"role": "user", "content": "q"}]})])  # plan, replayed
    out = provider.generate([json.dumps({"messages": [{"role": "user", "content": "q"}]})])
    assert out[0].text.startswith(FORCE_PREFILL[CODE])

def test_answer_max_turns():
    assert answer_max_turns(1) == 0 and answer_max_turns(3) == 2

def test_swap_only_first_live_code_call():
    ctrl = SwapReplayController(STEPS_WITH_CODE_AT_1, k=1, tokenizer=_RoundTripTokenizer())
    reg = SwapToolRegistry(_WeakRegistry(), ctrl, strong_code_tool=_Strong())
    ctrl.next_response("{}"); ctrl.next_response("{}")      # plan and the code decision, replayed
    assert reg.get("code_generator") is reg._strong          # live call k goes to the strong coder
    assert reg.get("code_generator").name == "weak_code"     # later calls are weak
    assert reg.swapped

def test_swap_leaves_search_on_weak():
    ...  # same set-up, first live lookup is web_search: weak tool returned, swapped stays False
```

- [ ] **Step 2: Run** in `agent_engine`. Expected: FAIL on import.
- [ ] **Step 3: Implement.** Keep `counterfactual.py` free of `fine_tuning.rollout` imports.
- [ ] **Step 4: Run tests.** Expected: pass.
- [ ] **Step 5: Stage.**

---

### Task 8: Fresh-group generator (research plan task 1.4)

**Files:**
- Create: `src/fine_tuning/idc/rollout.py`, `scripts/idc/generate_groups.py`, `jobs/fine_tuning/015_idc_fresh_groups.job`
- Test: `tests/unit/test_idc_generate_groups.py` (cosmas-train env; add `"test_idc_generate_groups": "agentops"` to `_REQUIRES` in `tests/conftest.py`)

**Interfaces:**
- Consumes: `OrchestratorRollout._run_episode`, the `_wrap_provider` / `_wrap_tools` / `_max_turns` seams, Task 7 shims.
- Produces:
  - `CounterfactualRollout(OrchestratorRollout)`: `__init__(*args, tokenizer, strong_code_tool=None, **kwargs)`. It reads an optional `task["cf"] = {"mode": "force" | "resample" | "swap", "steps": list, "k": int, "option": str | None}` and builds the matching controller and registry from Task 7. It overrides `_max_turns` to return `answer_max_turns(k)` for `force` with option ANSWER. With no `cf` key it behaves exactly like `OrchestratorRollout`. After each episode it exposes `last_option_probe` (Task 9) and `last_swapped`.
  - `scripts/idc/generate_groups.py --config experiments/configs/idc/week1.yaml --out experiments/results/idc_week1/fresh --n-per-question 8 --concurrency 32`: samples the Global Constraints question set and runs 8 episodes per question at temperature 0.7 via `await agent._run_episode(task, rollout_id, 0.7, endpoint, val=False)`. Records land in `<out>/train/idx_<domain>_<idx>/` in the existing format plus `steps`. It also writes `<out>/questions.parquet`.
  - Job file: one H100 node with 4 GPUs. GPU 0: `vllm serve <warm_start_ckpt> --served-model-name idc-warm-start --port 9999`. GPU 1: `vllm serve Qwen/Qwen3-1.7B --port 9998`. GPUs 2 and 3: `vllm serve Qwen/Qwen3-32B --tensor-parallel-size 2 --port 9997` (used by Task 9; started here so one allocation serves both). Export `BASE_MODEL=idc-warm-start`, `WEB_CACHE_DATASET=idc_week1`, `SUBAGENT_ENDPOINT=http://localhost:9998/v1`. The tokenizer comes from `week1.yaml`.

- [ ] **Step 1: Write the failing test** with a fake `_run_episode` that writes a record:

```python
def test_generator_writes_exact_groups(tmp_path):
    run_generator(questions=FAKE_QS[:3], out=tmp_path, n=8, episode_fn=fake_episode)
    dirs = list((tmp_path / "train").iterdir())
    assert len(dirs) == 3 and all(len(list(d.glob("*.json"))) == 8 for d in dirs)
    rec = json.loads(next(dirs[0].glob("*.json")).read_text())
    assert "steps" in rec
```

`run_generator(questions, out, n, episode_fn)` is the testable core of the script; `main()` only wires the real `episode_fn`.

- [ ] **Step 2: Run** in cosmas-train. Expected: FAIL on import.
- [ ] **Step 3: Implement** `CounterfactualRollout`, the script and the job.
- [ ] **Step 4: Run the test.** Expected: pass.
- [ ] **Step 5: Smoke run:** `--limit 2 --n-per-question 2` inside an interactive allocation. Expected: 2 directories of 2 records, non-empty `steps`, rewards in {0, 1}, at least one web search served from `cache/serper/idc_week1/`.
- [ ] **Step 6: Full run:** `sbatch jobs/fine_tuning/015_idc_fresh_groups.job`. Expected: 500 directories of 8 records. Then run `python scripts/failure_modes/fine_tuning/all_wrong.py --rollout-dir experiments/results/idc_week1/fresh/train --out data/results/idc_week1/fresh_composition.json`. Expected: all-wrong share roughly 15 to 35% per domain. If it is far outside that range, check reward parsing before going on.
- [ ] **Step 7: Run the Task 5 CLI** on the fresh groups.
- [ ] **Step 8: Stage** code, job, test and result JSONs (not the rollout directory).

---

### Task 9: Counterfactual runner, code swap (test 1) and forcing vs resampling (test 2) (research plan tasks 1.5 and 1.6)

**Files:**
- Create: `scripts/idc/run_counterfactuals.py`, `jobs/fine_tuning/016_idc_counterfactuals.job`
- Test: `tests/unit/test_idc_targets.py` (target selection is pure and lives in `src/agent_engine/analysis/idc/targets.py`, testable in `agent_engine`)

**Interfaces:**
- Consumes: fresh groups from Task 8; `CounterfactualRollout`; `decisions.decision_options`.
- Produces:
  - `targets.swap_targets(groups, cap=300, seed=0) -> list[dict]`: every CODE dispatch of a failed rollout in mixed or all-wrong groups, `{"rollout_id", "group_type", "k"}` where `k` is the step index of that dispatch (the plan is step 0). If there are more than `cap`, take a seeded random sample.
  - `targets.force_targets(groups, seed=0) -> list[dict]`: for each all-wrong group, one seeded random rollout with non-null `steps`. Its decision points are `first` (`k = 1`) and `last` (`k = len(steps) - 1`), deduplicated when equal. For each point: `factual` option, `force` = the two other options, `resample` = 2.
  - `scripts/idc/run_counterfactuals.py --groups ... --mode swap|force|probe --out data/results/idc_week1/<mode>.jsonl` writes one JSON line per continuation: `{"group_id", "rollout_id", "domain", "mode", "point", "k", "factual", "option", "reward", "swapped"}`. In `force` mode, resampling continuations are logged with `mode: "resample"` and `option` set to the resampled turn's parsed option.
  - `probe` mode, for the policy's choice probabilities at each force target: replay `steps[:k]`, then at decision `k` request 16 samples of the turn in one chat call (`n=16`, temperature 0.7, `max_tokens=512`). Classify each by `parse_tool_call` (SEARCH or CODE) or ANSWER otherwise, and write `pi_hat = {option: count / 16}`. This Monte Carlo estimate replaces header log-probabilities, because thinking-off tool turns usually open with a `<sub_goal>` line, so header probabilities at the turn start are not meaningful.
  - The strong coder is a `CodeGeneratorTool` whose `OpenAIProvider` points at `http://localhost:9997/v1`, model `Qwen/Qwen3-32B`, temperature 0.0, `max_tokens=2048`, thinking off.

- [ ] **Step 1: Write the failing tests** for target selection:

```python
def test_force_targets_first_last_dedup():
    g = all_wrong_group([steps_of_len(3), steps_of_len(2)])
    t = force_targets([g], seed=0)
    assert {x["point"] for x in t} <= {"first", "last"}
    assert all(set(x["force"]) == set(OPTIONS) - {x["factual"]} for x in t)

def test_force_targets_single_decision_dedup():
    t = force_targets([all_wrong_group([steps_of_len(2)] * 8)], seed=0)
    assert len(t) == 1  # plan + one decision: first == last

def test_swap_targets_cap_and_seed():
    a = swap_targets(many_groups_with_code(n=400), cap=300, seed=0)
    assert len(a) == 300 and a == swap_targets(many_groups_with_code(n=400), cap=300, seed=0)

def test_swap_targets_skip_correct_and_search():
    assert swap_targets([mixed_group_where_failed_rollout_only_searches()]) == []
```

- [ ] **Step 2: Run.** Expected: FAIL on import.
- [ ] **Step 3: Implement** `targets.py`, the runner and the job. The job reuses the Task 8 server layout and runs `swap`, then `force`, then `probe`.
- [ ] **Step 4: Run tests.** Expected: pass.
- [ ] **Step 5: Smoke:** `--limit 3` per mode. Expected: JSONL lines with rewards in {0, 1}; every `swap` line has `swapped: true`; each forced line's replayed prefix matches the recorded steps (spot-check one by printing the `[ReplayProvider]` lines).
- [ ] **Step 6: Full run:** `sbatch jobs/fine_tuning/016_idc_counterfactuals.job`.
- [ ] **Step 7: Stage** code, job, tests and JSONL outputs.

---

### Task 10: Week-1 report and branch decision (research plan task 1.8 input)

**Files:**
- Create: `src/agent_engine/analysis/idc/week1_report.py`
- Test: `tests/unit/test_idc_week1_report.py`

**Interfaces:**
- Consumes: `test0.json`, `coverage.json`, `search_adequacy.json`, `swap.jsonl`, `force.jsonl`, `probe.jsonl`, fresh groups (for the decision-point check).
- Produces:
  - `test1_result(search: dict, swap_lines: list[dict], n_code_failed: int) -> dict`. Code adequacy rate `r = mean(reward)` over swap lines. Pooled share = `(adequate_search + r * n_code_failed) / (eligible_search + n_code_failed)`. `pass` is `pooled >= 0.10`, `strong` is `pooled >= 0.20`. Reported per domain and pooled.
  - `test2_result(lines: list[dict], probe: list[dict]) -> dict`. A group is recovered by forcing if any `force` line of its rollout has reward 1, and likewise for resampling. `pass` requires forcing ≥ 0.15 and forcing minus resampling ≥ 0.05. `fail` is forcing ≤ resampling or forcing < 0.10. Anything between is `borderline` and treated as fail in the table, with a flag. Also reports the share of decision points where every forced Q̂ is 0, and the median `pi_hat` of recovering options.
  - `decide(t0_pass: bool, t1_pass: bool, t2_pass: bool) -> str`, which returns exactly one of the six outcome strings of research plan Section 8.1.
  - `decision_point_check(groups) -> dict`: over tool turns in the fresh steps, the share with non-empty `other_text` from `pretext`, the median length of `other_text` in characters, and the share with a non-empty sub-goal.
  - CLI: `python -m agent_engine.analysis.idc.week1_report --results data/results/idc_week1 --groups experiments/results/idc_week1/fresh/train --out data/results/idc_week1/week1_report.md`. The markdown has one table per test, the coverage table, the decision-point numbers, the 8.1 row that fired, and a caveats list. Caveats include the binning used for test 0, any runs that could not be found, borderline flags, skipped misaligned rollouts, and the no-sub-goal prefill.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("t0,t1,t2,expected", [
    (True, True, True, "Branch A: full IDC method paper"),
    (True, True, False, "Branch A without the control target"),
    (True, False, True, "Branch B: diagnostic paper with the control target as the fix"),
    (True, False, False, "Branch B, analysis only"),
    (False, True, True, "Branch C: control-credit paper"),
    (False, False, True, "Branch C: control-credit paper"),
    (False, True, False, "Stop this line"),
    (False, False, False, "Stop this line"),
])
def test_decide_covers_all_eight(t0, t1, t2, expected):
    assert decide(t0, t1, t2) == expected

def test_test1_pooled():
    r = test1_result({"adequate": 30, "eligible": 200}, [{"reward": 1.0}] * 10 + [{"reward": 0.0}] * 40, n_code_failed=100)
    assert r["pooled"] == pytest.approx((30 + 0.2 * 100) / 300) and r["pass"]

def test_test2_borderline():
    r = test2_result(lines_with_rates(force=0.12, resample=0.05), probe=[])
    assert r["status"] == "borderline" and r["pass"] is False
```

- [ ] **Step 2: Run.** Expected: FAIL on import.
- [ ] **Step 3: Implement.** Report prose has no em dashes.
- [ ] **Step 4: Run tests.** Expected: pass. Run the CLI and read the report end to end.
- [ ] **Step 5: Stage.**

---

### Task 11: Non-code week-1 items (research plan tasks 1.7 and 1.8)

- [ ] **Reading (1.7):** TWICE in full (method, budget rule, code release), ASCT ablations, CCPO method, TAPO, APPO, StepSearch, HiPRAG, Search-R2. Confirm the ICML 2027 call for papers and deadlines. Write a half-page note on anything that changes research plan Section 3, and append it to `data/results/idc_week1/week1_report.md` under "Reading".
- [ ] **Meeting (1.8):** present `week1_report.md`; agree on the branch; settle D10 (work split) and D11 (strong coder for training: switch to Qwen3-30B-A3B only if its labels agree with 32B's on the swap set, which needs one extra `swap` run with port 9997 serving 30B-A3B).

## Order and timing

| Day | Tasks | Needs a GPU |
|---|---|---|
| Mon | 1, 2, 3 (test 0 on existing data), 4 | No |
| Tue | 5, 6, 7 | No |
| Wed | 8 (generator, smoke, submit full run) | Yes, one node, about 4 to 8 hours |
| Thu | 9 (smoke, submit), 5 CLI on fresh groups | Yes, one node, about half a day |
| Fri | 10, 11 | No |

Test 0 and coverage give a first signal on Monday. If test 0 fails in every domain and coverage shows no missing options, raise it with Xinyi before spending GPU time: by Section 8.1 only Branch C remains, and it depends on test 2 alone.
