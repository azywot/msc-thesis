# Tool-Call Intervention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect Qwen3-8B trajectories (arm C), rescue failed ones by having a reflector fix one orchestrator tool call, gate on the net rescue rate, and train and evaluate arm C vs arm I (arm C plus rescued rows) with orchestrator-only LoRA.

**Architecture:** The orchestrator gains a `resume_batch` entry point that continues a trajectory from a rebuilt state, optionally injecting one action turn first. A new `src/rsi/intervention/` package turns collected JSONL into resumable states, asks the reflector for a fix, guards leakage, runs fix and no-fix control continuations, and reports the gate. Dataset building reuses `build_sft_parquet.py`'s folding; training and evaluation reuse the existing LoRA SFT job and the `lora_inference` config shape.

**Tech Stack:** Python 3.11, vLLM, pydantic, pandas, pytest, verl SFT (existing job), SLURM on Snellius.

**Spec:** `docs/superpowers/specs/2026-11-17-tool-call-intervention-design.md`

## Global Constraints

- Team: Qwen3-8B (`Qwen/Qwen3-8B`) in every role, `thinking_mode: "NO"`, `max_turns: 7`, tools `web_search` + `code_generator`, `direct_tool_call: false`, `web_tool_provider: "serper"`, `max_search_limit: 10` (copied from `experiments/configs/rsi/phase1/search_reflector_qwen3_8b.yaml`).
- Reflector: same model, `enable_thinking: true`, `max_model_len: 32768`, `max_tokens: 16384`, `input_budget_tokens: 14000`, `per_call_chars: 1500`; 32B fallback via the `openai` backend.
- Collection `--n-samples 3`; candidates are questions with 0 of 3 correct; at most 2 reflector attempts per question; one intervention per trajectory.
- Gate: net rescue rate (rescue rate minus no-fix control rate) of at least **10 points** in at least one domain.
- Seed prompts only. With no edits, every prompt the orchestrator sees must stay byte-identical to today's; `run_batch` behaviour must not change.
- Orchestrator-only training rows; no sub-agent rows; adapter unmerged, loaded on the orchestrator only at evaluation, sub-agents base Qwen3-8B.
- LoRA SFT with `jobs/fine_tuning/007_train_sft_folded.job` hyperparameters unchanged (`total_epochs=2`, `max_length=16384`) for both arms.
- Code style: `black` line length 100, `isort` black profile. Docs: plain English, no em dashes, no thesis chapter references.
- Commits: one local commit per task after review, message ending with the two attribution lines below. Never push.

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CTPoR7e2za4PHEXHgi5fAo
```

## Review Focus

1. **Short numeric gold answers** (math golds like `2`) appearing in a numbered code task (`1. ... 2. ...`): a reasonable person expects list numbering not to count as leakage, but `print the result, which is 2` to be rejected. Test in Task 4.
2. **Messy reflector replies** (thinking block, prose around the tags, two `<tool_call>` blocks, `\frac` inside JSON): expected to parse the last complete block and keep LaTeX intact, or fail cleanly as a parse error that counts as an attempt. Test in Task 4.
3. **A trajectory that already used all 7 turns:** a fix at step 7 has no turn left; the step must be capped at `max_turns - 1` and resuming at the cap must still produce an answer through the synthesis nudge. Tests in Tasks 1 and 4.
4. **A rescue job killed mid-attempt:** rerunning must skip finished `(question_id, attempt)` rows, never write duplicates, and never re-ask the reflector for a question already rescued. Test in Task 5.
5. **A rescued trajectory that does not fold as expected** (folded rows do not line up with `1 + step`, or a row exceeds `--max-length`): expected to be dropped and counted, never silently trained with the wrong target. Test in Task 7.

---

### Task 1: Orchestrator `resume_batch`

**Files:**
- Modify: `src/agent_engine/core/orchestrator.py` (`run_batch`, lines ~251-339)
- Test: `tests/unit/test_orchestrator_resume.py`

**Interfaces:**
- Produces: `AgenticOrchestrator.resume_batch(self, states: List[ExecutionState], injected_outputs: Sequence[Optional[str]]) -> List[ExecutionState]`. Each state arrives with `messages=[system, user]`, `query_analysis`, `action_history`, `output_messages`, `tool_calls`, `tool_counts`, `turn` already set (Task 3 builds them). For a non-`None` injected text the orchestrator spends one turn on it exactly as if the model had generated it (same tool path, same commit, sub-agent calls logged with the right `action_index`), then continues the normal loop and the max-turns synthesis. `None` means continue directly. Raises `ValueError` if lengths differ, a state is finished, an injected text has no parsable tool call, or a state with injected text has `turn >= max_turns`.
- Produces (private): `_finish_batch(self, states) -> None`, the loop-plus-synthesis tail of `run_batch`, now shared by both entry points.

- [ ] **Step 1: Write the failing tests** (fake provider and fake batched-free tool in the style of `tests/unit/test_acting_suffix_and_logging.py`)

```python
def test_run_batch_output_unchanged_after_refactor(): ...
    # scripted fake model: plan, one tool call, answer; assert answer, turn == 2,
    # len(action_history) == 1, output_messages roles == ["assistant", "tool", "assistant"]

def test_resume_injects_call_then_continues():
    # state with 1 prior action, turn=1; injected = '<sub_goal>g</sub_goal>\n\n<tool_call>\n{"name": "web_search", "arguments": {"query": "q2"}}\n</tool_call>'
    # fake model then emits "\\boxed{X}"
    assert st.answer == "X"
    assert st.turn == 3
    assert st.action_history[1]["sub_goal"] == "g"
    assert json.loads(st.action_history[1]["command"])["arguments"] == {"query": "q2"}
    assert st.output_messages[2] == {"role": "assistant", "content": injected}
    assert "Action Step 2:" in fake_model.prompts_seen[0]   # model saw the injected step

def test_resume_without_injection_generates_from_prefix():
    # injected None: first generated prompt's user turn has exactly the prefix actions

def test_resume_at_last_turn_uses_synthesis():
    # max_turns=7, turn=6, injected call; fake model never answers
    assert st.metadata["max_turns_reached"] is True and st.answer is not None and st.turn == 7

def test_resume_rejects_bad_input():
    # injected text without a tool call -> ValueError; turn == max_turns with injection -> ValueError
```

- [ ] **Step 2: Run them; expect failures** `pytest tests/unit/test_orchestrator_resume.py -v` → FAIL (`resume_batch` missing; the first test passes already).
- [ ] **Step 3: Extract `_finish_batch` from `run_batch` unchanged, then add `resume_batch`.** The injection step reuses the per-state tool routing of `_process_batch_turn` (`_classify_tool_call`, `_apply_immediate_results`, `flush_batches`) on the injected text instead of a generated one, for all injected states in one batch.
- [ ] **Step 4: Run** `pytest tests/unit/test_orchestrator_resume.py tests/unit tests/rsi -q` → all pass (1,231 before this plan, plus the new ones).
- [ ] **Step 5: Commit** `feat(core): resume_batch continues a trajectory from a rebuilt state`

---

### Task 2: Collection with per-pass seeds and resumable records

**Files:**
- Create: `src/rsi/intervention/__init__.py`, `src/rsi/intervention/records.py` (only `trajectory_messages` and `set_sampling_seed` in this task)
- Modify: `scripts/collect_sft_data.py`
- Create: `experiments/configs/rsi/phase2/collect_qwen3_8b.yaml`, `jobs/rsi/004_collect_8b.job`
- Test: `tests/rsi/test_collect_records.py`

**Interfaces:**
- Produces: `trajectory_messages(state: ExecutionState) -> list[dict]` (moved from `_build_sft_messages`, which now delegates to it; output identical).
- Produces: `set_sampling_seed(providers: Iterable, seed: int) -> None`: sets `config.seed` once per distinct provider object. `VLLMProvider.generate` reads `config.seed` per request, so this changes sampling for every later request.
- Produces: `collect_sft_data.make_record(row: dict, state: ExecutionState, pass_idx: int, seed: int, aliases: list[str], prediction: str, correct: bool) -> dict`, with the existing keys plus `"aliases"`, `"seed"`, `"query_analysis"`, `"raw_query_analysis"`, `"action_history"`, `"subagent_calls"` (from `state.metadata`, default `[]`).
- Collection behaviour: pass `k` (0-based) runs with seed `config.seed + k`; grading uses `evaluate_answer(prediction, str(row["result"]), answer_aliases=aliases)` with aliases from `extra_info["golden_answers"]`, as `rsi/data.py` does.

- [ ] **Step 1: Write the failing tests**

```python
def test_trajectory_messages_matches_old_builder():   # plan + 1 action + answer state
def test_set_sampling_seed_sets_each_provider_once(): # two roles sharing one provider, one separate
def test_make_record_has_resume_fields():
    assert rec["query_analysis"] == st.query_analysis and rec["action_history"] == st.action_history
    assert rec["subagent_calls"] == st.metadata["subagent_calls"] and rec["seed"] == 3
def test_passes_use_distinct_seeds(monkeypatch):      # main() with fakes, --n-samples 3
    assert seeds_seen_per_pass == [0, 1, 2]
def test_grading_accepts_alias():                     # gold "NYC", alias "New York City"
```

- [ ] **Step 2: Run** `pytest tests/rsi/test_collect_records.py -v` → FAIL.
- [ ] **Step 3: Implement** the helpers and wire them into `collect_sft_data.py` (seed set at the start of each pass; record built by `make_record`).
- [ ] **Step 4: Add the config and job.** Config = the Phase 1 8B config's `name`/`models`/`tools`/`max_turns`/`thinking_mode`/`seed`/`cache_dir`, no `search` block, `batch_size: 100`, no `system_prompt_suffix`. Job: 1 H100, 24 h, copy the environment setup of `jobs/fine_tuning/006_collect_sft_data.job`, run `python scripts/collect_sft_data.py --config experiments/configs/rsi/phase2/collect_qwen3_8b.yaml --output-dir data/training/rsi_phase2/collect_8b --n-samples 3`, log to `out/rsi/collect_8b_%A.log`.
- [ ] **Step 5: Run** `pytest tests/rsi tests/unit -q` → PASS. `bash -n jobs/rsi/004_collect_8b.job` → no output.
- [ ] **Step 6: Commit** `feat(rsi): 8B collection with per-pass seeds and resumable records`

---

### Task 3: Candidates and resume states

**Files:**
- Modify: `src/rsi/intervention/records.py`
- Test: `tests/rsi/test_intervention_records.py`

**Interfaces:**
- Consumes: Task 2's record keys; `rsi.data.DOMAIN_SOURCES`.
- Produces:
  - `load_collected(paths: Sequence[Path]) -> list[dict]`, deduplicated by `(question_id, pass)`, first occurrence wins.
  - `domain_of(data_source: str) -> str` (`"search"` or `"math"`).
  - `intervention_candidates(records: list[dict], n_passes: int) -> tuple[list[dict], dict]`: the pass-1 record of every question with exactly `n_passes` records, none correct. Skips, and counts per domain, records with an `"error"` key or empty `messages` (`"skipped_error"`) and records with `turns == 0` (planning answered, `"skipped_planning_answer"`). Second value: `{domain: {"zero_of_n": int, "skipped_error": int, "skipped_planning_answer": int, "candidates": int}}`.
  - `state_from_record(record: dict, step: int) -> ExecutionState`: `messages = record["messages"][:2]`; `query_analysis`, `raw_query_analysis` from the record; `action_history[:step]`; `output_messages` = the first `2 * step` messages after the planning message (the planning message is present iff `raw_query_analysis` is non-empty); `tool_calls` from each kept action's `command`; `tool_counts` recomputed from them; `turn = step`; `metadata = {"subagent_calls": [c for c in calls if c["action_index"] < step], "intervention_step": step}`. Raises `ValueError` if `step` is outside `[0, len(action_history)]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_candidates_need_all_passes_wrong():   # q1: 3 wrong -> candidate; q2: 2 wrong + 1 right -> not; q3: 2 records -> not
def test_candidate_is_pass_one_record():
def test_skips_counted_per_domain():           # turns == 0 and error records
def test_state_from_record_prefix():
    st = state_from_record(rec_with_3_actions, 2)
    assert st.turn == 2 and len(st.action_history) == 2 and len(st.output_messages) == 4
    assert st.tool_counts["web_search"] == 2
    assert AgenticOrchestrator(...)._build_memory_prompt(st, sys)[1]["content"] == \
        build_sft_parquet._memory_user_content(q, qa, rec["action_history"][:2])
def test_state_from_record_step_zero_and_out_of_range():
```

- [ ] **Step 2: Run** `pytest tests/rsi/test_intervention_records.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(rsi): intervention candidates and resume states from collected records`

---

### Task 4: The fix: reflector prompt, parsing, leakage

**Files:**
- Create: `src/rsi/intervention/fix.py`
- Modify: `src/rsi/leakage.py` (add `exempt_text`)
- Test: `tests/rsi/test_intervention_fix.py`, `tests/rsi/test_leakage.py` (one new test)

**Interfaces:**
- Consumes: `reflector_input._render_trajectory`, `reflector_input.LEAKAGE_INSTRUCTION`, `reflector.restore_latex_escapes`, `executor.TrajectoryRecord`, `agent_engine.analysis.failure_modes.classify_failure`, `agent_engine.utils.parsing.parse_tool_call`, `strip_thinking_tags`.
- Produces:
  - `TOOL_ARGS = {"web_search": "query", "code_generator": "task"}`
  - `@dataclass(frozen=True) class Fix: step: int; sub_goal: str; tool_call: dict` with property `text -> str` = `f"<sub_goal>{sub_goal}</sub_goal>\n\n<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"` (the orchestrator's own action format) and method `kind(n_actions: int) -> str` returning `"rewrite"` if `step < n_actions`, `"follow_up"` if `step == n_actions >= 1`, `"insert"` if `step == n_actions == 0`.
  - `class FixParseError(ValueError)`
  - `parse_fix(text: str, n_actions: int, max_turns: int) -> Fix`. Reply format the reflector is asked for: `<step>N</step>`, `<sub_goal>...</sub_goal>`, `<tool_call>{json}</tool_call>`. Strip thinking; use the last complete block of each tag; `step` an int in `[0, min(n_actions, max_turns - 1)]`; tool name in `TOOL_ARGS`; arguments exactly `{TOOL_ARGS[name]: non-empty str}` after `restore_latex_escapes`; sub-goal non-empty, at most 500 chars.
  - `to_trajectory_record(record: dict, show_gold: bool) -> TrajectoryRecord` (gold rendered as `"(hidden)"` when `show_gold` is false).
  - `build_fix_messages(record: dict, *, show_gold: bool = True, previous: Optional[dict] = None, per_call_chars: int = 1500, count_tokens: Optional[Callable[[list], int]] = None, budget_tokens: int = 14000) -> list[dict]`. System message: the task (find the first step that went wrong; return one corrected tool call; the three kinds; only the two tools and their one argument; the reply format) plus `LEAKAGE_INSTRUCTION`. User message: the rendered trajectory (`cut_actions=False`); when `previous = {"fix_text": str, "prediction": str}` is given, a closing section saying that fix was tried and the trajectory then answered `prediction`, which was wrong. If `count_tokens` reports more than `budget_tokens`, halve `per_call_chars` down to a floor of 200, then cut middle actions (`cut_actions=True`).
  - `fix_leakage_errors(fix: Fix, question: str, answers: Sequence[str], index: LeakageIndex) -> list[str]`. (a) `index.check([fix.sub_goal, arg], [], exempt_text=question)`. (b) For each answer: normalise with `normalize_answer`; skip empty and `GENERIC_ANSWERS`; skip if its token sequence occurs in the normalised question; else reject if it occurs as a token sequence in the normalised sub-goal or argument after removing line-start list numbering (`^\s*\d+[.)]\s+` per line). No length exemption.
- `LeakageIndex.check(self, texts, shown_answers, exempt_text=None)`: an n-gram that also occurs in `exempt_text` is not reported. Default `None` keeps today's behaviour.

- [ ] **Step 1: Write the failing tests**

```python
def test_fix_text_round_trips_through_orchestrator_parsers():
    f = Fix(1, "find X", {"name": "web_search", "arguments": {"query": "X born"}})
    assert parse_tool_call(f.text) == f.tool_call and AgenticOrchestrator._extract_sub_goal(f.text) == "find X"
def test_kind_labels(): assert [Fix(0,..).kind(0), Fix(1,..).kind(1), Fix(0,..).kind(2)] == ["insert", "follow_up", "rewrite"]
def test_parse_fix_messy_reply():          # <think> block, prose, two tool_call blocks -> last; "\\frac{1}{2}" survives
def test_parse_fix_rejects():              # step > n_actions; step == max_turns; unknown tool; extra arg; empty query
def test_parse_fix_caps_step_at_max_turns_minus_one():  # n_actions=7, max_turns=7, step 7 -> FixParseError, step 6 ok
def test_leak_numbering_is_not_leakage():  # gold "2", task "1. compute x\n2. print x" -> []
def test_leak_short_gold_in_text():        # gold "2", task "print x, which should be 2" -> rejected
def test_leak_gold_in_question_exempt():   # gold "Paris", question mentions Paris -> []
def test_leak_alias_rejected():            # alias "New York City" in query -> rejected
def test_own_question_ngrams_exempt_other_rejected():
def test_build_fix_messages_hides_gold():  # show_gold False -> gold string absent, "(hidden)" present
def test_build_fix_messages_previous_attempt():
def test_build_fix_messages_shrinks_to_budget():
# tests/rsi/test_leakage.py
def test_exempt_text_suppresses_shared_ngrams():
```

- [ ] **Step 2: Run** `pytest tests/rsi/test_intervention_fix.py tests/rsi/test_leakage.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/rsi -q` → PASS.
- [ ] **Step 5: Commit** `feat(rsi): reflector fix prompt, parsing and leakage guard for tool-call interventions`

---

### Task 5: Rescue loop with no-fix control

**Files:**
- Create: `src/rsi/intervention/rescue.py`
- Modify: `src/rsi/reflector.py` (add `sample_batch`)
- Test: `tests/rsi/test_intervention_rescue.py`, `tests/rsi/test_reflector.py` (one new test)

**Interfaces:**
- Consumes: Tasks 1, 3, 4; `ReflectorSettings`; `evaluate_answer`.
- Produces:
  - `InProcessReflector.sample_batch(self, messages_list: list[list[dict]], seeds: list[int]) -> list[dict]` (one `llm.generate` over all prompts) and `ReflectorClient.sample_batch` (same signature, sequential). Items have the same shape as `sample`.
  - `class RescueSettings(BaseModel): max_attempts: int = 2; attempt_seed_base: int = 1000; per_call_chars: int = 1500; input_budget_tokens: int = 14000`
  - `run_rescue(candidates: list[dict], *, orchestrator, providers: Sequence, reflector, index: LeakageIndex, settings: RescueSettings, out_path: Path, max_turns: int, count_tokens: Optional[Callable] = None) -> list[dict]`. Returns all rows in `out_path` (JSONL, appended and flushed per attempt batch).
- Algorithm, per attempt `a` in `1..max_attempts`:
  1. `pending` = candidates with no correct row yet and no row for `(question_id, a)` in `out_path`.
  2. Ask the reflector once per pending question (`sample_batch`, seed `attempt_seed_base + a`), passing `previous` from that question's attempt `a - 1` row when it has `fix_text`.
  3. Parse and leakage-check. Status `"ok"`, `"parse_error"` or `"leakage"` (with `reasons`).
  4. Control step = the fix's `step` when parsing succeeded, else `min(n_actions, max_turns - 1)`.
  5. `set_sampling_seed(providers, attempt_seed_base + a)`; `resume_batch` the `"ok"` fixes with their `text`; `resume_batch` every pending question at its control step with `None`. Same seed for fix and control (paired).
  6. Grade both with `evaluate_answer(prediction, gold, answer_aliases=aliases)`.
  7. Append one row per pending question: `question_id, data_source, domain, attempt, seed, status, reasons, reflector_text, step, kind, fix_text, correct, prediction, messages` (`trajectory_messages` of the fix state, `[]` unless `"ok"`), `control_step, control_correct, control_prediction`.

- [ ] **Step 1: Write the failing tests** (fake orchestrator recording `resume_batch` calls, fake reflector with scripted replies)

```python
def test_rescued_on_first_attempt_not_retried():
def test_second_attempt_sees_previous_fix():          # attempt-2 prompt contains attempt-1 fix_text
def test_parse_error_and_leakage_still_run_control(): # control_step == min(n_actions, max_turns - 1) for parse_error
def test_fix_and_control_share_seed():                # seeds set == [1001, 1001] for attempt 1
def test_resume_after_crash_skips_done_rows(tmp_path): # pre-seed out_path with attempt-1 rows; reflector not called for attempt 1; no duplicates
def test_rows_have_expected_fields():
# tests/rsi/test_reflector.py
def test_in_process_sample_batch_one_generate_call():
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/rsi -q` → PASS.
- [ ] **Step 5: Commit** `feat(rsi): rescue loop with paired no-fix control`

---

### Task 6: Gate report, blind check, CLI, configs and jobs

**Files:**
- Create: `src/rsi/intervention/report.py`, `scripts/run_call_intervention.py`, `experiments/configs/rsi/phase2/rescue_qwen3_8b.yaml`, `experiments/configs/rsi/phase2/rescue_qwen3_32b.yaml`, `jobs/rsi/005_rescue_8b.job`, `jobs/rsi/006_rescue_32b.job`
- Test: `tests/rsi/test_intervention_report.py`, `tests/rsi/test_run_call_intervention_cli.py`

**Interfaces:**
- Produces:
  - `summarize(candidates: list[dict], rows: list[dict], counts: dict, threshold_points: float = 10.0, blind_rows: Optional[list[dict]] = None) -> dict`: per domain `candidates, rescued, rescue_rate, control_solved, control_rate, net_points` (rates in percent; a question is rescued or control-solved if any of its attempts is), `attempts_used` (`{"1": n, "2": n}` among rescued), `parse_errors`, `leakage_rejections`, `leakage_by_reason`, `kinds` (`rewrite`/`follow_up`/`insert` among rescued), plus the `counts` skips; top level `gate = {"threshold_points": 10.0, "pass": bool, "passing_domains": [...]}` and, when `blind_rows` is given, `blind = {"n": int, "rescued": int, "rate": float}`.
  - `write_gate(out_dir: Path, summary: dict) -> None` → `gate.json`, `gate.md` (one table per domain, gate verdict first).
  - `blind_sample(rows: list[dict], n: int, seed: int = 0) -> list[int]`: question ids of rescued questions, seeded sample.
  - `InterventionConfig` (pydantic, in the CLI module): `data_path`, `n_passes: int = 3`, `rescue: RescueSettings`, `threshold_points: float = 10.0`, `blind_n: int = 100`, `ngram: int = 8`, `reflector: ReflectorSettings`, loaded from the YAML's `intervention:` block the way `run_prompt_search.load_search_config` loads `search:`.
- CLI: `python scripts/run_call_intervention.py --config <yaml> --collected <jsonl...> --out-dir <dir> [--subset N] [--report-only] [--blind]`. Writes `candidates.json`, `rescue.jsonl`, `gate.json`, `gate.md`; `--blind` runs one hidden-gold attempt (seed `attempt_seed_base + 99`) on `blind_sample(rows, blind_n)` into `blind.jsonl` and re-writes the gate files. Provider, tool and token-counter setup reuse `run_prompt_search.py`'s helpers (`_api_keys`, `_build_tools`, its in-process `count_tokens`). The leakage index covers every question in `data_path`.
- Configs: 8B = the Task 2 collection config plus `intervention:` with the Global Constraints reflector values and `data_path: data/training/train/combined_train.parquet`. 32B = same, reflector `backend: "openai"`, `model: "Qwen/Qwen3-32B"`, `base_url` as in `search_reflector_qwen3_32b.yaml`.
- Jobs: `005` = 1 H100, `--collected data/training/rsi_phase2/collect_8b/collected_*.jsonl --out-dir experiments/results/rsi/phase2/rescue_8b`, `BLIND=1` env adds `--blind`. `006` = `002_prompt_search_32b.job`'s reflector-server setup with this CLI and `--out-dir experiments/results/rsi/phase2/rescue_32b`.

- [ ] **Step 1: Write the failing tests**

```python
def test_summarize_net_points_and_gate():
    # search: 10 candidates, 4 rescued, 1 control-solved -> rescue 40.0, control 10.0, net 30.0
    # math: 10 candidates, 1 rescued, 1 control -> net 0.0
    assert s["gate"] == {"threshold_points": 10.0, "pass": True, "passing_domains": ["search"]}
def test_summarize_counts_attempts_kinds_and_rejections():
def test_gate_fails_below_threshold():     # net 9.9 in both -> pass False
def test_blind_sample_deterministic_and_rescued_only():
def test_cli_dry_end_to_end(tmp_path, monkeypatch):  # fakes for providers/reflector; asserts the four output files and gate.json keys
def test_cli_report_only_does_not_load_models(tmp_path, monkeypatch):
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/rsi -q` → PASS; `bash -n jobs/rsi/005_rescue_8b.job jobs/rsi/006_rescue_32b.job` → no output.
- [ ] **Step 5: Commit** `feat(rsi): rescue CLI, go/no-go report and blind-reflector check`

---

### Task 7: Arm C and arm I datasets

**Files:**
- Create: `scripts/build_arm_i_parquet.py`
- Test: `tests/rsi/test_build_arm_i.py`

**Interfaces:**
- Consumes: `build_sft_parquet._fold_trajectory`, `_strip_thinking`, `_default_planning_suffix`, `_drop_over_length`, `_extra_info_by_qid`, `_category`; Task 5 rows.
- Arm C is built by the existing script, unchanged: `python scripts/build_sft_parquet.py data/training/rsi_phase2/collect_8b/collected_*.jsonl --output-dir data/training/rsi_phase2/arm_c --output-name arm_c_folded_train.parquet --format folded --collection-config '' --reference-parquet data/training/train/combined_train.parquet --max-length 16384` → `arm_c_folded_train.parquet`, `arm_c_folded_val.parquet`.
- Produces:
  - `rescued_rows(row: dict, planning_suffix: str) -> Optional[list[list[dict]]]`: strip thinking, fold `row["messages"]`, return `rows[1 + row["step"]:]`. Returns `None` (caller counts it as `"misaligned"`) unless the fold has a planning row and `rows[1 + step]`'s assistant content equals `row["fix_text"]`.
  - `build_arm_i(arm_c_train: pd.DataFrame, rescue_rows: list[dict], extra_info: dict, planning_suffix: str) -> tuple[pd.DataFrame, dict]`: one rescued trajectory per question (the first correct `"ok"` attempt); new rows carry `data_source, question, result, extra_info, messages` like arm C's; output = arm C rows followed by the new rows. Raises `ValueError` if a rescued question appears in arm C. Stats: per arm `rows`, `questions`, `math_rows`, `search_rows`, `no_tool_rows` (assistant target without `<tool_call>`), and for arm I `rescued_questions`, `rescued_rows`, `by_kind`, `by_domain`, `misaligned`, `dropped_over_length`.
- CLI: `python scripts/build_arm_i_parquet.py --arm-c-dir data/training/rsi_phase2/arm_c --rescue experiments/results/rsi/phase2/rescue_8b/rescue.jsonl --reference-parquet data/training/train/combined_train.parquet --output-dir data/training/rsi_phase2/arm_i --max-length 16384 --tokenizer Qwen/Qwen3-8B` → `arm_i_folded_train.parquet`, `arm_i_folded_val.parquet` (byte copy of arm C's val), `arm_stats.json`. Over-length filtering applies to the new rows only.

- [ ] **Step 1: Write the failing tests**

```python
def test_rescued_rows_start_at_fix():             # step=1 of 2 original actions -> first kept row target == fix_text
def test_rescued_rows_insert_at_zero():           # step 0 -> rows[1:] kept; planning row dropped
def test_misaligned_trajectory_returns_none():    # fix_text not at rows[1+step]
def test_build_arm_i_appends_and_keeps_arm_c():   # arm C rows identical and first; val untouched
def test_one_trajectory_per_rescued_question():   # two correct attempts -> first kept
def test_overlap_with_arm_c_raises():
def test_stats_counts():                          # kinds, domains, no_tool_rows
```

- [ ] **Step 2: Run** `pytest tests/rsi/test_build_arm_i.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest tests/rsi tests/unit -q` → PASS.
- [ ] **Step 5: Commit** `feat(rsi): arm I dataset builder on top of the arm C folded build`

---

### Task 8: Parameterise the SFT job for two arms

**Files:**
- Modify: `jobs/fine_tuning/007_train_sft_folded.job` (lines ~138-164 and wherever the pre-flight check names data files)
- Test: `tests/unit/test_train_job_overrides.py`

**Interfaces:**
- `TRAIN_FILE`, `VAL_FILE`, `EXPERIMENT_NAME` become `VAR="${VAR:-<current default>}"`. Any pre-flight `check_sft_folded_format.py` call uses `$TRAIN_FILE`; its `--native` argument is passed only when `NATIVE_FILE` is set (default: the current native path, so the existing run is unchanged). Arm runs: `sbatch --export=ALL,TRAIN_FILE=data/training/rsi_phase2/arm_c/arm_c_folded_train.parquet,VAL_FILE=data/training/rsi_phase2/arm_c/arm_c_folded_val.parquet,EXPERIMENT_NAME=qwen3-8b-rsi-arm-c,NATIVE_FILE= jobs/fine_tuning/007_train_sft_folded.job` and the same with `arm_i` / `qwen3-8b-rsi-arm-i`.

- [ ] **Step 1: Write the failing test**: extract the assignment lines with `grep -E '^(TRAIN_FILE|VAL_FILE|EXPERIMENT_NAME|NATIVE_FILE)='`, source them in `bash` with and without the env vars, assert defaults equal today's values and overrides win.
- [ ] **Step 2: Run** `pytest tests/unit/test_train_job_overrides.py -v` → FAIL.
- [ ] **Step 3: Edit the job.**
- [ ] **Step 4: Run** the test and `bash -n jobs/fine_tuning/007_train_sft_folded.job` → PASS, no output.
- [ ] **Step 5: Commit** `feat(jobs): SFT folded job takes data files and experiment name from the environment`

---

### Task 9: Evaluation configs, GPU assignment fix and arm comparison

**Files:**
- Modify: `src/agent_engine/models/vllm_provider.py` (`resolve_gpu_assignments`), `src/agent_engine/runner/experiment.py:143-155`, `scripts/collect_sft_data.py` (the same lookup), `scripts/generate_configs.py`
- Create: `scripts/compare_arms.py`
- Test: `tests/unit/test_gpu_assignment_lora.py`, `tests/unit/test_generate_rsi_inference_configs.py`, `tests/rsi/test_compare_arms.py`

**Interfaces:**
- `model_instance_key(cfg) -> str` in `vllm_provider.py` = `f"{cfg.path_or_id}|lora:{cfg.lora_adapter_path or ''}"` (the format `runner/providers.py` caches by). `resolve_gpu_assignments` counts and keys distinct instances by it, not by path; the two consumers look up by it. Reason: the evaluation config loads a LoRA orchestrator and base sub-agents from the same `Qwen/Qwen3-8B`, which become two vLLM engines; keyed by path both get `util=0.95` on the same GPU.
- `generate_configs.py`: `VARIANTS_RSI_INFERENCE = [("qwen8B_sub8B_none", "8B", "8B", "NO")]`; suites `rsi_arm_c_inference` and `rsi_arm_i_inference` with `variant_type: "lora_inference"`, `config_subdir: "qwen3/rsi_inference/arm_c"` / `".../arm_i"`, `output_dir_root: "./experiments/results/rsi_inference/arm_c"` / `".../arm_i"`, `name_prefix: "RSI_ArmC_eval"` / `"RSI_ArmI_eval"`, `adapter_label: "ArmC"` / `"ArmI"`, `train_job: "007_train_sft_folded.job"`, `lora_adapter_path: "/gpfs/home3/xchen1/azywot/msc-thesis/data/adapters/qwen3-8b-rsi-arm-c/<run-tag>/best_adapter"` (and `arm-i`), datasets `["gaia", "hle", "gpqa", "aime", "musique"]`, `num_gpus: 2`.
- `compare_arms.py`: `load_run(path: Path) -> dict[int, dict]` (by `question_id` from `raw_results.json`); `compare(base: dict, arm_c: dict, arm_i: dict, benchmark: str) -> dict` with accuracy per arm, `mcnemar_p` (from `rsi.confirm`) for arm I vs arm C on shared ids, and behaviour: `code_call_rate` (share with `tool_counts["code_generator"] >= 1`) for `aime`, `multi_search_rate` (share with `tool_counts["web_search"] >= 2`) for `musique` and `gaia`. CLI takes `--base-root`, `--arm-c-root`, `--arm-i-root` (each `<root>/<benchmark>/<variant>/<run>/raw_results.json`, newest run) and writes `comparison.json` and `comparison.md` with the five-benchmark mean and the verdict `arm I mean > arm C mean`.

- [ ] **Step 1: Write the failing tests**

```python
def test_lora_and_base_same_path_get_separate_gpus(monkeypatch):   # 2 GPUs -> orch key [0], sub key [1]
def test_single_model_unchanged(monkeypatch):                      # existing behaviour, keyed by instance key
def test_rsi_suites_generate_ten_configs(tmp_path):                # 5 per arm; orchestrator has lora_adapter_path; subs Qwen/Qwen3-8B without it; thinking_mode "NO"
def test_rsi_configs_match_base_reference_splits():                # dataset.split equals experiments/configs/qwen3/agentflow/<bench>/qwen8B_subagent_tools_none.yaml
def test_compare_mean_mcnemar_and_behaviour(tmp_path):
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**, then `python scripts/generate_configs.py --suite rsi_arm_c_inference` and `--suite rsi_arm_i_inference`.
- [ ] **Step 4: Run** `pytest tests -q` → PASS. If `test_rsi_configs_match_base_reference_splits` fails, the base reference cannot be reused; add a `rsi_base_inference` suite (same shape, no adapter) and say so in the runbook.
- [ ] **Step 5: Commit** `feat(eval): arm C / arm I inference configs, per-instance GPU assignment, arm comparison`

---

### Task 10: Runbook

**Files:**
- Create: `docs/pipelines/rsi_intervention.md`
- Modify: `docs/pipelines/rsi_overview.md` (one line under next steps pointing to the new runbook; leave the existing uncommitted edits in that file as they are and commit them together only if the user agrees)

- [ ] **Step 1: Write the runbook**: the order (collect → build arm C → rescue → gate → [32B rescue if the gate fails] → blind check → build arm I → format check on both → train both → generate configs with the run tags → evaluate → `compare_arms.py`), the exact commands from Tasks 2 and 5 to 9, where each output lands, what the gate means, and a one-question GPU smoke run (`subset_num: 1`) of one arm C config before the full evaluation.
- [ ] **Step 2: Check** `grep -c "—" docs/pipelines/rsi_intervention.md` → `0`.
- [ ] **Step 3: Commit** `docs(rsi): tool-call intervention runbook`
