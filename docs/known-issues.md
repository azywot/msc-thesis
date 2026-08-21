# Known issues

Defects found while writing characterization tests during the repository
handover refactor (2026-08-16). They are recorded here rather than fixed,
because that refactor's contract was *no behaviour change* - fixing any of
these would move results that the thesis already reports.

Each entry names the test that pins the current behaviour, so the fix and the
test change land together.

---

## 1. BigCodeBench: a bare function body loses its indentation

**Severity:** high for BigCodeBench runs, none for other benchmarks.
**Status:** open. Pinned by `tests/unit/test_bigcodebench_scorer.py::test_a_bare_body_is_appended_after_the_stub` (`xfail(strict=True)`).

`_strip_markdown_fences` in `src/agent_engine/datasets/evaluators/bigcodebench_scorer.py`
ends in `.strip()`, applied both to the captured fence group and to the
unfenced fallback. `.strip()` removes leading whitespace from the *first* line:

```
"    return n * 2"          ->  "return n * 2"
"    x = 1\n    return x"   ->  "x = 1\n    return x"
```

`evaluate_bigcodebench` then takes the `else` branch of the full-definition
check and builds `code_prompt.rstrip() + "\n" + impl`, producing

```python
def task_func(n):
return n * 2
```

which raises `IndentationError` before any test runs. The task scores 0
regardless of whether the code was right.

This branch exists specifically to handle a prediction that is a bare function
body, so that whole path is unusable today. Only the full-definition branch -
where the model emits `def task_func(...)` itself - works, which is the common
case under the current prompts and is why the defect has not shown up in
reported numbers.

**Fix sketch:** dedent-preserving extraction - return the fence group with
trailing whitespace stripped and leading *newlines* stripped, but not leading
spaces (`group(1).strip("\n").rstrip()`). Re-scoring any affected BigCodeBench
run afterwards is required before comparing to earlier numbers.

---

## 2. `question_scorer` raises on a `None` ground truth

**Severity:** low - no loader produces one today.
**Status:** open by design (upstream parity). Pinned by `tests/unit/test_gaia_scorer.py::test_a_none_ground_truth_raises`.

`is_float` in `src/agent_engine/datasets/evaluators/gaia_scorer.py` catches only
`ValueError`, but `float(None)` raises `TypeError`. `question_scorer` calls
`is_float(ground_truth)` first, so a dataset row with a null answer aborts
scoring for the whole run instead of counting that row wrong.

The module header states it is an "exact copy from multi-agent-tools ... to
ensure consistent scoring", so diverging from upstream here is a deliberate
decision, not an obvious win. If a benchmark with nullable answers is ever
added, guard at the loader instead.

---

## 3. A thousands separator inside a list answer splits the number

**Severity:** low.
**Status:** open by design (upstream parity). Pinned by `tests/unit/test_gaia_scorer.py::test_a_thousands_separator_inside_a_list_answer_splits_the_number`.

`split_string` treats `,` as a delimiter and runs *before* the per-element
`normalize_number_str` that would have stripped it. So `"$1,000; 2"` splits into
three elements, fails the length check against a two-element ground truth, and
scores wrong for a formatting choice. Only affects list-valued ground truths; a
bare `"$1,000"` against `"1000"` takes the numeric path and passes.

---

## 4. `CacheManager.save_caches` skips the normalisation `save_search_cache` applies

**Severity:** cosmetic - self-correcting on the next load.
**Status:** open. Pinned by `tests/unit/test_cache_manager.py::test_save_search_cache_normalises_but_save_caches_does_not`.

`save_search_cache` passes `normalize=True`; `save_caches` writes the merged
dict straight out. A malformed search-cache value written through `save_caches`
therefore reaches disk intact and is only cleaned when a later process loads it.
Code reading `search_cache.json` directly - an analysis script, a cache
inspector - sees the difference; code going through `CacheManager` does not.

---

## 5. BigCodeBench evaluation runs under whatever `python` is on `PATH`

**Severity:** environment-dependent; can silently distort results.
**Status:** open. Noted by `tests/unit/test_bigcodebench_scorer.py::test_the_scorer_shells_out_to_a_bare_python`.

`evaluate_bigcodebench` calls `subprocess.Popen(["python", tmp_path])`, resolved
through `PATH` - not `sys.executable`. On the Snellius login node that resolves
to the system Python 3.9, several minor versions behind the project's 3.11, so a
prediction using newer syntax (`match`, `X | Y` unions) is scored wrong for
reasons unrelated to the model. Check `which python` before trusting a
BigCodeBench number, or change the call to `sys.executable`.

---

## 6. RL training records a prompt 4 tokens shorter than the one that generated it

**Severity:** low but pervasive. Affects every RL run in the repo: the GRPO
baseline, Prefix-RFT in both modes, and run 012.
**Status:** open, deliberately unfixed. Found 2026-08-19 while auditing
Prefix-RFT token mode.

`AgentModeDaemon`'s proxy builds the token IDs it injects into each response
with (`src/fine_tuning/agentflow/verl/daemon.py:216-225`):

```python
prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
```

It does not read the request's `chat_template_kwargs`. `OpenAIProvider` sends
`{"enable_thinking": use_thinking}` for every Qwen call
(`src/agent_engine/models/api_provider.py`), and the runs use `THINKING_MODE: NO`,
so vLLM renders with `enable_thinking=False`. Qwen3's template emits an empty
think block in that case and not otherwise:

| rendering | tokens | ends with |
|---|---|---|
| what the proxy records | 15 | `<\|im_start\|>assistant\n` |
| what vLLM conditioned on | 19 | `...assistant\n<think>\n\n</think>\n\n` |

So the recorded prompt is missing `<think>\n\n</think>\n\n`, four tokens, on
every turn.

**Why it is not a broken gradient.** verl recomputes both `old_log_prob` and the
current log-prob from the *recorded* prompt, so the PPO ratio stays
self-consistent. What is wrong is fidelity: training conditions on a context
four tokens shorter than the one that produced the text, and shorter than the
one used at inference, where the same provider sends `enable_thinking=False`
again. It is a fixed constant suffix on every prompt, which is why it has not
shown up as instability.

**Why it is not fixed.** Correcting it changes what the GRPO baseline trains on
and would break comparability with run 012 and every completed run the thesis
reports. Fixing it means re-running the baselines.

**Relevance to Prefix-RFT.** Both prefix modes inherit exactly this gap, in the
same place, and add no new one. `scripts/check_prefix_replay_tokenisation.py`
asserts the property that *does* matter for the prefix, namely that what vLLM
conditions on under `continue_final_message` is the rendered prompt followed by
precisely the teacher's tokens, so `prefix_mask` marks the right positions. The
comment there explains why the four-token gap is deliberately not asserted.
