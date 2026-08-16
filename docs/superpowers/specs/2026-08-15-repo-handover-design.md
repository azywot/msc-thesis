# Repository Handover: Clean, Modular, Extensible CoSMAS

**Date:** 2026-08-15
**Branch:** `chore/clean-the-code`
**Status:** design approved in chat; awaiting spec review

## Goal

Hand the repository to a new researcher/developer who will experiment with different
fine-tuning paradigms, add benchmarks, datasets, and sub-agents, and extend the system
in other ways. The repository must end up clean, modular, intuitive, well-tested,
reproducible, and clearly documented.

## The hard constraint

**Behaviour must be identical to the current implementation.** Not "equivalent",
not "improved": identical. Every phase below is gated on a recorded baseline
reproducing byte-for-byte. This constraint outranks every other goal in this
document. Where cleanliness and behaviour-identity conflict, behaviour-identity wins
and the ugliness is documented instead of fixed.

### Behaviour-preservation contract

These are the observable behaviours held fixed. They are the acceptance criteria, and
each is locked by a fixture in `tests/characterization/` (Phase 0) before any code moves.

| # | Observable | Locked by | Lifetime |
|---|---|---|---|
| B1 | `generate_configs.py` output, byte-for-byte against a snapshot of its *own* output recorded at Phase 0 | `test_configs_unchanged.py` | scaffolding |
| B2 | Exported prompt templates and tool schemas, byte-for-byte | `test_prompts_unchanged.py` | scaffolding |
| B3 | Orchestrator tool-call sequence, committed message text, memory contents, per-state token usage, and log ordering, for a mixed batch | `test_orchestrator_trace.py` | permanent |
| B4 | `_compute_metrics` output over existing `raw_results.json` | `test_metrics_replay.py` | permanent |
| B5 | `classify_failure` output and `breakdown.json` over the same runs | `test_failure_modes_replay.py` | permanent |
| B6 | The 496 currently-passing tests keep passing | existing suite | permanent |

Fixtures are committed and regenerated only via an explicit `--update-fixtures` flag.
Regenerating a fixture is a deliberate, reviewable act, never a side effect of a test run.

**Scaffolding gates are deleted at Task 20.** B1 and B2 fail on every *intended* change:
adding a benchmark changes the generated configs, editing a prompt changes the templates.
Since the point of this handover is that a new researcher does exactly those things,
leaving B1/B2 in place would fail on their first honest commit and teach them to run
`--update-fixtures` reflexively — at which point the gates protect nothing. They are
replaced by property tests (`tests/unit/test_wiring_invariants.py`) that assert every
dataset resolves to a loadable template and every registered tool exposes a well-formed
schema. Those catch the real failure modes and survive people adding things.

B3/B4/B5 are permanent. B3 goes red only when orchestrator *behaviour* changes, not when
the system grows. B4/B5 replay over frozen historical runs, so adding a benchmark cannot
invalidate them — they are what protects the thesis numbers.

**B1 compares the generator against itself, not against the committed configs.** The two
have drifted (see Open decisions), so a generator-vs-committed diff would fail at Phase 0
and prove nothing about the refactor. Phase 0 snapshots what `generate_configs.py`
produces *today* into `tests/characterization/fixtures/configs/`; every later phase must
reproduce that snapshot exactly. The drift itself is a pre-existing repository bug,
reported separately and never silently resolved by this work. The test writes to a
temporary directory and never touches `experiments/configs/`.

## Non-goals

- No renaming or merging of the four top-level packages (`agent_engine`, `fine_tuning`,
  `gepa_integration`, `verl_ext`). Considered and rejected as too large a diff to prove safe.
- No reorganisation of `experiments/configs/` and no un-committing of generated configs.
- No refactoring of vendored AgentFlow code (see "Vendored code").
- No new features, no performance work, no dependency upgrades.
- No changes to thesis numbers. Where the code is odd but load-bearing for published
  results, it is documented and locked, not corrected.

## Baseline as measured (2026-08-15)

- 34,015 Python LOC, 683 tracked files.
- 496 tests pass under the `agent_engine` conda env.
- `pytest` from the repo root **aborts during collection** in that env:
  `tests/unit/test_fine_tuning_rollout.py` imports `agentops`, which only exists in the
  `cosmas-train` env. There is currently no single command that runs the suite.
- All four packages are already installed editable
  (`__editable__.agent_engine-0.1.0.pth`), so the 14 `sys.path.insert(..., "src")` calls
  scattered through `scripts/`, `tests/`, and `src/gepa_integration/data/prepare.py` are
  redundant no-ops. The `sys.path.insert(..., "scripts")` calls are load-bearing until Phase 6.

## Problems being solved

1. **The library is incomplete; the wiring lives in a script.** `scripts/run_experiment.py`
   holds ~500 lines of logic (`setup_model_provider`, `setup_tools`, `run_experiment`,
   `_compute_metrics`). None of it is importable or unit-testable.
2. **Adding a sub-agent means editing a script.** `setup_tools()` is a 100-line `if/elif`
   chain over tool names.
3. **Adding a benchmark touches ~5 files** beyond the loader: `prompts/builder.py`
   (7 dataset-name literals), `config/schema.py`, `utils/wandb_logging.py`,
   `datasets/base.py`, `run_experiment.py`. `DatasetRegistry` exists but is not the seam.
4. **The orchestrator hardcodes tool identity.** `_WebJob` / `_CodeJob` and their
   schedule/flush methods make batching per-tool inside the 946-line core loop.
5. **The library depends on the scripts directory.** `src/gepa_integration/seed.py:21`
   inserts `scripts/` on `sys.path` so it can import the failure-mode classifier.
6. **Docs are large and stale.** 975-line README whose structure tree references `train/`,
   `experiments/configs/generate_configs.py`, `1_milestone_no_img_no_mindmap_AgentFlow/`,
   and `jobs/008_prepare_fine_tuning_data.job`, none of which exist. ~4,100 lines across 11
   `docs/*.md` files, mostly point-in-time status. 13 superseded plans/specs.
7. **Untested load-bearing code:** `caching/manager.py`, `gaia_scorer.py`,
   `bigcodebench_scorer.py`, all of `external/`, `models/mlx_provider.py`.
8. **`scripts/generate_configs.py` is not idempotent** (see Open decisions).

---

## Design

### 1. Packaging and imports

- Add `[project.scripts]` console entry points wrapping existing `main()` functions:
  `cosmas-run`, `cosmas-analyze`, `cosmas-configs`, `cosmas-gepa`, `cosmas-prompts`,
  `cosmas-datasets`. The `scripts/` paths keep working; entry points are additive.
- Delete the 14 redundant `sys.path.insert(..., "src")` lines. Provably inert: the
  packages resolve via the editable install with the inserts removed.
- Add `tests/conftest.py` defining a `training` marker that skips modules whose imports
  are absent (`agentops`, `verl`), so `pytest` from the repo root succeeds in either
  conda env. This changes test *collection*, not test *outcomes*: the same tests that
  pass today still pass, and the module that cannot even import today is skipped with a
  reason instead of aborting the run.
- Correct `[tool.mypy] python_version = "3.9"` and `[tool.black] target-version` to 3.11,
  matching `requires-python = ">=3.11"`. Formatting-only; no reformatting commit is made
  as part of this change.
- Replace the placeholder `[project.urls]` (`github.com/yourusername/agent-engine`) with
  the real repository URL.

### 2. Extension seams

Four seams, one per "add a thing" task. Each is the *only* file a newcomer edits.

**Tools / sub-agents.** New `src/agent_engine/tools/registry.py`:

```python
@register_tool("web_search")
def _build_web_search(config, deps) -> BaseTool: ...
```

`deps` is a frozen dataclass carrying `cache_manager`, `api_keys`, `model_providers`,
`mind_map_storage_path`. The five existing construction blocks are lifted **verbatim**
out of `setup_tools`'s `if/elif` into decorated factories in each tool's own module.
`build_tool_registry(config, deps)` iterates `config.tools.enabled_tools` and calls the
factory. `BaseTool` and `ToolRegistry` in `core/tool.py` are already clean and are not
touched.

Adding a sub-agent becomes: write the tool class, decorate a factory, name it in a config.

**Model families.** `models/base.py` is already well-factored via the `_*_FAMILIES`
frozensets. Mechanism unchanged. Add tests asserting every `ModelFamily` member *resolves*
to a `ToolCallFormat` through `get_tool_call_format`, and that no family table holds a
stale non-`ModelFamily` entry, so a half-added family fails loudly instead of silently.

Note the tables are **sparse on purpose**: `_TOOL_CALL_FORMAT` lists only the exceptions
and everything else defaults to JSON. A test requiring an entry per family would fail on
seven correct families, which is why the check is on resolution rather than membership.

**Datasets / benchmarks.** Per-dataset facts currently scattered as string literals move
onto the `DatasetRegistry` entry: prompt-template name, stratified-or-not, level-key
extractor, answer type. The literals in `_STRATIFIED`, `_level_key`, and
`prompts/builder.py`'s dispatch then read from one place. This is the one seam where
behaviour is *relocated* rather than merely moved, so B1/B2/B4 are the gate: the same
dataset names must produce the same templates, the same stratification, and the same
metrics.

**Runner.** `scripts/run_experiment.py` logic moves to `src/agent_engine/runner/`:

```
src/agent_engine/runner/
  experiment.py   run_experiment(), _make_run_dir, _write_json, _config_to_dict
  providers.py    setup_model_provider()  (instance cache, LoRA-aware cache key)
  tools.py        build_tool_registry()
  metrics.py      compute_metrics(), _level_key()
```

`scripts/run_experiment.py` becomes an argparse shim of roughly 40 lines. Same CLI flags,
same output paths, same log messages.

### 3. Orchestrator batching

`orchestrator.py:359-600` has two hardcoded deferral paths with the same shape:

| Stage | `web_search` | `code_generator` |
|---|---|---|
| short-circuit | `analysis_cache[query]` hit -> immediate | none |
| prepare | `search_and_format(query)` -> payload | `build_task_prompt(task, ctx)` -> prompt |
| pre-batch, across all jobs | batch-fetch URLs, update `url_cache`, save | none |
| group | by `id(tool.model_provider)` | by `id(tool.model_provider)` |
| prompt | `build_analysis_prompt(query, _format_results(...))` | the prepared prompt |
| finalize | strip -> write `analysis_cache` -> commit | strip -> `extract_code` -> log -> `execute(code=)` -> commit |

Both collapse into `src/agent_engine/core/batching.py` with a `BatchedTool` protocol:

```python
prepare(ctx)               -> BatchJob | ToolResult   # ToolResult short-circuits
pre_batch(jobs)            -> None                    # optional
batch_prompt(job)          -> str
finalize(job, generation)  -> ToolResult
batch_priority: int                                   # web=10, code=20
```

`_WebJob`, `_CodeJob`, and `_ImmediateResult` become one `BatchJob`. The orchestrator
holds a single `dict[tool_name, list[BatchJob]]` and knows nothing about web or code.

**Behaviour details that must be preserved, each a known trap:**

- **Flush order** stays immediates -> web -> code. Today this is hardcoded; generically it
  requires the explicit `batch_priority`, *not* registration or dict-insertion order.
- **Usage accounting asymmetry.** The deferred path accumulates only `generation.usage`
  and never the finalize `ToolResult.usage`; the immediate path accumulates
  `ToolResult.usage`. Unifying these naively double-counts code-generator tokens.
- **`strip_thinking_tags` runs twice on the web path** (in finalize, then again at commit).
  Idempotent, so routing web through the generic commit is identical. Recorded here so it
  is locked by B3 rather than "cleaned up" by a later reader.
- **Log ordering differs by tool.** `web_search` logs `Tool call:` for every job *before*
  URL fetching; `code_generator` logs *inside* finalize, after generation. Both preserved.
- **Grouping key** stays `id(tool.model_provider)`, including its identity (not equality)
  semantics.

This is the highest-risk phase. It proceeds only with B3 green before and after, where B3
covers: multiple states in one turn, web + code + immediate calls together, an
`analysis_cache` hit, a missing-argument error, and an exception raised in `prepare`.

If B3 cannot be made green, Phase 5 reverts alone and every other phase still lands.

### 4. Analysis code

`scripts/failure_modes/` moves to `src/agent_engine/analysis/`:

```
src/agent_engine/analysis/
  failure_modes.py   classify_failure(), moved byte-identical
  breakdown.py
  eval_runs/         baseline_counterfactual.py, retrieval_locus_split.py
  fine_tuning/       all_wrong.py, base_vs_lora.py, case_studies.py,
                     rollout_groups.py, runs.py
```

The current script paths remain as thin shims calling `main()`, accepting the same argv
and writing the same output paths, so commands recorded in thesis notes keep working.
`src/gepa_integration/seed.py` and the four test modules switch to real imports; their
`sys.path` hacks go. `classify_failure` is frozen: B5 locks its output over existing runs.

### 5. Vendored code

`src/fine_tuning/agentflow/` is a wholesale copy of the external AgentFlow repository
(per `docs/archive/superpowers/plans/2026-05-13-vendor-agentflow.md`). It is treated as vendored:
code untouched, excluded from restyling, and documented by a new
`src/fine_tuning/agentflow/VENDORED.md` recording upstream origin, version, and the exact
local modifications (the 8 fixed absolute imports and the removed `_agentflow_path.py`).

### 6. Tests

Beyond the Phase 0 characterization fixtures, add unit tests for the untested load-bearing
modules: `caching/manager.py`, `datasets/evaluators/gaia_scorer.py`,
`datasets/evaluators/bigcodebench_scorer.py`, `external/serper.py`, `external/tavily.py`,
`external/url_fetcher.py`. These are new tests over unchanged code, so they cannot alter
behaviour; if one fails on first run it has found a real pre-existing bug, which is
reported and **not** fixed as part of this work.

### 7. Documentation

```
README.md          ~250 lines: what it is, install, run one experiment, repo map
CONTRIBUTING.md    dev setup, running tests, style, how to add things
CHANGELOG.md       unchanged (real history)

docs/
  architecture.md          orchestrator loop, structured memory, batching, baseline vs AgentFlow
  configuration.md         full config schema reference
  guides/
    add-a-benchmark.md
    add-a-tool-or-subagent.md
    add-a-model-family.md
    add-an-adaptation-method.md
  pipelines/
    evaluation.md  gepa.md  rl.md  sft.md
  archive/                 the 11 current docs + 13 plans/specs, each with a
                           HISTORICAL banner naming its supersession date
```

Guides are written against the seams from Section 2 and must be walked through end-to-end
by actually following them, not written from memory.

---

## Phases

Each phase is one commit on `chore/clean-the-code`, ending with the full suite plus all
fixtures green. Phases are ordered so that the risky one is late and isolated.

| # | Phase | Gate | Risk |
|---|---|---|---|
| 0 | Characterization fixtures against current code, nothing else touched | fixtures pass against unmodified code | none |
| 1 | Packaging, entry points, `conftest.py`, drop dead `sys.path` inserts | B1-B6 | very low |
| 2 | Promote runner logic into `src/agent_engine/runner/` | B1-B6 | low |
| 3 | Tool factory registry; `setup_tools` if/elif dies | B2, B3, B6 | low |
| 4 | Dataset registry consolidation | B1, B2, B4, B6 | medium |
| 5 | Orchestrator batching collapse | B3 especially | high |
| 6 | Analysis move + shims | B5, B6 | low |
| 7 | Tests for untested load-bearing modules | new tests pass | none |
| 8 | Docs rewrite, archive, guides, `CONTRIBUTING.md`, `VENDORED.md` | guides walked end-to-end | none |

## Open decisions

1. **`scripts/generate_configs.py` is not idempotent.** Running it rewrites 10 committed
   `qwen3/lora_inference/*` configs, reverting a hand-edited `_v2` run
   (`.../qwen3-8b-grpo-search-math-v2/29-05-2026_11-36-23210365/global_step_40/...`) to
   `.../qwen3-8b-grpo-search-math/22-05-2026_00-01-23031012/global_step_20/...`. The
   committed configs and the generator have drifted; a new user following the documented
   regeneration step silently reverts the LoRA evaluation to an older checkpoint.
   Chapter 7's LoRA numbers depend on which adapter actually ran.
   **Blocked on Agata:** which is correct, `global_step_40`/`_v2` or `global_step_20`?
   The generator will be made to reproduce whichever is correct, in its own commit, and
   the B1 fixture will be regenerated with `--update-fixtures` as part of that commit so
   the change is visible in review.
   **Not a blocker for any phase.** B1 compares the generator to itself, so all nine
   phases proceed unaffected while this stays open. Nothing in this work regenerates
   `experiments/configs/` in place.

## Risks

| Risk | Mitigation |
|---|---|
| Phase 5 changes orchestrator behaviour subtly | B3 fixture covers the mixed batch, ordering, and usage asymmetry; phase reverts alone |
| Dataset registry consolidation drops a special case | B1/B2/B4 gate; the literals are enumerated from grep before moving, not from memory |
| A fixture is too weak and passes vacuously | Each fixture is validated by deliberately mutating the code it covers and confirming it fails |
| Fixtures drift silently over time | `--update-fixtures` is explicit; fixture files are committed and reviewed |
| Thesis analysis commands break | Script shims keep argv and output paths identical |
