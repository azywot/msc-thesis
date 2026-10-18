# msc-thesis repository status (2026-10-18)

A snapshot for picking the repository back up after about two months away. Everything
here was checked against the working copy, the git history and the files under
`experiments/results/` on 2026-10-18. Paths are relative to the repo root unless
absolute.

---

## 1. One-paragraph summary

`main` is clean, in sync with `origin/main`, and the test suite passes (936 passed,
3 skipped, 1 xfailed, about 65 s). The inference framework (CoSMAS: AgentFlow vs
baseline) is finished and its evaluation results are all on disk. Since mid-August the
work has been almost entirely on the **adaptation side**: SFT was fixed (folded format)
and evaluated, the repo was refactored for handover with byte-identical behaviour, and
**Prefix-RFT** was built as a fourth adaptation method in two modes (`steps`, `tokens`).
Prefix-RFT is verified on GPU at small scale, but **the production run (`013`) has never
been submitted**, so there are no Prefix-RFT benchmark numbers yet. The last commit
(today, PR #34) documents and prepares a new single training pool,
`data/traj_qwen32b_3k_v1/`, but **no config or job points at it yet**.

---

## 2. Where things live

| What | Where |
|---|---|
| Repo | `/gpfs/home3/xchen1/azywot/msc-thesis` (remote `https://github.com/azywot/msc-thesis`) |
| Python env | `/home/xchen1/.conda/envs/agent_engine/bin/python` (no `conda` on PATH in non-interactive shells) |
| Entry docs | `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md` |
| Reference docs | `docs/architecture.md`, `docs/configuration.md`, `docs/known-issues.md` |
| Pipeline docs | `docs/pipelines/{evaluation,gepa,sft,rl,prefix-rft}.md` |
| How-to guides | `docs/guides/add-a-{benchmark,tool-or-subagent,model-family}.md`, `add-an-adaptation-method.md`, `change-training-data.md` |
| Design specs and plans | `docs/superpowers/specs/`, `docs/superpowers/plans/` (handover refactor, Prefix-RFT, Prefix-RFT token mode) |
| Superseded docs | `docs/archive/` (HISTORICAL banner on each; numbers and paths may be stale). `docs/archive/sft_status.md` still holds the most detailed SFT record. |
| Results | `experiments/results/<suite>/<dataset>/<config>/<split>_<timestamp>_<jobid>/` |
| Training run outputs | `experiments/results/fine_tuning/<experiment>/<run-tag>/`, logs in `out/fine_tuning/` |
| Adapters | `data/adapters/qwen3-8b-sft-folded-v1/` (SFT, present); `/scratch-shared/azywot/fine_tuning/lora_adapters/` (only `qwen3-8b-prefix-rft-capped` left) |

### Source layout (after the August refactor)

```
src/agent_engine/      the framework
  config/              YAML schema + loader (schema.py)
  core/                orchestrator, execution state, batching.py (BatchJob / flush_batches)
  models/              model families (base.py quirks) + vLLM / MLX / API providers
  tools/               web_search, code_generator, mind_map, text/image inspector, registry.py (@register_tool)
  datasets/            loaders, spec.py (DatasetSpec table), evaluators
  prompts/             templates (*_dataset*.yaml = AgentFlow, *_baseline*.yaml = baseline) + builder.py
  runner/              experiment loop, providers, tool wiring, metrics (promoted out of scripts/)
  analysis/            failure-mode classifier (classify_failure is frozen) + analyses
  external/, caching/, utils/
src/fine_tuning/       GRPO + SFT; agentflow/ is vendored (see VENDORED.md, do not edit)
  prefix_rollout.py, prefix_replay.py   Prefix-RFT rollout + teacher replay
src/gepa_integration/  prompt optimisation
src/verl_ext/          verl extensions: folded_sft_dataset.py, checkpoint_utils.py, prefix_rft/
scripts/               thin CLI shims: run_experiment, analyze_results, generate_configs,
                       launch_verl, train_orchestrator, build_sft_parquet, build_prefix_demos,
                       finalize_sft_run, merge_lora, check_* gates, plots/, tables/, failure_modes/
jobs/                  SLURM: 001-007 setup, fine_tuning/000-014, gepa/, grpo_inference/, refactor_check/
tests/                 unit/ + characterization/ (behaviour-locking fixtures) + gepa_integration/
```

Things that bite if forgotten:

- **Most configs are generated.** Edit `scripts/generate_configs.py`, not the YAML. The
  generator now reproduces the committed tree exactly and a test guards that.
- **Typo'd config keys are silently ignored** (schema allows extra keys).
- **Everything is batched.** Use `batch_size: 1` to debug a single question.
- **Characterization tests** (`tests/characterization/`) lock behaviour. A failure means a
  change moved behaviour; investigate before regenerating fixtures.
- **Prefix-RFT settings live in two blocks** of one config: `env:` (`PREFIX_RFT`,
  `PREFIX_DEMOS_PATH`, `BASE_MODEL`) and `python_args:` (`prefix_rft.*`).
  `PREFIX_DEMOS_PATH` must equal `prefix_rft.demos_path`; `train_orchestrator.py`
  refuses to start otherwise.

---

## 3. Commit history

107 commits on `main`, first on 2026-02-17. Merged PRs in order:

| Date | PR | What landed |
|---|---|---|
| 2026-03-08 | #6 | Context injection, Tavily web provider option |
| 2026-03-18 | #8 | Context manager replaced by `mind_map`, AgentFlow alignment |
| 2026-03-21 | #10 | New vanilla baseline mode |
| 2026-03-25 | #11 | OLMo models |
| 2026-04-15 | #12 | BigCodeBench benchmark |
| 2026-04-20/21 | #14, #15 | OLMo configs; DeepSeek-R1-Distill + OLMo families v2 |
| 2026-04-22 | #16 | Thesis visualisations |
| 2026-04-25 | #19 | Repo versions investigation |
| 2026-05-12 | #21 | Failure case analysis |
| 2026-06-01 | #20 | Fine-tuning (GRPO / LoRA) + GEPA integration |
| 2026-07-21 | #23 | Failure-mode scripts reorganised into `eval_runs/` and `fine_tuning/`; GRPO group size inferred |
| 2026-08-12 | #24 | SFT folded format (fixes SFT landing below base) |
| 2026-08-16 | #25 | SFT code cleanup, full-parameter SFT support |
| 2026-08-19 | #32 | Handover refactor (squashes #26) + Prefix-RFT (squashes #28, #29, #31) |
| 2026-08-19 | (direct) | `scripts/upload_json_to_wandb.py` |
| 2026-08-21 | #33 | Prefix-RFT token mode |
| 2026-10-18 | #34 | `traj_qwen32b_3k_v1` verification, folded splits, `build_prefix_demos.py --from-parquet` |

### The last two months in more detail

**#24 SFT folded format (2026-08-12).** Root cause of SFT underperforming: training rows
were the native multi-turn transcript, but AgentFlow rebuilds a fresh `[system, user]`
memory prompt every turn (0/194 sampled decision points matched). `build_sft_parquet.py
--format folded` now writes one row per orchestrator decision using the orchestrator's own
formatting helpers. `FoldedSFTDataset` also fixes a 4-token generation-prompt offset.
`check_sft_folded_format.py` is a pre-flight gate the job runs. `finalize_sft_run.py` and
`sft_checkpoint_janitor.py` turn verl's 33 GB FSDP checkpoints into ~350 MB PEFT adapters
and keep only best and last. Trained run `06-08-2026_21-47-25300018` (val loss
0.5404 to 0.4234, 186 steps) and evaluated on all five benchmarks (results in section 6).

**#25 / #32 Handover refactor (2026-08-16 to 19).** Goal: clean, modular, documented, with
behaviour byte-identical. All 8 phases done (plan: `docs/superpowers/plans/2026-08-15-repo-handover.md`).
Runner moved into `src/agent_engine/runner/`, tool and dataset seams became registries
(`@register_tool`, `DatasetSpec`), orchestrator batching collapsed behind one protocol,
analysis moved into `src/agent_engine/analysis/` with CLI shims left in `scripts/`. Bugs
found during this were **recorded, not fixed** (see section 7). Also added
`007_train_sft_full.job` (full-parameter SFT, untested at scale).

**#32 Prefix-RFT (2026-08-17 to 19).** GRPO in which one of eight rollouts per prompt is
seeded with a prefix of a Qwen3-32B teacher demonstration and the policy writes the rest,
entropy-clipped (top 20%) and advantage-corrected per `papers/PrefixRFT_2507.01679v3.md`.
Runs on Flow GRPO (the multi-turn GRPO already used for the LoRA baseline). Code in
`src/verl_ext/prefix_rft/`, `src/fine_tuning/prefix_rollout.py`, `prefix_replay.py`.
Staged jobs `008` to `013` (section 5). Five launch- or replay-blocking bugs were found on
GPU and fixed. One advantage-fidelity fix for singleton groups (only reachable at
`rollout.n: 2`).

**#33 Prefix-RFT token mode (2026-08-21).** Adds `prefix_rft.mode: tokens` beside the
default `steps`. Token mode measures the prefix as a token fraction of the concatenated
demonstration, splits the straddling decision and lets vLLM finish it via
`continue_final_message`. It makes the 273 single-decision demonstrations prefixable
(coverage 1085 to 1358 of 1800), so the shipped token config sets
`min_demo_decisions: 2` to keep a steps-vs-tokens comparison fair. Select with
`--prefix-mode` on `scripts/launch_verl.py`; `013` takes `PREFIX_CONFIG`.

**#34 traj_qwen32b_3k_v1 (2026-08-24 work, merged 2026-10-18).** New README at
`data/traj_qwen32b_3k_v1/README.md` (only the README is tracked; parquets are on disk, not
in git). Verified schemas, counts, split disjointness, think stripping, and that stored
system prompts are byte-identical to what `PromptBuilder` renders now. Built
`folded/folded_train.parquet` (9457 rows) and `folded/folded_val.parquet` (628 rows; one
over-length row dropped), both passing all six SFT gates, and a demonstration store
`data/traj_qwen32b_3k_v1/prefix_demos.parquet` (3000 demos, 9457 decisions,
`check_prefix_demos` PASSED). `build_prefix_demos.py` gained `--from-parquet`. The README
states the intent that this is the single pool for SFT, Prefix-RFT and GRPO, and lists
what that costs (teacher-solved selection bias, SFT/GRPO overlap).

---

## 4. Branches

Branch cleanup was done on 2026-10-18 (see "Deleted branches" below). What is left:

| Branch | Where | Last commit | What it holds | Recommendation |
|---|---|---|---|---|
| `main` | GitHub + local | 2026-10-18 | Everything below is relative to this. | |
| `develop` | GitHub + local | 2026-08-18 (`41693bb`) | Integration branch for the Prefix-RFT work (#26, #28, #29, #31). Deleted during cleanup, then restored from the bundle on request the same day. Its content is fully in `main` (#32), and `main` is newer on every file. | Kept by choice. It cannot fast-forward to `main`: the squash merges left 4 commits on `develop` that `main` lacks by ancestry, though not by content. Before branching new work from it, reset it to `main` (`git checkout develop && git reset --hard main && git push --force-with-lease`) or merge `main` into it. |
| `gepa-common-suffix` | **local only** (its GitHub copy was deleted) | 2026-06-10 (`5fe0150`) | GEPA suffix-only mode (system prompt frozen, only `planning_suffix` optimised) on a common proxy set: spec `2026-06-09-gepa-common-suffix-design.md`, `experiments/configs/gepa/common_{no,orch}.yaml`, `splits/common_gepa_splits.json`, `jobs/gepa/008_run_gepa_common.job`, GEPA inference configs for AIME and GPQA, tests. **Its results are already on disk** (`experiments/results/gepa/common_*`, and the 2026-06-10 `gepa_inference/*/qwen8B_sub1_7b_none` runs). | Merge it (rebase onto `main` first: it predates the refactor, so `scripts/run_experiment.py` changes must move into `src/agent_engine/runner/`). Without it, the GEPA numbers reported from the common-suffix runs cannot be reproduced from `main`. Until then, push it back to GitHub so it is not single-copy. |
| `feat/clean-and-retain-inference-only` | **local only** (recreated from `5df6ecb` after its GitHub copy was deleted) | 2026-07-13 | One commit that strips RL, SFT and GEPA, keeping inference only (about 20k lines removed). | Keep as a parked variant, do not merge. Useful only if a slim inference-only release is wanted. |
| `feat/add-context-to-subagents` | **local only** (its GitHub copy was deleted) | 2026-04-25 (`f96e054`) | Sub-agent shared-memory experiment (results under `subagent_shared_memory_ablation/`). | Check whether its configs are in `main`; if the ablation is reported, merge the config generator change. |
| `chore/check-agentic-reasoning-prompts` | GitHub + local | 2026-04-22 | 2 commits checking prompts against the Agentic Reasoning repo. | Read once, then delete. |

No open PRs (checked via the GitHub API) and no git stashes.

### Deleted branches (2026-10-18)

Most of these had already been squash-merged, so git did not see them as merged. Before
deleting, each branch was checked by testing whether every file version it touched
appears somewhere in `main`'s history.

Between the first fetch of the day and the cleanup, most of them had already been
deleted on GitHub, outside this session. The cleanup then deleted `develop`,
`feat/prototype-age` and `sft` from GitHub, and all local copies. `develop` was restored
to GitHub and locally afterwards, at the same commit.

**Backup.** All 25 tips are in a verified bundle at
`/gpfs/home3/xchen1/azywot/archive/msc-thesis-deleted-branches-2026-10-18.bundle`.
To restore one (example):

```bash
git fetch /gpfs/home3/xchen1/azywot/archive/msc-thesis-deleted-branches-2026-10-18.bundle \
    refs/archive/fest/add-model-family:refs/heads/fest/add-model-family
```

| Branch | Tip | Landed via | Every file version in `main`? |
|---|---|---|---|
| `develop` (restored, see above), `feat/prototype-age` | `41693bb` | #32 | yes |
| `docs/prefix-rft-flow-grpo` | `01154a3` | #32 | yes |
| `feat/add-prefix-rft` | `e0ce7ce` | #32 | 1 doc version not, superseded by later edits |
| `chore/refactor-the-code` | `25b108b` | #32 | 4 files not (README, CHANGELOG, a guide, `train_orchestrator.py`), superseded by #32's later versions |
| `chore/clean-the-code` | `9274cef` | #25 | yes |
| `feat/prefix-rft-token-mode` | `0f9fe52` | #33 | yes |
| `docs/traj-3k-v1-readme-verification` | `1623c15` | #34 | yes |
| `feat/sft-folded-format` | `0d1344b` | #24 | yes |
| `feat/scripts-cleanup` | `eebd0b2` | #23 | yes |
| `sft` | `0b3ff07` | superseded by #24 | 3 files not: the old non-folded `007_train_sft.job` and its data prep, removed on purpose |
| `feat/fine-tuning` | `6612726` | #20 | yes |
| `feat/gepa-integration` | `64cf853` | #20 | about 48 files not: intermediate states of the GEPA/RL work, overwritten before #20 |
| `add/smoke-test-8b-example` | `96ac6ab` | #20 | 35 files not: same as above |
| `chore/failure-case-analysis` | `6d01f34` | #21 | yes |
| `repo-versions-investigation` | `b6be071` | #19 | yes |
| `add-ds-family-v2` | `8ae740d` | #15 | yes |
| `feat/add-ds-model-families` | `980f35a` | superseded by #15 | 13 files not: early DeepSeek configs (`ds32_test/`, `ds8_AgentFlow/`) and provider code, replaced by v2 |
| `fest/add-model-family` | `67cccc0` | superseded by #15 | 23 files not: **DeepSeek R1-0528 8B and R1-Distill 14B test configs** that never reached `main`. Restore from the bundle if those models come back. |
| `feat/add-olmo`, `feat/add-olmo-configs` | `f5b5b92`, `9cb1c3c` | #11, #14 | yes |
| `fest/add-coding-benchmark` | `067aee9` | #12 | yes |
| `feat/fix--baseline` | `d772f08` | #10 | yes |
| `feat/change-context-manager` | `4e68464` | #8 | yes |
| `feat/integrate-tavily` | `0f7a75a` | #6 | yes |

---

## 5. Pipeline status

| Pipeline | Code | Trained | Evaluated | Notes |
|---|---|---|---|---|
| Evaluation (AgentFlow vs baseline) | done | n/a | done for Qwen3 8B/32B, orchestrator capacity, ablations, DeepSeek (GAIA), OLMo (smoke only) | Section 6 |
| GEPA (full prompt) | done, in `main` | GAIA and math runs, May 2026 | 2026-05-29, GAIA-optimised prompt transferred to MuSiQue and HLE | |
| GEPA (suffix only, common set) | **only on `gepa-common-suffix`** (local branch only) | 2026-06-10 (`common_no`, `common_orch`) | NO regime on all five benchmarks (2026-06-10) | Merge pending |
| GRPO (Flow GRPO, LoRA) | done | v1 (`qwen3-8b-grpo-search-math`, May), v2 (`...-v2`, 2026-05-29) | done, both thinking regimes | **Adapters deleted from scratch**: the 10 `lora_inference` configs point at a path that no longer exists. Numbers on disk are the record. |
| SFT (folded, LoRA) | done | `06-08-2026_21-47-25300018` | done, NO regime only | Adapter kept at `data/adapters/qwen3-8b-sft-folded-v1/` |
| SFT (full parameter) | done (`007_train_sft_full.job`) | no | no | Untested at scale |
| Prefix-RFT, step mode | done | stages `008` to `012` green; `013` **not run** | no | Capped run `012` reached 9/10 steps; checkpoint at `global_step_5` in `/scratch-shared/.../qwen3-8b-prefix-rft-capped` |
| Prefix-RFT, token mode | done | `014` tiny run green (2026-08-21); `013` with token config **not run** | no | |

### Prefix-RFT staging, for reference

| Job | Purpose | Status |
|---|---|---|
| `008_build_prefix_demos.job` | Build demonstration store (1358 of 1800 questions, 1085 prefixable in step mode) | done |
| `009_run_tests_for_prefix_rft.job` | CPU verification suite | green |
| `010_smoke_prefix_rft.job` | 8B GPU smoke | green (25756950) |
| `011_tiny_prefix_rft.job` | 2 questions, 1 step, checks replay/masking/advantage/clip | green (25755605) |
| `012_capped_prefix_rft.job` | Production shape, 10 steps, checks the curriculum | 9/10 steps before 12 h wall (25762046); `low_t` 0.928 to 0.072, `off_ratio` at most 0.067, clip 20.1 to 20.3%, `kl_loss` 0.007 to 0.155, reward with prefix 0.74 vs 0.41 without |
| `014_tiny_prefix_rft_tokens.job` | Token-mode counterpart of `011` | green (25918774) |
| `013_train_prefix_rft.job` | Production: 1800 questions, 1 epoch, 56 steps, about 66 h of a 72 h wall on 4 H100 | **never submitted** |

The evaluation configs for a Prefix-RFT checkpoint were listed in August as "the one piece
of scaffolding still to write". There is no `experiments/configs/qwen3/prefix_rft_inference/`
yet.

---

## 6. Results tables

Accuracy in %, latest run per configuration, read from each run's `metrics.json`.
Benchmarks and splits: GAIA `all_validation` (165), HLE `test_subset_200` (200), GPQA
`diamond` (198), AIME (60), MuSiQue `validation_subset_200` (200). Thinking-mode suffixes:
`none` = thinking off everywhere, `orchestrator` = orchestrator only, `subagents` =
sub-agents only, `all` = everywhere.

To regenerate tables from the data: `python scripts/tables/main.py` (ablation and DeepSeek
tables), and `python scripts/analyze_results.py <run>/raw_results.json --by-level --tools`
for a single run.

### 6.1 Main comparison, Qwen3 8B and 32B

AgentFlow results are under `experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/`,
baseline results under `experiments/results/NEW_baseline/`.

| Model | Mode | Tools | Thinking | GAIA | HLE | GPQA | AIME | MuSiQue |
|---|---|---|---|---:|---:|---:|---:|---:|
| 8B | Baseline | none | none | 4.8 | 3.0 | 44.9 | 18.3 | 4.0 |
| 8B | Baseline | none | orch | 8.5 | 1.5 | 50.0 | 36.7 | 6.0 |
| 8B | Baseline | direct | none | 15.2 | 4.0 | 39.9 | 20.0 | 10.5 |
| 8B | Baseline | direct | orch | 21.8 | 3.0 | 45.5 | 38.3 | 13.0 |
| 8B | AgentFlow | none | none | 5.5 | 5.0 | 46.0 | 23.3 | 5.0 |
| 8B | AgentFlow | none | orch | 7.9 | 4.5 | 55.6 | 58.3 | 6.5 |
| 8B | AgentFlow | direct | none | 7.3 | 3.5 | 42.9 | 21.7 | 11.0 |
| 8B | AgentFlow | direct | orch | 23.0 | 2.5 | 54.5 | 55.0 | 16.5 |
| 8B | AgentFlow | sub-agent | none | 12.7 | 4.5 | 42.4 | 20.0 | 11.0 |
| 8B | AgentFlow | sub-agent | orch | 23.0 | 4.0 | 58.6 | 61.7 | 14.0 |
| 8B | AgentFlow | sub-agent | subagents | 15.8 | 3.5 | 41.9 | 25.0 | 14.0 |
| 8B | AgentFlow | sub-agent | all | 24.8 | 4.0 | 58.6 | 55.0 | 15.0 |
| 32B | Baseline | none | none | 9.1 | 1.0 | 53.5 | 21.7 | 7.5 |
| 32B | Baseline | none | orch | 11.5 | 2.5 | 61.1 | 45.0 | 9.0 |
| 32B | Baseline | direct | none | 20.6 | 2.0 | 47.0 | 28.3 | 8.0 |
| 32B | Baseline | direct | orch | 23.0 | 6.5 | 60.1 | 45.0 | 15.0 |
| 32B | AgentFlow | none | none | 9.1 | 3.0 | 52.5 | 23.3 | 6.0 |
| 32B | AgentFlow | none | orch | 9.1 | 4.5 | 64.1 | 68.3 | 8.5 |
| 32B | AgentFlow | direct | none | 26.1 | 5.0 | 54.5 | 36.7 | (unfinished) |
| 32B | AgentFlow | direct | orch | 26.1 | 6.0 | 67.2 | 58.3 | 17.0 |

The 32B AgentFlow direct/none MuSiQue run left a `raw_results.partial.json` and has no
`metrics.json`, so it did not finish.

### 6.2 Orchestrator capacity (`experiments/results/orchestrator_capacity/`)

AgentFlow, sub-agent tools. Orchestrator size x sub-agent size x thinking.

| Orch | Sub | Thinking | GAIA | HLE | GPQA | AIME | MuSiQue |
|---|---|---|---:|---:|---:|---:|---:|
| 8B | 1.7B | none | 7.9 | 3.0 | 41.9 | 23.3 | 9.5 |
| 8B | 1.7B | orch | 23.0 | 2.0 | 59.1 | 56.7 | 17.5 |
| 8B | 1.7B | all | 18.2 | 2.5 | 59.1 | 56.7 | 15.5 |
| 8B | 32B | none | 13.3 | 2.5 | 46.0 | 31.7 | 11.0 |
| 8B | 32B | orch | 23.6 | 2.0 | 58.6 | 56.7 | 16.5 |
| 8B | 32B | all | 24.8 | 2.0 | 59.1 | 56.7 | 17.5 |
| 32B | 1.7B | none | 15.2 | 3.0 | 50.5 | 21.7 | 13.5 |
| 32B | 1.7B | orch | 20.6 | 4.0 | 63.1 | 51.7 | 18.5 |
| 32B | 1.7B | all | 22.4 | 3.5 | 64.1 | 53.3 | 20.5 |
| 32B | 8B | none | 18.8 | 6.0 | 48.5 | 23.3 | 13.5 |
| 32B | 8B | orch | 21.8 | 4.0 | 63.6 | 53.3 | 19.0 |
| 32B | 8B | all | 21.2 | 3.5 | 64.1 | 58.3 | 17.0 |
| 32B | 32B | none | 15.8 | 4.5 | 55.1 | 35.0 | 16.0 |
| 32B | 32B | orch | 23.0 | 4.0 | 64.6 | 55.0 | 17.0 |
| 32B | 32B | all | 23.6 | 3.5 | 63.1 | 58.3 | 20.0 |

### 6.3 Ablations (Qwen3-8B, AgentFlow, sub-agent tools, thinking on orchestrator)

Reference row is 8B AgentFlow sub-agent/orch from 6.1 (23.0 / 4.0 / 58.6 / 61.7 / 14.0).

| Ablation | GAIA | HLE | GPQA | AIME | MuSiQue | MATH500 |
|---|---:|---:|---:|---:|---:|---:|
| No `code_generator` | 21.2 | 3.0 | 59.6 | 60.0 | 15.5 | 80.0 |
| No `web_search` | 13.3 | 3.0 | 57.1 | 58.3 | 4.0 | 80.0 |
| No `text_inspector` | 16.4 | 2.0 | | | | |
| Structured memory replaced by chat history | 21.8 | 2.5 | 50.5 | 36.7 | 18.0 | 77.0 |
| Sub-agent shared memory off | 22.4 | | | | 17.0 | |
| Sub-agent shared memory on | 20.0 | | | | 16.0 | |

### 6.4 Other model families

DeepSeek-R1-Distill-Qwen, GAIA (`experiments/results/deepseek/`):

| Model | Mode | Tools | Thinking | GAIA |
|---|---|---|---|---:|
| 7B | Baseline | none | none / orch | 0.6 / 4.2 |
| 7B | Baseline | direct | none / orch | 3.6 / 3.6 |
| 7B | AgentFlow | sub-agent | none / orch / subagents / all | 5.5 / 5.5 / 4.2 / 5.5 |
| 32B | Baseline | none | none / orch | 3.0 / 9.7 |
| 32B | Baseline | direct | none / orch | 13.9 / 14.5 |

Also on disk: DeepSeek 32B HLE aligned run (3.0), a 5-question MATH500 check. OLMo 3 has
only 10-question GAIA smoke runs (Think 10.0, Instruct 0.0); no full OLMo evaluation.

### 6.5 Adaptation methods (Qwen3-8B orchestrator + frozen Qwen3-1.7B sub-agents)

The matched baseline is the unadapted 8B/1.7B configuration from 6.2. All adaptation
comparisons are made with thinking off (`none`), because SFT data has thinking stripped.

| Method | Thinking | GAIA | HLE | GPQA | AIME | MuSiQue | Source |
|---|---|---:|---:|---:|---:|---:|---|
| Base (no adaptation) | none | 7.9 | 3.0 | 41.9 | 23.3 | 9.5 | `orchestrator_capacity/*/orch8b_sub1_7b_none` |
| GEPA, GAIA-optimised full prompt | none | 6.1 | 9.0 | | | 13.5 | `gepa_inference`, 2026-05-29 |
| GEPA, common suffix only | none | 12.1 | 8.5 | 41.9 | 16.7 | 9.5 | `gepa_inference`, 2026-06-10 (prompt from `gepa/common_no`) |
| GRPO-LoRA v1 | none | 12.7 | 10.0 | 46.5 | 18.3 | 14.0 | `lora_inference/*/qwen8B_sub1_7b_none` |
| GRPO-LoRA v2 | none | 10.9 | 6.5 | 44.9 | 18.3 | 14.0 | `lora_inference/*/qwen8B_sub1_7b_none_v2` |
| SFT, folded | none | 12.7 | 8.5 | 41.4 | 10.0 | 17.5 | `sft_inference/*/qwen8B_sub1_7b_none` |
| Prefix-RFT | none | | | | | | not trained yet |

With thinking on the orchestrator (for completeness; base is 23.0 / 2.0 / 59.1 / 56.7 / 17.5):

| Method | GAIA | HLE | GPQA | AIME | MuSiQue |
|---|---:|---:|---:|---:|---:|
| GEPA, GAIA-optimised full prompt | 13.9 | 7.5 | | | 15.0 |
| GRPO-LoRA v1 | 19.4 | 6.5 | 60.6 | 61.7 | 19.5 |
| GRPO-LoRA v2 | 20.0 | 7.0 | 55.1 | 55.0 | 19.5 |

Pattern worth remembering: every adaptation method helps GAIA, HLE and MuSiQue in the
no-thinking regime and none helps AIME (SFT drops it to 10.0, with tool calls on AIME
collapsing from 79 to 20). The AIME regression is the main open question on the
adaptation side; `docs/archive/sft_status.md` section 8 has the analysis.

---

## 7. Known issues and gaps

From `docs/known-issues.md` (recorded during the refactor, deliberately not fixed because
fixing them moves reported numbers):

1. **BigCodeBench bare function body loses indentation** (high for BigCodeBench). Pinned by
   a strict `xfail`. Fix sketch is in the doc; any BigCodeBench run must be re-scored after.
2. `question_scorer` raises on a `None` ground truth (low, upstream parity).
3. Thousands separator inside a list answer splits the number (low, upstream parity).
4. `CacheManager.save_caches` skips normalisation (cosmetic).
5. **BigCodeBench scorer runs whatever `python` is on PATH** (system 3.9 on the login node).
6. **RL records a prompt 4 tokens shorter than the one vLLM used** (missing empty think
   block) in every RL run, including Prefix-RFT. Not a gradient bug; fixing it means
   re-running the GRPO baselines.

Other gaps found while writing this:

- **The 3k pool is documented but not wired in.** `data/traj_qwen32b_3k_v1/README.md` says
  it is the single pool for SFT, Prefix-RFT and GRPO, but every config still reads
  `combined_train.parquet` and `data/training/prefix_rft/prefix_demos.parquet`. Note also
  that only 180 of the 3000 pool questions appear in `combined_train.parquet`, so
  switching changes the RL training distribution, not just the file.
- **Two incompatibilities to handle when mixing pools**: `extra_info.difficulty` is `double`
  in the 3k files vs `int64` in `data/training/`, and `extra_info` field order differs, so
  the documented `data.val_files` list raises in `concatenate_datasets` unless cast.
- **GRPO adapters are gone** from `/scratch-shared/azywot/fine_tuning/lora_adapters/`.
  The 10 `lora_inference` configs cannot be re-run. The SFT adapter is safe in `data/adapters/`.
- **GEPA common-suffix code is not on `main`** (section 4).
- **No Prefix-RFT eval configs** yet.
- One unfinished run: 32B AgentFlow direct/none MuSiQue (March).

---

## 8. Recommended next steps

In this order, because each one unblocks the next:

1. **Merge `gepa-common-suffix` into `main`.** It is the only branch with unmerged code
   whose results are already reported. Rebase onto `main` and port its
   `scripts/run_experiment.py` changes into `src/agent_engine/runner/`, then run
   `pytest -q` (target: 936+ passed) and the characterization gates.
2. **Decide the training pool for Prefix-RFT and write it into the configs.** The choice
   is between keeping `combined_train.parquet` (directly comparable with the existing
   GRPO-LoRA v1/v2, but only 1358/1800 questions have a demonstration) and switching to
   `traj_qwen32b_3k_v1` (full demonstration coverage, but a different, teacher-solved
   distribution that no existing GRPO number was trained on). My recommendation is to
   keep `combined_train.parquet` for the first production run, because the point of
   Prefix-RFT here is to beat the existing GRPO baseline, and that comparison only holds
   on the same pool. Merging the 3k demonstrations into the existing store only adds 19
   questions (1358 to 1377), so it is not worth doing.
3. **Write Prefix-RFT eval configs** (add a suite to `scripts/generate_configs.py`
   mirroring `sft_inference`, thinking `none`, five benchmarks).
4. **Dry-run then submit `013`** in step mode:
   `python scripts/launch_verl.py --config experiments/configs/fine_tuning/config_prefix_rft.yaml --dry-run`,
   then `sbatch jobs/fine_tuning/013_train_prefix_rft.job`. Watch `kl_loss` (it rose in
   `012`) and `off_ratio`. Token mode is a one-variable rerun
   (`PREFIX_CONFIG=.../config_prefix_rft_tokens.yaml`) once step mode has a number.
5. **Housekeeping**: merged branches were deleted on 2026-10-18, with a bundle backup
   (section 4). Still to do: push `gepa-common-suffix`, `feat/add-context-to-subagents` and
   `feat/clean-and-retain-inference-only` back to GitHub, since each now exists only in
   this clone. Copy any adapter you will need again off `/scratch-shared` into
   `data/adapters/`.

---

## 9. Quick commands

```bash
cd /gpfs/home3/xchen1/azywot/msc-thesis
PY=/home/xchen1/.conda/envs/agent_engine/bin/python

$PY -m pytest -q                                   # full suite (936 passed on 2026-10-18)
$PY -m pytest tests/characterization -q            # behaviour locks
$PY scripts/run_experiment.py --config <cfg>.yaml  # one run
$PY scripts/analyze_results.py <run>/raw_results.json --by-level --tools
$PY scripts/generate_configs.py                    # regenerate configs (idempotent now)
$PY scripts/tables/main.py                         # ablation + DeepSeek tables
$PY scripts/launch_verl.py --config <ft_cfg>.yaml --dry-run   # check an RL config without a GPU
```
