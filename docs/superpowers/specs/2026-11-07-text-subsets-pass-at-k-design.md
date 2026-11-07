# Text-only GAIA/HLE subsets and pass@k

**Date:** 2026-11-07
**Status:** approved 2026-11-07
**Update (2026-11-07, after implementation):** at the user's request, Qwen3 on vLLM now
defaults to the model-card sampling for every config that does not pin a sampling field,
not only the text configs. The "Qwen3 class default is not changed" decision below is
superseded; see `CHANGELOG.md` (Changed) and `docs/pipelines/evaluation.md`.
**Branch:** `feat/text-subsets-pass-at-k`

## Why

Two things are wrong with the current GAIA and HLE numbers for the paper.

1. **Unanswerable questions.** Every GAIA and HLE config runs with `image_inspector`
   disabled, so questions whose attachment is an image cannot be solved by any method.
   They count as failures for every system, which drags scores down and adds noise to
   comparisons. GAIA `all_validation` has 10 such questions (of 165); HLE
   `test_subset_200` has 21 (of 200).
2. **Only accuracy.** The paper reports single-run accuracy. pass@k (k in {1, 3, 5}),
   which is common in agentic papers, separates "the system can solve this" from "it
   solves it reliably", and pass@1 averaged over several attempts is more stable than one
   run.

## Goals

- A text-only GAIA validation split (GAIA-text) and a text-only HLE subset of 200
  questions (HLE-text), usable from a config like any other split.
- A way to run the same config several times with different seeds, and a script that
  turns those runs into pass@1, pass@3 and pass@5, overall and per GAIA level / HLE
  category.
- New Qwen3 configs for the text splits so the user can launch fresh runs (5 seeds each).

## Non-goals

- Reusing existing seed-0 runs as one of the attempts. The user will rerun everything
  fresh.
- Running the experiments. This work delivers the tooling and configs; the user launches
  the jobs.
- Enabling `image_inspector` or any vision model.
- Changing how a single answer is scored (`evaluate_gaia`, `evaluate_qa` stay as they are).
- Logging pass@k to W&B (possible follow-up).

## Anti-goals (would make this a failure even if it works)

- Changing the existing `gaia` / `hle` splits, results, or old configs. Plain GAIA and
  HLE must keep producing exactly what they produce today.
- An HLE-text subset that is not a uniform random sample of text-only HLE, or that
  cannot be regenerated deterministically.
- pass@k numbers computed over runs that silently differ in config (other than seed) or
  in question set.

## Decisions made with the user

- **GAIA-text drops only image questions.** Questions with other attachments
  (spreadsheets, PDFs, audio, zip, ...) stay, because the setup partly handles them.
  Result: 155 of 165 validation questions.
- **HLE-text is 200 text-only questions.** Built by taking the same seed-0 shuffle of
  full HLE that produced the existing `test_subset_200`, walking down it, and keeping
  text-only questions until there are 200. Consequences: it is a uniform random sample of
  text-only HLE, and its first 179 questions are exactly the 179 text-only questions of
  the existing subset (in the same order), followed by 21 new ones.
- **k in {1, 3, 5}**, from **fresh sampling**: 5 attempts per question.
- **Qwen3 only, for now.** Text configs are made for the Qwen3 `baseline` and `agentflow`
  suites only (8 + 4 configs per dataset, 24 in total, 120 jobs at 5 seeds). No DeepSeek
  or OLMo text configs; no capacity, ablation, LoRA or SFT suites.
- **Qwen3 samples with the model-card settings in the text configs.** Today every Qwen3
  run is greedy (`ModelConfig.temperature` defaults to `0.0` in
  `src/agent_engine/models/base.py` and no Qwen3 config overrides it), so five seeds would
  be identical. In the text configs, every Qwen3 role gets the Qwen3 model-card settings
  for its thinking state under the config's `thinking_mode`:
  - role thinks: `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`
  - role does not think: `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`

  Everything else (`repetition_penalty`, `max_tokens`, ...) stays as the existing configs
  have it. The values are written explicitly into the generated YAML; the Qwen3 class
  default is **not** changed, so plain GAIA/HLE stays greedy.

  *Mechanism (my call):* sub-agent roles without their own `models.<role>` entry reuse
  the orchestrator's provider, and vLLM sampling params come from the provider's
  `ModelConfig`, so one model instance serves both thinking and non-thinking calls. A
  per-role temperature therefore cannot live on the role; it is chosen **per call** from
  the call's thinking flag. `ModelConfig` gets an optional `sampling_by_thinking` block
  (`thinking` / `non_thinking`, each with `temperature`, `top_p`, `top_k`); when set, the
  vLLM provider uses the matching block for each prompt. When absent (all existing
  configs), behaviour is unchanged. vLLM backend only; other backends reject the field. Consequence for the paper:
  the new Qwen3 text numbers are sampled rather than greedy, so they replace the old table
  rather than sitting beside it.

## Design

### 1. Text-only subsets as new splits (not new dataset names)

*My call.* GAIA-text and HLE-text are new **splits** of the existing `gaia` and `hle`
datasets, i.e. new JSONL files next to the old ones:

| Subset | Config | File | Size |
|---|---|---|---|
| GAIA-text | `name: gaia`, `split: all_validation_text` | `data/GAIA/all_validation_text.jsonl` | 155 |
| HLE-text | `name: hle`, `split: test_text_subset_200` | `data/HLE/test_text_subset_200.jsonl` | 200 |

Why splits rather than new dataset names (`gaia_text`, `hle_text`): the dataset name is
used as a key in `src/agent_engine/datasets/spec.py` (prompt template, per-level
stratification), in the search cache path (`cache/<provider>/<dataset_name>/`), in W&B
logging, and in the analysis scripts. A new split needs none of those touched, keeps the
same prompts and scoring by construction, and reuses the existing search cache. In the
paper the subsets are still called GAIA-text and HLE-text.

**What counts as an image:**
- GAIA: the attachment's extension (from `file_name`, lowercased) is in
  `SUPPORTED_IMAGE_EXTS` from `src/agent_engine/tools/image_inspector.py`
  (`.jpg`, `.jpeg`, `.png`). Import that set rather than copying it.
- HLE: the source record has a non-empty `image` field (the same condition that sets
  `has_image` in the existing download code).

**Generating the files.** Add a `--text-only` flag to `scripts/download_datasets.py`
for `--dataset gaia` and `--dataset hle`.
- GAIA: filter the already-built records, write `<level>_<split>_text.jsonl`.
- HLE: shuffle full HLE exactly as today (`set_seed(seed); random.shuffle(examples)` on
  the full split), *then* filter to text-only, *then* truncate to `--subset`. Write
  `<split>_text_subset_<N>.jsonl`. The filter must sit between shuffle and truncate; that
  ordering is what makes the nesting property above true.
- HLE must be built from the **existing local snapshot** in `data/HLE/hle_hf_snapshot`
  (no re-download), because if upstream HLE has changed since the original subset was
  made, the shuffle no longer lines up. Provide a way to build from the local snapshot
  only (for example a `--local-only` flag or a small standalone script); builder's choice.

**Stable question identity.** `question_id` is the line index in the file, so the new
files renumber questions. To allow question-level comparison with old runs:
- GAIA-text records get a `source_index` field: the question's line index in
  `all_validation.jsonl`. The GAIA loader copies it into `metadata` when present.
- HLE already carries its own `id` (the HLE hash) in metadata; nothing extra needed.

**Acceptance checks for the files** (as a test or a one-off verification script):
- `all_validation_text.jsonl` has 155 lines, none with an image extension, and it equals
  `all_validation.jsonl` minus the 10 image questions, in the original order.
- `test_text_subset_200.jsonl` has 200 lines, none with an image; its first 179 `id`s
  equal, in order, the `id`s of the text-only rows of `test_subset_200.jsonl`; no `id`
  repeats.
- Re-running the generator gives byte-identical files.

### 2. Running several seeded attempts

Each attempt is a full, separate run of the same config with a different seed (0 to 4).
The runner already passes `config.seed` to vLLM's sampling params, so different seeds
give different samples **provided temperature > 0** (see the Qwen3 sampling decision
above). The pass@k script fails loudly if any model role in any input run has
`temperature: 0`.

- Add a `--seed N` CLI override to `scripts/run_experiment.py` /
  `src/agent_engine/runner/experiment.py` that replaces `config.seed` before anything is
  seeded. The config loader copies the top-level `seed` into each model config at load
  time (`src/agent_engine/config/loader.py`), so the override must also overwrite every
  model-level seed; otherwise vLLM keeps seed 0. A test must check this. The effective
  seed is already written to `config.json`.
- The run directory name should include the seed (for example `..._seed3_...`) so five
  runs of one config are easy to tell apart. Builder's choice of exact format; do not
  change the directory name when `--seed` is not given.
- A SLURM helper (extend `experiments/scripts/run_all_in_folder.sh` with a
  `--seeds 0,1,2,3,4` option, or a sibling script) submits one job per (config, seed).
  Seeds run as independent parallel jobs.

**Search cache across parallel seeds (my call: no code change).** All seeds share
`cache/serper/<dataset>/`. `CacheManager` is already safe for concurrent writers (fcntl
lock, atomic `os.replace`, merge on save), but each process loads the cache once at
startup and writes back every 10 questions and at the end. So seeds share every query
that was cached before they started; a query nobody has cached yet is fetched
independently by each seed. We accept this rather than change cache code that plain
runs also use. The paper statement is: "attempts share a common search cache; queries
not yet cached are fetched independently by each attempt."

### 3. pass@k script

New `scripts/pass_at_k.py`:

```
python scripts/pass_at_k.py <run_dir_or_raw_results.json> [...] --k 1 3 5 [--by-level] [--out pass_at_k.json]
```

- Inputs: the raw results of n runs of one config (n = 5 in practice).
- Validation before computing anything: all runs have the same config except seed;
  seeds are distinct; all runs cover the same question set (matched by `question_id`,
  cross-checked by question text). Any mismatch is an error unless `--allow-mismatch` is
  passed, in which case the script reports which questions or fields differ.
- *My call:* `config.json` is a partial dump (`_config_to_dict` in
  `src/agent_engine/runner/experiment.py` leaves out fields such as `web_tool_provider`,
  `gepa_prompt_path`, `lora_adapter_path`). So the runner adds one new key to
  `config.json`: a hash of the full resolved config with seeds (top-level and per model),
  output paths, run name and timestamps removed. This is purely additive; no existing key
  changes. The pass@k script compares this hash, and on mismatch also diffs the visible
  `config.json` fields to say what differs. Runs without the hash (old runs) are an
  error unless `--allow-mismatch` is passed.
- Per question: n_i attempts, c_i correct. pass@k uses the unbiased estimator from Chen
  et al. 2021 (Codex paper): `1 - C(n_i - c_i, k) / C(n_i, k)`, averaged over questions.
  Questions with n_i < k (a crashed run) are excluded from pass@k for that k and counted
  in the output.
- "Correct" is whatever the existing per-example scoring writes to `raw_results.json`
  (the same field `analyze_results.py` uses for accuracy). Do not re-score.
- Output: printed table and a JSON file with pass@k overall, per GAIA level / HLE
  category (via the existing level field from `spec.py`), number of questions, n, and the
  list of runs used. Also report the per-run accuracy for each seed and its mean and
  standard deviation, since that is what pass@1 should match.
- The estimator lives in an importable function (for example in
  `src/agent_engine/datasets/evaluators/metrics.py` or a new `pass_at_k.py` module) so
  other analysis scripts and tests can use it.

### 4. Configs

- `scripts/generate_configs.py`: generate text-split configs for the Qwen3 `baseline`
  and `agentflow` suites only, mirroring each existing Qwen3 GAIA/HLE config in those
  suites one to one (same models, tools, modes), with only these differences: the split,
  the per-role sampling settings above, and names / `output_dir`s that say `text`. They
  go in their own suite folders, `experiments/configs/qwen3/baseline_text/{gaia,hle}/`
  and `experiments/configs/qwen3/agentflow_text/{gaia,hle}/` (*my call*: the generator
  deletes every `*/*.yaml` under a suite's folder before writing, so a text suite sharing
  `qwen3/baseline/` would wipe the existing configs).
- No existing generated config may change. The generator currently reproduces the
  committed configs exactly, so the check is: regenerate, and `git diff` shows only new
  files.

### 5. Docs

- Update `docs/pipelines/evaluation.md` (and `CLAUDE.md`'s "Running experiments" /
  "Outputs" if needed) with: the two text splits and how they are built, the `--seed`
  flag, the multi-seed launcher, and `pass_at_k.py`.
- Do not refer to thesis chapters in repo docs.

## Tests

- Unit tests for the pass@k estimator: known values (for example n=5, c=1: pass@1 = 0.2,
  pass@5 = 1.0; c=0 gives 0; c=n gives 1), k > n handling, and averaging over questions.
- Unit tests for the GAIA image filter and the HLE shuffle-filter-truncate order, on
  small synthetic records (including the nesting property).
- Unit test for the pass@k input validation (config differing in something other than
  seed is rejected; differing question sets are rejected).
- Existing tests still pass; loading `gaia/all_validation` and `hle/test_subset_200`
  gives exactly what it gave before.

## Left to the builder

- Exact flag names and run directory format, as long as old behaviour is unchanged
  when the new flags are not used.
- Where the estimator function lives.
- Whether the HLE local-snapshot build is a flag or a separate script.
