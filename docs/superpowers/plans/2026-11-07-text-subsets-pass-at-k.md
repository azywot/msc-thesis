# Text-only GAIA/HLE subsets and pass@k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GAIA-text and HLE-text splits, seeded multi-attempt runs with per-thinking-state sampling for Qwen3, and a pass@k script, without changing anything plain GAIA/HLE runs produce.

**Architecture:** The text subsets are new JSONL splits of the existing `gaia` / `hle` datasets, built by pure selection helpers that `scripts/download_datasets.py` calls. Attempts are separate runs of one config with a `--seed` override, launched one SLURM job per (config, seed). A new analysis module computes the unbiased pass@k estimator over the runs' `raw_results.json`, after checking that the runs differ only in seed (via a config fingerprint the runner now writes).

**Tech Stack:** Python 3.11, pydantic v2 (`ModelConfig`), vLLM `SamplingParams`, pytest, bash (SLURM launchers), Jinja2 (job template).

**Spec:** `docs/superpowers/specs/2026-11-07-text-subsets-pass-at-k-design.md`

## Global Constraints

- Branch: `feat/text-subsets-pass-at-k`. **Do not run `git commit`**: stage with `git add` at the end of each task; the user approves commits.
- Plain GAIA/HLE must keep producing exactly what they produce today: no change to existing splits, data files, generated configs, scores, or existing `config.json` keys. New keys are allowed.
- Image extensions: import `SUPPORTED_IMAGE_EXTS` from `src/agent_engine/tools/image_inspector.py` (`{".jpg", ".jpeg", ".png"}`); never copy the set.
- GAIA-text: split `all_validation_text`, file `data/GAIA/all_validation_text.jsonl`, 155 rows.
- HLE-text: split `test_text_subset_200`, file `data/HLE/test_text_subset_200.jsonl`, 200 rows; first 179 `id`s equal the text-only `id`s of `test_subset_200.jsonl`, in order.
- Qwen3 sampling in text configs: thinking `temperature: 0.6, top_p: 0.95, top_k: 20`; non-thinking `temperature: 0.7, top_p: 0.8, top_k: 20`. Class defaults unchanged.
- k values: 1, 3, 5. Estimator: `1 - C(n-c, k) / C(n, k)` (Chen et al. 2021).
- Text configs: Qwen3 `baseline` and `agentflow` suites only; 24 configs.
- Style: `black` line length 100, `isort` black profile. No em dashes in docs or comments.
- Docs do not refer to thesis chapters.

## Review Focus

1. **A crashed seed** (run dir with only `raw_results.partial.json`, or a `raw_results.json` missing questions): pass@k must refuse or report clearly, never silently average over fewer attempts. Test in Task 6 (`test_missing_raw_results_rejected`, `test_question_short_of_k_excluded_and_counted`).
2. **Old runs passed to pass@k** (no `config_fingerprint` key): error unless `--allow-mismatch`. Test in Task 6 (`test_runs_without_fingerprint_rejected`).
3. **Greedy run passed to pass@k** (any role effective temperature 0, including plain Qwen3 configs): error. Test in Task 6 (`test_greedy_run_rejected`).
4. **`--seed` given but model declares its own seed in YAML**: the override still wins for every model. Test in Task 4 (`test_seed_override_beats_model_level_seed`).
5. **Regenerating all configs** wipes or alters existing suites: only new files may appear. Verification in Task 7 Step 4.

---

### Task 1: Text-only selection helpers and GAIA `source_index`

**Files:**
- Create: `src/agent_engine/datasets/text_only.py`
- Modify: `src/agent_engine/datasets/loaders/gaia.py` (metadata dict in `load`)
- Test: `tests/unit/test_text_only.py`

**Interfaces:**
- Produces:
  - `is_gaia_image_record(record: dict) -> bool`: true when `Path(record.get("file_name") or "").suffix.lower()` is in `SUPPORTED_IMAGE_EXTS`.
  - `filter_gaia_text(records: list[dict]) -> list[dict]`: drops image records, keeps order, returns **copies** with `source_index` = index in the input list.
  - `is_hle_image_example(example: dict) -> bool`: `bool(example.get("image"))`.
  - `select_hle_text(examples: list, subset: int, seed: int) -> list`: if `0 < subset < len(examples)`: `set_seed(seed); random.shuffle(copy)`; then drop image examples; then truncate to `subset`. With `subset <= 0`: no shuffle, filter only. Shuffle must be on the **full** list, before filtering, using the same call sequence as `download_hle` today (`set_seed(seed)` then `random.shuffle`).

- [ ] **Step 1: Write the failing tests**

```python
def test_gaia_image_detection_uses_supported_exts():
    assert is_gaia_image_record({"file_name": "a.PNG"})
    assert is_gaia_image_record({"file_name": "a.jpeg"})
    assert not is_gaia_image_record({"file_name": "a.xlsx"})
    assert not is_gaia_image_record({"file_name": ""})
    assert not is_gaia_image_record({"file_name": None})

def test_filter_gaia_text_keeps_order_and_source_index():
    recs = [{"file_name": ""}, {"file_name": "x.png"}, {"file_name": "y.pdf"}]
    out = filter_gaia_text(recs)
    assert [r["source_index"] for r in out] == [0, 2]
    assert "source_index" not in recs[0]  # input not mutated

def test_select_hle_text_nests_inside_original_subset():
    exs = [{"id": i, "image": "b64" if i % 4 == 0 else ""} for i in range(100)]
    set_seed(0); old = list(exs); random.shuffle(old); old = old[:20]
    new = select_hle_text(exs, subset=20, seed=0)
    old_text = [e["id"] for e in old if not e["image"]]
    assert len(new) == 20 and not any(e["image"] for e in new)
    assert [e["id"] for e in new][: len(old_text)] == old_text

def test_select_hle_text_without_subset_filters_only():
    exs = [{"id": 0, "image": ""}, {"id": 1, "image": "x"}, {"id": 2, "image": ""}]
    assert [e["id"] for e in select_hle_text(exs, subset=0, seed=0)] == [0, 2]

def test_gaia_loader_passes_source_index(tmp_path):
    # write data/GAIA/x.jsonl with one record carrying source_index=7 and one without;
    # load via GAIADataset; assert metadata["source_index"] == 7 for the first and the key
    # is absent for the second
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_text_only.py -v`
Expected: FAIL with `ModuleNotFoundError: agent_engine.datasets.text_only`

- [ ] **Step 3: Implement the four helpers in `text_only.py`; in `gaia.py`, add `"source_index": data["source_index"]` to `metadata` only when the key is present in the record**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_text_only.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/datasets/text_only.py src/agent_engine/datasets/loaders/gaia.py tests/unit/test_text_only.py
```

---

### Task 2: Build the text split files

**Files:**
- Modify: `scripts/download_datasets.py` (`download_gaia`, `download_hle`, CLI at ~line 613)
- Test: `tests/unit/test_text_subsets_data.py` (real-data acceptance; skips when files are absent)

**Interfaces:**
- Consumes: `filter_gaia_text`, `select_hle_text` (Task 1).
- Produces: CLI flags `--text-only` and `--local-only`; files `data/GAIA/all_validation_text.jsonl`, `data/HLE/test_text_subset_200.jsonl`.

Behaviour:
- `download_gaia(..., text_only=False, local_only=False)`: with `local_only`, skip `snapshot_download` and use `<output_dir>/gaia_hf_snapshot` as `snapshot_root`. With `text_only`, apply `filter_gaia_text` to `data` after building it and before `_apply_subset`, and save as `_subset_filename(f"{level}_{split}_text.jsonl", subset)`.
- `download_hle(..., text_only=False, local_only=False)`: with `local_only`, skip `snapshot_download` and `load_dataset(<output_dir>/hle_hf_snapshot, split=split)`. With `text_only`, replace the existing shuffle/truncate block by `examples = select_hle_text(examples, subset, seed)` and save as `_subset_filename(f"{split}_text.jsonl", subset)` (gives `test_text_subset_200.jsonl`). Without `text_only`, the existing code path is untouched.
- `--text-only` / `--local-only` are rejected (`parser.error`) for datasets other than `gaia` and `hle`.

- [ ] **Step 1: Write the acceptance tests** (each `pytest.mark.skipif` the files it reads are missing)

```python
def test_gaia_text_is_validation_minus_images():
    full = read_jsonl("data/GAIA/all_validation.jsonl")
    text = read_jsonl("data/GAIA/all_validation_text.jsonl")
    assert len(text) == 155
    assert not any(is_gaia_image_record(r) for r in text)
    assert [full[r["source_index"]]["Question"] for r in text] == [r["Question"] for r in text]
    assert [r["source_index"] for r in text] == sorted(r["source_index"] for r in text)

def test_hle_text_nests_existing_subset():
    old = read_jsonl("data/HLE/test_subset_200.jsonl")
    new = read_jsonl("data/HLE/test_text_subset_200.jsonl")
    old_text_ids = [r["id"] for r in old if not r["has_image"]]
    assert len(old_text_ids) == 179
    assert len(new) == 200 and len({r["id"] for r in new}) == 200
    assert not any(r["has_image"] for r in new)
    assert [r["id"] for r in new][:179] == old_text_ids

def test_existing_splits_load_unchanged():
    # GAIADataset(all_validation) -> 165 examples; HLEDataset(test_subset_200) -> 200,
    # and HLE split "test_subset_200" still resolves to test_subset_200.jsonl
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/unit/test_text_subsets_data.py -v`
Expected: the two text tests SKIP (files absent); `test_existing_splits_load_unchanged` PASS

- [ ] **Step 3: Implement the flags, then generate the files**

```bash
python scripts/download_datasets.py --dataset gaia --level all --split validation --text-only --local-only
python scripts/download_datasets.py --dataset hle --split test --subset 200 --seed 0 --text-only --local-only
```
Check the existing GAIA/HLE CLI arguments (`--level`, `--split`) match how the existing files were built before running. Then record checksums: `sha256sum data/GAIA/all_validation_text.jsonl data/HLE/test_text_subset_200.jsonl`, rerun both commands, and confirm the checksums are identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_text_subsets_data.py tests/unit/test_text_only.py -v`
Expected: PASS, no skips

- [ ] **Step 5: Stage** (data files are gitignored; stage code and tests only)

```bash
git add scripts/download_datasets.py tests/unit/test_text_subsets_data.py
```

---

### Task 3: Per-thinking-state sampling in `ModelConfig` and the vLLM provider

**Files:**
- Modify: `src/agent_engine/models/base.py` (`ModelConfig`, ~line 180)
- Modify: `src/agent_engine/models/vllm_provider.py` (both `SamplingParams` sites: multimodal path ~line 163 and `_generate_text` ~line 268)
- Modify: `src/agent_engine/runner/experiment.py` (`_config_to_dict._model_cfg`)
- Test: `tests/unit/test_sampling_by_thinking.py`

**Interfaces:**
- Produces:
  - `SamplingOverride(BaseModel)`: `temperature: float`, `top_p: float`, `top_k: int`.
  - `ThinkingSampling(BaseModel)`: `thinking: SamplingOverride`, `non_thinking: SamplingOverride`.
  - `ModelConfig.sampling_by_thinking: Optional[ThinkingSampling] = None`; a validator raises `ValueError` when set and `backend != "vllm"`.
  - `ModelConfig.sampling_for(use_thinking: bool) -> dict` returning `{"temperature", "top_p", "top_k"}`: from the matching override when set, else from the model's own fields.
  - `config.json` per-model dict gains key `"sampling_by_thinking"` (the override as a dict, or `None`).

- [ ] **Step 1: Write the failing tests**

```python
QWEN = dict(name="q", family="qwen3", path_or_id="Qwen/Qwen3-8B", role="orchestrator")
CARD = {"thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "non_thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}}

def test_sampling_for_without_override_uses_model_fields():
    m = ModelConfig(**QWEN)
    assert m.sampling_for(True) == {"temperature": 0.0, "top_p": 0.8, "top_k": 20}
    assert m.sampling_for(False) == m.sampling_for(True)

def test_sampling_for_picks_block_by_thinking():
    m = ModelConfig(**QWEN, sampling_by_thinking=CARD)
    assert m.sampling_for(True) == CARD["thinking"]
    assert m.sampling_for(False) == CARD["non_thinking"]

def test_override_rejected_on_non_vllm_backend():
    with pytest.raises(ValueError):
        ModelConfig(**QWEN, backend="mlx", sampling_by_thinking=CARD)

def test_vllm_text_path_builds_params_per_prompt(monkeypatch):
    # Build a VLLMProvider without loading a model (object.__new__ + set config, tokenizer
    # stub, llm stub recording sampling_params). Send two JSON payload prompts, one with
    # use_thinking=True and one with False. Assert recorded params[0].temperature == 0.6,
    # params[1].temperature == 0.7, top_p 0.95 / 0.8, and seed == config.seed on both.

def test_config_json_includes_sampling_by_thinking():
    # _config_to_dict on a config whose orchestrator has CARD: models.orchestrator
    # ["sampling_by_thinking"] == CARD; on a config without it: the key is None and every
    # previously existing key is unchanged
```
Check `tests/unit/test_runner_providers.py` and `tests/characterization/scripted_provider.py` for an existing way to build a provider without a GPU before writing the stub.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_sampling_by_thinking.py -v`
Expected: FAIL (`sampling_by_thinking` unknown field / `sampling_for` missing)

- [ ] **Step 3: Implement.** In both vLLM sites, replace the three `self.config.temperature/top_p/top_k` arguments with `**self.config.sampling_for(use_thinking)` using the per-prompt `use_thinking` already decoded from the payload. In the multimodal path, build one `SamplingParams` per prompt (a list) instead of one shared object, since prompts may differ in `use_thinking`. Leave `max_tokens`, `repetition_penalty`, `seed` and stop kwargs as they are.

- [ ] **Step 4: Run tests to verify they pass, plus the provider and characterization suites**

Run: `pytest tests/unit/test_sampling_by_thinking.py tests/unit/test_runner_providers.py tests/characterization -v`
Expected: PASS

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/models/base.py src/agent_engine/models/vllm_provider.py src/agent_engine/runner/experiment.py tests/unit/test_sampling_by_thinking.py
```

---

### Task 4: `--seed` override, seeded run dir, config fingerprint

**Files:**
- Modify: `src/agent_engine/config/loader.py` (new function)
- Modify: `src/agent_engine/runner/experiment.py` (`main`, `run_experiment`, `_make_run_dir`, config.json write)
- Test: `tests/unit/test_seed_override.py`

**Interfaces:**
- Produces:
  - `apply_seed_override(config: ExperimentConfig, seed: int) -> None` in `loader.py`: sets `config.seed` and **every** model's `seed` to `seed`, including models that declared their own.
  - `config_fingerprint(config: ExperimentConfig) -> str` in `loader.py`: sha256 hex of `json.dumps(d, sort_keys=True)` where `d = config.model_dump(mode="json")` with `seed`, `output_dir`, and every `models.<role>.seed` removed.
  - `_make_run_dir(output_dir: Path, split: str, seed: Optional[int] = None)`: name `{split}_seed{seed}_{date}_{job_id}` when `seed` is not None, else exactly today's `{split}_{date}_{job_id}`.
  - CLI: `--seed N` (int, optional) on `agent_engine.runner.experiment.main`.
  - `config.json` gains key `"config_fingerprint"` (always written, also without `--seed`).

Order inside `run_experiment`: load config, then `apply_seed_override` if `args.seed is not None`, then compute the fingerprint (before `_apply_gpu_assignment` mutates model configs), then `_make_run_dir(..., seed=args.seed)`, then everything else as today. `set_seed(config.seed)` already runs after this point.

- [ ] **Step 1: Write the failing tests**

```python
def test_seed_override_sets_all_seeds(tmp_path):
    cfg = load_experiment_config(write_yaml(tmp_path, seed=0, roles=["orchestrator", "web_search"]))
    apply_seed_override(cfg, 3)
    assert cfg.seed == 3 and all(m.seed == 3 for m in cfg.models.values())

def test_seed_override_beats_model_level_seed(tmp_path):
    cfg = load_experiment_config(write_yaml(tmp_path, seed=0, model_seed=11))
    apply_seed_override(cfg, 4)
    assert cfg.get_model("orchestrator").seed == 4

def test_fingerprint_ignores_seed_and_output_dir(tmp_path):
    a = load_experiment_config(write_yaml(tmp_path, seed=0)); b = load_experiment_config(write_yaml(tmp_path, seed=0))
    apply_seed_override(b, 2); b.output_dir = Path("/elsewhere")
    assert config_fingerprint(a) == config_fingerprint(b)

def test_fingerprint_sees_hidden_fields(tmp_path):
    a = load_experiment_config(write_yaml(tmp_path)); b = load_experiment_config(write_yaml(tmp_path))
    b.tools.web_tool_provider = "tavily"
    assert config_fingerprint(a) != config_fingerprint(b)

def test_run_dir_name_with_and_without_seed(tmp_path):
    assert _make_run_dir(tmp_path, "all_validation_text", seed=3)[0].name.startswith("all_validation_text_seed3_")
    name = _make_run_dir(tmp_path, "all_validation", seed=None)[0].name
    assert name.startswith("all_validation_") and "_seed" not in name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_seed_override.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_seed_override'`

- [ ] **Step 3: Implement the two loader functions, the `--seed` flag, the run-dir change and the `config_fingerprint` key**

- [ ] **Step 4: Run tests to verify they pass, plus the full unit suite**

Run: `pytest tests/unit -q`
Expected: PASS (no failures that are not also present on `main`)

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/config/loader.py src/agent_engine/runner/experiment.py tests/unit/test_seed_override.py
```

---

### Task 5: Multi-seed SLURM launching

**Files:**
- Modify: `jobs/scripts/generate_job.py`, `jobs/templates/experiment.job.j2`, `jobs/scripts/launch_experiment.sh`, `experiments/scripts/run_all_in_folder.sh`
- Test: `tests/unit/test_generate_job_seed.py`

**Interfaces:**
- Consumes: `--seed` on `scripts/run_experiment.py` (Task 4).
- Produces:
  - `generate_job(config_path, output_path=None, project_dir=None, seed: Optional[int] = None)` and CLI `--seed`. With `seed`: job file `jobs/generated/{name}_seed{seed}.job`, `#SBATCH --job-name={name}_seed{seed}`, log file `{name}_seed{seed}_%A.log`, `PYTHONHASHSEED={seed}`, and the run command gets `--seed {seed}`. Without `seed`: rendered job file byte-identical to today.
  - `launch_experiment.sh <config> [--seed N]` forwards `--seed` to `generate_job.py`.
  - `run_all_in_folder.sh <folder> [--dataset X] [--local] [--seeds 0,1,2,3,4]`: for each config, for each seed, submit (or run locally with `--seed`). Without `--seeds`, behaviour unchanged. Keep the 30 s stagger between submissions.

- [ ] **Step 1: Write the failing tests**

```python
def test_job_without_seed_is_unchanged(tmp_path):
    # render generate_job for an existing qwen3 baseline config with seed=None into tmp_path;
    # assert "--seed" not in text and the job-name line equals f"#SBATCH --job-name={cfg.name}"

def test_job_with_seed(tmp_path):
    # seed=3: assert f"--job-name={cfg.name}_seed3" in text, "--seed 3" in text,
    # "PYTHONHASHSEED=3" in text, and the default output path ends with f"{cfg.name}_seed3.job"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_generate_job_seed.py -v`
Expected: FAIL (`generate_job() got an unexpected keyword argument 'seed'`)

- [ ] **Step 3: Implement the four file changes.** In the template, render the seed argument from a variable that is the empty string when no seed is given, so the no-seed output stays byte-identical.

- [ ] **Step 4: Run tests, and dry-check the shell scripts**

Run: `pytest tests/unit/test_generate_job_seed.py -v && bash -n experiments/scripts/run_all_in_folder.sh && bash -n jobs/scripts/launch_experiment.sh`
Expected: PASS, no syntax errors

- [ ] **Step 5: Stage**

```bash
git add jobs/scripts/generate_job.py jobs/templates/experiment.job.j2 jobs/scripts/launch_experiment.sh experiments/scripts/run_all_in_folder.sh tests/unit/test_generate_job_seed.py
```

---

### Task 6: pass@k estimator and script

**Files:**
- Create: `src/agent_engine/analysis/pass_at_k.py`
- Create: `scripts/pass_at_k.py` (thin CLI)
- Test: `tests/unit/test_pass_at_k.py`

**Interfaces:**
- Consumes: `config.json` keys `config_fingerprint`, `seed`, `dataset.name`, `models.<role>.temperature` and `models.<role>.sampling_by_thinking` (Tasks 3, 4); `raw_results.json` fields `question_id`, `question`, `correct`, `metadata`; `get_spec` from `src/agent_engine/datasets/spec.py`.
- Produces:
  - `pass_at_k(n: int, c: int, k: int) -> float`; raises `ValueError` if `k > n`.
  - `class PassAtKInputError(Exception)`.
  - `@dataclass RunData`: `run_dir: Path`, `config: dict`, `results: list[dict]`, `metrics: dict | None`.
  - `load_run(path: Path) -> RunData`: accepts a run dir or a `raw_results.json` path; raises `PassAtKInputError` if `raw_results.json` is missing (also when only `raw_results.partial.json` exists).
  - `validate_runs(runs: list[RunData], allow_mismatch: bool = False) -> list[str]`: raises `PassAtKInputError` (unless `allow_mismatch`, in which case it returns the messages as warnings) on: fewer than 2 runs; duplicate seeds; differing or missing `config_fingerprint` (message also lists differing visible `config.json` fields); differing question sets by `question_id`; same `question_id` with different `question` text. Always raises (even with `allow_mismatch`) if any role's effective temperature is 0 (both blocks checked when `sampling_by_thinking` is set).
  - `compute_pass_at_k(runs: list[RunData], ks: list[int], by_level: bool = False) -> dict` with keys: `ks`, `n_runs`, `runs` (list of `{run_dir, seed, accuracy}`), `accuracy_mean`, `accuracy_std` (sample std, ddof=1), `overall` (`{"pass@k": float, ...}`, `num_questions`, `excluded_short_of_k: {k: count}`), and `per_level` (same shape, keyed by `str(metadata[level_field])` from `get_spec(dataset_name)`) when `by_level`. "Correct" is the result's `correct` field; questions missing from a run count as not attempted in that run (n_i lower), not as wrong.
  - CLI: `python scripts/pass_at_k.py RUN [RUN ...] [--k 1 3 5] [--by-level] [--out FILE] [--allow-mismatch]`; prints a table and writes JSON when `--out` is given.

- [ ] **Step 1: Write the failing tests**

```python
def test_estimator_known_values():
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(5, 0, 3) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 2, 3) == pytest.approx(1 - comb(3, 3) / comb(5, 3))  # 0.9
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)

def test_pass1_equals_mean_accuracy(tmp_path):
    runs = make_runs(tmp_path, correct=[[1, 0, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1]])
    out = compute_pass_at_k(runs, [1, 3, 5])
    assert out["overall"]["pass@1"] == pytest.approx(out["accuracy_mean"])
    assert out["overall"]["pass@5"] == pytest.approx(1.0)

def test_by_level_uses_spec_level_field(tmp_path):
    # gaia runs with metadata level 1/2: per_level has keys "1" and "2" with correct counts

def test_config_mismatch_rejected(tmp_path):        # different fingerprints -> PassAtKInputError
def test_duplicate_seeds_rejected(tmp_path):
def test_runs_without_fingerprint_rejected(tmp_path):
def test_question_set_mismatch_rejected(tmp_path):
def test_allow_mismatch_returns_warnings(tmp_path):
def test_greedy_run_rejected(tmp_path):              # temperature 0, no sampling_by_thinking
def test_greedy_rejected_even_with_allow_mismatch(tmp_path):
def test_missing_raw_results_rejected(tmp_path):     # dir with only raw_results.partial.json
def test_question_short_of_k_excluded_and_counted(tmp_path):
    # 5 runs, one lacks question 2 -> n=4 for q2: included for k=1,3, excluded for k=5;
    # excluded_short_of_k == {1: 0, 3: 0, 5: 1}
```
`make_runs` is a local helper writing `config.json` (same fingerprint, distinct seeds, temperature 0.6), `raw_results.json` and `metrics.json` per run.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_pass_at_k.py -v`
Expected: FAIL with `ModuleNotFoundError: agent_engine.analysis.pass_at_k`

- [ ] **Step 3: Implement the module and the CLI**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_pass_at_k.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Stage**

```bash
git add src/agent_engine/analysis/pass_at_k.py scripts/pass_at_k.py tests/unit/test_pass_at_k.py
```

---

### Task 7: Text-split Qwen3 configs

**Files:**
- Modify: `scripts/generate_configs.py` (`SUITES`, `make_config`)
- Create (generated): `experiments/configs/qwen3/baseline_text/{gaia,hle}/*.yaml` (8 + 8), `experiments/configs/qwen3/agentflow_text/{gaia,hle}/*.yaml` (4 + 4)
- Test: `tests/unit/test_generate_text_configs.py`

**Interfaces:**
- Consumes: `sampling_by_thinking` YAML block (Task 3).
- Produces: suites `baseline_text` and `agentflow_text`, each a copy of `baseline` / `agentflow` with:
  - `datasets: ["gaia", "hle"]`
  - `split_overrides: {"gaia": {"split": "all_validation_text", "display": "GAIA-text"}, "hle": {"split": "test_text_subset_200", "display": "HLE-text"}}`
  - `config_subdir`: `qwen3/baseline_text` / `qwen3/agentflow_text`
  - `name_prefix`: `TEXT_baseline` / `TEXT_AF`; `output_dir_root`: `./experiments/results/TEXT_baseline` / `./experiments/results/TEXT_agentflow`
  - `description_tag`: the parent's tag with `text-only, sampled` added
  - `wandb_project`: `benchmarks_text`
  - new key `"sampling_by_thinking": QWEN3_MODEL_CARD_SAMPLING` (module constant with the Global Constraints values); when present, `make_config` appends a `sampling_by_thinking:` block under the orchestrator entry of the models block. Only the orchestrator is declared in these suites; sub-agents fall back to it and get the right block per call.

- [ ] **Step 1: Write the failing tests**

```python
def test_text_suite_mirrors_parent(tmp_path):
    # generate_suite("baseline_text", tmp_path) and generate_suite("baseline", tmp_path);
    # same stems per dataset; for each pair, load both YAMLs and assert they differ only in
    # name, description, output_dir, wandb_project, dataset.split and
    # models.orchestrator.sampling_by_thinking

def test_text_suite_counts_and_values(tmp_path):
    # baseline_text: 8 gaia + 8 hle; agentflow_text: 4 + 4; every file loads with
    # load_experiment_config; orchestrator.sampling_by_thinking.thinking.temperature == 0.6
    # and non_thinking.temperature == 0.7; splits are all_validation_text / test_text_subset_200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_generate_text_configs.py -v`
Expected: FAIL (`KeyError: 'baseline_text'`)

- [ ] **Step 3: Implement the suites and the models-block addition**

- [ ] **Step 4: Regenerate everything and confirm only new files appear**

Run: `python scripts/generate_configs.py && git status --porcelain experiments/configs`
Expected: only `?? experiments/configs/qwen3/baseline_text/` and `?? experiments/configs/qwen3/agentflow_text/`; no ` M` or ` D` lines. Then `pytest tests/unit/test_generate_text_configs.py -v`: PASS.

- [ ] **Step 5: Stage**

```bash
git add scripts/generate_configs.py tests/unit/test_generate_text_configs.py experiments/configs/qwen3/baseline_text experiments/configs/qwen3/agentflow_text
```

---

### Task 8: Docs and end-to-end check

**Files:**
- Modify: `docs/pipelines/evaluation.md`, `CLAUDE.md` ("Running experiments", "Config essentials", "Outputs")

- [ ] **Step 1: Document**, in plain prose with no em dashes and no thesis-chapter references:
  - GAIA-text / HLE-text: what is excluded and why, sizes (155, 200), the nesting property of HLE-text, and the two regeneration commands from Task 2.
  - `sampling_by_thinking`: what it does, that it is vLLM-only, and that plain Qwen3 configs remain greedy.
  - `--seed`, the `_seed{N}_` run-dir naming, `config_fingerprint`, and `run_all_in_folder.sh --seeds 0,1,2,3,4`.
  - `scripts/pass_at_k.py` usage, the estimator, and the cache caveat sentence from the spec verbatim: "attempts share a common search cache; queries not yet cached are fetched independently by each attempt."

- [ ] **Step 2: Local smoke run of the whole path** (CPU-only machines: skip and say so)

The CLI has no `subset_num` override, so copy `experiments/configs/qwen3/baseline_text/gaia/qwen8B_no_tools_none.yaml` to the scratchpad, set `subset_num: 3` and `use_wandb: false` in the copy, run `python scripts/run_experiment.py --config <copy> --seed 0` and `--seed 1`, then `python scripts/pass_at_k.py <run0> <run1> --k 1 --by-level`.
Expected: both run dirs contain `_seed0_` / `_seed1_`, `config.json` has equal `config_fingerprint`, and pass@k prints a table.

- [ ] **Step 3: Full test suite**

Run: `pytest -q`
Expected: no failures beyond those present on `main` before this branch.

- [ ] **Step 4: Stage**

```bash
git add docs/pipelines/evaluation.md CLAUDE.md
```
