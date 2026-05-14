# Val Set Redesign: 50-Sample Mixed Val (20 AIME / 20 Search-R1 / 10 DeepMath)

**Date:** 2026-05-14
**Status:** Approved

## Motivation

The current validation set (100 Search-R1 + 100 DeepMath = 200 rows) makes per-epoch
validation slow. The goal is a smaller, higher-signal set that:
- Reduces val wall-clock time by ~4x
- Adds AIME questions as a harder math signal not currently represented in val
- Keeps combined-only W&B reporting (single `val_0` curve)

## Data Composition

| Source | Count | HF repo |
|---|---|---|
| AIME 2024 | 10 | `HuggingFaceH4/aime_2024` |
| AIME 2025 | 10 | `yentinglin/aime_2025` |
| Search-R1 | 20 | `PeterJinGo/nq_hotpotqa_train` |
| DeepMath | 10 | `zwhe99/DeepMath-103K` |
| **Total** | **50** | |

All 50 rows are written to a single `val_combined.parquet`. The separate
`val_search.parquet` and `val_deepmath.parquet` files are still produced for
offline inspection but are no longer referenced by VERL. The test set is
unchanged (100 Search-R1 + 100 DeepMath).

AIME answers are integers 0–999 — handled by the existing uniform
`evaluate_answer()` reward path with no code changes to `reward.py`.

## Code Changes

### `src/fine_tuning/data/prepare.py`

1. **`normalise_aime_row(raw, idx, year)`** — new normaliser mapping:
   - `raw["problem"]` → `question`
   - `raw["answer"]` → `result`
   - `data_source` → `"aime_2024"` or `"aime_2025"`
   - `extra_info` → `{"idx": idx, "groundtruth": answer, "year": year}`

2. **`_download_aime_val(n_2024, n_2025, seed)`** — loads from both HF repos
   (same pattern as `download_aime()` in `scripts/download_datasets.py`),
   shuffles each year's pool independently with `seed`, takes first `n` rows from
   each. Returns a flat list of normalised rows. Raises if either year cannot
   supply the requested count.

3. **New CLI args** (both default to 0, preserving old behaviour when omitted):
   - `--n-val-aime-2024`
   - `--n-val-aime-2025`

4. **`build_val_files(search_val_rows, deepmath_val_rows, output_dir, seed, aime_val_rows=None)`**
   — `aime_val_rows` defaults to `[]`. When provided, the rows are included in
   the combined shuffle before writing `val_combined.parquet`. The per-source
   `val_search.parquet` and `val_deepmath.parquet` outputs are unchanged.

### `experiments/configs/train/config.yaml`

Change `data.val_files` from the two-file list:
```yaml
data.val_files:
  - '${BASE_DATA_DIR}/val/val_search.parquet'
  - '${BASE_DATA_DIR}/val/val_deepmath.parquet'
```
to a single path:
```yaml
data.val_files: '${BASE_DATA_DIR}/val/val_combined.parquet'
```
This produces one combined W&B val curve (`val_0`) instead of two domain-specific
ones (`val_0`, `val_1`).

### `jobs/008_prepare_fine_tuning_data.job`

Update the full-data prepare invocation:
- `--n-val-search 100` → `--n-val-search 20`
- `--n-val-math 100` → `--n-val-math 10`
- Add `--n-val-aime-2024 10 --n-val-aime-2025 10`

## Backward Compatibility

- Default values of `--n-val-aime-2024 0` / `--n-val-aime-2025 0` mean running
  `prepare.py` without the new flags reproduces the old two-source behaviour.
- `val_search.parquet` and `val_deepmath.parquet` are still written; any tooling
  that reads them offline continues to work.
- `config_smoke.yaml` is not changed — smoke runs use `data/training/smoke/`
  which has its own tiny val.
