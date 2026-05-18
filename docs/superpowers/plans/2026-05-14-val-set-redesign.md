# Val Set Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the VERL validation set from 200 rows to 50 (20 Search-R1 + 10 DeepMath + 10 AIME 2024 + 10 AIME 2025) writing a single `val_combined.parquet` that VERL reads.

**Architecture:** Extend `prepare.py` with a new `normalise_aime_row` normaliser and `_download_aime_val` downloader following the same lazy-HF-import pattern as `_download_deepmath`. `build_val_files` gains an optional `aime_val_rows` arg. The VERL config switches from two separate val files to the single combined file. The smoke job section explicitly passes `--n-val-aime-2024 0 --n-val-aime-2025 0` so smoke val stays tiny.

**Tech Stack:** Python 3.11, pandas, HuggingFace `datasets`, pytest

---

## File Map

| File | Change |
|---|---|
| `src/fine_tuning/data/prepare.py` | Add `normalise_aime_row`, `_download_aime_val`; update `build_val_files` signature; add CLI args |
| `tests/unit/test_data_prepare.py` | Add `TestNormaliseAime`, `TestDownloadAimeVal`, `TestBuildValFilesWithAime` |
| `experiments/configs/train/config.yaml` | `data.val_files`: list of two → single string |
| `jobs/008_prepare_fine_tuning_data.job` | Full-data section: new flags; smoke section: disable AIME |

---

### Task 1: Add `normalise_aime_row` and its tests

**Files:**
- Modify: `src/fine_tuning/data/prepare.py`
- Modify: `tests/unit/test_data_prepare.py`

- [ ] **Step 1: Write the failing tests**

Open `tests/unit/test_data_prepare.py`. Add this import at the top alongside the existing ones:

```python
from fine_tuning.data.prepare import (
    normalise_search_r1_row,
    normalise_deepmath_row,
    normalise_aime_row,        # new
    validate_parquet_schema,
    REQUIRED_COLS,
)
```

Append this class at the bottom of the file:

```python
class TestNormaliseAime:
    def test_basic_row_2024(self):
        raw = {"problem": "Find the value of x", "answer": "42"}
        row = normalise_aime_row(raw, idx=0, year=2024)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "aime_2024"
        assert row["question"] == "Find the value of x"
        assert row["result"] == "42"
        assert row["extra_info"]["idx"] == 0
        assert row["extra_info"]["groundtruth"] == "42"
        assert row["extra_info"]["year"] == 2024

    def test_basic_row_2025(self):
        raw = {"problem": "Compute the sum", "answer": "113"}
        row = normalise_aime_row(raw, idx=3, year=2025)
        assert row["data_source"] == "aime_2025"
        assert row["extra_info"]["year"] == 2025
        assert row["extra_info"]["idx"] == 3

    def test_answer_coerced_to_str(self):
        raw = {"problem": "Q", "answer": 7}
        row = normalise_aime_row(raw, idx=0, year=2024)
        assert row["result"] == "7"
        assert row["extra_info"]["groundtruth"] == "7"

    def test_missing_fields_produce_empty_strings(self):
        row = normalise_aime_row({}, idx=0, year=2024)
        assert row["question"] == ""
        assert row["result"] == ""
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /gpfs/home3/xchen1/azywot/msc-thesis
pytest tests/unit/test_data_prepare.py::TestNormaliseAime -v
```

Expected: `ImportError: cannot import name 'normalise_aime_row'`

- [ ] **Step 3: Implement `normalise_aime_row` in `prepare.py`**

In `src/fine_tuning/data/prepare.py`, add `Optional` to the typing import:

```python
from typing import Any, Dict, List, Optional, Tuple
```

Then insert `normalise_aime_row` directly after `normalise_deepmath_row` (around line 162):

```python
def normalise_aime_row(raw: Dict[str, Any], idx: int, year: int) -> Dict[str, Any]:
    """Convert an AIME row (HuggingFaceH4/aime_2024 or yentinglin/aime_2025) to VERL schema.

    Both repos use `problem` + `answer` fields.
    """
    question = str(raw.get("problem") or "")
    answer = str(raw.get("answer") if raw.get("answer") is not None else "")
    return {
        "data_source": f"aime_{year}",
        "question": question,
        "result": answer,
        "extra_info": {"idx": idx, "groundtruth": answer, "year": year},
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_data_prepare.py::TestNormaliseAime -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/fine_tuning/data/prepare.py tests/unit/test_data_prepare.py
git commit -m "feat: add normalise_aime_row to prepare.py"
```

---

### Task 2: Add `_download_aime_val` and its tests

**Files:**
- Modify: `src/fine_tuning/data/prepare.py`
- Modify: `tests/unit/test_data_prepare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_prepare.py`:

```python
class TestDownloadAimeVal:
    """Tests for _download_aime_val — HF calls are mocked."""

    def _fake_rows(self, n):
        return [{"problem": f"Problem {i}", "answer": str(i)} for i in range(n)]

    def test_returns_correct_count_and_sources(self):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(30)
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake):
            rows = _download_aime_val(n_2024=10, n_2025=10, seed=42)

        assert len(rows) == 20
        assert sum(1 for r in rows if r["data_source"] == "aime_2024") == 10
        assert sum(1 for r in rows if r["data_source"] == "aime_2025") == 10

    def test_raises_if_not_enough_rows(self):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(5)  # only 5 available, need 10
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake):
            with pytest.raises(RuntimeError, match="only 5 rows available"):
                _download_aime_val(n_2024=10, n_2025=10, seed=42)

    def test_zero_n_skips_that_year(self):
        from unittest.mock import patch, call
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(30)
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake) as mock_load:
            rows = _download_aime_val(n_2024=0, n_2025=10, seed=42)

        assert len(rows) == 10
        assert all(r["data_source"] == "aime_2025" for r in rows)
        assert mock_load.call_count == 1  # only called for 2025

    def test_indices_are_per_year(self):
        from unittest.mock import patch
        from fine_tuning.data.prepare import _download_aime_val

        fake = self._fake_rows(30)
        with patch("fine_tuning.data.prepare._load_dataset_for_aime", return_value=fake):
            rows = _download_aime_val(n_2024=3, n_2025=3, seed=42)

        rows_2024 = [r for r in rows if r["data_source"] == "aime_2024"]
        rows_2025 = [r for r in rows if r["data_source"] == "aime_2025"]
        assert [r["extra_info"]["idx"] for r in rows_2024] == [0, 1, 2]
        assert [r["extra_info"]["idx"] for r in rows_2025] == [0, 1, 2]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_data_prepare.py::TestDownloadAimeVal -v
```

Expected: `ImportError` or `AttributeError` — `_download_aime_val` and `_load_dataset_for_aime` don't exist yet.

- [ ] **Step 3: Implement `_load_dataset_for_aime` and `_download_aime_val` in `prepare.py`**

Add both functions after `normalise_aime_row`. The thin `_load_dataset_for_aime` wrapper exists purely so tests can patch it without patching the global `datasets` module:

```python
def _load_dataset_for_aime(repo_id: str) -> List[Dict[str, Any]]:
    """Load an AIME HF dataset to a plain list. Exists as a seam for testing."""
    from datasets import load_dataset
    return list(load_dataset(repo_id, split="train"))


def _download_aime_val(
    n_2024: int,
    n_2025: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Download AIME 2024 and 2025 val rows from HuggingFace.

    Shuffles each year's pool independently before taking the first n rows
    so the sample is random but deterministic given the seed.
    """
    rows: List[Dict[str, Any]] = []
    for repo_id, year, n in [
        ("HuggingFaceH4/aime_2024", 2024, n_2024),
        ("yentinglin/aime_2025", 2025, n_2025),
    ]:
        if n == 0:
            continue
        pool = _load_dataset_for_aime(repo_id)
        rng = random.Random(seed)
        rng.shuffle(pool)
        if len(pool) < n:
            raise RuntimeError(
                f"AIME {year}: only {len(pool)} rows available, need {n}. "
                f"The dataset may have changed — lower --n-val-aime-{year}."
            )
        for i, raw in enumerate(pool[:n]):
            rows.append(normalise_aime_row(dict(raw), idx=i, year=year))
    return rows
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_data_prepare.py::TestDownloadAimeVal -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/fine_tuning/data/prepare.py tests/unit/test_data_prepare.py
git commit -m "feat: add _download_aime_val to prepare.py"
```

---

### Task 3: Update `build_val_files` to accept AIME rows and test it

**Files:**
- Modify: `src/fine_tuning/data/prepare.py`
- Modify: `tests/unit/test_data_prepare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_data_prepare.py`:

```python
class TestBuildValFilesWithAime:
    """Integration tests for build_val_files with the new aime_val_rows arg."""

    def _search_row(self, i):
        return {
            "data_source": "hotpotqa",
            "question": f"Q{i}",
            "result": f"A{i}",
            "extra_info": {"idx": i, "groundtruth": f"A{i}", "golden_answers": []},
        }

    def _math_row(self, i):
        return {
            "data_source": "deepmath",
            "question": f"Q{i}",
            "result": f"A{i}",
            "extra_info": {"idx": i, "groundtruth": f"A{i}"},
        }

    def _aime_row(self, i, year=2024):
        return {
            "data_source": f"aime_{year}",
            "question": f"Q{i}",
            "result": f"A{i}",
            "extra_info": {"idx": i, "groundtruth": f"A{i}", "year": year},
        }

    def test_combined_includes_aime_rows(self, tmp_path):
        from fine_tuning.data.prepare import build_val_files

        search = [self._search_row(i) for i in range(3)]
        math = [self._math_row(i) for i in range(2)]
        aime = [self._aime_row(i) for i in range(2)]

        paths = build_val_files(search, math, tmp_path, seed=42, aime_val_rows=aime)

        combined = pd.read_parquet(paths["combined"])
        assert len(combined) == 7  # 3 + 2 + 2
        sources = set(combined["data_source"].tolist())
        assert "aime_2024" in sources

    def test_combined_indices_are_unique(self, tmp_path):
        from fine_tuning.data.prepare import build_val_files

        search = [self._search_row(i) for i in range(3)]
        math = [self._math_row(i) for i in range(2)]
        aime = [self._aime_row(i) for i in range(2)]

        paths = build_val_files(search, math, tmp_path, seed=42, aime_val_rows=aime)

        combined = pd.read_parquet(paths["combined"])
        indices = [row["idx"] for row in combined["extra_info"].tolist()]
        assert len(indices) == len(set(indices)), "idx values must be unique in combined"

    def test_no_aime_arg_backward_compat(self, tmp_path):
        from fine_tuning.data.prepare import build_val_files

        search = [self._search_row(0)]
        math = [self._math_row(0)]

        paths = build_val_files(search, math, tmp_path, seed=42)

        combined = pd.read_parquet(paths["combined"])
        assert len(combined) == 2
        assert "aime_2024" not in set(combined["data_source"].tolist())

    def test_separate_search_deepmath_files_unchanged(self, tmp_path):
        from fine_tuning.data.prepare import build_val_files

        search = [self._search_row(i) for i in range(3)]
        math = [self._math_row(i) for i in range(2)]
        aime = [self._aime_row(0)]

        paths = build_val_files(search, math, tmp_path, seed=42, aime_val_rows=aime)

        # Per-source files should not include AIME rows
        search_df = pd.read_parquet(paths["search"])
        math_df = pd.read_parquet(paths["deepmath"])
        assert len(search_df) == 3
        assert len(math_df) == 2
        assert "aime_2024" not in set(search_df["data_source"].tolist())
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_data_prepare.py::TestBuildValFilesWithAime -v
```

Expected: `TypeError` — `build_val_files` does not accept `aime_val_rows`.

- [ ] **Step 3: Update `build_val_files` in `prepare.py`**

Replace the existing `build_val_files` signature and combined-building block:

```python
def build_val_files(
    search_val_rows: List[Dict[str, Any]],
    deepmath_val_rows: List[Dict[str, Any]],
    output_dir: Path,
    seed: int,
    aime_val_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    """Write val parquet files under <output_dir>/val/.

    Returns a dict with keys "search", "deepmath", "combined".

    val_search.parquet   — Search-R1 held-out
    val_deepmath.parquet — DeepMath held-out
    val_combined.parquet — all sources merged and shuffled; used by VERL's data.val_files
    """
    val_dir = output_dir / "val"

    search_path = _write_val(search_val_rows, val_dir / "val_search.parquet")
    deepmath_path = _write_val(deepmath_val_rows, val_dir / "val_deepmath.parquet")

    all_rows = search_val_rows + deepmath_val_rows + (aime_val_rows or [])
    combined = []
    for i, row in enumerate(all_rows):
        row = dict(row)
        row["extra_info"] = {**row["extra_info"], "idx": i}
        combined.append(row)
    rng = random.Random(seed)
    rng.shuffle(combined)
    combined_path = _write_val(combined, val_dir / "val_combined.parquet")

    return {"search": search_path, "deepmath": deepmath_path, "combined": combined_path}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/unit/test_data_prepare.py::TestBuildValFilesWithAime -v
```

Expected: 4 passed

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest tests/unit/test_data_prepare.py -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/fine_tuning/data/prepare.py tests/unit/test_data_prepare.py
git commit -m "feat: build_val_files accepts optional aime_val_rows"
```

---

### Task 4: Wire the new CLI args into `main()` in `prepare.py`

**Files:**
- Modify: `src/fine_tuning/data/prepare.py`

No new tests — existing unit tests cover the functions; CLI parsing is tested implicitly.

- [ ] **Step 1: Add the two new arguments to the `argparse` block in `main()`**

In `main()`, after the `--n-val-math` argument block (~line 545), add:

```python
    parser.add_argument(
        "--n-val-aime-2024", type=int, default=10,
        help="Number of AIME 2024 rows held out for validation (default: 10)",
    )
    parser.add_argument(
        "--n-val-aime-2025", type=int, default=10,
        help="Number of AIME 2025 rows held out for validation (default: 10)",
    )
```

- [ ] **Step 2: Call `_download_aime_val` and pass rows to `build_val_files`**

In `main()`, after the DeepMath download block and before `build_combined_train(...)`, add:

```python
    aime_val_rows: List[Dict[str, Any]] = []
    if args.n_val_aime_2024 > 0 or args.n_val_aime_2025 > 0:
        print(
            f"Downloading AIME val "
            f"({args.n_val_aime_2024} from 2024, {args.n_val_aime_2025} from 2025)..."
        )
        aime_val_rows = _download_aime_val(
            args.n_val_aime_2024,
            args.n_val_aime_2025,
            args.seed,
        )
```

Then update the `build_val_files` call (currently `build_val_files(search_val_rows, math_val_rows, output_dir, args.seed)`) to:

```python
    build_val_files(search_val_rows, math_val_rows, output_dir, args.seed, aime_val_rows=aime_val_rows)
```

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/unit/test_data_prepare.py -v
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add src/fine_tuning/data/prepare.py
git commit -m "feat: add --n-val-aime-2024/2025 CLI args to prepare.py"
```

---

### Task 5: Update `config.yaml` — single val file

**Files:**
- Modify: `experiments/configs/train/config.yaml`

- [ ] **Step 1: Replace the two-file list with a single path**

In `experiments/configs/train/config.yaml`, find (~line 49):

```yaml
  data.val_files:                                           # two files → VERL logs val_0 (search) and val_1 (math) separately in W&B
    - '${BASE_DATA_DIR}/val/val_search.parquet'
    - '${BASE_DATA_DIR}/val/val_deepmath.parquet'
```

Replace with:

```yaml
  data.val_files: '${BASE_DATA_DIR}/val/val_combined.parquet'
```

- [ ] **Step 2: Commit**

```bash
git add experiments/configs/train/config.yaml
git commit -m "config: switch data.val_files to single val_combined.parquet"
```

---

### Task 6: Update `jobs/008_prepare_fine_tuning_data.job`

**Files:**
- Modify: `jobs/008_prepare_fine_tuning_data.job`

- [ ] **Step 1: Update the full-data section**

Find the full-data `python src/fine_tuning/data/prepare.py` call (~line 72). Replace:

```bash
python src/fine_tuning/data/prepare.py \
    --n-search       900 \
    --n-math         900 \
    --n-val-search   100 \
    --n-val-math     100 \
    --n-test-search  100 \
    --n-test-math    100 \
    --search-source  both \
    --hotpot-ratio   0.85 \
    --deepmath-min-difficulty 5 \
    --output-dir data/training \
    --seed 42
```

with:

```bash
python src/fine_tuning/data/prepare.py \
    --n-search          900 \
    --n-math            900 \
    --n-val-search       20 \
    --n-val-math         10 \
    --n-val-aime-2024    10 \
    --n-val-aime-2025    10 \
    --n-test-search     100 \
    --n-test-math       100 \
    --search-source  both \
    --hotpot-ratio   0.85 \
    --deepmath-min-difficulty 5 \
    --output-dir data/training \
    --seed 42
```

- [ ] **Step 2: Update the smoke section to disable AIME**

Find the smoke `python src/fine_tuning/data/prepare.py` call (~line 98). Replace:

```bash
python src/fine_tuning/data/prepare.py \
    --n-search    50 \
    --n-math      50 \
    --n-val-search 3 \
    --n-val-math   3 \
    --output-dir data/training/smoke \
    --seed 42
```

with:

```bash
python src/fine_tuning/data/prepare.py \
    --n-search         50 \
    --n-math           50 \
    --n-val-search      3 \
    --n-val-math        3 \
    --n-val-aime-2024   0 \
    --n-val-aime-2025   0 \
    --output-dir data/training/smoke \
    --seed 42
```

- [ ] **Step 3: Update the inline comments in the job header**

Find (~line 12):
```bash
#   data/training/val/val_*.parquet             (100 search + 100 math = 200 val)
```
Replace with:
```bash
#   data/training/val/val_combined.parquet      (20 search + 10 math + 10 aime2024 + 10 aime2025 = 50 val)
```

- [ ] **Step 4: Commit**

```bash
git add jobs/008_prepare_fine_tuning_data.job
git commit -m "job: update 008 for new 50-sample val set (20 search / 10 math / 10+10 AIME)"
```
