# GEPA Data Plan Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the GEPA optimization data pipeline — download Search-R1 + DeepMath in the
right ratios, split into D_feedback / D_pareto / Test, and wire new GEPA configs for `gaia.yaml`
and `math.yaml` that use this clean data instead of the thesis benchmark splits.

**Architecture:** A new `src/gepa_integration/data/` package handles download + split + disk
I/O. A thin loader converts the JSON to `DatasetExample` objects. `run_gepa.py` gains a
`gepa_data_file` code path in `_load_examples` and `run_splits`; everything else (optimize,
evaluate, diff modes) is unchanged. New GEPA configs replace `gaia.yaml` and introduce
`math.yaml`.

**Tech Stack:** Python 3.11+, HuggingFace `datasets` (streaming), pytest, PyYAML. Existing
`_download_search_r1` / `_download_deepmath` from `src/fine_tuning/data/prepare.py` are
reused via import (both are under `src/` on the same `sys.path`).

---

## File Map

| File | Role |
|------|------|
| `src/gepa_integration/data/__init__.py` | Create — package marker (empty) |
| `src/gepa_integration/data/prepare.py` | Create — download + split + save GEPA data |
| `src/gepa_integration/data/loader.py` | Create — load JSON examples as `DatasetExample` |
| `experiments/configs/gepa/gaia.yaml` | Modify — replace old thesis-data config with Search-R1/DeepMath |
| `experiments/configs/gepa/math.yaml` | Create — new GEPA config for AIME prompt optimization |
| `scripts/run_gepa.py` | Modify — add `gepa_data_file` pathway in `_load_examples` + `run_splits` |
| `jobs/gepa/005_prep_gepa_data.job` | Create — SLURM job for data download on cluster |
| `jobs/gepa/000_prep_gepa_data.job` | Modify — remove gaia section (superseded by job 005) |
| `tests/gepa_integration/test_gepa_data.py` | Create — tests for prepare + loader |

---

## Task 1: Create `src/gepa_integration/data/` package

**Files:**
- Create: `src/gepa_integration/data/__init__.py`

- [ ] **Step 1: Create the package marker**

Create `src/gepa_integration/data/__init__.py` with content:

```python
"""GEPA optimization data: download, split, and load Search-R1 + DeepMath."""
```

- [ ] **Step 2: Commit**

```bash
git add src/gepa_integration/data/__init__.py
git commit -m "chore(gepa): add gepa_integration/data package skeleton"
```

---

## Task 2: Write failing tests for `prepare.py`

**Files:**
- Create: `tests/gepa_integration/test_gepa_data.py`

Before writing any implementation, write all tests. The tests mock HuggingFace downloads
so they run fully offline.

- [ ] **Step 1: Create the test file**

Create `tests/gepa_integration/test_gepa_data.py`:

```python
"""Tests for src/gepa_integration/data/prepare.py and loader.py.

All HuggingFace downloads are mocked — tests run fully offline.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ---------------------------------------------------------------------------
# Helpers shared by multiple tests
# ---------------------------------------------------------------------------

def _make_search_rows(n, source="hotpotqa"):
    """n fake Search-R1 rows."""
    return [
        {
            "question_id": i,
            "question": f"Search Q {i}",
            "answer": f"Answer {i}",
            "answer_aliases": [f"Answer {i}", f"Alt {i}"],
            "data_source": source,
        }
        for i in range(n)
    ]


def _make_math_rows(n):
    """n fake DeepMath rows."""
    return [
        {
            "question_id": i,
            "question": f"Math Q {i}",
            "answer": f"{i}",
            "answer_aliases": [f"{i}"],
            "data_source": "deepmath",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# prepare.py: _make_gepa_splits
# ---------------------------------------------------------------------------

class TestMakeGEPASplits:
    def _import(self):
        from gepa_integration.data.prepare import make_gepa_splits
        return make_gepa_splits

    def test_correct_split_sizes(self):
        make = self._import()
        all_ids = list(range(300))
        splits = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        assert len(splits["train"]) == 150
        assert len(splits["val"]) == 50
        assert len(splits["test"]) == 100

    def test_no_overlap(self):
        make = self._import()
        all_ids = list(range(300))
        splits = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert len(all_assigned) == 300
        assert set(splits["train"]) & set(splits["val"]) == set()
        assert set(splits["train"]) & set(splits["test"]) == set()
        assert set(splits["val"]) & set(splits["test"]) == set()

    def test_deterministic_given_seed(self):
        make = self._import()
        all_ids = list(range(300))
        s1 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        s2 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        assert s1 == s2

    def test_different_seeds_give_different_splits(self):
        make = self._import()
        all_ids = list(range(300))
        s1 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=1)
        s2 = make(all_ids, n_feedback=150, n_pareto=50, n_test=100, seed=2)
        assert s1["train"] != s2["train"]

    def test_raises_if_ids_too_few(self):
        make = self._import()
        with pytest.raises(ValueError, match="n_feedback.*n_pareto.*n_test"):
            make(list(range(10)), n_feedback=150, n_pareto=50, n_test=100, seed=1)


# ---------------------------------------------------------------------------
# prepare.py: build_gepa_examples
# ---------------------------------------------------------------------------

class TestBuildGEPAExamples:
    def _import(self):
        from gepa_integration.data.prepare import build_gepa_examples
        return build_gepa_examples

    def test_assigns_sequential_ids(self):
        build = self._import()
        search = _make_search_rows(5)
        math = _make_math_rows(3)
        examples = build(
            feedback_search=search[:2], feedback_math=math[:1],
            pareto_search=search[2:4], pareto_math=math[1:2],
            test_search=search[4:], test_math=math[2:],
            seed=1,
        )
        ids = [ex["question_id"] for ex in examples]
        assert ids == list(range(len(examples)))

    def test_total_example_count(self):
        build = self._import()
        examples = build(
            feedback_search=_make_search_rows(3), feedback_math=_make_math_rows(1),
            pareto_search=_make_search_rows(2), pareto_math=_make_math_rows(1),
            test_search=_make_search_rows(2), test_math=_make_math_rows(1),
            seed=1,
        )
        assert len(examples) == 10  # 4 + 3 + 3

    def test_required_fields_present(self):
        build = self._import()
        examples = build(
            feedback_search=_make_search_rows(1), feedback_math=_make_math_rows(1),
            pareto_search=_make_search_rows(1), pareto_math=[],
            test_search=_make_search_rows(1), test_math=[],
            seed=1,
        )
        for ex in examples:
            assert "question_id" in ex
            assert "question" in ex
            assert "answer" in ex
            assert "answer_aliases" in ex
            assert "data_source" in ex

    def test_question_id_type_is_int(self):
        build = self._import()
        examples = build(
            feedback_search=_make_search_rows(2), feedback_math=[],
            pareto_search=[], pareto_math=[],
            test_search=[], test_math=[],
            seed=1,
        )
        for ex in examples:
            assert isinstance(ex["question_id"], int)


# ---------------------------------------------------------------------------
# prepare.py: save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_save_and_reload(self, tmp_path):
        from gepa_integration.data.prepare import save_gepa_data

        examples = [
            {"question_id": 0, "question": "Q?", "answer": "A",
             "answer_aliases": ["A"], "data_source": "hotpotqa"},
        ]
        splits = {"train": [0], "val": [], "test": []}
        data_file = tmp_path / "all_examples.json"
        splits_file = tmp_path / "splits.json"

        save_gepa_data(examples, splits, data_file, splits_file)

        loaded_examples = json.loads(data_file.read_text())
        loaded_splits = json.loads(splits_file.read_text())

        assert loaded_examples == examples
        assert loaded_splits == splits

    def test_output_dir_created_if_missing(self, tmp_path):
        from gepa_integration.data.prepare import save_gepa_data

        data_file = tmp_path / "sub" / "nested" / "data.json"
        splits_file = tmp_path / "splits.json"
        save_gepa_data([], {"train": [], "val": [], "test": []}, data_file, splits_file)
        assert data_file.exists()


# ---------------------------------------------------------------------------
# loader.py: load_gepa_examples
# ---------------------------------------------------------------------------

class TestLoadGEPAExamples:
    def _write_examples(self, tmp_path, examples):
        p = tmp_path / "all_examples.json"
        p.write_text(json.dumps(examples))
        return p

    def test_loads_by_id(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [
            {"question_id": 0, "question": "Q0", "answer": "A0",
             "answer_aliases": ["A0"], "data_source": "hotpotqa"},
            {"question_id": 1, "question": "Q1", "answer": "A1",
             "answer_aliases": [], "data_source": "deepmath"},
            {"question_id": 2, "question": "Q2", "answer": "A2",
             "answer_aliases": ["A2", "alt"], "data_source": "nq"},
        ]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [0, 2])
        assert len(examples) == 2
        assert {ex.question_id for ex in examples} == {0, 2}

    def test_returns_dataset_examples(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        from agent_engine.datasets.base import DatasetExample
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": [], "data_source": "hotpotqa"}]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [0])
        assert isinstance(examples[0], DatasetExample)

    def test_question_and_answer_set_correctly(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 5, "question": "What is X?", "answer": "42",
                "answer_aliases": ["forty-two"], "data_source": "nq"}]
        p = self._write_examples(tmp_path, raw)
        ex = load_gepa_examples(p, [5])[0]
        assert ex.question == "What is X?"
        assert ex.answer == "42"
        assert ex.question_id == 5

    def test_answer_aliases_in_metadata(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": ["A", "alt"], "data_source": "hotpotqa"}]
        p = self._write_examples(tmp_path, raw)
        ex = load_gepa_examples(p, [0])[0]
        assert ex.metadata["answer_aliases"] == ["A", "alt"]

    def test_unknown_ids_ignored(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": [], "data_source": "nq"}]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [0, 99])  # 99 doesn't exist
        assert len(examples) == 1

    def test_empty_ids_returns_empty(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        raw = [{"question_id": 0, "question": "Q", "answer": "A",
                "answer_aliases": [], "data_source": "nq"}]
        p = self._write_examples(tmp_path, raw)
        examples = load_gepa_examples(p, [])
        assert examples == []

    def test_file_not_found_raises(self, tmp_path):
        from gepa_integration.data.loader import load_gepa_examples
        with pytest.raises(FileNotFoundError):
            load_gepa_examples(tmp_path / "missing.json", [0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/gepa_integration/test_gepa_data.py -v 2>&1 | head -40
```

Expected: `ImportError: No module named 'gepa_integration.data.prepare'` (and similar for loader).

---

## Task 3: Implement `src/gepa_integration/data/loader.py`

**Files:**
- Create: `src/gepa_integration/data/loader.py`

- [ ] **Step 1: Implement loader**

Create `src/gepa_integration/data/loader.py`:

```python
"""Load GEPA data JSON files as DatasetExample objects."""
from __future__ import annotations

import json
from pathlib import Path

from agent_engine.datasets.base import DatasetExample


def load_gepa_examples(data_file: Path, question_ids: list[int]) -> list[DatasetExample]:
    """Load examples from a GEPA data JSON file, filtered by question_id.

    Args:
        data_file: Path to ``all_examples.json`` written by ``prepare.py``.
        question_ids: IDs to load. Unknown IDs are silently ignored.

    Returns:
        List of ``DatasetExample`` objects in the order they appear in the file.
    """
    if not data_file.exists():
        raise FileNotFoundError(f"GEPA data file not found: {data_file}")

    with open(data_file, encoding="utf-8") as f:
        all_examples = json.load(f)

    id_set = set(question_ids)
    result = []
    for raw in all_examples:
        if raw["question_id"] not in id_set:
            continue
        result.append(
            DatasetExample(
                question_id=raw["question_id"],
                question=raw["question"],
                answer=raw["answer"],
                metadata={
                    "answer_aliases": raw.get("answer_aliases", []),
                    "data_source": raw["data_source"],
                },
            )
        )
    return result
```

- [ ] **Step 2: Run loader tests**

```bash
pytest tests/gepa_integration/test_gepa_data.py::TestLoadGEPAExamples -v
```

Expected: all 7 loader tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/gepa_integration/data/__init__.py src/gepa_integration/data/loader.py \
    tests/gepa_integration/test_gepa_data.py
git commit -m "feat(gepa): add gepa_integration/data loader + tests"
```

---

## Task 4: Implement `src/gepa_integration/data/prepare.py`

**Files:**
- Create: `src/gepa_integration/data/prepare.py`

- [ ] **Step 1: Implement prepare module**

Create `src/gepa_integration/data/prepare.py`:

```python
"""GEPA optimization data preparation.

Downloads Search-R1 (HotpotQA + NQ) and DeepMath-103K from HuggingFace,
splits into D_feedback / D_pareto / Test, and saves to disk.

Two presets are supported — configured via the ``--preset`` CLI flag:

  gaia   — 75 % Search-R1 (85 % HotpotQA / 15 % NQ) + 25 % DeepMath (no difficulty filter)
           Targets GAIA / HLE / MuSiQue failure modes: retrieval chaining, single-shot
           tool trust, evidence failure.

  math   — 75 % DeepMath (difficulty ≥ 5) + 25 % Search-R1
           Targets AIME failure mode: direct reasoning without Coder delegation.

Split sizes (both presets):
  D_feedback (train): 150   D_pareto (val): 50   Test: 100   Total: 300

Outputs (relative to --output-dir):
  all_examples.json     — all 300 examples with sequential question_id 0..299
  (splits written to the path given by --splits-out)

Usage:
  python src/gepa_integration/data/prepare.py \\
      --preset gaia \\
      --output-dir data/gepa/gaia \\
      --splits-out experiments/configs/gepa/splits/gaia_gepa_splits.json \\
      --seed 1

  python src/gepa_integration/data/prepare.py \\
      --preset math \\
      --output-dir data/gepa/math \\
      --splits-out experiments/configs/gepa/splits/math_gepa_splits.json \\
      --seed 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Make fine_tuning importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from fine_tuning.data.prepare import (
    _download_deepmath,
    _download_search_r1,
)

# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

# Each preset maps to (search_n, deepmath_n, deepmath_min_difficulty) per split.
# Tuple layout: (search_feedback, search_pareto, search_test,
#                math_feedback,   math_pareto,   math_test,
#                deepmath_min_difficulty)

_PRESETS: dict[str, dict] = {
    "gaia": {
        "search": {"feedback": 112, "pareto": 37, "test": 75},
        "math":   {"feedback": 38,  "pareto": 13, "test": 25},
        "deepmath_min_difficulty": 1,   # no meaningful filter
        "hotpot_ratio": 0.85,
        "description": "75% Search-R1 + 25% DeepMath (no difficulty filter)",
    },
    "math": {
        "search": {"feedback": 38,  "pareto": 13, "test": 25},
        "math":   {"feedback": 112, "pareto": 37, "test": 75},
        "deepmath_min_difficulty": 5,   # difficulty >= 5
        "hotpot_ratio": 0.85,
        "description": "75% DeepMath (difficulty ≥ 5) + 25% Search-R1",
    },
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def make_gepa_splits(
    all_ids: list[int],
    n_feedback: int,
    n_pareto: int,
    n_test: int,
    seed: int = 1,
) -> dict[str, list[int]]:
    """Randomly partition ``all_ids`` into feedback / pareto / test splits.

    Args:
        all_ids:    All example IDs (must have length >= n_feedback + n_pareto + n_test).
        n_feedback: D_feedback size (GEPA ``train`` split).
        n_pareto:   D_pareto size (GEPA ``val`` split).
        n_test:     Held-out test size.
        seed:       Random seed for reproducibility.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` — splits use the
        same key names as the existing thesis-benchmark splits so ``run_gepa.py``
        can read them unchanged.
    """
    total = n_feedback + n_pareto + n_test
    if len(all_ids) < total:
        raise ValueError(
            f"n_feedback ({n_feedback}) + n_pareto ({n_pareto}) + n_test ({n_test}) "
            f"= {total} exceeds available IDs ({len(all_ids)})"
        )
    ids = list(all_ids)
    random.Random(seed).shuffle(ids)
    return {
        "train": sorted(ids[:n_feedback]),
        "val":   sorted(ids[n_feedback : n_feedback + n_pareto]),
        "test":  sorted(ids[n_feedback + n_pareto : total]),
    }


def _norm_to_example(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalised fine_tuning row to GEPA example schema."""
    return {
        "question": row["question"],
        "answer": row["result"],
        "answer_aliases": row["extra_info"].get("golden_answers", [row["result"]]),
        "data_source": row["data_source"],
    }


def build_gepa_examples(
    feedback_search: list[dict],
    feedback_math: list[dict],
    pareto_search: list[dict],
    pareto_math: list[dict],
    test_search: list[dict],
    test_math: list[dict],
    seed: int = 1,
) -> list[dict]:
    """Combine and shuffle split buckets, then assign sequential ``question_id``.

    Within each split, search and math rows are shuffled together so the
    GEPA minibatch sampler sees a mixed distribution. Splits are concatenated
    in order: feedback → pareto → test, so IDs 0..149 are D_feedback,
    150..199 are D_pareto, and 200..299 are Test (for the default 150/50/100
    split; sizes depend on preset).

    Args:
        feedback_search / feedback_math: rows for D_feedback split
        pareto_search   / pareto_math:   rows for D_pareto split
        test_search     / test_math:     rows for Test split
        seed: random seed for within-split shuffling

    Returns:
        List of example dicts with ``question_id`` 0..N-1.
    """
    rng = random.Random(seed)

    def _mix(search_rows, math_rows):
        combined = [_norm_to_example(r) for r in search_rows + math_rows]
        rng.shuffle(combined)
        return combined

    ordered: list[dict] = (
        _mix(feedback_search, feedback_math)
        + _mix(pareto_search, pareto_math)
        + _mix(test_search, test_math)
    )
    for i, ex in enumerate(ordered):
        ex["question_id"] = i
    return ordered


def save_gepa_data(
    examples: list[dict],
    splits: dict[str, list[int]],
    data_file: Path,
    splits_file: Path,
) -> None:
    """Write examples and splits to disk.

    Args:
        examples:    List of example dicts (with ``question_id``).
        splits:      ``{"train": [...], "val": [...], "test": [...]}``
        data_file:   Output path for ``all_examples.json``.
        splits_file: Output path for the splits JSON.
    """
    data_file.parent.mkdir(parents=True, exist_ok=True)
    splits_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------

def prepare(
    preset_name: str,
    output_dir: Path,
    splits_out: Path,
    seed: int = 1,
) -> tuple[list[dict], dict[str, list[int]]]:
    """Download, split, and save GEPA data for the given preset.

    Args:
        preset_name: ``"gaia"`` or ``"math"``.
        output_dir:  Directory for ``all_examples.json``.
        splits_out:  Path for the splits JSON (``{"train":..., "val":..., "test":...}``).
        seed:        Random seed (default 1 — matches GEPA config default).

    Returns:
        ``(examples, splits)`` tuple for inspection / testing.
    """
    if preset_name not in _PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Choose from: {list(_PRESETS)}")

    p = _PRESETS[preset_name]
    search_cfg = p["search"]
    math_cfg = p["math"]

    print(f"Preparing GEPA data for preset '{preset_name}': {p['description']}")

    print(
        f"  Downloading Search-R1 "
        f"({search_cfg['feedback']} feedback + {search_cfg['pareto']} pareto "
        f"+ {search_cfg['test']} test, hotpot_ratio={p['hotpot_ratio']})..."
    )
    search_feedback, search_pareto, search_test = _download_search_r1(
        n_train=search_cfg["feedback"],
        n_val=search_cfg["pareto"],
        n_test=search_cfg["test"],
        seed=seed,
        search_source="both",
        hotpot_ratio=p["hotpot_ratio"],
    )

    print(
        f"  Downloading DeepMath-103K "
        f"({math_cfg['feedback']} feedback + {math_cfg['pareto']} pareto "
        f"+ {math_cfg['test']} test, "
        f"min_difficulty={p['deepmath_min_difficulty']})..."
    )
    math_feedback, math_pareto, math_test = _download_deepmath(
        n_train=math_cfg["feedback"],
        n_val=math_cfg["pareto"],
        n_test=math_cfg["test"],
        seed=seed,
        min_difficulty=p["deepmath_min_difficulty"],
    )

    examples = build_gepa_examples(
        feedback_search=search_feedback, feedback_math=math_feedback,
        pareto_search=search_pareto,   pareto_math=math_pareto,
        test_search=search_test,       test_math=math_test,
        seed=seed,
    )

    n_total = len(examples)
    n_feedback = search_cfg["feedback"] + math_cfg["feedback"]
    n_pareto = search_cfg["pareto"] + math_cfg["pareto"]
    n_test = search_cfg["test"] + math_cfg["test"]

    splits = make_gepa_splits(
        all_ids=[ex["question_id"] for ex in examples],
        n_feedback=n_feedback,
        n_pareto=n_pareto,
        n_test=n_test,
        seed=seed,
    )

    data_file = Path(output_dir) / "all_examples.json"
    save_gepa_data(examples, splits, data_file, Path(splits_out))

    print(f"  Wrote {n_total} examples to {data_file}")
    print(f"  D_feedback (train): {len(splits['train'])}")
    print(f"  D_pareto   (val):   {len(splits['val'])}")
    print(f"  Test:               {len(splits['test'])}")
    print(f"  Splits saved to:    {splits_out}")

    return examples, splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and split GEPA optimization data (Search-R1 + DeepMath)."
    )
    parser.add_argument(
        "--preset", choices=list(_PRESETS), required=True,
        help="Data preset: 'gaia' (75%% Search-R1) or 'math' (75%% DeepMath diff≥5)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for all_examples.json",
    )
    parser.add_argument(
        "--splits-out", type=Path, required=True,
        help="Output path for splits JSON (train/val/test question IDs)",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    prepare(
        preset_name=args.preset,
        output_dir=args.output_dir,
        splits_out=args.splits_out,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run prepare tests**

```bash
pytest tests/gepa_integration/test_gepa_data.py -v
```

Expected: all tests pass (download tests are mocked via the `build_gepa_examples` / `make_gepa_splits` / `save_gepa_data` public interface — no network calls).

- [ ] **Step 3: Commit**

```bash
git add src/gepa_integration/data/prepare.py tests/gepa_integration/test_gepa_data.py
git commit -m "feat(gepa): GEPA data prepare module (Search-R1 + DeepMath)"
```

---

## Task 5: Update `scripts/run_gepa.py` — `gepa_data_file` support

**Files:**
- Modify: `scripts/run_gepa.py` (two functions: `_load_examples`, `run_splits`)

The change adds a `gepa_data_file` code path. When present in the config, examples are
loaded from the local JSON instead of DatasetRegistry. `run_splits` just verifies the
splits file exists (splits are pre-generated by `prepare.py`).

- [ ] **Step 1: Update `_load_examples`**

Edit `scripts/run_gepa.py`. Find:

```python
def _load_examples(cfg: dict, question_ids: list) -> list:
    ds_cfg = DatasetConfig(
        name=cfg["benchmark"],
        split=cfg.get("dataset_split", _get_default_split(cfg["benchmark"])),
        data_dir=Path(cfg.get("data_dir", "./data")),
        subset_num=-1,
    )
    dataset = DatasetRegistry.get(ds_cfg)
    all_examples = list(dataset)
    id_set = set(question_ids)
    return [ex for ex in all_examples if ex.question_id in id_set]
```

Replace with:

```python
def _load_examples(cfg: dict, question_ids: list) -> list:
    gepa_data_file = cfg.get("gepa_data_file")
    if gepa_data_file:
        from gepa_integration.data.loader import load_gepa_examples
        data_path = Path(gepa_data_file)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        return load_gepa_examples(data_path, question_ids)

    ds_cfg = DatasetConfig(
        name=cfg["benchmark"],
        split=cfg.get("dataset_split", _get_default_split(cfg["benchmark"])),
        data_dir=Path(cfg.get("data_dir", "./data")),
        subset_num=-1,
    )
    dataset = DatasetRegistry.get(ds_cfg)
    all_examples = list(dataset)
    id_set = set(question_ids)
    return [ex for ex in all_examples if ex.question_id in id_set]
```

- [ ] **Step 2: Update `run_splits`**

Edit `scripts/run_gepa.py`. Find the start of `run_splits`:

```python
def run_splits(cfg: dict, config_path: Path) -> None:
    root = _repo_root(config_path)
    existing_results = Path(cfg["existing_results"])
    if not existing_results.is_absolute():
        existing_results = root / existing_results
```

Replace through the closing `print(f"  Saved: {splits_file}")` with:

```python
def run_splits(cfg: dict, config_path: Path) -> None:
    root = _repo_root(config_path)

    # New path: splits were pre-generated by src/gepa_integration/data/prepare.py.
    if cfg.get("gepa_data_file"):
        splits_file = Path(cfg["splits_file"])
        if not splits_file.is_absolute():
            splits_file = root / splits_file
        if splits_file.exists():
            with open(splits_file) as f:
                splits = json.load(f)
            print(f"Splits file already exists: {splits_file}")
            print(f"  train (D_feedback): {len(splits['train'])}")
            print(f"  val   (D_pareto):   {len(splits['val'])}")
            print(f"  test:               {len(splits['test'])}")
            print("To regenerate, re-run: python src/gepa_integration/data/prepare.py")
        else:
            print(f"ERROR: splits file not found: {splits_file}")
            print(
                "Run prepare.py first:\n"
                "  python src/gepa_integration/data/prepare.py "
                f"--preset {cfg['benchmark']} "
                f"--output-dir <dir> --splits-out {splits_file}"
            )
            sys.exit(1)
        return

    existing_results = Path(cfg["existing_results"])
    if not existing_results.is_absolute():
        existing_results = root / existing_results

    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = root / splits_file

    split_cfg = cfg.get("splits", {})
    train_n = split_cfg.get("train_n", 80)
    val_n = split_cfg.get("val_n", 45)
    test_n = split_cfg.get("test_n")  # None → all remaining
    seed = cfg.get("seed", 1)

    print(f"Building splits for {cfg['benchmark']}...")
    print(f"  Source: {existing_results}")
    print(f"  Train: {train_n}, Val: {val_n}, Test: {test_n or 'remainder'}, seed: {seed}")

    splits = build_splits(
        raw_results_path=existing_results,
        train_n=train_n,
        val_n=val_n,
        seed=seed,
        output_path=splits_file,
        test_n=test_n,
    )

    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val:   {len(splits['val'])} examples")
    print(f"  Test:  {len(splits['test'])} examples")
    print(f"  Saved: {splits_file}")
```

- [ ] **Step 3: Verify diagnostics no longer appear**

The Pylance warning `"config_path" is not accessed` at line 413 is a false positive —
`config_path` is used via `_repo_root(config_path)` two lines below. No code change needed
here; the warning may clear after the `run_splits` restructuring.

Run the test that exercises `_load_examples` (existing seed tests cover the DatasetRegistry
path; we test the `gepa_data_file` path in integration tests implicitly):

```bash
pytest tests/gepa_integration/ -v -k "not test_gepa_data" 2>&1 | tail -20
```

Expected: all pre-existing GEPA tests still pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_gepa.py
git commit -m "feat(gepa): gepa_data_file loading path in run_gepa.py

When a GEPA config has 'gepa_data_file', _load_examples reads from
the local JSON (written by prepare.py) instead of DatasetRegistry.
run_splits similarly checks / reports the pre-generated splits file."
```

---

## Task 6: Update `experiments/configs/gepa/gaia.yaml`

**Files:**
- Modify: `experiments/configs/gepa/gaia.yaml`

Replace the existing thesis-benchmark config with the Search-R1/DeepMath data config.

- [ ] **Step 1: Replace gaia.yaml**

Overwrite `experiments/configs/gepa/gaia.yaml` with:

```yaml
name: "GEPA_gaia_qwen3_8b"
description: >
  GEPA two-component prompt optimisation for gaia.yaml (GAIA / HLE / MuSiQue prompts).
  Uses Search-R1 (75%) + DeepMath (25%, no difficulty filter) as clean,
  non-overlapping optimization data. See docs/superpowers/plans/2026-05-18-gepa-data-plan.md.

benchmark: "gaia"
thinking_mode: "ORCHESTRATOR_ONLY"
seed: 1
max_turns: 15
cache_dir: "./cache"

# GEPA data — generated by src/gepa_integration/data/prepare.py
# Run: python src/gepa_integration/data/prepare.py \
#          --preset gaia \
#          --output-dir data/gepa/gaia \
#          --splits-out experiments/configs/gepa/splits/gaia_gepa_splits.json \
#          --seed 1
gepa_data_file: "data/gepa/gaia/all_examples.json"
splits_file: "experiments/configs/gepa/splits/gaia_gepa_splits.json"

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  family: "qwen3"
  role: "orchestrator"

# Reflector runs on a separate vLLM instance.
# Start before --mode optimize:
#   vllm serve Qwen/Qwen3-32B --port 8001 --enable-thinking
reflector:
  name: "Qwen3-32B"
  path_or_id: "Qwen/Qwen3-32B"
  host: "localhost"
  port: 8001

tools:
  enabled_tools:
    - web_search
    - code_generator
    - text_inspector
  direct_tool_call: false
  web_tool_provider: "serper"
  max_search_limit: 10

gepa:
  # rollout_budget >= 15-30 × len(D_pareto). D_pareto=50 → 15×=750.
  rollout_budget: 750
  # b=3 per mutation step — matches GEPA paper setup (Section 3).
  # Smaller than previous value of 10: GEPA traces with thinking enabled
  # are long; b=3 keeps the reflective prompt under the reflector context limit.
  minibatch_size: 3
  merge_proposer: true
  track_best_outputs: true
  run_dir: "experiments/results/gepa/gaia"

wandb:
  enabled: true
  project: "gepa"
  name: "gepa_gaia_qwen3_8b"
  tags: ["gaia", "gepa", "qwen3-8b", "search-r1", "deepmath"]

slurm:
  partition: "gpu_h100"
  num_gpus: 4
  ntasks: 1
  cpus_per_task: 16
  time: "12:00:00"
  conda_env: "agent_engine"
```

- [ ] **Step 2: Verify config loads without error**

```bash
python -c "
import yaml, sys
from pathlib import Path
cfg = yaml.safe_load(open('experiments/configs/gepa/gaia.yaml'))
assert cfg['benchmark'] == 'gaia'
assert 'gepa_data_file' in cfg
assert cfg['gepa']['minibatch_size'] == 3
print('gaia.yaml OK')
"
```

Expected: `gaia.yaml OK`.

- [ ] **Step 3: Commit**

```bash
git add experiments/configs/gepa/gaia.yaml
git commit -m "feat(gepa): update gaia.yaml to use Search-R1/DeepMath data

Replaces thesis-benchmark (gaia raw_results.json) with clean
non-overlapping Search-R1 + DeepMath optimization data per the
GEPA data plan (docs/superpowers/plans/2026-05-18-gepa-data-plan.md).
Also sets minibatch_size=3 (GEPA paper default, from 10)."
```

---

## Task 7: Create `experiments/configs/gepa/math.yaml`

**Files:**
- Create: `experiments/configs/gepa/math.yaml`

- [ ] **Step 1: Create math.yaml**

Create `experiments/configs/gepa/math.yaml`:

```yaml
name: "GEPA_math_qwen3_8b"
description: >
  GEPA two-component prompt optimisation for math.yaml (AIME prompts).
  Uses DeepMath-103K (difficulty >= 5, 75%) + Search-R1 (25%) as clean,
  non-overlapping optimization data. Targets AIME failure mode: direct
  reasoning without Coder delegation (55.8% of failures).
  See docs/superpowers/plans/2026-05-18-gepa-data-plan.md.

benchmark: "math"
thinking_mode: "ORCHESTRATOR_ONLY"
seed: 1
max_turns: 15
cache_dir: "./cache"

# GEPA data — generated by src/gepa_integration/data/prepare.py
# Run: python src/gepa_integration/data/prepare.py \
#          --preset math \
#          --output-dir data/gepa/math \
#          --splits-out experiments/configs/gepa/splits/math_gepa_splits.json \
#          --seed 1
gepa_data_file: "data/gepa/math/all_examples.json"
splits_file: "experiments/configs/gepa/splits/math_gepa_splits.json"

model:
  name: "Qwen3-8B"
  path_or_id: "Qwen/Qwen3-8B"
  family: "qwen3"
  role: "orchestrator"

# Reflector runs on a separate vLLM instance.
# Start before --mode optimize:
#   vllm serve Qwen/Qwen3-32B --port 8001 --enable-thinking
reflector:
  name: "Qwen3-32B"
  path_or_id: "Qwen/Qwen3-32B"
  host: "localhost"
  port: 8001

tools:
  enabled_tools:
    - web_search
    - code_generator
    - text_inspector
  direct_tool_call: false
  web_tool_provider: "serper"
  max_search_limit: 10

gepa:
  # rollout_budget >= 15-30 × len(D_pareto). D_pareto=50 → 15×=750.
  rollout_budget: 750
  # b=3 per mutation step — GEPA paper default (Section 3).
  minibatch_size: 3
  merge_proposer: true
  track_best_outputs: true
  run_dir: "experiments/results/gepa/math"

wandb:
  enabled: true
  project: "gepa"
  name: "gepa_math_qwen3_8b"
  tags: ["math", "aime", "gepa", "qwen3-8b", "deepmath", "search-r1"]

slurm:
  partition: "gpu_h100"
  num_gpus: 4
  ntasks: 1
  cpus_per_task: 16
  time: "12:00:00"
  conda_env: "agent_engine"
```

- [ ] **Step 2: Verify config loads without error**

```bash
python -c "
import yaml
cfg = yaml.safe_load(open('experiments/configs/gepa/math.yaml'))
assert cfg['benchmark'] == 'math'
assert 'gepa_data_file' in cfg
assert cfg['gepa']['minibatch_size'] == 3
print('math.yaml OK')
"
```

Expected: `math.yaml OK`.

- [ ] **Step 3: Commit**

```bash
git add experiments/configs/gepa/math.yaml
git commit -m "feat(gepa): add math.yaml GEPA config for AIME prompt optimization

New config optimizes the math.yaml orchestrator prompt using DeepMath
(75%, difficulty>=5) + Search-R1 (25%). Targets the dominant AIME
failure mode: direct reasoning without Coder delegation (55.8%)."
```

---

## Task 8: Create SLURM job `jobs/gepa/005_prep_gepa_data.job`

**Files:**
- Create: `jobs/gepa/005_prep_gepa_data.job`
- Modify: `jobs/gepa/000_prep_gepa_data.job`

- [ ] **Step 1: Create 005_prep_gepa_data.job**

Create `jobs/gepa/005_prep_gepa_data.job`:

```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=PrepGEPADataV2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=out/gepa/prep_gepa_data_v2_%A.log

# Download and split GEPA optimization data (Search-R1 + DeepMath).
#
# Generates:
#   data/gepa/gaia/all_examples.json          — 300 examples (75% Search-R1, 25% DeepMath)
#   data/gepa/math/all_examples.json          — 300 examples (75% DeepMath diff>=5, 25% Search-R1)
#   experiments/configs/gepa/splits/gaia_gepa_splits.json
#   experiments/configs/gepa/splits/math_gepa_splits.json
#
# CPU-only; no GPU required. Requires HuggingFace Hub access (HF_TOKEN if needed).
# Safe to re-run: overwrites output files with a deterministic result (seed=1).
#
# Prerequisites: conda env 'agent_engine' with datasets + pandas installed.
#   (Created by 001_install_gepa_deps.job or existing setup.)

set -euo pipefail

mkdir -p out/gepa

echo "=========================================="
echo "MSC Thesis — GEPA data preparation v2"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}  Node: $(hostname)"
echo ""

module purge
module load 2025
module load Miniconda3/25.5.1-1

PROJECT_DIR="${PROJECT_DIR:-$HOME/azywot/msc-thesis}"
cd "$PROJECT_DIR" || exit 1
echo "Project directory: $PROJECT_DIR"

ENV_NAME="${ENV_NAME:-agent_engine}"
echo "Activating environment: $ENV_NAME"
source activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

echo ""
echo "── gaia preset (75% Search-R1, 25% DeepMath) ───────"
python src/gepa_integration/data/prepare.py \
    --preset gaia \
    --output-dir data/gepa/gaia \
    --splits-out experiments/configs/gepa/splits/gaia_gepa_splits.json \
    --seed 1

echo ""
echo "── math preset (75% DeepMath diff>=5, 25% Search-R1) ──"
python src/gepa_integration/data/prepare.py \
    --preset math \
    --output-dir data/gepa/math \
    --splits-out experiments/configs/gepa/splits/math_gepa_splits.json \
    --seed 1

echo ""
echo "── Verification ──────────────────────────────────────"
python - <<'PY'
import json
from pathlib import Path

for name in ("gaia", "math"):
    splits_path = Path(f"experiments/configs/gepa/splits/{name}_gepa_splits.json")
    data_path   = Path(f"data/gepa/{name}/all_examples.json")

    splits   = json.loads(splits_path.read_text())
    examples = json.loads(data_path.read_text())

    train, val, test = splits["train"], splits["val"], splits["test"]
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    assert not overlap, f"{name}: split overlap!"
    assert len(train) + len(val) + len(test) == len(examples), f"{name}: count mismatch"
    print(
        f"  {name.upper()}: {len(examples)} total, "
        f"D_feedback={len(train)}, D_pareto={len(val)}, test={len(test)} — OK"
    )
PY

echo ""
echo "Done at: $(date)"
```

- [ ] **Step 2: Update `000_prep_gepa_data.job` to remove the gaia section**

Edit `jobs/gepa/000_prep_gepa_data.job`. Find the gaia section:

```bash
echo ""
echo "── GAIA splits ──────────────────────────────────────"
python scripts/run_gepa.py --mode splits \
    --config experiments/configs/gepa/gaia.yaml
```

Replace with a note:

```bash
echo ""
echo "── GAIA splits ──────────────────────────────────────"
echo "NOTE: gaia.yaml now uses Search-R1/DeepMath data (not thesis splits)."
echo "      Run jobs/gepa/005_prep_gepa_data.job to prepare gaia data."
echo "      Skipping gaia splits generation here."
```

- [ ] **Step 3: Commit**

```bash
git add jobs/gepa/005_prep_gepa_data.job jobs/gepa/000_prep_gepa_data.job
git commit -m "feat(gepa): add SLURM job for GEPA data prep (005_prep_gepa_data.job)

Job 005 downloads Search-R1 + DeepMath for both gaia and math presets
and writes all_examples.json + splits JSON files. Job 000 updated to
note that gaia splits are now handled by job 005."
```

---

## Task 9: Run full test suite

- [ ] **Step 1: Run all GEPA tests**

```bash
pytest tests/gepa_integration/ -v
```

Expected: all tests pass (new `test_gepa_data.py` tests + all pre-existing GEPA tests).

- [ ] **Step 2: Run unit tests for fine_tuning data (sanity check no regressions)**

```bash
pytest tests/unit/test_data_prepare.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Verify imports are clean**

```bash
python -c "
from gepa_integration.data.prepare import make_gepa_splits, build_gepa_examples, save_gepa_data, _PRESETS
from gepa_integration.data.loader import load_gepa_examples
print('Imports OK')
print('Presets:', list(_PRESETS))
"
```

Expected:
```
Imports OK
Presets: ['gaia', 'math']
```

---

## Self-Review

### 1. Spec coverage

| Spec requirement | Covered by |
|---|---|
| gaia: 75% Search-R1 (85/15 HotpotQA/NQ), 25% DeepMath, no difficulty filter | `_PRESETS["gaia"]` in prepare.py |
| math: 75% DeepMath difficulty≥5, 25% Search-R1 | `_PRESETS["math"]` in prepare.py |
| D_feedback=150, D_pareto=50, Test=100 for both | `make_gepa_splits` + `_PRESETS` counts |
| gaia split: 112+38 / 37+13 / 75+25 | `_PRESETS["gaia"]["search"]["feedback"]=112` etc. |
| math split: 38+112 / 13+37 / 25+75 | `_PRESETS["math"]["search"]["feedback"]=38` etc. |
| Minibatch size b=3 (GEPA paper default) | `minibatch_size: 3` in both configs |
| GEPA+Merge variant | `merge_proposer: true` in both configs |
| Pareto-based sampling | default in `gepa` library (no config needed) |
| No overlap with thesis benchmark IDs | Search-R1/DeepMath are entirely disjoint by design |
| Thinking truncated to 800 chars | already implemented in adapter.py (Iteration 2) |

**GPQA exclusion**: spec explicitly says "GPQA is excluded from this optimization round" —
no `gpqa.yaml` GEPA config is created or modified. ✓

### 2. Placeholder scan

No TBD, TODO, or "similar to Task N" phrases present.

### 3. Type / signature consistency

- `make_gepa_splits(all_ids: list[int], n_feedback, n_pareto, n_test, seed)` → called in `prepare()` with `[ex["question_id"] for ex in examples]` ✓
- `build_gepa_examples` inputs are raw fine_tuning rows (have `"question"`, `"result"`, `"data_source"`, `"extra_info"`) — `_norm_to_example` converts via those exact fields ✓
- `load_gepa_examples(data_file: Path, question_ids: list[int])` — called in `_load_examples` with `Path(gepa_data_file)` and `question_ids: list` ✓
- `DatasetExample.question_id: int` — `raw["question_id"]` is `int` (assigned by `build_gepa_examples`) ✓
- `run_gepa.py` `run_splits` new branch: reads `splits_file` from cfg same as existing code ✓
