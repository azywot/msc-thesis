# Orchestrator RL Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a GRPO-based RL fine-tuning pipeline for the CoSMAS orchestrator (Qwen3-8B) using VERL via the shin-ee-chen/AgentFlow library.

**Architecture:** The VERL server runs as a Ray cluster managing FSDP actor/ref model workers and an `AgentModeDaemon` on port 9999. A separate `scripts/train_orchestrator.py` process runs `OrchestratorRollout(LitAgent)` workers that call the VERL vLLM endpoint, execute the full `AgenticOrchestrator` loop, compute binary rewards via `metrics.py`, and return results to the daemon via the AgentFlow `@reward` decorator. LoRA adapters are saved every 2 steps; a merge step produces a standard HF model usable in existing inference configs.

**Tech Stack:** Python 3.11, VERL 0.5.0, agentflow (local clone), vLLM 0.9.2, PyTorch 2.7.0, LoRA rank-64, GRPO, W&B, Snellius SLURM, HuggingFace datasets, pyarrow parquet.

**Spec:** `docs/superpowers/specs/2026-05-06-orchestrator-finetuning-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/agent_engine/models/api_provider.py` | Modify | Add optional `base_url` to `OpenAIProvider` |
| `src/fine_tuning/__init__.py` | Modify | Export public classes |
| `src/fine_tuning/config.py` | Implement | `FinetuningConfig` dataclass |
| `src/fine_tuning/reward.py` | Implement | `OrchestratorReward` — routes to `evaluate_answer()` |
| `src/fine_tuning/data/__init__.py` | Create | Empty package marker |
| `src/fine_tuning/data/prepare.py` | Create | Download Search-R1 + DeepMath → parquet |
| `src/fine_tuning/rollout.py` | Implement | `OrchestratorRollout(LitAgent)` |
| `scripts/launch_verl.py` | Create | Parse config YAML + launch `python -m agentflow.verl` |
| `scripts/train_orchestrator.py` | Create | NullTracer + `agentflow.Trainer.fit()` |
| `train/config.yaml` | Create | VERL + agentflow env vars for 4-GPU run |
| `jobs/train_orchestrator.sh` | Create | SLURM job script |
| `jobs/environment_train.yml` | Create | Conda environment for training |
| `pyproject.toml` | Modify | Add `[training]` optional deps |
| `tests/unit/test_api_provider_base_url.py` | Create | Unit test for base_url extension |
| `tests/unit/test_fine_tuning_config.py` | Create | Unit tests for FinetuningConfig |
| `tests/unit/test_fine_tuning_reward.py` | Create | Unit tests for OrchestratorReward |
| `tests/unit/test_data_prepare.py` | Create | Unit tests for prepare.py schema normalisation |
| `tests/unit/test_fine_tuning_rollout.py` | Create | Unit tests for OrchestratorRollout (mocked) |

---

## Task 1: Extend `OpenAIProvider` with optional `base_url`

The rollout needs to call the VERL-served vLLM endpoint (an OpenAI-compatible API). The existing `OpenAIProvider` hard-codes `api.openai.com`. Add an optional `base_url` parameter.

**Files:**
- Modify: `src/agent_engine/models/api_provider.py:21-32`
- Create: `tests/unit/test_api_provider_base_url.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/unit/test_api_provider_base_url.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from unittest.mock import patch, MagicMock
from agent_engine.models.api_provider import OpenAIProvider
from agent_engine.models.base import ModelConfig, ModelFamily


def _config():
    return ModelConfig(
        name="test",
        family=ModelFamily.QWEN3,
        path_or_id="Qwen/Qwen3-8B",
        role="orchestrator",
    )


def test_default_base_url_is_none():
    with patch("agent_engine.models.api_provider.OpenAI") as mock_openai:
        OpenAIProvider(_config(), api_key="k")
        _, kwargs = mock_openai.call_args
        assert kwargs.get("base_url") is None


def test_custom_base_url_is_forwarded():
    with patch("agent_engine.models.api_provider.OpenAI") as mock_openai:
        OpenAIProvider(_config(), api_key="k", base_url="http://localhost:9000/v1")
        _, kwargs = mock_openai.call_args
        assert kwargs["base_url"] == "http://localhost:9000/v1"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
pytest tests/unit/test_api_provider_base_url.py -v
```

Expected: FAIL — `OpenAIProvider.__init__() got an unexpected keyword argument 'base_url'`

- [ ] **Step 1.3: Implement the change**

In `src/agent_engine/models/api_provider.py`, change `OpenAIProvider.__init__`:

```python
def __init__(self, config: ModelConfig, api_key: str = None, base_url: str = None):
    """Initialize OpenAI provider.

    Args:
        config: ModelConfig with model ID and generation settings
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url: Optional base URL for OpenAI-compatible APIs (e.g. vLLM server).
                  When set, overrides the default openai.com endpoint.
    """
    super().__init__(config)
    self.client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        base_url=base_url,
    )
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
pytest tests/unit/test_api_provider_base_url.py -v
```

Expected: PASS — 2 tests passed

- [ ] **Step 1.5: Commit**

```bash
git add src/agent_engine/models/api_provider.py tests/unit/test_api_provider_base_url.py
git commit -m "feat(models): add optional base_url to OpenAIProvider for vLLM-compatible endpoints"
```

---

## Task 2: `FinetuningConfig` dataclass

**Files:**
- Implement: `src/fine_tuning/config.py`
- Create: `tests/unit/test_fine_tuning_config.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/unit/test_fine_tuning_config.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import tempfile, yaml
from fine_tuning.config import FinetuningConfig


def test_defaults():
    cfg = FinetuningConfig(
        base_model="Qwen/Qwen3-8B",
        train_data="data/training/train/combined_train.parquet",
        val_data="data/training/val/aime24.parquet",
        output_dir="experiments/results/training/test_run",
    )
    assert cfg.lora_rank == 64
    assert cfg.lora_alpha == 16
    assert cfg.lora_target_modules == "all-linear"
    assert cfg.train_temperature == 0.7
    assert cfg.test_temperature == 0.0
    assert cfg.n_gpus == 4
    assert cfg.seed == 42


def test_from_dict():
    data = {
        "base_model": "Qwen/Qwen3-8B",
        "train_data": "data/train.parquet",
        "val_data": "data/val.parquet",
        "output_dir": "/tmp/run",
        "lora_rank": 32,
        "seed": 7,
    }
    cfg = FinetuningConfig(**data)
    assert cfg.lora_rank == 32
    assert cfg.seed == 7


def test_from_yaml():
    content = {
        "base_model": "Qwen/Qwen3-8B",
        "train_data": "data/train.parquet",
        "val_data": "data/val.parquet",
        "output_dir": "/tmp/run",
        "wandb_project": "cosmas-test",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(content, f)
        path = f.name
    cfg = FinetuningConfig.from_yaml(path)
    assert cfg.base_model == "Qwen/Qwen3-8B"
    assert cfg.wandb_project == "cosmas-test"
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
pytest tests/unit/test_fine_tuning_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tuning.config'` or `ImportError`

- [ ] **Step 2.3: Implement `src/fine_tuning/config.py`**

```python
"""Configuration dataclass for the orchestrator fine-tuning pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class FinetuningConfig:
    """All hyperparameters and paths for one fine-tuning run.

    Required fields must be supplied; optional fields carry sensible defaults
    that mirror AgentFlow's train/config.yaml.
    """

    # ── required ────────────────────────────────────────────────────────────
    base_model: str                # HF model ID, e.g. "Qwen/Qwen3-8B"
    train_data: str                # path to combined_train.parquet
    val_data: str                  # path to validation parquet
    output_dir: str                # checkpoint + config output directory

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_target_modules: str = "all-linear"

    # ── generation temperatures ──────────────────────────────────────────────
    train_temperature: float = 0.7
    test_temperature: float = 0.0

    # ── hardware ────────────────────────────────────────────────────────────
    n_gpus: int = 4
    rollout_tp_size: int = 2        # tensor parallel size for vLLM rollout

    # ── reproducibility ─────────────────────────────────────────────────────
    seed: int = 42

    # ── W&B ─────────────────────────────────────────────────────────────────
    wandb_project: str = "cosmas-rl-finetuning"
    wandb_run_name: str = "qwen3-8b-grpo-search-math"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuningConfig":
        """Load config from a YAML file. Unknown keys are silently ignored."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
```

- [ ] **Step 2.4: Run test to verify it passes**

```bash
pytest tests/unit/test_fine_tuning_config.py -v
```

Expected: PASS — 3 tests passed

- [ ] **Step 2.5: Commit**

```bash
git add src/fine_tuning/config.py tests/unit/test_fine_tuning_config.py
git commit -m "feat(fine_tuning): implement FinetuningConfig dataclass"
```

---

## Task 3: `OrchestratorReward`

Routes reward computation to `evaluate_answer()` from `metrics.py`, keyed by `data_source`.

**Files:**
- Implement: `src/fine_tuning/reward.py`
- Create: `tests/unit/test_fine_tuning_reward.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/unit/test_fine_tuning_reward.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from fine_tuning.reward import OrchestratorReward

reward_fn = OrchestratorReward()


class TestSearchDomain:
    def test_correct_nq(self):
        assert reward_fn("Paris", "Paris", "nq") == 1.0

    def test_partial_containment_nq(self):
        # containment-based: prediction contains ground truth → correct
        assert reward_fn("The capital is Paris, France", "Paris", "nq") == 1.0

    def test_wrong_nq(self):
        assert reward_fn("London", "Paris", "nq") == 0.0

    def test_correct_hotpotqa(self):
        assert reward_fn("yes", "yes", "hotpotqa") == 1.0

    def test_wrong_hotpotqa(self):
        assert reward_fn("no", "yes", "hotpotqa") == 0.0


class TestMathDomain:
    def test_correct_math(self):
        assert reward_fn("42", "42", "math") == 1.0

    def test_correct_deepmath(self):
        assert reward_fn("7", "7", "deepmath") == 1.0

    def test_wrong_math(self):
        assert reward_fn("43", "42", "math") == 0.0

    def test_correct_aime(self):
        assert reward_fn("120", "120", "aime") == 1.0


class TestEdgeCases:
    def test_empty_prediction_returns_zero(self):
        assert reward_fn("", "Paris", "nq") == 0.0

    def test_none_prediction_returns_zero(self):
        assert reward_fn(None, "Paris", "nq") == 0.0

    def test_unknown_data_source_falls_back(self):
        # unknown source defaults to evaluate_answer with no mode hints → still works
        result = reward_fn("Paris", "Paris", "unknown_dataset")
        assert result == 1.0
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
pytest tests/unit/test_fine_tuning_reward.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tuning.reward'`

- [ ] **Step 3.3: Implement `src/fine_tuning/reward.py`**

```python
"""Reward function for the orchestrator fine-tuning pipeline.

Routes correctness evaluation to the existing metrics.py evaluate_answer()
function, selecting the right behaviour based on data_source.
"""

from typing import Optional
from agent_engine.datasets.evaluators.metrics import evaluate_answer

# Data sources that use containment-based QA scoring
_QA_SOURCES = frozenset({"nq", "hotpotqa", "triviaqa", "bamboogle", "2wikimultihopqa", "musique"})
# Data sources that use math / exact-match scoring
_MATH_SOURCES = frozenset({"math", "deepmath", "aime", "amc", "math500", "gpqa"})


class OrchestratorReward:
    """Computes binary reward (1.0 / 0.0) for a predicted answer.

    Routes to evaluate_answer() from metrics.py. The data_source field in the
    parquet determines which evaluation path is used.
    """

    def __call__(
        self,
        prediction: Optional[str],
        ground_truth: str,
        data_source: str,
    ) -> float:
        """Return 1.0 if prediction is correct, 0.0 otherwise.

        Args:
            prediction: Extracted answer string (may be None).
            ground_truth: Reference answer from the dataset.
            data_source: Dataset name string (e.g. "nq", "math", "deepmath").

        Returns:
            1.0 if correct, 0.0 if wrong or prediction is None/empty.
        """
        if not prediction:
            return 0.0

        result = evaluate_answer(
            prediction=str(prediction),
            ground_truth=str(ground_truth),
        )
        return 1.0 if result["correct"] else 0.0
```

- [ ] **Step 3.4: Run test to verify it passes**

```bash
pytest tests/unit/test_fine_tuning_reward.py -v
```

Expected: PASS — all tests passed

- [ ] **Step 3.5: Commit**

```bash
git add src/fine_tuning/reward.py tests/unit/test_fine_tuning_reward.py
git commit -m "feat(fine_tuning): implement OrchestratorReward using evaluate_answer()"
```

---

## Task 4: Data preparation script

Downloads Search-R1 and DeepMath from HuggingFace and converts to VERL parquet schema.

**Files:**
- Create: `src/fine_tuning/data/__init__.py`
- Create: `src/fine_tuning/data/prepare.py`
- Create: `tests/unit/test_data_prepare.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/unit/test_data_prepare.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import pandas as pd
from fine_tuning.data.prepare import (
    normalise_search_r1_row,
    normalise_deepmath_row,
    validate_parquet_schema,
)

REQUIRED_COLS = {"data_source", "question", "result", "extra_info"}


class TestNormaliseSearchR1:
    def test_nq_row(self):
        raw = {"question": "Who wrote Hamlet?", "answer": "Shakespeare", "dataset": "nq"}
        row = normalise_search_r1_row(raw, idx=0)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "nq"
        assert row["question"] == "Who wrote Hamlet?"
        assert row["result"] == "Shakespeare"
        assert row["extra_info"]["idx"] == 0
        assert row["extra_info"]["groundtruth"] == "Shakespeare"

    def test_hotpotqa_row(self):
        raw = {"question": "Where was X born?", "answer": "London", "dataset": "hotpotqa"}
        row = normalise_search_r1_row(raw, idx=5)
        assert row["data_source"] == "hotpotqa"
        assert row["extra_info"]["idx"] == 5


class TestNormaliseDeepMath:
    def test_basic_row(self):
        raw = {"problem": "What is 2+2?", "answer": "4", "source": "math"}
        row = normalise_deepmath_row(raw, idx=3)
        assert set(row.keys()) == REQUIRED_COLS
        assert row["data_source"] == "deepmath"
        assert row["question"] == "What is 2+2?"
        assert row["result"] == "4"
        assert row["extra_info"]["idx"] == 3

    def test_missing_source_defaults(self):
        raw = {"problem": "Solve x=2", "answer": "2"}
        row = normalise_deepmath_row(raw, idx=0)
        assert row["data_source"] == "deepmath"


class TestValidateSchema:
    def test_valid_df_passes(self):
        df = pd.DataFrame([
            {
                "data_source": "nq",
                "question": "Q?",
                "result": "A",
                "extra_info": {"idx": 0, "groundtruth": "A"},
            }
        ])
        validate_parquet_schema(df)  # should not raise

    def test_missing_column_raises(self):
        df = pd.DataFrame([{"data_source": "nq", "question": "Q?", "result": "A"}])
        with pytest.raises(ValueError, match="Missing columns"):
            validate_parquet_schema(df)
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
pytest tests/unit/test_data_prepare.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tuning.data'`

- [ ] **Step 4.3: Create `src/fine_tuning/data/__init__.py`**

```python
```

(empty file)

- [ ] **Step 4.4: Implement `src/fine_tuning/data/prepare.py`**

```python
"""Data preparation: download Search-R1 + DeepMath and write VERL parquet files.

Usage:
    python src/fine_tuning/data/prepare.py \\
        --n-search 10000 --n-math 10000 \\
        --output-dir data/training --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Row normalisers
# ---------------------------------------------------------------------------

def normalise_search_r1_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a Search-R1 row to VERL schema.

    Search-R1 (PeterJinGo/SearchR1-nq_hotpotqa_train) columns:
        question, answer, dataset (one of "nq", "hotpotqa")
    """
    answer = str(raw.get("answer") or raw.get("answers") or "")
    # answers field can be a list
    if isinstance(raw.get("answer"), list):
        answer = raw["answer"][0] if raw["answer"] else ""
    if isinstance(raw.get("answers"), list):
        answer = raw["answers"][0] if raw["answers"] else ""

    return {
        "data_source": str(raw.get("dataset", "nq")).lower(),
        "question": str(raw.get("question", "")),
        "result": answer,
        "extra_info": {"idx": idx, "groundtruth": answer},
    }


def normalise_deepmath_row(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a DeepMath-103K row to VERL schema.

    DeepMath (zwhe99/DeepMath-103K) columns: problem, answer, source (optional)
    """
    answer = str(raw.get("answer", ""))
    return {
        "data_source": "deepmath",
        "question": str(raw.get("problem", "")),
        "result": answer,
        "extra_info": {"idx": idx, "groundtruth": answer},
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"data_source", "question", "result", "extra_info"}


def validate_parquet_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if df is missing required VERL columns."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_search_r1(n: int, seed: int) -> list[Dict[str, Any]]:
    from datasets import load_dataset  # lazy import — not needed for unit tests
    ds = load_dataset("PeterJinGo/SearchR1-nq_hotpotqa_train", split="train")
    ds = ds.shuffle(seed=seed)
    if n < len(ds):
        ds = ds.select(range(n))
    return [normalise_search_r1_row(row, idx=i) for i, row in enumerate(ds)]


def _download_deepmath(n: int, seed: int) -> list[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=seed)
    if n < len(ds):
        ds = ds.select(range(n))
    return [normalise_deepmath_row(row, idx=i) for i, row in enumerate(ds)]


def _download_aime24() -> list[Dict[str, Any]]:
    from datasets import load_dataset
    # Standard AIME24 dataset used by AgentFlow for validation
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    rows = []
    for i, row in enumerate(ds):
        answer = str(row.get("answer", ""))
        rows.append({
            "data_source": "aime",
            "question": str(row.get("problem", "")),
            "result": answer,
            "extra_info": {"idx": i, "groundtruth": answer},
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_combined_train(
    n_search: int,
    n_math: int,
    output_dir: Path,
    seed: int,
) -> Path:
    """Download, normalise, mix, and write combined_train.parquet."""
    print(f"Downloading Search-R1 ({n_search} examples)...")
    search_rows = _download_search_r1(n_search, seed)

    print(f"Downloading DeepMath ({n_math} examples)...")
    math_rows = _download_deepmath(n_math, seed)

    combined = search_rows + math_rows
    rng = random.Random(seed)
    rng.shuffle(combined)

    df = pd.DataFrame(combined)
    validate_parquet_schema(df)

    out_path = output_dir / "train" / "combined_train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def build_aime24_val(output_dir: Path) -> Path:
    """Download AIME24 and write aime24.parquet."""
    print("Downloading AIME24 validation set...")
    rows = _download_aime24()
    df = pd.DataFrame(rows)
    validate_parquet_schema(df)

    out_path = output_dir / "val" / "aime24.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for orchestrator fine-tuning.")
    parser.add_argument("--n-search", type=int, default=10_000,
                        help="Number of Search-R1 rows (default: 10000)")
    parser.add_argument("--n-math", type=int, default=10_000,
                        help="Number of DeepMath rows (default: 10000)")
    parser.add_argument("--output-dir", type=str, default="data/training",
                        help="Output directory for parquet files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    build_combined_train(args.n_search, args.n_math, output_dir, args.seed)
    build_aime24_val(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.5: Run test to verify it passes**

```bash
pytest tests/unit/test_data_prepare.py -v
```

Expected: PASS — all tests passed

- [ ] **Step 4.6: Commit**

```bash
git add src/fine_tuning/data/ tests/unit/test_data_prepare.py
git commit -m "feat(fine_tuning): implement data preparation script for Search-R1 + DeepMath"
```

---

## Task 5: `OrchestratorRollout(LitAgent)`

The core rollout: wraps `AgenticOrchestrator`, calls the VERL vLLM endpoint, computes rewards.

**Files:**
- Implement: `src/fine_tuning/rollout.py`
- Create: `tests/unit/test_fine_tuning_rollout.py`

- [ ] **Step 5.1: Write the failing tests**

Create `tests/unit/test_fine_tuning_rollout.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fine_tuning.rollout import OrchestratorRollout, _build_tool_registry


class TestBuildToolRegistry:
    def test_returns_registry_with_web_search(self):
        with patch.dict("os.environ", {"SERPER_API_KEY": "test_key"}):
            registry = _build_tool_registry()
        assert "web_search" in registry

    def test_missing_api_key_raises(self):
        import os
        # Remove both keys if present
        env = {k: v for k, v in os.environ.items()
               if k not in ("SERPER_API_KEY", "TAVILY_API_KEY")}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(EnvironmentError, match="SERPER_API_KEY"):
                _build_tool_registry()


class TestOrchestratorRolloutInit:
    def test_instantiation(self):
        rollout = OrchestratorRollout(
            rollout_dir="/tmp/test_rollouts",
            rollout_n=4,
            train_temperature=0.7,
            test_temperature=0.0,
            max_turns=5,
        )
        assert rollout.max_turns == 5
        assert rollout.train_temperature == 0.7


class TestTrainingRolloutAsync:
    def test_correct_answer_yields_reward_1(self):
        rollout = OrchestratorRollout(rollout_dir="/tmp/test_rollouts", rollout_n=1)

        # Mock the orchestrator state
        mock_state = MagicMock()
        mock_state.answer = "Paris"
        mock_state.output_messages = [{"role": "assistant", "content": "Paris"}]

        # Mock the orchestrator run
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.return_value = mock_state

        task = {
            "question": "What is the capital of France?",
            "result": "Paris",
            "extra_info": {"idx": 0, "groundtruth": "Paris", "data_source": "nq"},
        }
        resources = MagicMock()
        resources.get.return_value = MagicMock(model="Qwen/Qwen3-8B", endpoint="http://localhost:9000/v1")

        with patch("fine_tuning.rollout.AgenticOrchestrator", return_value=mock_orchestrator), \
             patch("fine_tuning.rollout._build_tool_registry", return_value=MagicMock()), \
             patch("fine_tuning.rollout.OpenAIProvider", return_value=MagicMock()), \
             patch.dict("os.environ", {"SERPER_API_KEY": "k"}):
            captured_reward = []

            async def _run():
                with patch("fine_tuning.rollout._reward_fn") as mock_reward:
                    mock_reward.return_value = asyncio.coroutine(lambda: 1.0)()
                    await rollout.training_rollout_async(task, "rollout_0", resources)

            asyncio.run(_run())
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
pytest tests/unit/test_fine_tuning_rollout.py::TestBuildToolRegistry -v
pytest tests/unit/test_fine_tuning_rollout.py::TestOrchestratorRolloutInit -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fine_tuning.rollout'`

- [ ] **Step 5.3: Implement `src/fine_tuning/rollout.py`**

```python
"""OrchestratorRollout — AgentFlow LitAgent wrapping AgenticOrchestrator.

This module runs the full msc-thesis orchestration loop as a VERL rollout worker.
It connects to the VERL-served vLLM endpoint for model generation and uses the
existing WebSearchTool (direct mode) for web search during training.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock

from agentflow import LitAgent, reward
from agentflow.types import NamedResources

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.api_provider import OpenAIProvider
from agent_engine.models.base import ModelConfig, ModelFamily
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools import WebSearchTool
from agent_engine.utils.parsing import extract_answer

from .reward import OrchestratorReward

_reward_fn_instance = OrchestratorReward()


@reward
async def _reward_fn(
    question: str,
    ground_truth: str,
    prediction: str,
    data_source: str,
    val: bool = False,
) -> float:
    """AgentFlow reward function — registered with VERL via @reward decorator."""
    return _reward_fn_instance(prediction, ground_truth, data_source)


def _build_tool_registry() -> ToolRegistry:
    """Build a minimal ToolRegistry for rollout workers.

    Uses WebSearchTool in direct mode (no sub-agent LLM). Requires SERPER_API_KEY
    or TAVILY_API_KEY in the environment.
    """
    api_key = os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SERPER_API_KEY (or TAVILY_API_KEY) must be set for rollout workers."
        )
    provider = "tavily" if os.environ.get("TAVILY_API_KEY") and not os.environ.get("SERPER_API_KEY") else "serper"
    registry = ToolRegistry()
    registry.register(WebSearchTool(api_key=api_key, provider=provider, top_k=5))
    return registry


def _make_model_config(model_id: str, temperature: float, max_tokens: int = 2048) -> ModelConfig:
    """Build a ModelConfig for the VERL-served model."""
    return ModelConfig(
        name="verl_orchestrator",
        family=ModelFamily.QWEN3,
        path_or_id=model_id,
        role="orchestrator",
        temperature=temperature,
        max_tokens=max_tokens,
        backend="openai",  # use OpenAI-compatible API
    )


class OrchestratorRollout(LitAgent):
    """AgentFlow LitAgent that runs AgenticOrchestrator as the rollout agent.

    Mirrors AgentFlow's Rollout class in train/rollout.py, replacing
    construct_solver() with AgenticOrchestrator.
    """

    def __init__(
        self,
        rollout_dir: str = "./rollout_data",
        rollout_n: int = 8,
        train_temperature: float = 0.7,
        test_temperature: float = 0.0,
        max_turns: int = 5,
        max_tokens: int = 2048,
    ):
        super().__init__()
        self.rollout_dir = Path(rollout_dir)
        self.rollout_n = rollout_n
        self.train_temperature = train_temperature
        self.test_temperature = test_temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._prompt_builder = PromptBuilder()
        self._tool_registry: Optional[ToolRegistry] = None
        self._train_model: Optional[OpenAIProvider] = None
        self._val_model: Optional[OpenAIProvider] = None

    # ------------------------------------------------------------------ #
    # LitAgent interface                                                   #
    # ------------------------------------------------------------------ #

    async def training_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources, val: bool = False
    ) -> None:
        temperature = self.test_temperature if val else self.train_temperature
        endpoint = resources.get("main_llm").endpoint
        await self._run_episode(task, rollout_id, temperature, endpoint=endpoint, val=val)

    async def validation_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources
    ) -> None:
        endpoint = resources.get("main_llm").endpoint
        await self._run_episode(task, rollout_id, self.test_temperature, endpoint=endpoint, val=True)

    # ------------------------------------------------------------------ #
    # Core episode logic                                                   #
    # ------------------------------------------------------------------ #

    async def _run_episode(
        self, task: Any, rollout_id: str, temperature: float, endpoint: str, val: bool
    ) -> None:
        question_text = task.get("question", "")
        ground_truth = str(task.get("result", ""))
        extra = task.get("extra_info", {}) or {}
        data_source = str(extra.get("data_source", "nq"))
        idx = extra.get("idx", 0)

        # Append output format instruction (mirrors AgentFlow rollout)
        output_fmt = (
            " When ready, output the final answer enclosed in "
            "<answer> and </answer> tags."
        )
        prompt = question_text + output_fmt

        answer = "None"
        output_messages: list = []
        try:
            provider = self._build_provider(endpoint, temperature)
            tool_registry = self._get_or_build_tools()
            system_prompt = self._prompt_builder.build_system_prompt(
                dataset_name="gaia",
                tool_schemas=tool_registry.get_all_schemas(),
                direct_tool_call=True,
            )
            orchestrator = AgenticOrchestrator(
                model_provider=provider,
                tool_registry=tool_registry,
                max_turns=self.max_turns,
            )
            state = orchestrator.run(
                question=prompt,
                question_id=idx,
                system_prompt=system_prompt,
            )
            output_messages = state.output_messages
            answer = state.answer or "None"
        except Exception as exc:
            print(f"[OrchestratorRollout] Episode failed: {exc}")
            answer = "None"

        # Register reward with VERL via @reward decorator
        reward_value = await _reward_fn(
            question_text, ground_truth, answer, data_source, val
        )
        print(f"answer={answer!r}  gt={ground_truth!r}  reward={reward_value}")

        # Save rollout data to disk for debugging (mirrors AgentFlow)
        self._save_rollout(
            idx=idx,
            rollout_id=rollout_id,
            question=question_text,
            ground_truth=ground_truth,
            answer=answer,
            reward=reward_value,
            output_messages=output_messages,
            val=val,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_provider(self, endpoint: str, temperature: float) -> OpenAIProvider:
        """Build an OpenAIProvider pointing at the VERL vLLM endpoint."""
        config = _make_model_config(
            model_id=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"),
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        return OpenAIProvider(config, api_key="EMPTY", base_url=endpoint)

    def _get_or_build_tools(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = _build_tool_registry()
        return self._tool_registry

    # ------------------------------------------------------------------ #
    # Rollout persistence                                                  #
    # ------------------------------------------------------------------ #

    def _save_rollout(
        self,
        *,
        idx: int,
        rollout_id: str,
        question: str,
        ground_truth: str,
        answer: str,
        reward: float,
        output_messages: list,
        val: bool,
    ) -> None:
        split = "val" if val else "train"
        save_dir = self.rollout_dir / split / f"idx_{idx}"
        save_dir.mkdir(parents=True, exist_ok=True)

        lock_path = self.rollout_dir / f".{split}.lock"
        with FileLock(str(lock_path), timeout=30):
            existing = sum(1 for _ in save_dir.glob("rollout_*.json"))
            assert existing < self.rollout_n, (
                f"Too many rollouts for idx {idx}: {existing} >= {self.rollout_n}"
            )

        record = {
            "idx": idx,
            "rollout_id": rollout_id,
            "question": question,
            "groundtruth": ground_truth,
            "answer_extracted": answer,
            "reward": reward,
            "output_messages": output_messages,
            "timestamp": datetime.now().isoformat(),
        }
        out = save_dir / f"rollout_{uuid.uuid4().hex[:8]}.json"
        with open(out, "w") as f:
            json.dump(record, f, indent=2, default=str)
```

- [ ] **Step 5.4: Run tests to verify they pass**

```bash
pytest tests/unit/test_fine_tuning_rollout.py::TestBuildToolRegistry -v
pytest tests/unit/test_fine_tuning_rollout.py::TestOrchestratorRolloutInit -v
```

Expected: PASS

- [ ] **Step 5.5: Commit**

```bash
git add src/fine_tuning/rollout.py tests/unit/test_fine_tuning_rollout.py
git commit -m "feat(fine_tuning): implement OrchestratorRollout(LitAgent)"
```

---

## Task 6: Update `src/fine_tuning/__init__.py`

Export the public API.

**Files:**
- Modify: `src/fine_tuning/__init__.py`

- [ ] **Step 6.1: Implement**

```python
"""Orchestrator fine-tuning pipeline for CoSMAS."""

from .config import FinetuningConfig
from .reward import OrchestratorReward
from .rollout import OrchestratorRollout

__all__ = ["FinetuningConfig", "OrchestratorReward", "OrchestratorRollout"]
```

- [ ] **Step 6.2: Run existing tests to verify nothing broke**

```bash
pytest tests/unit/ -v --tb=short
```

Expected: All tests pass

- [ ] **Step 6.3: Commit**

```bash
git add src/fine_tuning/__init__.py
git commit -m "feat(fine_tuning): export public API from __init__.py"
```

---

## Task 7: `scripts/launch_verl.py` — VERL server entry point

Mirrors `AgentFlow/train/train_agent.py` exactly. Parses the training config YAML and launches `python -m agentflow.verl key=value ...`.

**Files:**
- Create: `scripts/launch_verl.py`

- [ ] **Step 7.1: Implement `scripts/launch_verl.py`**

```python
"""Launch the VERL training server.

Mirrors AgentFlow's train/train_agent.py: reads train/config.yaml, sets
environment variables, and spawns `python -m agentflow.verl key=value ...`.

Usage:
    python scripts/launch_verl.py --config train/config.yaml
"""

import argparse
import os
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch VERL training server.")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    _, unknown = parser.parse_known_args()
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set environment variables
    for key, value in config.get("env", {}).items():
        os.environ[key] = str(value)
        print(f"  Exported {key}={value}")

    # Build python -m agentflow.verl key=value ... command
    command = [sys.executable, "-m", "agentflow.verl"]
    for key, value in config.get("python_args", {}).items():
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            command.append(f"{key}={expanded}")
        else:
            command.append(f"{key}={value}")

    command.extend(unknown)

    print("Launching VERL server:")
    print(" ".join(str(x) for x in command))
    print("-" * 60)

    try:
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"VERL server exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Verify the script is importable**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
python -c "import scripts.launch_verl" 2>&1 || python scripts/launch_verl.py --help
```

Expected: prints usage help without errors

- [ ] **Step 7.3: Commit**

```bash
git add scripts/launch_verl.py
git commit -m "feat(scripts): add launch_verl.py to start VERL training server"
```

---

## Task 8: `scripts/train_orchestrator.py` — rollout worker entry point

**Files:**
- Create: `scripts/train_orchestrator.py`

- [ ] **Step 8.1: Implement `scripts/train_orchestrator.py`**

```python
"""Start OrchestratorRollout workers and connect them to the VERL daemon.

Usage:
    python scripts/train_orchestrator.py --config train/config.yaml

This script:
  1. Reads train/config.yaml
  2. Sets environment variables
  3. Copies the config to output_dir for reproducibility
  4. Logs git commit + SLURM_JOB_ID to W&B run metadata
  5. Starts agentflow.Trainer with a NullTracer (no AgentOps required)
  6. Runs OrchestratorRollout workers connected to VERL daemon at agentflow.port
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Add src to path (matches run_experiment.py convention)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Start OrchestratorRollout workers.")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ── 1. Set environment variables ────────────────────────────────────────
    for key, value in config.get("env", {}).items():
        os.environ[key] = str(value)

    # ── 2. Pull settings from config ────────────────────────────────────────
    env = config.get("env", {})
    python_args = config.get("python_args", {})

    port = int(str(python_args.get("agentflow.port", 9999)))
    n_workers = int(str(env.get("N_WORKERS", 1)))
    rollout_n = int(str(python_args.get("actor_rollout_ref.rollout.n", 8)))
    train_temperature = float(str(env.get("TRAIN_TEMPERATURE", 0.7)))
    test_temperature = float(str(env.get("TEST_TEMPERATURE", 0.0)))
    max_turns = int(str(env.get("TOOL_STEPS", 5)))
    max_tokens = int(str(python_args.get("data.max_response_length", 2048)))
    experiment_name = str(env.get("EXPERIMENT_NAME", "cosmas-train"))
    output_dir = Path("experiments/results/training") / experiment_name

    # ── 3. Save config to output dir ────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    # ── 4. Log reproducibility info to stdout (captured by SLURM log) ───────
    #       W&B is initialised by VERL's Tracking class, not here.
    print(f"git_commit={_get_git_hash()}  slurm_job_id={os.environ.get('SLURM_JOB_ID', 'local')}  config={args.config}")

    # ── 5. Build NullTracer ─────────────────────────────────────────────────
    from agentflow.tracer.base import BaseTracer

    class NullTracer(BaseTracer):
        """No-op tracer — avoids AgentOps dependency."""
        def init(self): pass
        def teardown(self): pass
        def init_worker(self, worker_id): pass
        def teardown_worker(self, worker_id): pass

    # ── 6. Instantiate rollout agent ────────────────────────────────────────
    from fine_tuning.rollout import OrchestratorRollout
    from agentflow import Trainer

    rollout_dir = str(output_dir / "rollout_data")
    agent = OrchestratorRollout(
        rollout_dir=rollout_dir,
        rollout_n=rollout_n,
        train_temperature=train_temperature,
        test_temperature=test_temperature,
        max_turns=max_turns,
        max_tokens=max_tokens,
    )

    # ── 7. Start trainer ────────────────────────────────────────────────────
    trainer = Trainer(n_workers=n_workers, tracer=NullTracer())
    print(f"Connecting to VERL daemon at http://localhost:{port}/")
    trainer.fit(agent, f"http://localhost:{port}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Verify the script parses without error (agentflow not required yet)**

```bash
python -c "
import ast, sys
with open('scripts/train_orchestrator.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 8.3: Commit**

```bash
git add scripts/train_orchestrator.py
git commit -m "feat(scripts): add train_orchestrator.py rollout worker entrypoint"
```

---

## Task 9: `train/config.yaml`

**Files:**
- Create: `train/config.yaml`

- [ ] **Step 9.1: Create `train/config.yaml`**

```yaml
# train/config.yaml
# Training configuration for CoSMAS orchestrator RL fine-tuning.
# Mirrors AgentFlow's train/config.yaml structure.
# Launch with: python scripts/launch_verl.py --config train/config.yaml
#              python scripts/train_orchestrator.py --config train/config.yaml

env:
  BASE_MODEL: 'Qwen/Qwen3-8B'
  N_GPUS: 4
  ROLLOUT_TP_SIZE: 2
  EXPERIMENT_NAME: 'qwen3-8b-grpo-search-math'
  PROJECT_NAME: 'cosmas-rl-finetuning'
  BASE_DATA_DIR: 'data/training'
  ENABLE_TOOLS: ["web_search"]
  TOOL_STEPS: 5
  TRAIN_TEMPERATURE: 0.7
  TEST_TEMPERATURE: 0.0
  N_WORKERS: 1
  VERBOSITY: 'INFO'

python_args:
  agentflow.port: 9999
  algorithm.adv_estimator: 'grpo'
  data.train_files: '${BASE_DATA_DIR}/train/combined_train.parquet'
  data.val_files: '${BASE_DATA_DIR}/val/aime24.parquet'
  data.train_batch_size: 32
  data.train_max_samples: 128
  data.max_prompt_length: 18432
  data.max_response_length: 2048
  data.truncation: 'truncate'
  actor_rollout_ref.rollout.n: 8
  actor_rollout_ref.actor.ppo_mini_batch_size: 8
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu: 4
  actor_rollout_ref.rollout.multi_turn.format: 'hermes'
  actor_rollout_ref.model.path: '${BASE_MODEL}'
  actor_rollout_ref.model.lora_rank: 64
  actor_rollout_ref.model.lora_alpha: 16
  actor_rollout_ref.model.lora_target_modules: 'all-linear'
  actor_rollout_ref.model.enable_gradient_checkpointing: true
  actor_rollout_ref.model.use_remove_padding: true
  actor_rollout_ref.actor.optim.lr: 1.0e-6
  actor_rollout_ref.actor.use_kl_loss: true
  actor_rollout_ref.actor.kl_loss_coef: 0.001
  actor_rollout_ref.actor.clip_ratio_low: 0.2
  actor_rollout_ref.actor.clip_ratio_high: 0.3
  actor_rollout_ref.actor.fsdp_config.param_offload: false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload: false
  actor_rollout_ref.ref.fsdp_config.param_offload: false
  actor_rollout_ref.rollout.name: 'vllm'
  actor_rollout_ref.rollout.gpu_memory_utilization: 0.6
  actor_rollout_ref.rollout.tensor_model_parallel_size: '${ROLLOUT_TP_SIZE}'
  trainer.n_gpus_per_node: '${N_GPUS}'
  trainer.logger: ['console', 'wandb']
  trainer.project_name: '${PROJECT_NAME}'
  trainer.experiment_name: '${EXPERIMENT_NAME}'
  trainer.save_freq: 2
  trainer.test_freq: 2
  trainer.total_epochs: 5
  trainer.val_before_train: true
  trainer.critic_warmup: 0
  algorithm.use_kl_in_reward: false
```

- [ ] **Step 9.2: Verify YAML parses cleanly**

```bash
python -c "import yaml; cfg = yaml.safe_load(open('train/config.yaml')); print('env keys:', list(cfg['env'].keys())[:5])"
```

Expected: `env keys: ['BASE_MODEL', 'N_GPUS', 'ROLLOUT_TP_SIZE', 'EXPERIMENT_NAME', 'PROJECT_NAME']`

- [ ] **Step 9.3: Commit**

```bash
git add train/config.yaml
git commit -m "feat(train): add VERL training config for Qwen3-8B GRPO with LoRA"
```

---

## Task 10: Infrastructure — SLURM script and conda environment

**Files:**
- Create: `jobs/train_orchestrator.sh`
- Create: `jobs/environment_train.yml`

- [ ] **Step 10.1: Create `jobs/train_orchestrator.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=cosmas-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/%j_train.log
#SBATCH --error=jobs/logs/%j_train.err

set -euo pipefail

module load 2023
module load CUDA/12.1.1

conda activate cosmas-train

cd "$HOME/azywot/msc-thesis"

# Ensure log dir exists
mkdir -p jobs/logs

echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES"

# ── 1. Start VERL server in background ──────────────────────────────────────
echo "Starting VERL server..."
python scripts/launch_verl.py --config train/config.yaml > jobs/logs/${SLURM_JOB_ID}_verl.log 2>&1 &
VERL_PID=$!

# Give Ray + vLLM time to initialise (adjust if needed)
echo "Waiting 60s for VERL server to be ready..."
sleep 60

# ── 2. Start rollout workers ─────────────────────────────────────────────────
echo "Starting rollout workers..."
python scripts/train_orchestrator.py --config train/config.yaml

# ── 3. Wait for VERL ─────────────────────────────────────────────────────────
wait $VERL_PID
echo "Training complete."
```

- [ ] **Step 10.2: Create `jobs/environment_train.yml`**

```yaml
# Conda environment for orchestrator RL fine-tuning.
# Pin versions to match AgentFlow's setup_stable_gpu.sh exactly.
#
# Create with:
#   conda env create -f jobs/environment_train.yml
#   conda activate cosmas-train
#
# Then install AgentFlow:
#   pip install -e $HOME/azywot/AgentFlow/agentflow
#   pip install -e .   # msc-thesis itself

name: cosmas-train
channels:
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    # Core ML stack — pinned to AgentFlow's setup_stable_gpu.sh
    - torch==2.7.0
    - torchvision==0.22.0
    - torchaudio==2.7.0
    - transformers==4.53.3
    - flash-attn==2.8.1 --no-build-isolation
    - vllm==0.9.2
    - verl==0.5.0
    # AgentFlow (local clone)
    - agentflow @ file://$HOME/azywot/AgentFlow/agentflow
    # msc-thesis itself
    - -e .
    # Supporting deps
    - datasets>=3.0.0
    - pyarrow>=15.0.0
    - wandb>=0.16.0
    - filelock>=3.13.0
    - sympy>=1.13.0
    - omegaconf>=2.3.0
    - codetiming>=1.4.0
    - numpy
    - pandas
    - pyyaml>=6.0
    - python-dotenv>=1.0.0
    - requests>=2.31.0
```

- [ ] **Step 10.3: Commit**

```bash
git add jobs/train_orchestrator.sh jobs/environment_train.yml
git commit -m "feat(jobs): add SLURM job script and training conda environment"
```

---

## Task 11: Add `[training]` optional deps to `pyproject.toml`

Documents the training stack as an installable extras group.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 11.1: Add the `training` extras group**

In `pyproject.toml`, add inside `[project.optional-dependencies]`:

```toml
# RL fine-tuning dependencies (training env only — see jobs/environment_train.yml)
# Install with: pip install -e ".[training]"
# Note: torch, vllm, flash-attn, verl versions are pinned in environment_train.yml
training = [
    "verl==0.5.0",
    "filelock>=3.13.0",
    "omegaconf>=2.3.0",
    "codetiming>=1.4.0",
]
```

- [ ] **Step 11.2: Verify pyproject.toml still parses**

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb')); print('OK')" 2>/dev/null || \
python -c "import tomli; tomli.load(open('pyproject.toml', 'rb')); print('OK')"
```

Expected: `OK`

- [ ] **Step 11.3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(deps): add [training] optional extras group for RL fine-tuning pipeline"
```

---

## Task 12: End-to-end smoke test (dry run, no GPU)

Verifies the full import chain and config loading work correctly without needing a GPU or VERL server.

- [ ] **Step 12.1: Run all unit tests**

```bash
cd /Users/agatazywot/Desktop/uni/YEAR2/thesis/msc-thesis
pytest tests/unit/ -v --tb=short
```

Expected: All tests pass (no failures)

- [ ] **Step 12.2: Verify train_orchestrator.py config loading**

```bash
python -c "
import yaml, os
cfg = yaml.safe_load(open('train/config.yaml'))
env = cfg['env']
pa = cfg['python_args']
print('BASE_MODEL:', env['BASE_MODEL'])
print('lora_rank:', pa['actor_rollout_ref.model.lora_rank'])
print('adv_estimator:', pa['algorithm.adv_estimator'])
print('Config OK')
"
```

Expected:
```
BASE_MODEL: Qwen/Qwen3-8B
lora_rank: 64
adv_estimator: grpo
Config OK
```

- [ ] **Step 12.3: Verify FinetuningConfig loads from YAML**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from fine_tuning.config import FinetuningConfig
cfg = FinetuningConfig(
    base_model='Qwen/Qwen3-8B',
    train_data='data/training/train/combined_train.parquet',
    val_data='data/training/val/aime24.parquet',
    output_dir='experiments/results/training/smoke_test',
)
print('lora_rank:', cfg.lora_rank)
print('seed:', cfg.seed)
print('FinetuningConfig OK')
"
```

Expected:
```
lora_rank: 64
seed: 42
FinetuningConfig OK
```

- [ ] **Step 12.4: Verify OrchestratorReward**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from fine_tuning.reward import OrchestratorReward
r = OrchestratorReward()
print('nq correct:', r('Paris', 'Paris', 'nq'))
print('math wrong:', r('43', '42', 'math'))
print('OrchestratorReward OK')
"
```

Expected:
```
nq correct: 1.0
math wrong: 0.0
OrchestratorReward OK
```

- [ ] **Step 12.5: Final commit**

```bash
git add -A
git commit -m "test(fine_tuning): end-to-end smoke test passes — pipeline ready for cluster"
```

---

## Post-Training: Merge LoRA for Inference

After training completes on Snellius, merge the LoRA adapter to produce a standard HF model.

- [ ] **Run the merge (on Snellius, in `cosmas-train` conda env)**

```bash
conda activate cosmas-train
python $HOME/azywot/AgentFlow/util/model_merger.py \
  --base_model Qwen/Qwen3-8B \
  --lora_path experiments/results/training/qwen3-8b-grpo-search-math/checkpoint_best/actor/lora_weights.pt \
  --output_dir experiments/results/training/qwen3-8b-grpo-search-math/merged_model/
```

- [ ] **Use merged model in inference — update one line in any existing config YAML**

```yaml
# experiments/configs/qwen3/agentflow/qwen3_8b_gaia_ft.yaml
model:
  path_or_id: /home/azywot/msc-thesis/experiments/results/training/qwen3-8b-grpo-search-math/merged_model/
  # all other fields (family, role, generation params) unchanged
```

No changes to `VLLMProvider`, `AgenticOrchestrator`, or evaluation scripts.
