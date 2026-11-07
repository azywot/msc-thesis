"""Real-data acceptance tests for the text-only GAIA/HLE splits.

Each test skips when the data files it reads are absent (data/ is gitignored).
"""

import json
from pathlib import Path

import pytest

from agent_engine.config.schema import DatasetConfig
from agent_engine.datasets.loaders.gaia import GAIADataset
from agent_engine.datasets.loaders.hle import HLEDataset
from agent_engine.datasets.text_only import is_gaia_image_record

DATA = Path(__file__).resolve().parents[2] / "data"
GAIA_FULL = DATA / "GAIA" / "all_validation.jsonl"
GAIA_TEXT = DATA / "GAIA" / "all_validation_text.jsonl"
HLE_OLD = DATA / "HLE" / "test_subset_200.jsonl"
HLE_TEXT = DATA / "HLE" / "test_text_subset_200.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.skipif(not (GAIA_FULL.exists() and GAIA_TEXT.exists()), reason="GAIA files absent")
def test_gaia_text_is_validation_minus_images():
    full = read_jsonl(GAIA_FULL)
    text = read_jsonl(GAIA_TEXT)
    assert len(text) == 155
    assert not any(is_gaia_image_record(r) for r in text)
    assert [full[r["source_index"]]["Question"] for r in text] == [r["Question"] for r in text]
    assert [r["source_index"] for r in text] == sorted(r["source_index"] for r in text)


@pytest.mark.skipif(not (HLE_OLD.exists() and HLE_TEXT.exists()), reason="HLE files absent")
def test_hle_text_nests_existing_subset():
    old = read_jsonl(HLE_OLD)
    new = read_jsonl(HLE_TEXT)
    old_text_ids = [r["id"] for r in old if not r["has_image"]]
    assert len(old_text_ids) == 179
    assert len(new) == 200 and len({r["id"] for r in new}) == 200
    assert not any(r["has_image"] for r in new)
    assert [r["id"] for r in new][:179] == old_text_ids


@pytest.mark.skipif(not (GAIA_FULL.exists() and HLE_OLD.exists()), reason="existing files absent")
def test_existing_splits_load_unchanged():
    gaia = GAIADataset(DatasetConfig(name="gaia", split="all_validation", data_dir=DATA)).load()
    assert len(gaia) == 165
    hle = HLEDataset(DatasetConfig(name="hle", split="test_subset_200", data_dir=DATA)).load()
    assert len(hle) == 200
    assert HLEDataset(
        DatasetConfig(name="hle", split="test_subset_200", data_dir=DATA)
    )._resolve_data_path() == (DATA / "HLE" / "test_subset_200.jsonl")
