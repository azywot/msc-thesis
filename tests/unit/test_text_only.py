import json
import random

from agent_engine.config.schema import DatasetConfig
from agent_engine.datasets.loaders.gaia import GAIADataset
from agent_engine.datasets.text_only import (
    filter_gaia_text,
    is_gaia_image_record,
    select_hle_text,
)
from agent_engine.utils import set_seed


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
    set_seed(0)
    old = list(exs)
    random.shuffle(old)
    old = old[:20]
    new = select_hle_text(exs, subset=20, seed=0)
    old_text = [e["id"] for e in old if not e["image"]]
    assert len(new) == 20 and not any(e["image"] for e in new)
    assert [e["id"] for e in new][: len(old_text)] == old_text


def test_select_hle_text_without_subset_filters_only():
    exs = [{"id": 0, "image": ""}, {"id": 1, "image": "x"}, {"id": 2, "image": ""}]
    assert [e["id"] for e in select_hle_text(exs, subset=0, seed=0)] == [0, 2]


def test_gaia_loader_passes_source_index(tmp_path):
    (tmp_path / "GAIA").mkdir()
    rows = [
        {"Question": "q1", "Answer": "a1", "source_index": 7},
        {"Question": "q2", "Answer": "a2"},
    ]
    with open(tmp_path / "GAIA" / "x.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cfg = DatasetConfig(name="gaia", split="x", data_dir=tmp_path)
    exs = GAIADataset(cfg).load()
    assert exs[0].metadata["source_index"] == 7
    assert "source_index" not in exs[1].metadata
