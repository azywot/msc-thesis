import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import tempfile
import yaml
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


def test_custom_fields():
    cfg = FinetuningConfig(
        base_model="Qwen/Qwen3-8B",
        train_data="data/train.parquet",
        val_data="data/val.parquet",
        output_dir="/tmp/run",
        lora_rank=32,
        seed=7,
    )
    assert cfg.lora_rank == 32
    assert cfg.seed == 7


def test_from_yaml():
    content = {
        "base_model": "Qwen/Qwen3-8B",
        "train_data": "data/train.parquet",
        "val_data": "data/val.parquet",
        "output_dir": "/tmp/run",
        "wandb_project": "cosmas-test",
        "unknown_key": "ignored",   # should be silently dropped
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(content, f)
        path = f.name
    cfg = FinetuningConfig.from_yaml(path)
    assert cfg.base_model == "Qwen/Qwen3-8B"
    assert cfg.wandb_project == "cosmas-test"
    assert cfg.lora_rank == 64  # default applied
