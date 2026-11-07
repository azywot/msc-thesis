"""Pure helpers for selecting text-only (no image) GAIA and HLE questions."""

import random
from pathlib import Path
from typing import List

from ..tools.image_inspector import SUPPORTED_IMAGE_EXTS
from ..utils import set_seed


def is_gaia_image_record(record: dict) -> bool:
    """True when the GAIA record's attachment is an image the image tool supports."""
    return Path(record.get("file_name") or "").suffix.lower() in SUPPORTED_IMAGE_EXTS


def filter_gaia_text(records: List[dict]) -> List[dict]:
    """Drop image records; return copies tagged with their index in the input list."""
    out = []
    for idx, record in enumerate(records):
        if is_gaia_image_record(record):
            continue
        copy = dict(record)
        copy["source_index"] = idx
        out.append(copy)
    return out


def is_hle_image_example(example: dict) -> bool:
    """True when the HLE example carries an image."""
    return bool(example.get("image"))


def select_hle_text(examples: list, subset: int, seed: int) -> list:
    """Select up to `subset` text-only HLE examples.

    Shuffles the full list first (same call sequence as the original subset
    download) so the text-only selection nests inside the original subset.
    """
    pool = list(examples)
    if 0 < subset < len(pool):
        set_seed(seed)
        random.shuffle(pool)
    pool = [e for e in pool if not is_hle_image_example(e)]
    if subset > 0:
        pool = pool[:subset]
    return pool
