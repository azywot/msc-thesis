#!/usr/bin/env python3
"""
Run all table generators under scripts/tables/.

  1. generate_ablation_tables.py  — tool and structured-memory ablation tables
  2. ds_table.py                  — DeepSeek GAIA results table

Usage (from repo root):

    python scripts/tables/main.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS = (
    "generate_ablation_tables.py",
    "ds_table.py",
)


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    for name in _SCRIPTS:
        script = here / name
        if not script.is_file():
            print(f"Missing script: {script}", file=sys.stderr)
            sys.exit(1)
        print(f"\n{'=' * 60}\n→ {name}\n{'=' * 60}", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(root),
            check=True,
        )
    print(f"\n{'=' * 60}\nAll table scripts finished.\n{'=' * 60}")


if __name__ == "__main__":
    main()
