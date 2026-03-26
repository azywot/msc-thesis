#!/usr/bin/env python3
"""
Run all figure / table generators under scripts/plots/.

  1. efficiency_plots.py           — token, latency, tool-call figures
  2. orchestrator_capabilities_figure.py
  3. generate_results_table.py

Usage (from repo root):

    python scripts/plots/main.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPTS = (
    "efficiency_plots.py",
    "orchestrator_capabilities_figure.py",
    "generate_results_table.py",
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
    print(f"\n{'=' * 60}\nAll plot scripts finished.\n{'=' * 60}")


if __name__ == "__main__":
    main()
