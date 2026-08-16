#!/usr/bin/env python
"""Thin CLI wrapper. The implementation lives in agent_engine.runner.experiment.

Kept so existing commands and SLURM job scripts continue to work unchanged:

    python scripts/run_experiment.py --config <config.yaml>

The installed ``cosmas-run`` entry point invokes the same ``main``.
"""

from agent_engine.runner.experiment import main

if __name__ == "__main__":
    main()
