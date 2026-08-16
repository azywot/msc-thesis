#!/usr/bin/env python
"""Thin CLI wrapper.  Implementation lives in :mod:`agent_engine.analysis.eval_runs.baseline_counterfactual`.

Kept at this path because thesis notes, job files and existing shell history
invoke it here.  Same argv, same output paths.
"""

from agent_engine.analysis.eval_runs.baseline_counterfactual import main

if __name__ == "__main__":
    main()
