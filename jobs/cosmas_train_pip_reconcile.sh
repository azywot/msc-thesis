#!/usr/bin/env bash
# Resolve one stubborn pip-resolver conflict that survives `pip install -e ".[training]"`:
#
#   math-verify[antlr4_13_2]>=0.7.0  (core dep, pyproject.toml)  → antlr4-python3-runtime==4.13.2
#   omegaconf>=2.3.0                  (training extra)            → antlr4-python3-runtime~=4.9.3
#
# pip picks one and silently lets the other break at import time. Force the omegaconf-side
# version (4.9.x) since VERL/Hydra are the load-bearing consumers; math-verify uses antlr only
# for its LaTeX parser, which works with 4.9.x in practice.
#
# Everything else the previous version of this script did is now redundant:
#   - openai pin: vllm 0.17.0 declares `openai>=1.99.1,<2.25.0` in common.txt; let pip resolve.
#   - agentops/flask/setproctitle force-reinstall: now in pyproject.toml [training] extras.
#
# Run once after `pip install -e ".[training]"` on the cosmas-train env.
set -euo pipefail
echo "Reconciling antlr4-python3-runtime (math-verify ↔ omegaconf)..."
python -m pip install --force-reinstall 'antlr4-python3-runtime>=4.9.0,<4.10'
