#!/usr/bin/env bash
# After `pip install -e .` on cosmas-train, pip may upgrade packages that break the
# training stack:
#   - openai 2.x vs vllm 0.9.2 (requires openai<=1.90.0,>=1.52.0)
#   - antlr4-python3-runtime 4.13 vs hydra-core / omegaconf (require 4.9.*)
#
# Run this once after installing agent-engine into the same env as VERL/vLLM.
set -euo pipefail
echo "Reconciling cosmas-train pins (openai ↔ vLLM; antlr ↔ Hydra/omegaconf)..."
pip install \
  'openai>=1.52.0,<=1.90.0' \
  'antlr4-python3-runtime>=4.9.0,<4.10' \
  --force-reinstall
