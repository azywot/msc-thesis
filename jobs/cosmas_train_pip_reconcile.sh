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
# Also pin peft<0.19. peft 0.19.x introduced _maybe_shard_state_dict_for_tp() which
# unconditionally imports EmbeddingParallel from transformers.integrations.tensor_parallel
# (peft/utils/save_and_load.py:500-505); EmbeddingParallel does not exist in any released
# transformers version (checked 4.57.6 and 5.0.0). The call site is gated by
# torch.distributed.is_initialized() and only fires when PeftModel.from_pretrained() loads
# a saved adapter — so fresh-train works but resume-from-checkpoint crashes with ImportError.
# verl declares peft with no version constraint, so pip otherwise resolves to 0.19.1 (latest).
#
# Everything else the previous version of this script did is now redundant:
#   - openai pin: vllm 0.17.0 declares `openai>=1.99.1,<2.25.0` in common.txt; let pip resolve.
#   - agentops/flask/setproctitle force-reinstall: now in pyproject.toml [training] extras.
#
# Run once after `pip install -e ".[training]"` on the cosmas-train env.
set -euo pipefail
echo "Reconciling antlr4-python3-runtime (math-verify ↔ omegaconf)..."
python -m pip install --force-reinstall 'antlr4-python3-runtime>=4.9.0,<4.10'
echo "Pinning peft<0.19 (workaround for EmbeddingParallel ImportError in peft 0.19.x)..."
# CRITICAL: --no-deps prevents pip from re-resolving the whole transitive graph. Without it,
# pip picks the LATEST torch / transformers / numpy / huggingface_hub because pyproject.toml
# does not constrain those upper bounds, blowing away the env's working pinned stack
# (torch==2.7.1, transformers<5, numpy==1.26.4 etc.). peft 0.18.1's runtime deps are already
# satisfied by the canonical env, so --no-deps is safe here.
python -m pip install --force-reinstall --no-deps 'peft==0.18.1'
