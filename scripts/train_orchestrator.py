"""Start OrchestratorRollout workers and connect them to the VERL daemon.

Usage:
    python scripts/train_orchestrator.py --config train/config.yaml

This script:
  1. Reads train/config.yaml and sets environment variables
  2. Copies the config to output_dir for reproducibility
  3. Starts agentflow.Trainer with a NullTracer (no AgentOps required)
  4. Runs OrchestratorRollout workers connected to the VERL daemon
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Add src to path (matches run_experiment.py convention)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Start OrchestratorRollout workers.")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ── 1. Set environment variables ────────────────────────────────────────
    for key, value in config.get("env", {}).items():
        os.environ[key] = str(value)

    # ── 2. Pull settings from config ────────────────────────────────────────
    env = config.get("env", {})
    python_args = config.get("python_args", {})

    port = int(str(python_args.get("agentflow.port", 9999)))
    n_workers = int(str(env.get("N_WORKERS", 1)))
    rollout_n = int(str(python_args.get("actor_rollout_ref.rollout.n", 8)))
    train_temperature = float(str(env.get("TRAIN_TEMPERATURE", 0.7)))
    test_temperature = float(str(env.get("TEST_TEMPERATURE", 0.0)))
    max_turns = int(str(env.get("TOOL_STEPS", 5)))
    max_tokens = int(str(python_args.get("data.max_response_length", 2048)))
    thinking_mode = str(env.get("THINKING_MODE", "NO")).upper()
    use_thinking = thinking_mode in ("ORCHESTRATOR_ONLY", "ALL")
    experiment_name = str(env.get("EXPERIMENT_NAME", "cosmas-train"))
    output_dir = Path("experiments/results/training") / experiment_name

    # ── 3. Save config to output dir ────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    # ── 4. Log reproducibility info (captured by SLURM log) ─────────────────
    #       W&B is initialised by VERL's Tracking class, not by this script.
    print(
        f"git_commit={_get_git_hash()}  "
        f"slurm_job_id={os.environ.get('SLURM_JOB_ID', 'local')}  "
        f"config={args.config}"
    )

    # ── 5. Build NullTracer ─────────────────────────────────────────────────
    from agentflow.tracer.base import BaseTracer

    class NullTracer(BaseTracer):
        """No-op tracer — avoids AgentOps dependency."""
        def init(self): pass
        def teardown(self): pass
        def init_worker(self, worker_id): pass
        def teardown_worker(self, worker_id): pass

    # ── 6. Instantiate rollout agent ────────────────────────────────────────
    from fine_tuning.rollout import OrchestratorRollout
    from agentflow import Trainer

    rollout_dir = str(output_dir / "rollout_data")
    agent = OrchestratorRollout(
        rollout_dir=rollout_dir,
        rollout_n=rollout_n,
        train_temperature=train_temperature,
        test_temperature=test_temperature,
        max_turns=max_turns,
        max_tokens=max_tokens,
        use_thinking=use_thinking,
    )

    # ── 7. Start trainer ────────────────────────────────────────────────────
    trainer = Trainer(n_workers=n_workers, tracer=NullTracer())
    print(f"Connecting to VERL daemon at http://localhost:{port}/")
    trainer.fit(agent, f"http://localhost:{port}/")


if __name__ == "__main__":
    main()
