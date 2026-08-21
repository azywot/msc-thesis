"""Start OrchestratorRollout workers and connect them to the VERL daemon.

Note: upstream ``agentflow.Trainer`` imports ``AgentOpsTracer`` at package import time,
so ``agentops``, ``flask``, and ``setproctitle`` must be installed even though this
script only uses ``NullTracer`` at runtime.

Usage:
    python scripts/train_orchestrator.py --config experiments/configs/fine_tuning/config.yaml

This script:
  1. Reads the training config and sets environment variables
  2. Validates SUBAGENT_ENDPOINT (frozen sub-agent vLLM server must be running)
  3. Copies the config to output_dir for reproducibility
  4. Starts agentflow.Trainer with a NullTracer (no AgentOps required)
  5. Runs OrchestratorRollout workers connected to the VERL daemon
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

# Route INFO/DEBUG to stdout (→ SLURM .log); WARNING+ stay on stderr (→ SLURM .err).
# force=True replaces any handlers that packages may have already installed.
class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        self.max_level = max_level
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.DEBUG)
_stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[_stdout_handler, _stderr_handler],
    force=True,
)



def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Start OrchestratorRollout workers.")
    parser.add_argument("--config", type=str, default="experiments/configs/fine_tuning/config.yaml")
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
    subagent_endpoint = str(env.get("SUBAGENT_ENDPOINT", os.environ.get("SUBAGENT_ENDPOINT", "")))
    subagent_model = str(env.get("SUBAGENT_MODEL", os.environ.get("SUBAGENT_MODEL", "Qwen/Qwen3-1.7B")))
    experiment_name = str(env.get("EXPERIMENT_NAME", "cosmas-train"))
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_subdir = os.environ.get("VERL_RUN_TAG") or f"{datetime.now().strftime('%d-%m-%Y_%H-%M')}-{job_id}"
    output_dir = Path("experiments/results/fine_tuning") / experiment_name / run_subdir

    # ── 3. Validate sub-agent endpoint ──────────────────────────────────────
    if not subagent_endpoint:
        print(
            "ERROR: SUBAGENT_ENDPOINT is not set.\n"
            "  Sub-agents must use a separate frozen vLLM server (not the VERL endpoint).\n"
            "  Start one first:\n"
            f"    vllm serve {subagent_model} --port 9998 --tensor-parallel-size 1 \\\n"
            f"      --gpu-memory-utilization 0.15 --max-model-len 8192\n"
            "  Then export SUBAGENT_ENDPOINT=http://localhost:9998/v1  (or set it in the config)."
        )
        sys.exit(1)

    # ── 4. Save config to output dir ────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    # ── 5. Log reproducibility info (captured by SLURM log) ─────────────────
    #       W&B is initialised by VERL's Tracking class, not by this script.
    print(
        f"git_commit={_get_git_hash()}  "
        f"slurm_job_id={os.environ.get('SLURM_JOB_ID', 'local')}  "
        f"config={args.config}  "
        f"subagent_model={subagent_model}  "
        f"subagent_endpoint={subagent_endpoint}"
    )

    # ── 6. Build NullTracer ─────────────────────────────────────────────────
    from fine_tuning.agentflow.tracer.base import BaseTracer

    class NullTracer(BaseTracer):
        """No-op tracer — avoids AgentOps dependency."""
        def init(self): pass
        def teardown(self): pass
        def init_worker(self, worker_id): pass
        def teardown_worker(self, worker_id): pass

        @contextmanager
        def trace_context(self, name=None):
            yield

        def get_last_trace(self):
            return []

    # ── 7. Instantiate rollout agent ────────────────────────────────────────
    from fine_tuning.rollout import OrchestratorRollout
    from fine_tuning.agentflow import Trainer

    rollout_dir = str(output_dir / "rollout_data")
    common = dict(
        rollout_dir=rollout_dir,
        rollout_n=rollout_n,
        train_temperature=train_temperature,
        test_temperature=test_temperature,
        max_turns=max_turns,
        max_tokens=max_tokens,
        use_thinking=use_thinking,
        subagent_endpoint=subagent_endpoint,
        subagent_model=subagent_model,
    )

    # Prefix-RFT replaces the rollout agent so the first k decisions of a hybrid rollout
    # are replayed from a teacher demonstration instead of generated. k arrives per
    # rollout in the task payload; k = 0 behaves exactly like OrchestratorRollout.
    prefix_rft = os.environ.get("PREFIX_RFT", "").strip().lower() in ("1", "true", "yes", "on")
    if prefix_rft:
        from fine_tuning.prefix_rollout import PrefixOrchestratorRollout

        demos_path = os.environ.get(
            "PREFIX_DEMOS_PATH", "data/training/prefix_rft/prefix_demos.parquet"
        )
        if not Path(demos_path).exists():
            raise FileNotFoundError(
                f"PREFIX_RFT is on but the demonstration store is missing: {demos_path}\n"
                "Build it with: sbatch jobs/fine_tuning/008_build_prefix_demos.job"
            )
        # The store is named twice in every config: PREFIX_DEMOS_PATH under env: for
        # these workers, and prefix_rft.demos_path under python_args: for the driver.
        # The worker ignores the Hydra key entirely, so setting only that one leaves the
        # worker on the default path. If the default happens to exist, the driver
        # dispatches prefixes against one store while the worker replays from another
        # and nothing says so. Compare them here rather than let that run.
        configured = str(python_args.get("prefix_rft.demos_path", "") or "")
        if configured and os.path.abspath(configured) != os.path.abspath(demos_path):
            raise ValueError(
                "The demonstration store is configured twice and the two disagree:\n"
                f"  env.PREFIX_DEMOS_PATH        = {demos_path}\n"
                f"  python_args.prefix_rft.demos_path = {configured}\n"
                "The rollout workers use the first and the driver uses the second, so "
                "this run would dispatch prefixes for one store and replay from another. "
                "Set both to the same path."
            )
        agent = PrefixOrchestratorRollout(
            demos_path=demos_path,
            base_model=str(env.get("BASE_MODEL", "Qwen/Qwen3-8B")),
            **common,
        )
        print(f"  Prefix-RFT enabled: PrefixOrchestratorRollout, demos={demos_path}")
    else:
        agent = OrchestratorRollout(**common)

    # ── 8. Start trainer ────────────────────────────────────────────────────
    trainer = Trainer(n_workers=n_workers, tracer=NullTracer())
    print(f"Connecting to VERL daemon at http://localhost:{port}/")
    trainer.fit(agent, f"http://localhost:{port}/")


if __name__ == "__main__":
    main()
