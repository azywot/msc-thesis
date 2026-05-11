"""Pre-flight smoke test for the fine-tuning pipeline.

Validates imports, reward routing, config loading, parquet schema, and
OrchestratorRollout instantiation WITHOUT requiring a running VERL server
or GPU.  Run this before submitting the training job.

Usage:
    conda activate cosmas-train
    python scripts/test_ft_smoke.py [--data-dir data/training/smoke]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []  # (check, status, detail)


def check(name: str):
    """Decorator that records pass/fail for each named check."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs) or ""
                results.append((name, PASS, detail))
            except Exception as exc:
                results.append((name, FAIL, f"{type(exc).__name__}: {exc}"))
                traceback.print_exc()
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@check("import agentflow")
def check_agentflow():
    import agentflow  # noqa: F401
    from agentflow import LitAgent, Trainer, reward  # noqa: F401
    return f"agentflow imported"


@check("import fine_tuning")
def check_fine_tuning():
    from fine_tuning import FinetuningConfig, OrchestratorReward  # noqa: F401
    from fine_tuning.rollout import OrchestratorRollout  # noqa: F401
    return "FinetuningConfig, OrchestratorReward, OrchestratorRollout imported"


@check("import agent_engine")
def check_agent_engine():
    from agent_engine.core.orchestrator import AgenticOrchestrator  # noqa: F401
    from agent_engine.models.api_provider import OpenAIProvider  # noqa: F401
    return "AgenticOrchestrator, OpenAIProvider imported"


@check("reward routing — nq/hotpotqa → mode=qa")
def check_reward_qa():
    from fine_tuning.reward import OrchestratorReward
    r = OrchestratorReward()
    score = r("Paris", "Paris", "nq")
    assert score == 1.0, f"expected 1.0, got {score}"
    score = r("wrong", "Paris", "hotpotqa")
    assert score == 0.0, f"expected 0.0, got {score}"
    return "nq correct=1.0, hotpotqa wrong=0.0"


@check("reward routing — deepmath → mode=gen")
def check_reward_gen():
    from fine_tuning.reward import OrchestratorReward
    r = OrchestratorReward()
    score = r("42", "42", "deepmath")
    assert score == 1.0, f"expected 1.0, got {score}"
    score = r("None", "42", "deepmath")
    assert score == 0.0, f"expected 0.0, got {score}"
    return "deepmath correct=1.0, wrong=0.0"


@check("reward routing — empty/None prediction → 0.0")
def check_reward_none():
    from fine_tuning.reward import OrchestratorReward
    r = OrchestratorReward()
    for pred in ("", "None", None):
        score = r(pred, "42", "deepmath")
        assert score == 0.0, f"expected 0.0 for pred={pred!r}, got {score}"
    return "empty/None/None all → 0.0"


@check("config loading — smoke config")
def check_config(config_path: str):
    import yaml
    from fine_tuning.config import FinetuningConfig
    p = Path(config_path)
    if not p.exists():
        return f"SKIP — {p} not found (create it first)"
    with open(p) as f:
        raw = yaml.safe_load(f)
    # FinetuningConfig reads from a flat dict; just verify YAML parses cleanly
    env = raw.get("env", {})
    assert "BASE_MODEL" in env, "BASE_MODEL missing from env block"
    assert "THINKING_MODE" in env, "THINKING_MODE missing from env block"
    assert "SUBAGENT_MODEL" in env, "SUBAGENT_MODEL missing from env block"
    return (
        f"YAML valid, BASE_MODEL={env['BASE_MODEL']}, "
        f"THINKING_MODE={env['THINKING_MODE']}, "
        f"SUBAGENT_MODEL={env['SUBAGENT_MODEL']}"
    )


@check("OrchestratorRollout — instantiation (no GPU)")
def check_rollout_init():
    from fine_tuning.rollout import OrchestratorRollout
    agent = OrchestratorRollout(
        rollout_dir="/tmp/cosmas_smoke_rollout",
        rollout_n=2,
        train_temperature=0.7,
        test_temperature=0.0,
        max_turns=2,
        max_tokens=512,
        use_thinking=True,
        subagent_endpoint="http://localhost:9998/v1",
        subagent_model="Qwen/Qwen3-1.7B",
    )
    assert agent.use_thinking is True
    assert agent.max_turns == 2
    assert agent.subagent_model == "Qwen/Qwen3-1.7B"
    assert agent.subagent_endpoint == "http://localhost:9998/v1"
    return "OrchestratorRollout instantiated with frozen subagent_endpoint"


@check("data parquet schema — smoke training data")
def check_parquet(data_dir: str):
    import pandas as pd
    from fine_tuning.data.prepare import validate_parquet_schema
    base = Path(data_dir)
    paths = [
        base / "train" / "combined_train.parquet",
        base / "val" / "val_search.parquet",
        base / "val" / "val_deepmath.parquet",
        base / "val" / "val_combined.parquet",
    ]
    msgs = []
    for p in paths:
        if not p.exists():
            msgs.append(f"SKIP {p.name} (not prepared yet)")
            continue
        df = pd.read_parquet(p)
        validate_parquet_schema(df)
        sources = df["data_source"].unique().tolist()
        msgs.append(f"{p.name}: {len(df)} rows, sources={sources}")
    return " | ".join(msgs) if msgs else "no parquet files found"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/training/smoke",
                        help="Directory containing smoke train/val parquet files")
    parser.add_argument("--config", default="experiments/configs/train/config_smoke.yaml",
                        help="Smoke training config YAML to validate")
    args = parser.parse_args()

    print("=" * 60)
    print("CoSMAS fine-tuning pre-flight smoke test")
    print("=" * 60)

    check_agentflow()
    check_fine_tuning()
    check_agent_engine()
    check_reward_qa()
    check_reward_gen()
    check_reward_none()
    check_config(args.config)
    check_rollout_init()
    check_parquet(args.data_dir)

    print()
    print(f"{'Check':<45} {'Status':<6} Detail")
    print("-" * 100)
    n_fail = 0
    for name, status, detail in results:
        print(f"{name:<45} {status:<6} {detail}")
        if status == FAIL:
            n_fail += 1

    print()
    if n_fail == 0:
        print(f"ALL {len(results)} checks passed — safe to submit training job.")
    else:
        print(f"{n_fail}/{len(results)} checks FAILED — fix before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
