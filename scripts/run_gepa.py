"""GEPA prompt optimisation CLI.

Modes:
  splits   — generate train/val/test split files from existing raw_results.json
  optimize — run GEPA optimisation loop, save best candidate
  evaluate — evaluate best candidate on held-out test set
  diff     — print diff between seed and best candidate prompts

Usage:
  python scripts/run_gepa.py --mode splits   --config experiments/configs/gepa/gaia.yaml
  python scripts/run_gepa.py --mode optimize --config experiments/configs/gepa/gaia.yaml
  python scripts/run_gepa.py --mode evaluate --config experiments/configs/gepa/gaia.yaml
  python scripts/run_gepa.py --mode diff     --config experiments/configs/gepa/gaia.yaml
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add src/ and scripts/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import yaml

from agent_engine.config.schema import DatasetConfig
from agent_engine.core import AgenticOrchestrator, ToolRegistry
from agent_engine.datasets import DatasetRegistry
from agent_engine.tools import CodeGeneratorTool, TextInspectorTool, WebSearchTool
from agent_engine.utils import set_seed, setup_logging
from gepa_integration.seed import build_seed_candidate, build_splits


def load_gepa_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_default_split(benchmark: str) -> str:
    defaults = {
        "gaia": "all_validation",
        "gpqa": "diamond",
        "math500": "test",
        "aime": "train",
    }
    return defaults.get(benchmark, "validation")


def _load_examples(cfg: dict, question_ids: list) -> list:
    ds_cfg = DatasetConfig(
        name=cfg["benchmark"],
        split=cfg.get("dataset_split", _get_default_split(cfg["benchmark"])),
        data_dir=Path(cfg.get("data_dir", "./data")),
        subset_num=-1,
    )
    dataset = DatasetRegistry.get(ds_cfg)
    all_examples = list(dataset)
    id_set = set(question_ids)
    return [ex for ex in all_examples if ex.question_id in id_set]


def _build_tool_registry(cfg: dict) -> ToolRegistry:
    import os
    tools = ToolRegistry()
    enabled = cfg.get("tools", {}).get("enabled_tools", ["web_search", "code_generator"])
    direct = cfg.get("tools", {}).get("direct_tool_call", True)
    provider = cfg.get("tools", {}).get("web_tool_provider", "serper")

    if "web_search" in enabled:
        tools.register(WebSearchTool(
            api_key=os.environ.get("SERPER_API_KEY" if provider == "serper" else "TAVILY_API_KEY", ""),
            provider=provider,
            direct_mode=direct,
        ))
    if "code_generator" in enabled:
        tools.register(CodeGeneratorTool(direct_mode=direct))
    if "text_inspector" in enabled:
        tools.register(TextInspectorTool())

    return tools


# ─────────────────────────────────────────────── MODE: splits ──────────────

def run_splits(cfg: dict, config_path: Path) -> None:
    existing_results = Path(cfg["existing_results"])
    if not existing_results.is_absolute():
        existing_results = config_path.parent.parent.parent / existing_results

    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = config_path.parent.parent.parent / splits_file

    split_cfg = cfg.get("splits", {})
    train_n = split_cfg.get("train_n", 80)
    val_n = split_cfg.get("val_n", 45)
    seed = cfg.get("seed", 1)

    print(f"Building splits for {cfg['benchmark']}...")
    print(f"  Source: {existing_results}")
    print(f"  Train: {train_n}, Val: {val_n}, seed: {seed}")

    splits = build_splits(
        raw_results_path=existing_results,
        train_n=train_n,
        val_n=val_n,
        seed=seed,
        output_path=splits_file,
    )

    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val:   {len(splits['val'])} examples")
    print(f"  Test:  {len(splits['test'])} examples")
    print(f"  Saved: {splits_file}")


# ─────────────────────────────────────────────── MODE: optimize ────────────

def run_optimize(cfg: dict, config_path: Path) -> None:
    from gepa.api import optimize

    gepa_cfg = cfg["gepa"]
    run_dir = Path(gepa_cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = config_path.parent.parent.parent / splits_file
    if not splits_file.exists():
        print(f"ERROR: splits file not found: {splits_file}")
        print("Run --mode splits first.")
        sys.exit(1)

    with open(splits_file) as f:
        splits = json.load(f)

    train_examples = _load_examples(cfg, splits["train"])
    val_examples = _load_examples(cfg, splits["val"])
    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples.")

    tool_registry = _build_tool_registry(cfg)

    from agent_engine.models.base import ModelConfig, ModelFamily, get_tool_call_format
    from agent_engine.models.vllm_provider import VLLMProvider

    model_cfg_raw = cfg["model"]
    model_cfg = ModelConfig(
        name=model_cfg_raw["name"],
        path_or_id=model_cfg_raw["path_or_id"],
        family=ModelFamily.QWEN3,
        role="orchestrator",
        use_thinking=True,
    )
    model_provider = VLLMProvider(model_cfg)

    from gepa_integration.adapter import AgentGEPAAdapter
    adapter = AgentGEPAAdapter(
        model_provider=model_provider,
        tool_registry=tool_registry,
        use_thinking=True,
        max_turns=cfg.get("max_turns", 15),
    )

    tool_schemas = tool_registry.get_all_schemas()
    tcf = get_tool_call_format(model_cfg.family)
    seed = build_seed_candidate(
        benchmark=cfg["benchmark"],
        tool_schemas=tool_schemas,
        direct_tool_call=cfg.get("tools", {}).get("direct_tool_call", True),
        tool_call_format=tcf,
    )
    print(f"Seed candidate built. system_prompt length: {len(seed['system_prompt'])} chars.")

    reflector_cfg = cfg.get("reflector", {})
    reflector_model = reflector_cfg.get("path_or_id", "Qwen/Qwen3-32B")
    reflector_host = reflector_cfg.get("host", "localhost")
    reflector_port = reflector_cfg.get("port", 8001)

    print(f"Starting GEPA optimisation: budget={gepa_cfg['rollout_budget']}, "
          f"minibatch={gepa_cfg['minibatch_size']}")

    result = optimize(
        seed_candidate=seed,
        trainset=train_examples,
        valset=val_examples,
        adapter=adapter,
        reflection_lm=f"openai/{reflector_model}",
        reflection_lm_kwargs={
            "base_url": f"http://{reflector_host}:{reflector_port}/v1",
            "api_key": "EMPTY",
        },
        max_metric_calls=gepa_cfg["rollout_budget"],
        reflection_minibatch_size=gepa_cfg["minibatch_size"],
        use_merge=gepa_cfg.get("merge_proposer", True),
        run_dir=str(run_dir),
        seed=cfg.get("seed", 1),
        raise_on_exception=False,
        display_progress_bar=True,
    )

    best = result.best_candidate
    best_path = run_dir / "best_candidate.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    seed_path = run_dir / "seed_candidate.json"
    with open(seed_path, "w") as f:
        json.dump(seed, f, indent=2)

    print(f"\nOptimisation complete.")
    print(f"  Best candidate: {best_path}")
    print(f"  Seed candidate: {seed_path}")
    print(f"  system_prompt length: {len(best['system_prompt'])} chars")
    print(f"  planning_suffix length: {len(best['planning_suffix'])} chars")


# ─────────────────────────────────────────────── MODE: evaluate ────────────

def run_evaluate(cfg: dict, config_path: Path) -> None:
    gepa_cfg = cfg["gepa"]
    run_dir = Path(gepa_cfg["run_dir"])
    best_path = run_dir / "best_candidate.json"

    if not best_path.exists():
        print(f"ERROR: {best_path} not found. Run --mode optimize first.")
        sys.exit(1)

    with open(best_path) as f:
        best = json.load(f)

    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = config_path.parent.parent.parent / splits_file

    with open(splits_file) as f:
        splits = json.load(f)

    test_examples = _load_examples(cfg, splits["test"])
    print(f"Evaluating on {len(test_examples)} held-out test examples...")

    tool_registry = _build_tool_registry(cfg)

    from agent_engine.models.base import ModelConfig, ModelFamily
    from agent_engine.models.vllm_provider import VLLMProvider

    model_cfg_raw = cfg["model"]
    model_cfg = ModelConfig(
        name=model_cfg_raw["name"],
        path_or_id=model_cfg_raw["path_or_id"],
        family=ModelFamily.QWEN3,
        role="orchestrator",
        use_thinking=True,
    )
    model_provider = VLLMProvider(model_cfg)

    orchestrator = AgenticOrchestrator(
        model_provider=model_provider,
        tool_registry=tool_registry,
        max_turns=cfg.get("max_turns", 15),
        use_thinking=True,
        planning_suffix=best["planning_suffix"],
    )

    states = orchestrator.run_batch(
        questions=[ex.question for ex in test_examples],
        question_ids=[ex.question_id for ex in test_examples],
        system_prompts=[best["system_prompt"]] * len(test_examples),
        attachments=[ex.get_attachments() or None for ex in test_examples],
    )

    from agent_engine.datasets.evaluators.metrics import evaluate_answer

    results = []
    n_correct = 0
    for state, ex in zip(states, test_examples):
        prediction = state.answer or ""
        choices = ex.metadata.get("choices")
        eval_result = evaluate_answer(prediction, ex.answer, choices=choices)
        correct = bool(eval_result["correct"])
        if correct:
            n_correct += 1
        results.append({
            "question_id": ex.question_id,
            "question": ex.question,
            "prediction": prediction,
            "answer": ex.answer,
            "correct": correct,
            "accuracy": eval_result["accuracy"],
            "turns": state.turn,
            "tool_counts": dict(state.tool_counts),
            "action_history": state.action_history,
            "metadata": state.metadata,
        })

    accuracy = n_correct / len(results) if results else 0.0
    print(f"\nTest accuracy: {n_correct}/{len(results)} = {accuracy:.1%}")

    out_path = run_dir / "gepa_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


# ─────────────────────────────────────────────── MODE: diff ────────────────

def run_diff(cfg: dict, config_path: Path) -> None:
    import difflib

    run_dir = Path(gepa_cfg["run_dir"]) if (gepa_cfg := cfg.get("gepa")) else Path(".")
    best_path = run_dir / "best_candidate.json"
    seed_path = run_dir / "seed_candidate.json"

    if not best_path.exists():
        print(f"No best_candidate.json found at {best_path}. Run --mode optimize first.")
        sys.exit(1)
    if not seed_path.exists():
        print(f"No seed_candidate.json found at {seed_path}. Run --mode optimize first.")
        sys.exit(1)

    with open(best_path) as f:
        best = json.load(f)
    with open(seed_path) as f:
        seed = json.load(f)

    for component in ("system_prompt", "planning_suffix"):
        print(f"\n{'='*60}")
        print(f"DIFF: {component}")
        print("=" * 60)
        seed_lines = seed[component].splitlines(keepends=True)
        best_lines = best[component].splitlines(keepends=True)
        diff = list(difflib.unified_diff(seed_lines, best_lines, fromfile="seed", tofile="best", n=3))
        if diff:
            print("".join(diff))
        else:
            print("(no change)")


# ─────────────────────────────────────────────── main ──────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA prompt optimisation")
    parser.add_argument("--mode", choices=["splits", "optimize", "evaluate", "diff"], required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    cfg = load_gepa_config(config_path)
    setup_logging()
    set_seed(cfg.get("seed", 1))

    if args.mode == "splits":
        run_splits(cfg, config_path)
    elif args.mode == "optimize":
        run_optimize(cfg, config_path)
    elif args.mode == "evaluate":
        run_evaluate(cfg, config_path)
    elif args.mode == "diff":
        run_diff(cfg, config_path)


if __name__ == "__main__":
    main()
