"""GEPA pre-flight smoke test — no GPU required.

Checks:
  1. All GEPA + gepa_integration imports resolve.
  2. build_seed_candidate produces the right structure for both benchmarks
     and the seed planning_suffix matches the orchestrator constant exactly.
  3. Source raw_results.json files exist and are valid JSON.
  4. Splits files exist; train/val/test have no overlaps and cover the right counts.
  5. All split question IDs exist in the actual dataset.
  6. evaluate_answer spot-checks match expected outcomes.

Run:
    python scripts/smoke_gepa.py
    python scripts/smoke_gepa.py --config-dir experiments/configs/gepa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv()

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)


# ─────────────────────────── 1. imports ──────────────────────────────────────

def test_imports() -> None:
    print("\n[1] Imports")
    try:
        from gepa.api import optimize  # noqa: F401
        check("gepa.api.optimize", True)
    except Exception as e:
        check("gepa.api.optimize", False, str(e))

    try:
        from gepa.core.adapter import EvaluationBatch  # noqa: F401
        check("gepa.core.adapter.EvaluationBatch", True)
    except Exception as e:
        check("gepa.core.adapter.EvaluationBatch", False, str(e))

    try:
        from gepa_integration.adapter import AgentGEPAAdapter  # noqa: F401
        check("gepa_integration.adapter.AgentGEPAAdapter", True)
    except Exception as e:
        check("gepa_integration.adapter.AgentGEPAAdapter", False, str(e))

    try:
        from gepa_integration.seed import build_seed_candidate, build_splits  # noqa: F401
        check("gepa_integration.seed", True)
    except Exception as e:
        check("gepa_integration.seed", False, str(e))


# ─────────────────────────── 2. seed candidate ───────────────────────────────

def test_seed_candidate() -> None:
    print("\n[2] Seed candidate")
    from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
    from agent_engine.models.base import ToolCallFormat
    from gepa_integration.seed import build_seed_candidate

    for benchmark in ("gaia", "gpqa"):
        try:
            cand = build_seed_candidate(
                benchmark=benchmark,
                tool_schemas=[],
                direct_tool_call=True,
                tool_call_format=ToolCallFormat.JSON,
                max_search_limit=10,
            )
            check(f"{benchmark}: returns system_prompt + planning_suffix",
                  set(cand.keys()) == {"system_prompt", "planning_suffix"})
            check(f"{benchmark}: system_prompt non-empty",
                  len(cand["system_prompt"]) > 100)
            check(f"{benchmark}: planning_suffix == _DEFAULT_PLANNING_SUFFIX_TOOLS",
                  cand["planning_suffix"] == _DEFAULT_PLANNING_SUFFIX_TOOLS,
                  f"got: {cand['planning_suffix'][:60]!r}")
        except Exception as e:
            check(f"{benchmark}: build_seed_candidate", False, str(e))

    # With a real tool schema the prompt should embed tool JSON
    tool_schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    cand_tools = build_seed_candidate("gaia", tool_schemas=[tool_schema])
    check("gaia with tools: <tools> block present", "<tools>" in cand_tools["system_prompt"])
    check("gaia with tools: web_search in prompt", "web_search" in cand_tools["system_prompt"])


# ─────────────────────────── 3. source raw_results ───────────────────────────

def test_source_results(config_dir: Path) -> None:
    print("\n[3] Source raw_results.json files")
    import yaml

    for yaml_file in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_file.read_text())
        benchmark = cfg.get("benchmark", yaml_file.stem)
        raw_path = Path(cfg["existing_results"])
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        if not raw_path.exists():
            check(f"{benchmark}: {raw_path.name} exists", False, str(raw_path))
            continue
        try:
            records = json.loads(raw_path.read_text())
            check(f"{benchmark}: {raw_path.name} valid JSON ({len(records)} records)", True)
        except Exception as e:
            check(f"{benchmark}: {raw_path.name} valid JSON", False, str(e))


# ─────────────────────────── 4. splits integrity ─────────────────────────────

def test_splits(config_dir: Path) -> None:
    print("\n[4] Splits integrity")
    import yaml

    for yaml_file in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_file.read_text())
        benchmark = cfg.get("benchmark", yaml_file.stem)

        splits_file = Path(cfg["splits_file"])
        if not splits_file.is_absolute():
            splits_file = ROOT / splits_file
        if not splits_file.exists():
            check(f"{benchmark}: splits file exists", False, str(splits_file))
            continue

        try:
            splits = json.loads(splits_file.read_text())
        except Exception as e:
            check(f"{benchmark}: splits file valid JSON", False, str(e))
            continue

        train = set(splits.get("train", []))
        val   = set(splits.get("val", []))
        test  = set(splits.get("test", []))

        check(f"{benchmark}: has train/val/test keys",
              {"train", "val", "test"} == set(splits.keys()))
        check(f"{benchmark}: train non-empty", len(train) > 0)
        check(f"{benchmark}: val non-empty",   len(val)   > 0)
        check(f"{benchmark}: test non-empty",  len(test)  > 0)
        check(f"{benchmark}: no train/val overlap",  not train & val)
        check(f"{benchmark}: no train/test overlap", not train & test)
        check(f"{benchmark}: no val/test overlap",   not val   & test)

        # Check target sizes from config
        split_cfg = cfg.get("splits", {})
        expected_train = split_cfg.get("train_n")
        expected_val   = split_cfg.get("val_n")
        if expected_train:
            check(f"{benchmark}: train size == {expected_train}",
                  len(train) == expected_train,
                  f"got {len(train)}")
        if expected_val:
            check(f"{benchmark}: val size == {expected_val}",
                  len(val) == expected_val,
                  f"got {len(val)}")

        print(f"         train={len(train)}, val={len(val)}, test={len(test)}")


# ─────────────────────────── 5. dataset loading ──────────────────────────────

def test_dataset_loading(config_dir: Path) -> None:
    print("\n[5] Dataset example loading")
    import yaml
    from agent_engine.config.schema import DatasetConfig
    from agent_engine.datasets import DatasetRegistry

    for yaml_file in sorted(config_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yaml_file.read_text())
        benchmark = cfg.get("benchmark", yaml_file.stem)

        splits_file = Path(cfg["splits_file"])
        if not splits_file.is_absolute():
            splits_file = ROOT / splits_file
        if not splits_file.exists():
            check(f"{benchmark}: skipped (no splits file)", False, str(splits_file))
            continue

        splits = json.loads(splits_file.read_text())
        all_ids = set(splits["train"] + splits["val"] + splits["test"])

        try:
            ds_cfg = DatasetConfig(
                name=cfg["benchmark"],
                split=cfg.get("dataset_split", "validation"),
                data_dir=Path(cfg.get("data_dir", "./data")),
                subset_num=-1,
            )
            dataset = DatasetRegistry.get(ds_cfg)
            examples = list(dataset)
            dataset_ids = {ex.question_id for ex in examples}

            missing = all_ids - dataset_ids
            check(f"{benchmark}: {len(examples)} examples loaded", len(examples) > 0)
            check(f"{benchmark}: all split IDs exist in dataset",
                  len(missing) == 0,
                  f"{len(missing)} missing IDs: {sorted(missing)[:5]}")
        except Exception as e:
            check(f"{benchmark}: dataset load", False, str(e))


# ─────────────────────────── 6. evaluate_answer spot-checks ──────────────────

def test_evaluator() -> None:
    print("\n[6] evaluate_answer spot-checks")
    from agent_engine.datasets.evaluators.metrics import evaluate_answer

    cases = [
        # (prediction, ground_truth, choices, expected_correct, label)
        ("Paris",  "Paris",  None, True,  "exact match"),
        ("paris",  "Paris",  None, True,  "case-insensitive"),
        ("Berlin", "Paris",  None, False, "wrong answer"),
        ("42",     "42",     None, True,  "numeric exact"),
        ("42.0",   "42",     None, True,  "numeric float vs int"),
        ("\\text{No}", "No", None, True,  "LaTeX wrapper stripped"),
        ("A",      "A",      ["opt1","opt2","opt3","opt4"], True,  "MC correct letter"),
        ("B",      "A",      ["opt1","opt2","opt3","opt4"], False, "MC wrong letter"),
    ]
    for pred, gt, choices, expected, label in cases:
        try:
            result = evaluate_answer(pred, gt, choices=choices)
            correct = result.get("correct", False)
            check(label, correct == expected,
                  f"expected correct={expected}, got {result}")
        except Exception as e:
            check(label, False, str(e))


# ─────────────────────────── main ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA pre-flight smoke test")
    parser.add_argument("--config-dir", type=Path,
                        default=ROOT / "experiments/configs/gepa",
                        help="Directory containing GEPA YAML configs")
    args = parser.parse_args()

    config_dir = args.config_dir
    if not config_dir.exists():
        print(f"Config dir not found: {config_dir}")
        sys.exit(1)

    print("=" * 55)
    print("GEPA smoke test")
    print("=" * 55)

    test_imports()
    test_seed_candidate()
    test_source_results(config_dir)
    test_splits(config_dir)
    test_dataset_loading(config_dir)
    test_evaluator()

    print("\n" + "=" * 55)
    if _failures:
        print(f"FAILED ({len(_failures)} checks):")
        for f in _failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("All checks passed. Safe to submit jobs/gepa/003_smoke_gepa_gpu.job.")
    print("=" * 55)


if __name__ == "__main__":
    main()
