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

from agent_engine.caching import CacheManager
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
    gepa_data_file = cfg.get("gepa_data_file")
    if gepa_data_file:
        from gepa_integration.data.loader import load_gepa_examples
        data_path = Path(gepa_data_file)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_path
        return load_gepa_examples(data_path, question_ids)

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


def _build_tool_registry(
    cfg: dict,
    model_provider=None,
    cache_manager=None,
    sub_agent_provider=None,
) -> ToolRegistry:
    import os
    tools = ToolRegistry()
    tools_cfg = cfg.get("tools", {})
    enabled = tools_cfg.get("enabled_tools", ["web_search", "code_generator"])
    provider = tools_cfg.get("web_tool_provider", "serper")
    top_k = tools_cfg.get("top_k_results", 5)
    max_doc_len = tools_cfg.get("max_doc_len", 3000)

    # Sub-agent thinking: only when thinking_mode includes sub-agents.
    thinking_mode = cfg.get("thinking_mode", "NO").upper()
    use_subagent_thinking = thinking_mode in ("SUBAGENTS_ONLY", "ALL")
    direct = tools_cfg.get("direct_tool_call", True)

    # In direct mode the tool handles everything itself (model_provider=None).
    # In sub-agent mode: use dedicated sub_agent_provider if given, else the orchestrator.
    sub_agent = None if direct else (sub_agent_provider or model_provider)

    if "web_search" in enabled:
        api_key_env = "SERPER_API_KEY" if provider == "serper" else "TAVILY_API_KEY"
        tools.register(WebSearchTool(
            api_key=os.environ.get(api_key_env, ""),
            provider=provider,
            top_k=top_k,
            max_doc_len=max_doc_len,
            model_provider=sub_agent,
            use_thinking=use_subagent_thinking,
            search_cache=cache_manager.search_cache if cache_manager else None,
            url_cache=cache_manager.url_cache if cache_manager else None,
            cache_manager=cache_manager,
        ))
    if "code_generator" in enabled:
        tools.register(CodeGeneratorTool(
            model_provider=sub_agent,
            use_thinking=use_subagent_thinking,
        ))
    if "text_inspector" in enabled:
        tools.register(TextInspectorTool(
            model_provider=sub_agent,
            use_thinking=use_subagent_thinking,
        ))

    return tools


def _build_sub_agent_provider(cfg: dict):
    """Create an OpenAI-compatible provider for the sub-agent vLLM server, if configured."""
    sa_cfg = cfg.get("sub_agent")
    if not sa_cfg:
        return None

    from agent_engine.models.base import ModelConfig, ModelFamily
    from agent_engine.models.api_provider import OpenAIProvider

    host = sa_cfg.get("host", "localhost")
    port = sa_cfg.get("port", 9998)
    family_str = sa_cfg.get("family", "qwen3")

    sa_model_cfg = ModelConfig(
        name=sa_cfg["name"],
        path_or_id=sa_cfg["path_or_id"],
        family=ModelFamily(family_str),
        role="sub_agent",
        backend="openai",
    )
    provider = OpenAIProvider(
        sa_model_cfg,
        api_key="EMPTY",
        base_url=f"http://{host}:{port}/v1",
    )
    print(f"Sub-agent provider: {sa_cfg['name']} at http://{host}:{port}/v1")
    return provider


# ─────────────────────────────────────────────── MODE: splits ──────────────

def _repo_root(config_path: Path) -> Path:
    """Return the repo root (4 levels up from experiments/configs/gepa/foo.yaml)."""
    return config_path.parent.parent.parent.parent


def run_splits(cfg: dict, config_path: Path) -> None:
    root = _repo_root(config_path)

    # New path: splits were pre-generated by src/gepa_integration/data/prepare.py.
    if cfg.get("gepa_data_file"):
        splits_file = Path(cfg["splits_file"])
        if not splits_file.is_absolute():
            splits_file = root / splits_file
        if splits_file.exists():
            with open(splits_file) as f:
                splits = json.load(f)
            print(f"Splits file already exists: {splits_file}")
            print(f"  train (D_feedback): {len(splits['train'])}")
            print(f"  val   (D_pareto):   {len(splits['val'])}")
            print(f"  test:               {len(splits['test'])}")
            print("To regenerate, re-run: python src/gepa_integration/data/prepare.py")
        else:
            print(f"ERROR: splits file not found: {splits_file}")
            print(
                "Run prepare.py first:\n"
                "  python src/gepa_integration/data/prepare.py "
                f"--preset {cfg['benchmark']} "
                f"--output-dir <dir> --splits-out {splits_file}"
            )
            sys.exit(1)
        return

    existing_results = Path(cfg["existing_results"])
    if not existing_results.is_absolute():
        existing_results = root / existing_results

    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = root / splits_file

    split_cfg = cfg.get("splits", {})
    train_n = split_cfg.get("train_n", 80)
    val_n = split_cfg.get("val_n", 45)
    test_n = split_cfg.get("test_n")  # None → all remaining
    seed = cfg.get("seed", 1)

    print(f"Building splits for {cfg['benchmark']}...")
    print(f"  Source: {existing_results}")
    print(f"  Train: {train_n}, Val: {val_n}, Test: {test_n or 'remainder'}, seed: {seed}")

    splits = build_splits(
        raw_results_path=existing_results,
        train_n=train_n,
        val_n=val_n,
        seed=seed,
        output_path=splits_file,
        test_n=test_n,
    )

    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Val:   {len(splits['val'])} examples")
    print(f"  Test:  {len(splits['test'])} examples")
    print(f"  Saved: {splits_file}")

class ProgressSaverCallback:
    def __init__(self, max_metric_calls: int, run_dir: Path):
        self._max = max_metric_calls
        self._run_dir = run_dir

        self._best_score = float("-inf")
        self._best_iteration = -1

        self._tree = []

        self._run_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, gepa_state) -> bool:
        iteration = getattr(gepa_state, "i", -1)

        scores = list(getattr(gepa_state, "program_full_scores_val_set", []) or [])
        candidates = list(getattr(gepa_state, "program_candidates", []) or [])

        total = getattr(gepa_state, "total_num_evals", 0)
        pct = 100.0 * total / self._max if self._max else 0.0

        best_val = max(scores) if scores else float("-inf")

        print(
            f"[gepa] iter={iteration} "
            f"budget={total}/{self._max} ({pct:.1f}%) "
            f"best_val={best_val:.4f}",
            flush=True,
        )

        # ─────────────────────────────────────────────
        # TREE LOG (FULL EVOLUTION HISTORY)
        # ─────────────────────────────────────────────
        node = {
            "iteration": iteration,
            "total_evals": total,
            "best_score": best_val,
            "scores": scores,

            # IMPORTANT: 2-PROMPT STRUCTURE AWARE
            "candidates": [
                {
                    "idx": i,
                    "system_prompt": c.get("system_prompt"),
                    "planning_suffix": c.get("planning_suffix"),
                }
                for i, c in enumerate(candidates)
            ],
        }

        self._tree.append(node)

        tree_path = self._run_dir / "gepa_tree.json"
        tmp_tree_path = self._run_dir / "gepa_tree.tmp.json"

        try:
            with open(tmp_tree_path, "w") as f:
                json.dump(self._tree, f, indent=2)

            tmp_tree_path.replace(tree_path)

        except Exception as e:
            print(f"[gepa] tree save failed: {e}", flush=True)

        # ─────────────────────────────────────────────
        # BEST CANDIDATE CHECKPOINT (WITH METADATA)
        # ─────────────────────────────────────────────
        if scores and candidates:
            best_idx = max(range(len(scores)), key=scores.__getitem__)
            best_candidate = candidates[best_idx]

            if best_val > self._best_score:
                self._best_score = best_val
                self._best_iteration = iteration

                # Top-level system_prompt / planning_suffix so that --mode evaluate
                # and --mode diff can read this checkpoint directly without
                # completing the full optimize() run.
                payload = {
                    "system_prompt": best_candidate.get("system_prompt"),
                    "planning_suffix": best_candidate.get("planning_suffix"),
                    # metadata (ignored by evaluate/diff, useful for debugging)
                    "best_val": best_val,
                    "best_iteration": iteration,
                    "best_idx": best_idx,
                    "total_evals": total,
                }

                best_path = self._run_dir / "best_candidate.json"
                tmp_best_path = self._run_dir / "best_candidate.tmp.json"

                try:
                    with open(tmp_best_path, "w") as f:
                        json.dump(payload, f, indent=2)

                    tmp_best_path.replace(best_path)

                    print(
                        f"[gepa] checkpoint → {best_path} "
                        f"(iter={iteration}, val={best_val:.4f})",
                        flush=True,
                    )

                except Exception as e:
                    print(f"[gepa] checkpoint save failed: {e}", flush=True)

        return False

# ─────────────────────────────────────────────── MODE: optimize ────────────

def run_optimize(cfg: dict, config_path: Path) -> None:
    from gepa.api import optimize

    gepa_cfg = cfg["gepa"]
    run_dir = Path(gepa_cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    splits_file = Path(cfg["splits_file"])
    if not splits_file.is_absolute():
        splits_file = _repo_root(config_path) / splits_file
    if not splits_file.exists():
        print(f"ERROR: splits file not found: {splits_file}")
        print("Run --mode splits first.")
        sys.exit(1)

    with open(splits_file) as f:
        splits = json.load(f)

    train_examples = _load_examples(cfg, splits["train"])
    val_examples = _load_examples(cfg, splits["val"])
    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples.")

    from agent_engine.models.base import ModelConfig, ModelFamily, get_tool_call_format
    from agent_engine.models.vllm_provider import VLLMProvider

    model_cfg_raw = cfg["model"]
    family_str = model_cfg_raw.get("family", "qwen3")
    model_cfg = ModelConfig(
        name=model_cfg_raw["name"],
        path_or_id=model_cfg_raw["path_or_id"],
        family=ModelFamily(family_str),
        role="orchestrator",
        gpu_memory_utilization=model_cfg_raw.get("gpu_memory_utilization"),
    )
    model_provider = VLLMProvider(model_cfg)

    sub_agent_provider = _build_sub_agent_provider(cfg)

    provider = cfg.get("tools", {}).get("web_tool_provider", "serper")
    cache_manager = CacheManager(
        cache_dir=cfg.get("cache_dir", "./cache"),
        web_tool_provider=provider,
        dataset_name=cfg["benchmark"],
    )
    print(f"Search cache: ./cache/{provider}/{cfg['benchmark']}/")
    tool_registry = _build_tool_registry(
        cfg, model_provider=model_provider, cache_manager=cache_manager,
        sub_agent_provider=sub_agent_provider,
    )

    thinking_mode = cfg.get("thinking_mode", "NO").upper()
    use_orchestrator_thinking = thinking_mode in ("ORCHESTRATOR_ONLY", "ALL")

    from gepa_integration.adapter import AgentGEPAAdapter
    max_search_limit = cfg.get("tools", {}).get("max_search_limit", 10)
    adapter = AgentGEPAAdapter(
        model_provider=model_provider,
        tool_registry=tool_registry,
        use_thinking=use_orchestrator_thinking,
        max_turns=cfg.get("max_turns", 15),
        tool_limits={"web_search": max_search_limit},
    )

    tool_schemas = tool_registry.get_all_schemas()
    tcf = get_tool_call_format(model_cfg.family)
    seed = build_seed_candidate(
        benchmark=cfg["benchmark"],
        tool_schemas=tool_schemas,
        direct_tool_call=cfg.get("tools", {}).get("direct_tool_call", True),
        tool_call_format=tcf,
        max_search_limit=cfg.get("tools", {}).get("max_search_limit", 10),
    )
    print(f"Seed candidate built. system_prompt length: {len(seed['system_prompt'])} chars.")

    # Save seed immediately — before optimize() starts — so --mode diff works
    # even if the optimization is aborted and only the callback checkpoint exists.
    seed_path = run_dir / "seed_candidate.json"
    with open(seed_path, "w") as f:
        json.dump(seed, f, indent=2)

    reflector_cfg = cfg.get("reflector", {})
    reflector_model = reflector_cfg.get("path_or_id", "Qwen/Qwen3-32B")
    reflector_host = reflector_cfg.get("host", "localhost")
    reflector_port = reflector_cfg.get("port", 8001)

    # Qwen3-32B native context is 40960. Keep some headroom for the response so
    # the reflector is never rejected for context overflow during a long trace.
    REFLECTOR_CTX = int(reflector_cfg.get("max_model_len", 40960))
    REFLECTOR_RESPONSE_BUDGET = int(reflector_cfg.get("max_tokens", 4096))
    REFLECTOR_INPUT_BUDGET = REFLECTOR_CTX - REFLECTOR_RESPONSE_BUDGET - 256  # safety margin
    # Qwen3-32B generating REFLECTOR_RESPONSE_BUDGET tokens on 2×H100 takes ~2-4 min.
    # 15 min is a generous upper bound; prevents a stalled reflector from freezing the job.
    REFLECTOR_TIMEOUT_S = int(reflector_cfg.get("timeout_s", 900))

    from transformers import AutoTokenizer
    try:
        _reflector_tok = AutoTokenizer.from_pretrained(reflector_model, trust_remote_code=True)
    except Exception as e:
        print(f"WARN: could not load reflector tokenizer ({e}); falling back to char heuristic.")
        _reflector_tok = None

    from gepa_integration.reflection import trim_prompt

    import litellm

    def _reflection_lm(prompt: str) -> str:
        trimmed = trim_prompt(
            prompt,
            budget_tokens=REFLECTOR_INPUT_BUDGET,
            tokenizer=_reflector_tok,
            on_trim=print,
        )
        completion = litellm.completion(
            model=f"openai/{reflector_model}",
            messages=[{"role": "user", "content": trimmed}],
            base_url=f"http://{reflector_host}:{reflector_port}/v1",
            api_key="EMPTY",
            max_tokens=REFLECTOR_RESPONSE_BUDGET,
            timeout=REFLECTOR_TIMEOUT_S,
        )
        return completion.choices[0].message.content

    import os
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = wandb_cfg.get("enabled", False)
    wandb_api_key = os.environ.get("WANDB_API_KEY") if use_wandb else None
    wandb_init_kwargs = {
        "project": wandb_cfg.get("project", "gepa"),
        "name": wandb_cfg.get("name", f"gepa_{cfg['benchmark']}"),
        "tags": wandb_cfg.get("tags", [cfg["benchmark"], "gepa"]),
        "config": {
            "benchmark": cfg["benchmark"],
            "model": cfg["model"]["name"],
            "rollout_budget": gepa_cfg["rollout_budget"],
            "minibatch_size": gepa_cfg["minibatch_size"],
            "seed": cfg.get("seed", 1),
        },
    } if use_wandb else None

    print(f"Starting GEPA optimisation: budget={gepa_cfg['rollout_budget']}, "
          f"minibatch={gepa_cfg['minibatch_size']}, wandb={use_wandb}")

    progress_saver = ProgressSaverCallback(
        max_metric_calls=gepa_cfg["rollout_budget"],
        run_dir=run_dir,
    )

    result = optimize(
        seed_candidate=seed,
        trainset=train_examples,
        valset=val_examples,
        adapter=adapter,
        reflection_lm=_reflection_lm,
        max_metric_calls=gepa_cfg["rollout_budget"],
        reflection_minibatch_size=gepa_cfg["minibatch_size"],
        use_merge=gepa_cfg.get("merge_proposer", True),
        track_best_outputs=gepa_cfg.get("track_best_outputs", False),
        run_dir=str(run_dir),
        seed=cfg.get("seed", 1),
        raise_on_exception=gepa_cfg.get("raise_on_exception", True),
        display_progress_bar=True,
        use_wandb=use_wandb,
        wandb_api_key=wandb_api_key,
        wandb_init_kwargs=wandb_init_kwargs,
        stop_callbacks=[progress_saver],
    )

    best = result.best_candidate
    best_path = run_dir / "best_candidate.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

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
        splits_file = _repo_root(config_path) / splits_file

    with open(splits_file) as f:
        splits = json.load(f)

    test_examples = _load_examples(cfg, splits["test"])
    print(f"Evaluating on {len(test_examples)} held-out test examples...")

    from agent_engine.models.base import ModelConfig, ModelFamily
    from agent_engine.models.vllm_provider import VLLMProvider

    model_cfg_raw = cfg["model"]
    family_str = model_cfg_raw.get("family", "qwen3")
    model_cfg = ModelConfig(
        name=model_cfg_raw["name"],
        path_or_id=model_cfg_raw["path_or_id"],
        family=ModelFamily(family_str),
        role="orchestrator",
        gpu_memory_utilization=model_cfg_raw.get("gpu_memory_utilization"),
    )
    model_provider = VLLMProvider(model_cfg)

    sub_agent_provider = _build_sub_agent_provider(cfg)

    provider = cfg.get("tools", {}).get("web_tool_provider", "serper")
    cache_manager = CacheManager(
        cache_dir=cfg.get("cache_dir", "./cache"),
        web_tool_provider=provider,
        dataset_name=cfg["benchmark"],
    )
    tool_registry = _build_tool_registry(
        cfg, model_provider=model_provider, cache_manager=cache_manager,
        sub_agent_provider=sub_agent_provider,
    )

    thinking_mode = cfg.get("thinking_mode", "NO").upper()
    use_orchestrator_thinking = thinking_mode in ("ORCHESTRATOR_ONLY", "ALL")

    orchestrator = AgenticOrchestrator(
        model_provider=model_provider,
        tool_registry=tool_registry,
        max_turns=cfg.get("max_turns", 15),
        tool_limits={"web_search": cfg.get("tools", {}).get("max_search_limit", 10)},
        use_thinking=use_orchestrator_thinking,
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
        aliases = ex.metadata.get("answer_aliases") or []
        eval_result = evaluate_answer(prediction, ex.answer, choices=choices, answer_aliases=aliases)
        correct = bool(eval_result["correct"])
        if correct:
            n_correct += 1
        results.append({
            "question_id": ex.question_id,
            "question": ex.question,
            "prediction": prediction,
            "ground_truth": ex.answer,
            "correct": correct,
            "evaluation": eval_result,
            "query_analysis": state.query_analysis or "",
            "action_history": state.action_history,
            "output_messages": state.output_messages,
            "turns": state.turn,
            "tool_counts": dict(state.tool_counts),
            "token_usage": state.metadata.get("token_usage", {}),
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

    gepa_cfg = cfg.get("gepa")
    if not gepa_cfg:
        print("ERROR: config has no 'gepa' section.")
        sys.exit(1)
    run_dir = Path(gepa_cfg["run_dir"])
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
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override gepa.run_dir from config. Job scripts use this to inject a "
             "timestamped + job-id subdirectory so optimize/evaluate/diff all read "
             "and write the same run.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    cfg = load_gepa_config(config_path)
    if args.run_dir is not None:
        cfg.setdefault("gepa", {})["run_dir"] = args.run_dir
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
