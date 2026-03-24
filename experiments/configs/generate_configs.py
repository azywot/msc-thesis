"""Generate experiment configs for all suites.

Run from the repo root:
    python experiments/configs/generate_configs.py                  # all suites
    python experiments/configs/generate_configs.py --suite baseline
    python experiments/configs/generate_configs.py --suite agentflow
    python experiments/configs/generate_configs.py --suite orch_capacity
"""
import argparse
from pathlib import Path

CONFIGS_ROOT = Path(__file__).parent

# ── dataset metadata (defaults; suites may override per-dataset fields) ────────
DATASETS = {
    "gaia": {
        "display": "GAIA",
        "split": "all_validation",
        "tools": ["web_search", "code_generator", "text_inspector"],
    },
    "hle": {
        "display": "HLE",
        "split": "test_subset_200",
        "tools": ["web_search", "code_generator", "text_inspector"],
    },
    "gpqa": {
        "display": "GPQA",
        "split": "diamond",
        "tools": ["web_search", "code_generator"],
    },
    "aime": {
        "display": "AIME",
        "split": "train",
        "tools": ["web_search", "code_generator"],
    },
    "musique": {
        "display": "MuSiQue",
        "split": "validation_subset_200",
        "tools": ["web_search", "code_generator"],
    },
}

# ── model definitions ──────────────────────────────────────────────────────────
MODELS = {
    "1.7B": {
        "name": "Qwen3-1.7B",
        "family": "qwen3",
        "path_or_id": "Qwen/Qwen3-1.7B",
        "tp": None,
    },
    "8B": {
        "name": "Qwen3-8B",
        "family": "qwen3",
        "path_or_id": "Qwen/Qwen3-8B",
        "tp": None,
    },
    "32B": {
        "name": "Qwen3-32B",
        "family": "qwen3",
        "path_or_id": "Qwen/Qwen3-32B",
        "tp": 2,
    },
}

# ── all possible variants ──────────────────────────────────────────────────────
# (filename_stem, model_key, direct_tool_call, enabled_tools_key, thinking_mode)
# enabled_tools_key: "none" → [], "tools" → dataset-specific tools list
VARIANTS_ALL = [
    # 8B — no tools
    ("qwen8B_no_tools_none",              "8B",  True,  "none",  "NO"),
    ("qwen8B_no_tools_orchestrator",      "8B",  True,  "none",  "ORCHESTRATOR_ONLY"),
    # 8B — direct tools
    ("qwen8B_direct_tools_none",          "8B",  True,  "tools", "NO"),
    ("qwen8B_direct_tools_orchestrator",  "8B",  True,  "tools", "ORCHESTRATOR_ONLY"),
    # 8B — sub-agent tools
    ("qwen8B_subagent_tools_none",        "8B",  False, "tools", "NO"),
    ("qwen8B_subagent_tools_orchestrator","8B",  False, "tools", "ORCHESTRATOR_ONLY"),
    ("qwen8B_subagent_tools_subagents",   "8B",  False, "tools", "SUBAGENTS_ONLY"),
    ("qwen8B_subagent_tools_all",         "8B",  False, "tools", "ALL"),
    # 32B — no tools
    ("qwen32B_no_tools_none",             "32B", True,  "none",  "NO"),
    ("qwen32B_no_tools_orchestrator",     "32B", True,  "none",  "ORCHESTRATOR_ONLY"),
    # 32B — direct tools
    ("qwen32B_direct_tools_none",         "32B", True,  "tools", "NO"),
    ("qwen32B_direct_tools_orchestrator", "32B", True,  "tools", "ORCHESTRATOR_ONLY"),
]

VARIANTS_ALL_BASELINE = [v for v in VARIANTS_ALL if v[2] == True]
VARIANTS_ALL_SUBAGENTS = [v for v in VARIANTS_ALL if v[2] == False]
VARIANTS_32B = [v for v in VARIANTS_ALL if v[1] == "32B"]
VARIANTS_8B = [v for v in VARIANTS_ALL if v[1] == "8B"]

# ── orchestrator-capacity variants ─────────────────────────────────────────────
# (filename_stem, orch_key, sub_key, thinking_mode)
# Always sub-agent tool mode (direct_tool_call: false) with full tools.
_THINK_SHORT = {"NO": "none", "ORCHESTRATOR_ONLY": "orchestrator", "ALL": "all"}
VARIANTS_ORCH_CAPACITY = [
    (
        f"orch{ok.lower().replace('.','_')}_sub{sk.lower().replace('.','_')}_{_THINK_SHORT[t]}",
        ok, sk, t,
    )
    for ok in ["8B", "32B"]
    for sk in ["1.7B", "8B", "32B"]
    for t  in ["NO", "ORCHESTRATOR_ONLY", "ALL"]
]

# ── human-readable labels ──────────────────────────────────────────────────────
THINKING_LABELS = {
    "NO":                "thinking disabled",
    "ORCHESTRATOR_ONLY": "orchestrator thinking",
    "SUBAGENTS_ONLY":    "sub-agents thinking only",
    "ALL":               "orchestrator + sub-agents thinking",
}

TOOL_MODE_LABELS = {
    True:  "direct tools",
    False: "sub-agent tools",
}

# ── suite definitions ──────────────────────────────────────────────────────────
SUITES = {
    "baseline": {
        "description_tag": "[Baseline; NO image_inspector, NO mindmap]",
        "name_prefix":     "NEW_baseline",
        "output_dir_root": "./experiments/results/NEW_baseline",
        "config_subdir":   "baseline",
        "baseline":        True,
        "variants":        VARIANTS_ALL_BASELINE,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        # Override splits that differ from the defaults above
        "split_overrides": {
            # "musique": {"split": "validation_subset_200"},
        },
    },
    "agentflow": {
        "description_tag": "[AgentFlow; NO image_inspector, NO mindmap]",
        "name_prefix":     "AF_no_img_no_mm",
        "output_dir_root": "./experiments/results/1_milestone_no_img_no_mindmap_AgentFlow",
        "config_subdir":   "1_milestone_no_img_no_mindmap_AgentFlow",
        "baseline":        False,
        "variants":        VARIANTS_ALL,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "orch_capacity": {
        "description_tag": "[OrchestratorCapacity; NO image_inspector, NO mindmap]",
        "name_prefix":     "OC",
        "output_dir_root": "./experiments/results/orchestrator_capacity",
        "config_subdir":   "orchestrator_capacity",
        "baseline":        False,
        "variant_type":    "orch_capacity",
        "variants":        VARIANTS_ORCH_CAPACITY,
        "datasets":        ["gaia"],
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _tools_block(direct: bool, enabled: list[str]) -> str:
    if not enabled:
        return "tools:\n  enabled_tools: []\n  direct_tool_call: true"
    lines = ["tools:", "  enabled_tools:"]
    for t in enabled:
        lines.append(f"    - {t}")
    lines.append(f"  direct_tool_call: {'true' if direct else 'false'}")
    return "\n".join(lines)


def _model_block(key: str) -> str:
    m = MODELS[key]
    lines = [
        "models:",
        "  orchestrator:",
        f'    name: "{m["name"]}"',
        f'    family: "{m["family"]}"',
        f'    path_or_id: "{m["path_or_id"]}"',
        '    role: "orchestrator"',
    ]
    if m["tp"]:
        lines.append(f"    tensor_parallel_size: {m['tp']}  # 64 heads; TP must divide 64.")
    return "\n".join(lines)


def _model_block_with_subagent(orch_key: str, sub_key: str, tool_roles: list[str]) -> str:
    """Build a models block with explicit per-tool-role sub-agent entries."""
    orch = MODELS[orch_key]
    sub  = MODELS[sub_key]
    lines = [
        "models:",
        "  orchestrator:",
        f'    name: "{orch["name"]}"',
        f'    family: "{orch["family"]}"',
        f'    path_or_id: "{orch["path_or_id"]}"',
        '    role: "orchestrator"',
    ]
    if orch["tp"]:
        lines.append(f"    tensor_parallel_size: {orch['tp']}  # TP must divide num attention heads.")
    for role in tool_roles:
        lines += [
            f"  {role}:",
            f'    name: "{sub["name"]}"',
            f'    family: "{sub["family"]}"',
            f'    path_or_id: "{sub["path_or_id"]}"',
            f'    role: "{role}"',
        ]
        if sub["tp"]:
            lines.append(f"    tensor_parallel_size: {sub['tp']}  # TP must divide num attention heads.")
    return "\n".join(lines)


def make_config(suite: dict, dataset: str, stem: str, model_key: str,
                direct: bool, tools_key: str, thinking: str) -> str:
    ds = {**DATASETS[dataset], **suite["split_overrides"].get(dataset, {})}
    m = MODELS[model_key]
    enabled = ds["tools"] if tools_key == "tools" else []

    tool_desc = "no tools" if not enabled else TOOL_MODE_LABELS[direct]
    think_desc = THINKING_LABELS[thinking]
    comment_line = f"# {ds['display']} — {m['name']}, {tool_desc}, {think_desc}"

    exp_name = f"{suite['name_prefix']}_{dataset}_{stem}"
    description = (
        f"{suite['description_tag']} "
        f"{ds['display']} {ds['split']} with {m['name']}, "
        f"{tool_desc}, {think_desc}"
    )
    output_dir = f"{suite['output_dir_root']}/{dataset}/{stem}"

    baseline_line = "baseline: true\n" if suite["baseline"] else ""
    num_gpus = suite["num_gpus"]
    wandb_project = suite["wandb_project"]

    return f"""{comment_line}

name: "{exp_name}"
description: "{description}"

slurm:
  partition: "gpu_h100"
  num_gpus: {num_gpus}
  ntasks: 1
  cpus_per_task: 8
  time: "16:00:00"
  conda_env: "agent_engine"

{_model_block(model_key)}

{_tools_block(direct, enabled)}

dataset:
  name: "{dataset}"
  split: "{ds['split']}"
  data_dir: "./data"
  subset_num: -1

seed: 0
thinking_mode: "{thinking}"
{baseline_line}output_dir: "{output_dir}"
use_wandb: true
wandb_project: "{wandb_project}"

cache_dir: "./cache"
"""


def make_config_orch_capacity(suite: dict, dataset: str, stem: str,
                              orch_key: str, sub_key: str, thinking: str) -> str:
    ds = {**DATASETS[dataset], **suite["split_overrides"].get(dataset, {})}
    orch = MODELS[orch_key]
    sub  = MODELS[sub_key]
    think_desc = THINKING_LABELS[thinking]

    comment_line = (
        f"# {ds['display']} — orch {orch['name']} / sub-agent {sub['name']}, "
        f"sub-agent tools, {think_desc}"
    )
    exp_name    = f"{suite['name_prefix']}_{dataset}_{stem}"
    description = (
        f"{suite['description_tag']} "
        f"{ds['display']} {ds['split']} with orch {orch['name']} / sub-agent {sub['name']}, "
        f"sub-agent tools, {think_desc}"
    )
    output_dir  = f"{suite['output_dir_root']}/{dataset}/{stem}"
    num_gpus    = suite["num_gpus"]
    wandb_project = suite["wandb_project"]

    tools_block  = _tools_block(direct=False, enabled=ds["tools"])
    models_block = _model_block_with_subagent(orch_key, sub_key, ds["tools"])

    return f"""{comment_line}

name: "{exp_name}"
description: "{description}"

slurm:
  partition: "gpu_h100"
  num_gpus: {num_gpus}
  ntasks: 1
  cpus_per_task: 8
  time: "16:00:00"
  conda_env: "agent_engine"

{models_block}

{tools_block}

dataset:
  name: "{dataset}"
  split: "{ds['split']}"
  data_dir: "./data"
  subset_num: -1

seed: 0
thinking_mode: "{thinking}"
output_dir: "{output_dir}"
use_wandb: true
wandb_project: "{wandb_project}"

cache_dir: "./cache"
"""


def generate_suite(suite_name: str) -> None:
    suite = SUITES[suite_name]
    suite_dir = CONFIGS_ROOT / suite["config_subdir"]

    # Remove stale configs
    removed = sum(1 for p in suite_dir.glob("*/*.yaml") if p.unlink() or True)
    removed += sum(1 for p in suite_dir.glob("*/*.yml") if p.unlink() or True)

    datasets_to_run = suite.get("datasets", list(DATASETS.keys()))
    variant_type    = suite.get("variant_type", "standard")

    created = 0
    for dataset in datasets_to_run:
        dataset_dir = suite_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if variant_type == "orch_capacity":
            for stem, orch_key, sub_key, thinking in suite["variants"]:
                content = make_config_orch_capacity(
                    suite, dataset, stem, orch_key, sub_key, thinking
                )
                path = dataset_dir / f"{stem}.yaml"
                path.write_text(content)
                print(f"  wrote {path.relative_to(CONFIGS_ROOT.parent)}")
                created += 1
        else:
            for stem, model_key, direct, tools_key, thinking in suite["variants"]:
                content = make_config(suite, dataset, stem, model_key, direct, tools_key, thinking)
                path = dataset_dir / f"{stem}.yaml"
                path.write_text(content)
                print(f"  wrote {path.relative_to(CONFIGS_ROOT.parent)}")
                created += 1

    print(f"\n[{suite_name}] removed {removed} old, created {created} configs.")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configs.")
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()),
        default=None,
        help="Suite to generate (default: all suites). "
             f"Available: {', '.join(SUITES.keys())}",
    )
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else list(SUITES.keys())
    for suite_name in suites_to_run:
        print(f"\n=== Generating suite: {suite_name} ===")
        generate_suite(suite_name)

    print("\nAll done.")


if __name__ == "__main__":
    main()
