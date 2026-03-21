"""Generate baseline experiment configs mirroring 1_milestone_no_img_no_mindmap_AgentFlow.

Run from the repo root:
    python experiments/configs/baseline/generate_configs.py
"""
from pathlib import Path

BASE = Path(__file__).parent

# ── dataset metadata ──────────────────────────────────────────────────────────
DATASETS = {
    "gaia": {
        "display": "GAIA",
        "split": "all_validation",
        "split_desc": "all_validation",
        "tools": ["web_search", "code_generator", "text_inspector"],
    },
    # "gpqa": {
    #     "display": "GPQA",
    #     "split": "diamond",
    #     "split_desc": "diamond",
    #     "tools": ["web_search", "code_generator"],
    # },
    # "hle": {
    #     "display": "HLE",
    #     "split": "test_subset_200",
    #     "split_desc": "test_subset_200",
    #     "tools": ["web_search", "code_generator", "text_inspector"],
    # },
    # "musique": {
    #     "display": "MuSiQue",
    #     "split": "validation_subset_200",
    #     "split_desc": "validation_subset_200",
    #     "tools": ["web_search", "code_generator"],
    # },
    # "aime": {
    #     "display": "AIME",
    #     "split": "train",
    #     "split_desc": "train",
    #     "tools": ["web_search", "code_generator"],
    # },
}

# ── variant definitions ───────────────────────────────────────────────────────
# (filename_stem, model_key, direct_tool_call, enabled_tools_key, thinking_mode)
# enabled_tools_key: "none" → [], "tools" → dataset tools
VARIANTS = [
    # # 8B — no tools
    # ("qwen8B_no_tools_none",          "8B", True,  "none",  "NO"),
    # ("qwen8B_no_tools_orchestrator",  "8B", True,  "none",  "ORCHESTRATOR_ONLY"),
    # # 8B — direct tools
    # ("qwen8B_direct_tools_none",      "8B", True,  "tools", "NO"),
    # ("qwen8B_direct_tools_orchestrator", "8B", True, "tools", "ORCHESTRATOR_ONLY"),
    # # 8B — sub-agent tools
    # ("qwen8B_subagent_tools_none",        "8B", False, "tools", "NO"),
    # ("qwen8B_subagent_tools_orchestrator","8B", False, "tools", "ORCHESTRATOR_ONLY"),
    # ("qwen8B_subagent_tools_subagents",   "8B", False, "tools", "SUBAGENTS_ONLY"),
    # ("qwen8B_subagent_tools_all",         "8B", False, "tools", "ALL"),
    # 32B — no tools
    ("qwen32B_no_tools_none",         "32B", True,  "none",  "NO"),
    ("qwen32B_no_tools_orchestrator", "32B", True,  "none",  "ORCHESTRATOR_ONLY"),
    # 32B — direct tools
    ("qwen32B_direct_tools_none",         "32B", True, "tools", "NO"),
    ("qwen32B_direct_tools_orchestrator", "32B", True, "tools", "ORCHESTRATOR_ONLY"),
]

MODELS = {
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

THINKING_LABELS = {
    "NO": "no thinking",
    "ORCHESTRATOR_ONLY": "orchestrator thinking",
    "SUBAGENTS_ONLY": "sub-agents thinking only",
    "ALL": "orchestrator + sub-agents thinking",
}

TOOL_LABELS = {
    True:  "direct tools",
    False: "sub-agent tools",
}


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


def make_config(dataset: str, stem: str, model_key: str, direct: bool,
                tools_key: str, thinking: str) -> str:
    ds = DATASETS[dataset]
    m = MODELS[model_key]
    enabled = ds["tools"] if tools_key == "tools" else []

    tool_desc = "no tools" if not enabled else (TOOL_LABELS[direct])
    think_desc = THINKING_LABELS[thinking]
    comment_line = f"# {ds['display']} — {m['name']}, {tool_desc}, {think_desc}"

    exp_name = f"NEW_baseline_{dataset}_{stem}"
    description = (
        f"[Baseline; NO image_inspector, NO mindmap] "
        f"{ds['display']} {ds['split_desc']} with {m['name']}, "
        f"{tool_desc}, {think_desc}"
    )
    output_dir = f"./experiments/results/NEW_baseline/{dataset}/{stem}"

    return f"""{comment_line}

name: "{exp_name}"
description: "{description}"

slurm:
  partition: "gpu_h100"
  num_gpus: 2
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
baseline: true

output_dir: "{output_dir}"
use_wandb: true
wandb_project: "benchmarks"

cache_dir: "./cache"
"""


def main():
    removed = 0
    for old_cfg in BASE.glob("*/*.yaml"):
        old_cfg.unlink()
        removed += 1
    for old_cfg in BASE.glob("*/*.yml"):
        old_cfg.unlink()
        removed += 1

    created = 0
    for dataset in DATASETS:
        dataset_dir = BASE / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for stem, model_key, direct, tools_key, thinking in VARIANTS:
            content = make_config(dataset, stem, model_key, direct, tools_key, thinking)
            path = dataset_dir / f"{stem}.yaml"
            path.write_text(content)
            print(f"  wrote {path.relative_to(BASE.parent.parent)}")
            created += 1
    print(f"\nDone — removed {removed} old configs, created {created} configs.")


if __name__ == "__main__":
    main()
