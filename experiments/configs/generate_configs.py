"""Generate experiment configs for all suites.

Run from the repo root:
    python experiments/configs/generate_configs.py                  # all suites
    python experiments/configs/generate_configs.py --suite baseline
    python experiments/configs/generate_configs.py --suite agentflow
    python experiments/configs/generate_configs.py --suite orch_capacity
    python experiments/configs/generate_configs.py --suite subagent_orchestrator_ablation
        (subagent_orchestrator_ablation: leave-one-out no_<tool> ablations per dataset, 8B only)
    python experiments/configs/generate_configs.py --suite structured_memory_ablation
        (8B: subagent tools + orch thinking + baseline: true — no query analysis / structured memory)
"""
import argparse
from pathlib import Path
from typing import List, Optional

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
    "bigcodebench": {
        "display": "BigCodeBench",
        "split": "v0.1.4_subset_200",
        "tools": ["code_generator"],
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
        "gpus": 1,
    },
    "32B": {
        "name": "Qwen3-32B",
        "family": "qwen3",
        "path_or_id": "Qwen/Qwen3-32B",
        "tp": 2,
    },
    "olmo-7b": {
        "name": "OLMo-3-7B-Think",
        "family": "olmo-think",
        "path_or_id": "allenai/Olmo-3-7B-Think",
        "tp": None,
        "gpus": 1,
    },
    "olmo-32b": {
        "name": "OLMo-3.1-32B-Think",
        "family": "olmo-think",
        "path_or_id": "allenai/Olmo-3.1-32B-Think",
        "tp": 2,
        "gpus": 2,
    },
    "olmo-instruct-7b": {
        "name": "OLMo-3-7B-Instruct",
        "family": "olmo-instruct",
        "path_or_id": "allenai/Olmo-3-7B-Instruct",
        "tp": None,
        "gpus": 1,
    },
    "olmo-instruct-32b": {
        "name": "OLMo-3.1-32B-Instruct",
        "family": "olmo-instruct",
        "path_or_id": "allenai/Olmo-3.1-32B-Instruct",
        "tp": 2,
        "gpus": 2,
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

# Used only as documentation; subagent_orchestrator_ablation suite uses variant_type ablation instead.
_SUBAGENT_ORCH_MODEL_STEMS = (("8B", "qwen8B"),)

# AF-style sub-agent tools + orchestrator thinking, but baseline: true (plain message history;
# skips planning / query analysis and structured AgentFlow memory).
VARIANTS_STRUCTURED_MEMORY_ABLATION = [
    ("qwen8B_subagent_orch_baseline_chat", "8B", False, "tools", "ORCHESTRATOR_ONLY"),
]

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

# ── olmo variants ──────────────────────────────────────────────────────────────
# OLMo-Think models always produce <think> output regardless of config, so all
# Think variants use thinking_mode="ALL" for consistency.
# (filename_stem, model_key, direct_tool_call, tools_key, thinking_mode)
VARIANTS_OLMO_THINK = [
    # # 7B — no tools
    # ("olmo7b_no_tools",        "olmo-7b",  True,  "none",  "ALL"),
    # # 7B — direct tools
    # ("olmo7b_direct_tools",    "olmo-7b",  True,  "tools", "ALL"),
    # 7B — sub-agent tools (self as sub-agent)
    ("olmo7b_subagent_tools",  "olmo-7b",  False, "tools", "ALL"),
    # # 32B — no tools
    # ("olmo32b_no_tools",       "olmo-32b", True,  "none",  "ALL"),
    # # 32B — direct tools
    # ("olmo32b_direct_tools",   "olmo-32b", True,  "tools", "ALL"),
    # 32B — sub-agent tools (self as sub-agent)
    ("olmo32b_subagent_tools", "olmo-32b", False, "tools", "ALL"),
]

VARIANTS_OLMO_INSTRUCT = [
    # # 7B — no tools
    # ("olmo_instruct7b_no_tools",        "olmo-instruct-7b",  True,  "none",  "NO"),
    # # 7B — direct tools
    # ("olmo_instruct7b_direct_tools",    "olmo-instruct-7b",  True,  "tools", "NO"),
    # 7B — sub-agent tools
    ("olmo_instruct7b_subagent_tools",  "olmo-instruct-7b",  False, "tools", "NO"),
    # # 32B — no tools
    # ("olmo_instruct32b_no_tools",        "olmo-instruct-32b", True,  "none",  "NO"),
    # # 32B — direct tools
    # ("olmo_instruct32b_direct_tools",    "olmo-instruct-32b", True,  "tools", "NO"),
    # 32B — sub-agent tools
    ("olmo_instruct32b_subagent_tools", "olmo-instruct-32b", False, "tools", "NO"),
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
        # num_gpus is computed per-combo in make_config_orch_capacity
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "subagent_orchestrator_ablation": {
        "description_tag": (
            "[Subagent tools + orchestrator thinking; tool ablations (leave-one-out); "
            "NO image_inspector, NO mindmap]"
        ),
        "name_prefix":     "AF_subagent_orch",
        "output_dir_root": "./experiments/results/subagent_orchestrator_ablation",
        "config_subdir":   "subagent_orchestrator_ablation",
        "baseline":        False,
        "variant_type":    "subagent_orch_ablation",
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "structured_memory_ablation": {
        "description_tag": (
            "[Structured-memory ablation: subagent tools + orch thinking, baseline chat "
            "(no query analysis / structured memory); NO image_inspector, NO mindmap]"
        ),
        "name_prefix":     "AF_struct_mem_ablation",
        "output_dir_root": "./experiments/results/structured_memory_ablation",
        "config_subdir":   "structured_memory_ablation",
        "baseline":        True,
        "variants":        VARIANTS_STRUCTURED_MEMORY_ABLATION,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "olmo-think": {
        "description_tag": "[OLMo-Think; NO image_inspector, NO mindmap]",
        "name_prefix":     "OLMo_Think",
        "output_dir_root": "./experiments/results/olmo/think",
        "config_subdir":   "olmo/think",
        "baseline":        False,
        "variants":        VARIANTS_OLMO_THINK,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "olmo-instruct": {
        "description_tag": "[OLMo-Instruct; NO image_inspector, NO mindmap]",
        "name_prefix":     "OLMo_Instruct",
        "output_dir_root": "./experiments/results/olmo/instruct",
        "config_subdir":   "olmo/instruct",
        "baseline":        False,
        "variants":        VARIANTS_OLMO_INSTRUCT,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────

def _tools_block(direct: bool, enabled: list[str], return_code: bool = False) -> str:
    if not enabled:
        return "tools:\n  enabled_tools: []\n  direct_tool_call: true"
    lines = ["tools:", "  enabled_tools:"]
    for t in enabled:
        lines.append(f"    - {t}")
    lines.append(f"  direct_tool_call: {'true' if direct else 'false'}")
    if return_code:
        lines.append("  return_code: true")
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


def make_config(
    suite: dict,
    dataset: str,
    stem: str,
    model_key: str,
    direct: bool,
    tools_key: str,
    thinking: str,
    *,
    enabled_tools_override: Optional[List[str]] = None,
) -> str:
    ds = {**DATASETS[dataset], **suite["split_overrides"].get(dataset, {})}
    m = MODELS[model_key]
    full_tools = ds["tools"]
    if enabled_tools_override is not None:
        enabled = list(enabled_tools_override)
    else:
        enabled = full_tools if tools_key == "tools" else []

    if not enabled:
        tool_desc = "no tools"
    else:
        base = TOOL_MODE_LABELS[direct]
        if enabled_tools_override is not None:
            omitted = [t for t in full_tools if t not in enabled]
            if not omitted:
                tool_desc = f"{base} (full tool set)"
            elif len(omitted) == 1:
                tool_desc = f"{base} (without {omitted[0]})"
            else:
                tool_desc = f"{base} (enabled: {', '.join(enabled)})"
        else:
            tool_desc = base
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
    num_gpus = m.get("gpus", suite["num_gpus"])
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


def _num_gpus_for_combo(orch_key: str, sub_key: str) -> int:
    """Compute the SLURM GPU request for an orchestrator/sub-agent model combo.

    Large models (32B; tp=2) need 2 GPUs each.  When the orchestrator and
    sub-agent share the same model path only one model instance is loaded, so
    we count that model once.
    """
    orch_m = MODELS[orch_key]
    sub_m  = MODELS[sub_key]
    orch_gpus = orch_m["tp"] or 1
    sub_gpus  = sub_m["tp"] or 1
    if orch_m["path_or_id"] == sub_m["path_or_id"]:
        # Single shared instance — only count once.
        return max(2, orch_gpus)
    return max(2, orch_gpus + sub_gpus)


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
    output_dir    = f"{suite['output_dir_root']}/{dataset}/{stem}"
    wandb_project = suite["wandb_project"]
    num_gpus      = _num_gpus_for_combo(orch_key, sub_key)
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


def make_config_bigcodebench(
    suite: dict,
    stem: str,
    model_key: str,
    direct: bool,
    tools_key: str,
    thinking: str,
    baseline: bool,
    return_code: bool,
) -> str:
    """Generate a BigCodeBench experiment config.

    tools_key:
        "subagent"  — code_generator, direct=False, return_code=True
        "direct"    — code_generator, direct=True,  return_code=True
        "none"      — no tools (model outputs code directly)
    """
    ds = DATASETS["bigcodebench"]
    m = MODELS[model_key]

    if tools_key == "none":
        enabled = []
        tool_desc = "no tools"
    else:
        enabled = ["code_generator"]
        tool_desc = "direct code_generator" if direct else "sub-agent code_generator"

    think_desc = THINKING_LABELS[thinking]
    baseline_desc = "baseline" if baseline else "AgentFlow"
    comment_line = f"# BigCodeBench — {m['name']}, {tool_desc}, {think_desc}, {baseline_desc}"

    exp_name = f"{suite['name_prefix']}_bigcodebench_{stem}"
    description = (
        f"{suite['description_tag']} "
        f"BigCodeBench {ds['split']} with {m['name']}, "
        f"{tool_desc}, {think_desc}, {baseline_desc}"
    )
    output_dir = f"{suite['output_dir_root']}/bigcodebench/{stem}"
    baseline_line = "baseline: true\n" if baseline else ""
    num_gpus = m.get("gpus", suite.get("num_gpus", 2))
    wandb_project = suite.get("wandb_project", "benchmarks")

    tools_blk = _tools_block(direct, enabled, return_code=return_code and bool(enabled))

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

{tools_blk}

dataset:
  name: "bigcodebench"
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


# BigCodeBench variants:
# (stem, model_key, direct_tool_call, tools_key, thinking_mode, baseline, return_code)
_BCB_VARIANTS = []
for _mk in ["8B", "32B"]:
    for _think in ["NO", "ORCHESTRATOR_ONLY"]:
        _short_think = {"NO": "none", "ORCHESTRATOR_ONLY": "orchestrator"}[_think]
        _mk_slug = _mk.lower().replace(".", "_")
        # mode 1: sub-agent, return_code=True, AgentFlow + baseline
        for _bl in [False, True]:
            _bl_slug = "baseline" if _bl else "af"
            _BCB_VARIANTS.append((
                f"qwen{_mk_slug}_subagent_{_short_think}_{_bl_slug}",
                _mk, False, "subagent", _think, _bl, True,
            ))
        # mode 2: direct, return_code=True, AgentFlow + baseline
        for _bl in [False, True]:
            _bl_slug = "baseline" if _bl else "af"
            _BCB_VARIANTS.append((
                f"qwen{_mk_slug}_direct_{_short_think}_{_bl_slug}",
                _mk, True, "direct", _think, _bl, True,
            ))
        # mode 3: no tools, AgentFlow + baseline
        for _bl in [False, True]:
            _bl_slug = "baseline" if _bl else "af"
            _BCB_VARIANTS.append((
                f"qwen{_mk_slug}_no_tools_{_short_think}_{_bl_slug}",
                _mk, True, "none", _think, _bl, False,
            ))

SUITES["bigcodebench"] = {
    "description_tag": "[BigCodeBench; code generation benchmark]",
    "name_prefix": "BCB",
    "output_dir_root": "./experiments/results/bigcodebench",
    "config_subdir": "bigcodebench",
    "num_gpus": 2,
    "wandb_project": "benchmarks",
    "split_overrides": {},
}


def generate_suite(suite_name: str) -> None:
    suite = SUITES[suite_name]
    suite_dir = CONFIGS_ROOT / suite["config_subdir"]

    # Remove stale configs
    removed = sum(1 for p in suite_dir.glob("*/*.yaml") if p.unlink() or True)
    removed += sum(1 for p in suite_dir.glob("*/*.yml") if p.unlink() or True)

    datasets_to_run = suite.get("datasets", list(DATASETS.keys()))
    variant_type    = suite.get("variant_type", "standard")

    created = 0

    # BigCodeBench has its own generation logic (configs in suite_dir directly)
    if suite_name == "bigcodebench":
        suite_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale top-level yaml files
        removed = sum(1 for p in suite_dir.glob("*.yaml") if p.unlink() or True)
        removed += sum(1 for p in suite_dir.glob("*.yml") if p.unlink() or True)
        for stem, model_key, direct, tools_key, thinking, baseline, return_code in _BCB_VARIANTS:
            content = make_config_bigcodebench(
                suite, stem, model_key, direct, tools_key, thinking, baseline, return_code
            )
            path = suite_dir / f"{stem}.yaml"
            path.write_text(content)
            print(f"  wrote {path.relative_to(CONFIGS_ROOT.parent)}")
            created += 1
        print(f"\n[{suite_name}] removed {removed} old, created {created} configs.")
        return

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
        elif variant_type == "subagent_orch_ablation":
            ds = {**DATASETS[dataset], **suite["split_overrides"].get(dataset, {})}
            full_tools = list(ds["tools"])
            thinking = "ORCHESTRATOR_ONLY"
            for model_key, stem_prefix in _SUBAGENT_ORCH_MODEL_STEMS:
                for absent in full_tools:
                    stem = f"{stem_prefix}_subagent_orch_no_{absent}"
                    ablated = [t for t in full_tools if t != absent]
                    content = make_config(
                        suite,
                        dataset,
                        stem,
                        model_key,
                        False,
                        "tools",
                        thinking,
                        enabled_tools_override=ablated,
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
        help="Suite to generate (default: all suites).",
    )
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else list(SUITES.keys())
    for suite_name in suites_to_run:
        print(f"\n=== Generating suite: {suite_name} ===")
        generate_suite(suite_name)

    print("\nAll done.")


if __name__ == "__main__":
    main()
