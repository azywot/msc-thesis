"""Generate experiment configs for all suites.

Run from the repo root:
    python scripts/generate_configs.py                  # all suites
    python scripts/generate_configs.py --suite baseline
    python scripts/generate_configs.py --suite agentflow
    python scripts/generate_configs.py --suite orch_capacity
    python scripts/generate_configs.py --suite subagent_orchestrator_ablation
        (leave-one-out no_<tool> ablations; 8B only; single-tool datasets skip empty ablation)
    python scripts/generate_configs.py --suite structured_memory_ablation
        (8B: subagent tools + orch thinking + baseline: true — no query analysis / structured memory)
    python scripts/generate_configs.py --suite olmo-think-baseline
    python scripts/generate_configs.py --suite olmo-think-agentflow
    python scripts/generate_configs.py --suite olmo-instruct-baseline
    python scripts/generate_configs.py --suite olmo-instruct-agentflow
    python scripts/generate_configs.py --suite lora_inference
        (LoRA-adapted Qwen3-8B orchestrator + Qwen3-1.7B subagents; no thinking / orch thinking)
        Before running, set LORA_ADAPTER_PLACEHOLDER to the actual adapter path.
    python scripts/generate_configs.py --suite sft_inference
        (SFT-adapted Qwen3-8B orchestrator + Qwen3-1.7B subagents; NO thinking only)
        Before running, set SFT_ADAPTER_PLACEHOLDER to the actual adapter path.

Config output layout:
    experiments/configs/qwen3/<suite>/<dataset>/<variant>.yaml
    experiments/configs/olmo3/think/baseline/<dataset>/<variant>.yaml
    experiments/configs/olmo3/think/agentflow/<dataset>/<variant>.yaml
    experiments/configs/olmo3/instruct/baseline/<dataset>/<variant>.yaml
    experiments/configs/olmo3/instruct/agentflow/<dataset>/<variant>.yaml
"""
import argparse
from pathlib import Path
from typing import List, Optional

CONFIGS_ROOT = Path(__file__).parent.parent / "experiments" / "configs"

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
    "math500": {
        "display": "MATH500",
        "split": "test_subset_200",
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
        "tp_comment": "64 heads; TP must divide 64.",
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
    "deepseek-7b": {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "family": "deepseek",
        "path_or_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "tp": None,
        "gpus": 1,
    },
    "deepseek-32b": {
        "name": "DeepSeek-R1-Distill-Qwen-32B",
        "family": "deepseek",
        "path_or_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "tp": 2,
        "gpus": 2,
    },
}

# ── LoRA inference ─────────────────────────────────────────────────────────────
# Replace <run-tag> with the value printed by 005_train.job ("Run tag: ...") once
# training is complete.  VERL writes the best-tracked adapter to:
#   /scratch-shared/azywot/fine_tuning/lora_adapters/<experiment>/<run-tag>/best_checkpoint/actor/lora_adapter
# The adapter the committed lora_inference configs evaluate: the *v2* GRPO run at
# global_step_40.  It was v1/global_step_20 here while the generated configs were
# hand-edited to v2, so regenerating silently reverted all ten of them -- the
# generator and its own output disagreed.  Keep this in step with LORA_RUN_SUFFIX
# below; the two describe the same run.
#
# NOTE: this path no longer resolves.  /scratch-shared is not durable and the
# qwen3-8b-grpo-search-math{,-v2} adapters were purged from it, which is why the
# LoRA evaluations cannot be re-run (see jobs/fine_tuning/007_train_sft_folded.job,
# which now archives adapters to data/adapters/ for exactly this reason).  It is
# recorded rather than corrected because it is what the reported v2 numbers were
# produced with; point it at a durable copy when one exists.
LORA_ADAPTER_PLACEHOLDER = (
    "/scratch-shared/azywot/fine_tuning/lora_adapters/"
    "qwen3-8b-grpo-search-math-v2/29-05-2026_11-36-23210365/global_step_40/actor/lora_adapter"
)

# Appended to the experiment *name* and *output_dir* of the lora_inference suite,
# not to its filenames -- which is exactly how the hand-edit had it.  It keeps the
# v2 results in their own directory alongside v1's, and
# src/agent_engine/analysis/fine_tuning/base_vs_lora.py reads both paths to compare
# the two runs, so the suffix is load-bearing: dropping it would orphan the v2
# results and break that analysis.
LORA_RUN_SUFFIX = "_v2"

# SFT adapters are archived off scratch by 007_train_sft_folded.job, because scratch is not
# durable: the GRPO adapter directories were purged from /scratch-shared while the configs above
# still pointed at them. Replace <run-tag> with the value the training job prints.
SFT_ADAPTER_PLACEHOLDER = (
    "/gpfs/home3/xchen1/azywot/msc-thesis/data/adapters/"
    "qwen3-8b-sft-folded-v1/06-08-2026_21-47-25300018/best_adapter"
)

# (stem, orch_key, sub_key, thinking_mode)
VARIANTS_LORA_INFERENCE = [
    ("qwen8B_sub1_7b_none",         "8B", "1.7B", "NO"),
    ("qwen8B_sub1_7b_orchestrator", "8B", "1.7B", "ORCHESTRATOR_ONLY"),
]

# NO thinking only. The SFT data has thinking stripped at build time and the folded prompt
# renders an empty <think>\n\n</think> block, so an ORCHESTRATOR_ONLY eval would sample the
# adapter off the distribution it was trained on — the exact mismatch the folded format fixes.
VARIANTS_SFT_INFERENCE = [
    ("qwen8B_sub1_7b_none", "8B", "1.7B", "NO"),
]

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
# AgentFlow BigCodeBench: sub-agent code_generator only (no direct / no-tools / 32B sweeps).
VARIANTS_QWEN8B_SUBAGENT_TOOLS_ONLY = [
    v for v in VARIANTS_ALL if v[0].startswith("qwen8B_subagent_tools_")
]
# Documented in README as a smaller variant grid example (not used by any built-in suite).
VARIANTS_32B = [v for v in VARIANTS_ALL if v[1] == "32B"]

# (model_key, filename prefix) for subagent_orchestrator_ablation leave-one-out configs.
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
# Orchestrator capacity: Qwen3 8B/32B orchestrators × 1.7B/8B/32B sub-agents × 3 thinking
# modes, skipping orch 8B + sub 8B (same path / no distinct sub sweep) → 15 experiments (GAIA).
VARIANTS_ORCH_CAPACITY = [
    (
        f"orch{ok.lower().replace('.','_')}_sub{sk.lower().replace('.','_')}_{_THINK_SHORT[t]}",
        ok, sk, t,
    )
    for ok in ["8B", "32B"]
    for sk in ["1.7B", "8B", "32B"]
    for t in ["NO", "ORCHESTRATOR_ONLY", "ALL"]
    if not (ok == "8B" and sk == "8B")
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

# ── olmo variants mirroring full qwen3 grid (no thinking_mode distinctions) ───
# These omit thinking_mode from the YAML entirely; stems have no thinking suffix.
# Baseline: direct_tool_call=True only (mirrors VARIANTS_ALL_BASELINE).
VARIANTS_OLMO_THINK_BASELINE = [
    ("olmo7b_no_tools",      "olmo-7b",   True,  "none",  "NO"),
    ("olmo7b_direct_tools",  "olmo-7b",   True,  "tools", "NO"),
    ("olmo32b_no_tools",     "olmo-32b",  True,  "none",  "NO"),
    ("olmo32b_direct_tools", "olmo-32b",  True,  "tools", "NO"),
]

# AgentFlow: subagent_tools for 7B only.
VARIANTS_OLMO_THINK_AGENTFLOW = [
    ("olmo7b_subagent_tools", "olmo-7b", False, "tools", "NO"),
]

VARIANTS_OLMO_INSTRUCT_BASELINE = [
    ("olmo_instruct7b_no_tools",      "olmo-instruct-7b",   True,  "none",  "NO"),
    ("olmo_instruct7b_direct_tools",  "olmo-instruct-7b",   True,  "tools", "NO"),
    ("olmo_instruct32b_no_tools",     "olmo-instruct-32b",  True,  "none",  "NO"),
    ("olmo_instruct32b_direct_tools", "olmo-instruct-32b",  True,  "tools", "NO"),
]

# AgentFlow: subagent_tools for 7B only.
VARIANTS_OLMO_INSTRUCT_AGENTFLOW = [
    ("olmo_instruct7b_subagent_tools", "olmo-instruct-7b", False, "tools", "NO"),
]

# BigCodeBench in agentflow: only 7B sub-agent tools (mirrors VARIANTS_QWEN8B_SUBAGENT_TOOLS_ONLY).
VARIANTS_OLMO_THINK_7B_SUBAGENT_ONLY = [
    v for v in VARIANTS_OLMO_THINK_AGENTFLOW if v[0] == "olmo7b_subagent_tools"
]
VARIANTS_OLMO_INSTRUCT_7B_SUBAGENT_ONLY = [
    v for v in VARIANTS_OLMO_INSTRUCT_AGENTFLOW if v[0] == "olmo_instruct7b_subagent_tools"
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
        "config_subdir":   "qwen3/baseline",
        "baseline":        True,
        "force_num_gpus":  True,
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
        "config_subdir":   "qwen3/agentflow",
        "baseline":        False,
        "force_num_gpus":  True,
        "variants":        VARIANTS_QWEN8B_SUBAGENT_TOOLS_ONLY,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "orch_capacity": {
        "description_tag": "[OrchestratorCapacity; NO image_inspector, NO mindmap]",
        "name_prefix":     "OC",
        "output_dir_root": "./experiments/results/orchestrator_capacity",
        "config_subdir":   "qwen3/orchestrator_capacity",
        "baseline":        False,
        "variant_type":    "orch_capacity",
        "variants":        VARIANTS_ORCH_CAPACITY,
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
        "config_subdir":   "qwen3/subagent_orchestrator_ablation",
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
        "config_subdir":   "qwen3/structured_memory_ablation",
        "baseline":        True,
        "variants":        VARIANTS_STRUCTURED_MEMORY_ABLATION,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "olmo-think-baseline": {
        "description_tag": "[OLMo-Think Baseline; NO image_inspector, NO mindmap]",
        "name_prefix":     "OLMo_Think_baseline",
        "output_dir_root": "./experiments/results/olmo/think/baseline",
        "config_subdir":   "olmo3/think/baseline",
        "baseline":        True,
        "force_num_gpus":  True,
        "no_thinking_mode": True,
        "variants":        VARIANTS_OLMO_THINK_BASELINE,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "olmo-think-agentflow": {
        "description_tag": "[OLMo-Think AgentFlow; NO image_inspector, NO mindmap]",
        "name_prefix":     "OLMo_Think_AF",
        "output_dir_root": "./experiments/results/olmo/think/agentflow",
        "config_subdir":   "olmo3/think/agentflow",
        "baseline":        False,
        "force_num_gpus":  True,
        "no_thinking_mode": True,
        "variants":        VARIANTS_OLMO_THINK_AGENTFLOW,
        "variants_by_dataset": {
            "bigcodebench": VARIANTS_OLMO_THINK_7B_SUBAGENT_ONLY,
        },
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "olmo-instruct-baseline": {
        "description_tag": "[OLMo-Instruct Baseline; NO image_inspector, NO mindmap]",
        "name_prefix":     "OLMo_Instruct_baseline",
        "output_dir_root": "./experiments/results/olmo/instruct/baseline",
        "config_subdir":   "olmo3/instruct/baseline",
        "baseline":        True,
        "force_num_gpus":  True,
        "no_thinking_mode": True,
        "variants":        VARIANTS_OLMO_INSTRUCT_BASELINE,
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "olmo-instruct-agentflow": {
        "description_tag": "[OLMo-Instruct AgentFlow; NO image_inspector, NO mindmap]",
        "name_prefix":     "OLMo_Instruct_AF",
        "output_dir_root": "./experiments/results/olmo/instruct/agentflow",
        "config_subdir":   "olmo3/instruct/agentflow",
        "baseline":        False,
        "force_num_gpus":  True,
        "no_thinking_mode": True,
        "variants":        VARIANTS_OLMO_INSTRUCT_AGENTFLOW,
        "variants_by_dataset": {
            "bigcodebench": VARIANTS_OLMO_INSTRUCT_7B_SUBAGENT_ONLY,
        },
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
    },
    "lora_inference": {
        "description_tag": "[LoRA inference]",
        "name_prefix":     "LORA_eval",
        "output_dir_root": "./experiments/results/lora_inference",
        "config_subdir":   "qwen3/lora_inference",
        "baseline":        False,
        "variant_type":    "lora_inference",
        "variants":        VARIANTS_LORA_INFERENCE,
        "datasets":        ["gaia", "hle", "gpqa", "aime", "musique"],
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
        # Replace <run-tag> with the value from the training job log before running inference.
        "lora_adapter_path": LORA_ADAPTER_PLACEHOLDER,
        # Names and output dirs carry the adapter version; filenames do not.
        "run_suffix":        LORA_RUN_SUFFIX,
    },
    "sft_inference": {
        "description_tag": "[SFT inference]",
        "name_prefix":     "SFT_eval",
        "output_dir_root": "./experiments/results/sft_inference",
        "config_subdir":   "qwen3/sft_inference",
        "baseline":        False,
        "variant_type":    "lora_inference",  # same YAML shape: adapter on the orchestrator
        "variants":        VARIANTS_SFT_INFERENCE,
        "datasets":        ["gaia", "hle", "gpqa", "aime", "musique"],
        "num_gpus":        2,
        "wandb_project":   "benchmarks",
        "split_overrides": {},
        "adapter_label":   "SFT",
        "adapter_desc":    "SFT-adapted",
        "train_job":       "007_train_sft_folded.job",
        # Replace <run-tag> with the value from the training job log before running inference.
        "lora_adapter_path": SFT_ADAPTER_PLACEHOLDER,
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
        comment = m.get("tp_comment", "TP must divide num attention heads.")
        lines.append(f"    tensor_parallel_size: {m['tp']}  # {comment}")
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
        comment = orch.get("tp_comment", "TP must divide num attention heads.")
        lines.append(f"    tensor_parallel_size: {orch['tp']}  # {comment}")
    for role in tool_roles:
        lines += [
            f"  {role}:",
            f'    name: "{sub["name"]}"',
            f'    family: "{sub["family"]}"',
            f'    path_or_id: "{sub["path_or_id"]}"',
            f'    role: "{role}"',
        ]
        if sub["tp"]:
            comment = sub.get("tp_comment", "TP must divide num attention heads.")
            lines.append(f"    tensor_parallel_size: {sub['tp']}  # {comment}")
    return "\n".join(lines)


def _model_block_lora_orchestrator(
    orch_key: str, sub_key: str, tool_roles: list[str], lora_adapter_path: str,
    adapter_label: str = "LoRA",
) -> str:
    """Models block with an adapter-annotated orchestrator and explicit per-role subagent entries.

    `adapter_label` becomes the model-name suffix (Qwen3-8B-LoRA / Qwen3-8B-SFT). It is what
    lands in the `model_name` column of the W&B export, so the two adaptation runs stay
    distinguishable in orchestrator_ft_results.csv.
    """
    orch = MODELS[orch_key]
    sub  = MODELS[sub_key]
    lines = [
        "models:",
        "  orchestrator:",
        f'    name: "{orch["name"]}-{adapter_label}"',
        f'    family: "{orch["family"]}"',
        f'    path_or_id: "{orch["path_or_id"]}"',
        '    role: "orchestrator"',
        f'    lora_adapter_path: "{lora_adapter_path}"',
    ]
    if orch["tp"]:
        comment = orch.get("tp_comment", "TP must divide num attention heads.")
        lines.append(f"    tensor_parallel_size: {orch['tp']}  # {comment}")
    for role in tool_roles:
        lines += [
            f"  {role}:",
            f'    name: "{sub["name"]}"',
            f'    family: "{sub["family"]}"',
            f'    path_or_id: "{sub["path_or_id"]}"',
            f'    role: "{role}"',
        ]
        if sub["tp"]:
            comment = sub.get("tp_comment", "TP must divide num attention heads.")
            lines.append(f"    tensor_parallel_size: {sub['tp']}  # {comment}")
    return "\n".join(lines)


def make_config_lora_inference(
    suite: dict, dataset: str, stem: str,
    orch_key: str, sub_key: str, thinking: str,
) -> str:
    ds = {**DATASETS[dataset], **suite["split_overrides"].get(dataset, {})}
    orch = MODELS[orch_key]
    sub  = MODELS[sub_key]
    think_desc    = THINKING_LABELS[thinking]
    lora_path     = suite["lora_adapter_path"]
    adapter_label = suite.get("adapter_label", "LoRA")
    adapter_desc  = suite.get("adapter_desc", "LoRA-adapted")
    train_job     = suite.get("train_job", "005_train.job")
    tools         = ds["tools"]
    return_code   = dataset == "bigcodebench" and bool(tools)
    tools_block   = _tools_block(direct=False, enabled=tools, return_code=return_code)
    models_block  = _model_block_lora_orchestrator(
        orch_key, sub_key, tools, lora_path, adapter_label
    )
    # Applied to the name and the output dir but deliberately not to the filename,
    # so an adapter version gets its own results directory and its own W&B rows
    # while the config keeps a stable path that job scripts can reference.
    run_suffix    = suite.get("run_suffix", "")
    exp_name      = f"{suite['name_prefix']}_{dataset}_{stem}{run_suffix}"
    output_dir    = f"{suite['output_dir_root']}/{dataset}/{stem}{run_suffix}"
    wandb_project = suite["wandb_project"]

    comment_line = (
        f"# {ds['display']} — {adapter_desc} {orch['name']} orchestrator "
        f"+ {sub['name']} subagents, {think_desc}"
    )
    description = (
        f"{suite['description_tag']} "
        f"{ds['display']} {ds['split']} with {adapter_desc} {orch['name']} orchestrator "
        f"({think_desc}) + {sub['name']} subagents (no thinking)."
    )

    return f"""{comment_line}
# Update lora_adapter_path once training is complete (replace <run-tag> printed by {train_job}).

name: "{exp_name}"
description: "{description}"

slurm:
  partition: "gpu_h100"
  num_gpus: {suite['num_gpus']}
  ntasks: 1
  cpus_per_task: 8
  time: "24:00:00"
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

    no_thinking_mode = suite.get("no_thinking_mode", False)
    if no_thinking_mode:
        comment_line = f"# {ds['display']} — {m['name']}, {tool_desc}"
    else:
        think_desc = THINKING_LABELS[thinking]
        comment_line = f"# {ds['display']} — {m['name']}, {tool_desc}, {think_desc}"

    exp_name = f"{suite['name_prefix']}_{dataset}_{stem}"
    if no_thinking_mode:
        description = (
            f"{suite['description_tag']} "
            f"{ds['display']} {ds['split']} with {m['name']}, {tool_desc}"
        )
    else:
        description = (
            f"{suite['description_tag']} "
            f"{ds['display']} {ds['split']} with {m['name']}, "
            f"{tool_desc}, {think_desc}"
        )
    output_dir = f"{suite['output_dir_root']}/{dataset}/{stem}"

    baseline_line = "baseline: true\n" if suite["baseline"] else ""
    thinking_line = "" if no_thinking_mode else f'thinking_mode: "{thinking}"\n'
    num_gpus = suite["num_gpus"] if suite.get("force_num_gpus", False) else m.get("gpus", suite["num_gpus"])
    wandb_project = suite["wandb_project"]
    # BigCodeBench evaluation is done externally via test harness; the tool must
    # return generated code rather than executing it.
    return_code = dataset == "bigcodebench" and bool(enabled)

    return f"""{comment_line}

name: "{exp_name}"
description: "{description}"

slurm:
  partition: "gpu_h100"
  num_gpus: {num_gpus}
  ntasks: 1
  cpus_per_task: 8
  time: "24:00:00"
  conda_env: "agent_engine"

{_model_block(model_key)}

{_tools_block(direct, enabled, return_code=return_code)}

dataset:
  name: "{dataset}"
  split: "{ds['split']}"
  data_dir: "./data"
  subset_num: -1

seed: 0
{thinking_line}{baseline_line}output_dir: "{output_dir}"
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
  time: "24:00:00"
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


def generate_suite(suite_name: str, configs_root: Path = CONFIGS_ROOT) -> None:
    suite = SUITES[suite_name]
    suite_dir = configs_root / suite["config_subdir"]

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
                print(f"  wrote {path.relative_to(configs_root.parent)}")
                created += 1
        elif variant_type == "lora_inference":
            for stem, orch_key, sub_key, thinking in suite["variants"]:
                content = make_config_lora_inference(
                    suite, dataset, stem, orch_key, sub_key, thinking
                )
                path = dataset_dir / f"{stem}.yaml"
                path.write_text(content)
                print(f"  wrote {path.relative_to(configs_root.parent)}")
                created += 1
        elif variant_type == "subagent_orch_ablation":
            ds = {**DATASETS[dataset], **suite["split_overrides"].get(dataset, {})}
            full_tools = list(ds["tools"])
            thinking = "ORCHESTRATOR_ONLY"
            for model_key, stem_prefix in _SUBAGENT_ORCH_MODEL_STEMS:
                for absent in full_tools:
                    ablated = [t for t in full_tools if t != absent]
                    if not ablated:
                        # Single-tool datasets (e.g. BigCodeBench): skip "no tools" leave-one-out.
                        continue
                    stem = f"{stem_prefix}_subagent_orch_no_{absent}"
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
                    print(f"  wrote {path.relative_to(configs_root.parent)}")
                    created += 1
        else:
            variants = suite.get("variants_by_dataset", {}).get(dataset, suite["variants"])
            for stem, model_key, direct, tools_key, thinking in variants:
                content = make_config(suite, dataset, stem, model_key, direct, tools_key, thinking)
                path = dataset_dir / f"{stem}.yaml"
                path.write_text(content)
                print(f"  wrote {path.relative_to(configs_root.parent)}")
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CONFIGS_ROOT,
        help="Directory to write configs into (default: experiments/configs).",
    )
    args = parser.parse_args()

    suites_to_run = [args.suite] if args.suite else list(SUITES.keys())
    for suite_name in suites_to_run:
        print(f"\n=== Generating suite: {suite_name} ===")
        generate_suite(suite_name, args.output_root)

    print("\nAll done.")


if __name__ == "__main__":
    main()
