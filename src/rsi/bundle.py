"""The prompt bundle: the plain-data object the reflector edits.

A bundle holds every editable prompt string for one domain. The seed bundle renders
byte-identically to today's prompts.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from agent_engine.core.orchestrator import _DEFAULT_PLANNING_SUFFIX_TOOLS
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools.code_generator import CODE_GENERATOR_TASK_TEMPLATE
from agent_engine.tools.web_search import WEB_SEARCH_ANALYSIS_TEMPLATE

DOMAINS = {"search": "gaia", "math": "deepmath"}
UNITS = ("orchestrator", "web_search", "code_generator")

_ORCH_PART_KEYS = ("guidance_opening", "guidance_closing", "example", "tool_use_note")


@dataclass
class PromptBundle:
    domain: str
    orchestrator: dict
    subagents: dict

    def to_dict(self):
        return copy.deepcopy(
            {
                "domain": self.domain,
                "orchestrator": self.orchestrator,
                "subagents": self.subagents,
            }
        )

    @classmethod
    def from_dict(cls, d):
        d = copy.deepcopy(d)
        return cls(domain=d["domain"], orchestrator=d["orchestrator"], subagents=d["subagents"])


def _descriptions_from_schemas(tool_schemas):
    out = {}
    for schema in tool_schemas:
        fn = schema["function"]
        props = fn.get("parameters", {}).get("properties", {})
        out[fn["name"]] = {
            "description": fn.get("description", ""),
            "parameters": {k: v.get("description", "") for k, v in props.items()},
        }
    return out


def seed_bundle(domain, tool_schemas, direct_tool_call=False):
    parts = PromptBuilder().resolve_orchestrator_parts(
        DOMAINS[domain], tool_schemas, direct_tool_call
    )
    orchestrator = {
        "guidance_opening": parts["guidance_opening"],
        "guidance_closing": parts["guidance_closing"],
        "example": copy.deepcopy(parts["example"]),
        "turn0_prompt": _DEFAULT_PLANNING_SUFFIX_TOOLS.removeprefix("\n\n"),
        "turn1n_prompt": "",
        "tool_descriptions": _descriptions_from_schemas(tool_schemas),
        "tool_use_note": parts["tool_use_note"],
    }
    subagents = {
        "web_search": WEB_SEARCH_ANALYSIS_TEMPLATE,
        "code_generator": CODE_GENERATOR_TASK_TEMPLATE,
    }
    return PromptBundle(domain=domain, orchestrator=orchestrator, subagents=subagents)


def save_bundle(bundle, path):
    Path(path).write_text(
        json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_bundle(path):
    return PromptBundle.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_tool_descriptions(schemas, tool_descriptions):
    """Return a deep copy of *schemas* with only the description strings replaced."""
    out = copy.deepcopy(schemas)
    for schema in out:
        fn = schema["function"]
        entry = tool_descriptions.get(fn["name"])
        if not entry:
            continue
        if entry.get("description") is not None:
            fn["description"] = entry["description"]
        props = fn.get("parameters", {}).get("properties", {})
        for pname, text in (entry.get("parameters") or {}).items():
            if pname in props and text is not None:
                props[pname]["description"] = text
    return out


def render_system_prompt(bundle, tool_schemas, tool_call_format):
    orch = bundle.orchestrator
    schemas = apply_tool_descriptions(tool_schemas, orch.get("tool_descriptions") or {})
    overrides = {k: copy.deepcopy(orch[k]) for k in _ORCH_PART_KEYS if k in orch}
    return PromptBuilder().build_system_prompt(
        DOMAINS[bundle.domain],
        schemas,
        direct_tool_call=False,
        tool_call_format=tool_call_format,
        overrides=overrides,
    )


def planning_suffix(bundle):
    return "\n\n" + bundle.orchestrator["turn0_prompt"].strip()


def acting_suffix(bundle):
    text = bundle.orchestrator["turn1n_prompt"].strip()
    return "\n\n" + text if text else ""


def apply_subagent_templates(bundle, registry):
    ws = registry.get("web_search")
    if ws is not None:
        ws.set_analysis_template(bundle.subagents["web_search"])
    cg = registry.get("code_generator")
    if cg is not None:
        cg.set_task_template(bundle.subagents["code_generator"])


def replace_unit(bundle, unit, payload):
    if unit not in UNITS:
        raise ValueError(f"unknown unit: {unit}")
    new = PromptBundle.from_dict(bundle.to_dict())
    if unit == "orchestrator":
        new.orchestrator.update(copy.deepcopy(payload))
    else:
        new.subagents[unit] = payload
    return new


def unit_text(bundle, unit):
    if unit == "orchestrator":
        return json.dumps(bundle.orchestrator, indent=2, ensure_ascii=False, sort_keys=True)
    return bundle.subagents[unit]


def editable_texts(bundle):
    """Every editable string in the bundle, for the leakage check."""
    orch = bundle.orchestrator
    texts = []
    for key in (
        "guidance_opening",
        "guidance_closing",
        "turn0_prompt",
        "turn1n_prompt",
        "tool_use_note",
    ):
        texts.append(orch.get(key) or "")
    example = orch.get("example")
    if example:
        texts.append(example.get("question") or "")
        for step in example.get("steps", []):
            for v in step.values():
                if isinstance(v, str):
                    texts.append(v)
    for entry in (orch.get("tool_descriptions") or {}).values():
        texts.append(entry.get("description") or "")
        texts.extend((entry.get("parameters") or {}).values())
    texts.extend(bundle.subagents.values())
    return [t for t in texts if t]
