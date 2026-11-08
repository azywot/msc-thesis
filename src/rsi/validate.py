"""Candidate validation: reject malformed reflector proposals with named reasons."""

import copy
import json

from agent_engine.prompts.templating import find_placeholders
from agent_engine.tools.code_generator import CODE_GENERATOR_PLACEHOLDERS
from agent_engine.tools.web_search import WEB_SEARCH_PLACEHOLDERS
from rsi.bundle import UNITS

ORCH_PARTS = (
    "guidance_opening",
    "guidance_closing",
    "example",
    "turn0_prompt",
    "turn1n_prompt",
    "tool_descriptions",
    "tool_use_note",
)
_MAY_BE_EMPTY = ("turn1n_prompt", "guidance_closing")
_STRING_PARTS = (
    "guidance_opening",
    "guidance_closing",
    "turn0_prompt",
    "turn1n_prompt",
    "tool_use_note",
)
_SUBAGENT_PLACEHOLDERS = {
    "web_search": WEB_SEARCH_PLACEHOLDERS,
    "code_generator": CODE_GENERATOR_PLACEHOLDERS,
}


def normalize_example(example):
    """Return a copy with tool_call as a JSON string and non-final steps numbered 1..n."""
    out = copy.deepcopy(example)
    if not isinstance(out, dict) or not isinstance(out.get("steps"), list):
        return out
    n = 0
    for step in out.get("steps", []):
        if not isinstance(step, dict) or step.get("step") == "final":
            continue
        n += 1
        step["step"] = n
        call = step.get("tool_call")
        if isinstance(call, (dict, list)):
            step["tool_call"] = json.dumps(call, ensure_ascii=False)
    return out


def _schema_index(tool_schemas):
    index = {}
    for schema in tool_schemas:
        fn = schema["function"]
        params = fn.get("parameters", {})
        index[fn["name"]] = (set(params.get("properties", {})), list(params.get("required", [])))
    return index


def _validate_example(example, index):
    if not isinstance(example, dict):
        return ["example must be an object"]
    errors = []
    question = example.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append("example: question must be a non-empty string")
    steps = example.get("steps")
    if not isinstance(steps, list) or not steps:
        return errors + ["example: steps must be a non-empty list"]
    if not all(isinstance(s, dict) for s in steps):
        return errors + ["example: every step must be an object"]
    non_final = [s for s in steps if s.get("step") != "final"]
    if not non_final:
        errors.append("example: needs at least one non-final step")
    for i, step in enumerate(non_final, 1):
        errors.extend(_validate_call(step.get("tool_call"), index, i))
    last = steps[-1]
    if last.get("step") != "final":
        errors.append('example: last step must be {"step": "final", "answer": ...}')
    else:
        answer = last.get("answer")
        if not isinstance(answer, str) or "\\boxed{" not in answer:
            errors.append("example: final answer must contain \\boxed{")
    return errors


def _validate_call(call, index, i):
    label = f"example step {i}"
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except ValueError:
            return [f"{label}: tool_call is not valid JSON"]
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return [f'{label}: tool_call must be {{"name", "arguments"}}']
    name, args = call["name"], call["arguments"]
    if not isinstance(name, str):
        return [f"{label}: tool name must be a string"]
    if name not in index:
        return [f"{label}: tool {name!r} is not enabled (enabled: {sorted(index)})"]
    if not isinstance(args, dict):
        return [f"{label}: arguments must be an object"]
    props, required = index[name]
    errors = []
    for key in sorted(set(args) - props):
        errors.append(f"{label}: argument {key!r} is not a parameter of {name} {sorted(props)}")
    for key in required:
        if key not in args:
            errors.append(f"{label}: missing required argument {key!r} of {name}")
    return errors


def _validate_tool_descriptions(td, seed):
    if not isinstance(td, dict):
        return ["tool_descriptions must be an object"]
    seed_td = seed.orchestrator["tool_descriptions"]
    errors = []
    if set(td) != set(seed_td):
        errors.append(f"tool_descriptions tools {sorted(td)} differ from seed {sorted(seed_td)}")
    for name in sorted(set(td) & set(seed_td)):
        entry = td[name]
        if not isinstance(entry, dict):
            errors.append(f"tool_descriptions[{name}] must be an object")
            continue
        desc = entry.get("description")
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"tool_descriptions[{name}]: description must be a non-empty string")
        params = entry.get("parameters")
        seed_params = seed_td[name]["parameters"]
        if not isinstance(params, dict):
            errors.append(f"tool_descriptions[{name}]: parameters must be an object")
            continue
        if set(params) != set(seed_params):
            errors.append(
                f"tool_descriptions[{name}]: parameter names {sorted(params)} "
                f"differ from seed {sorted(seed_params)}"
            )
        for pname, text in params.items():
            if not isinstance(text, str) or not text.strip():
                errors.append(f"tool_descriptions[{name}].{pname}: must be a non-empty string")
    return errors


def _validate_orchestrator(payload, seed, index):
    if not isinstance(payload, dict):
        return ["orchestrator payload must be an object"]
    errors = []
    for key in sorted(set(payload) - set(ORCH_PARTS)):
        errors.append(f"unknown orchestrator part {key!r}")
    for key in _STRING_PARTS:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, str):
            errors.append(f"{key} must be a string")
        elif not value.strip() and key not in _MAY_BE_EMPTY:
            errors.append(f"{key} must be non-empty")
    if "example" in payload:
        errors.extend(_validate_example(payload["example"], index))
    if "tool_descriptions" in payload:
        errors.extend(_validate_tool_descriptions(payload["tool_descriptions"], seed))
    return errors


def _validate_template(unit, template):
    if not isinstance(template, str):
        return [f"{unit} template must be a string"]
    required = _SUBAGENT_PLACEHOLDERS[unit]
    found = set(find_placeholders(template))
    errors = [f"{unit} template is missing placeholder {{{p}}}" for p in required if p not in found]
    errors += [
        f"{unit} template has unknown placeholder {{{p}}}" for p in sorted(found - set(required))
    ]
    return errors


def validate_unit(unit, payload, seed, tool_schemas):
    """Return a list of error strings for a candidate payload; empty means valid."""
    if unit not in UNITS:
        return [f"unknown unit {unit!r}"]
    index = _schema_index(tool_schemas)
    if unit == "orchestrator":
        return _validate_orchestrator(payload, seed, index)
    if unit not in index:
        return [f"tool {unit!r} is not enabled, so its sub-agent prompt cannot be edited"]
    return _validate_template(unit, payload)
