"""Team edits: the reflector rewrites the whole team's prompts in one plain-text reply.

The team is shown as one document with a section per prompt. The reflector returns the same
sections; a section it leaves out or marks UNCHANGED keeps the champion's text. The reply is
plain text, not JSON, so long prompts need no escaping.

The reflector writes instructions only. Fixed parts are inserted by the system:

- the orchestrator system prompt keeps the {tools} and {answer_format} placeholders of full mode
  (see rsi.bundle), so the tool list, call format and answer rule cannot be broken;
- each sub-agent section may contain {inputs} once to mark where the sub-agent's inputs go
  (search query and fetched pages, or context and task); without it they are appended at the
  end. The input blocks are the seed templates' own text, so the seed renders byte-identically.
"""

import difflib
import re

from agent_engine.prompts.templating import find_placeholders
from agent_engine.tools.code_generator import (
    CODE_GENERATOR_PLACEHOLDERS,
    CODE_GENERATOR_TASK_TEMPLATE,
)
from agent_engine.tools.web_search import WEB_SEARCH_ANALYSIS_TEMPLATE, WEB_SEARCH_PLACEHOLDERS
from rsi.bundle import FULL_PROMPT_KEY, PromptBundle, is_full

TEAM_UNIT = "team"
INPUTS = "inputs"
UNCHANGED = "UNCHANGED"
RATIONALE = "RATIONALE"

# (key, header) in document order
ORCHESTRATOR_SECTIONS = (
    (FULL_PROMPT_KEY, "ORCHESTRATOR SYSTEM PROMPT"),
    ("turn0_prompt", "PLANNING TURN"),
    ("turn1n_prompt", "AFTER EACH TOOL RESULT"),
)
SUBAGENT_SECTIONS = (
    ("web_search", "WEB_SEARCH SUB-AGENT"),
    ("code_generator", "CODE_GENERATOR SUB-AGENT"),
)
HEADERS = dict(ORCHESTRATOR_SECTIONS + SUBAGENT_SECTIONS)

# What each section is for, shown to the reflector.
SECTION_NOTES = {
    FULL_PROMPT_KEY: (
        "The orchestrator's system prompt: it plans, calls the sub-agents as tools and gives "
        "the final answer. Must contain {tools} exactly once and {answer_format} exactly once; "
        "the system replaces them with the tool list and call format, and with the final-answer "
        "rule."
    ),
    "turn0_prompt": "Appended to the question on the planning turn, before any tool call. "
    "Must not be empty.",
    "turn1n_prompt": "Appended on every later turn, after each tool result. May be empty.",
    "web_search": (
        "The web_search sub-agent reads the pages fetched for ONE search query. Its reply is "
        "the tool result the orchestrator sees. It receives the query and the pages through "
        "{inputs}."
    ),
    "code_generator": (
        "The code_generator sub-agent writes a Python script for one task. The script is run "
        "and its printed output is the tool result the orchestrator sees. It receives optional "
        "context and the task through {inputs}. It must reply with code only."
    ),
}

_WS_INPUTS_AT = WEB_SEARCH_ANALYSIS_TEMPLATE.index("**Inputs:**")
INPUT_BLOCKS = {
    "web_search": WEB_SEARCH_ANALYSIS_TEMPLATE[_WS_INPUTS_AT:],
    "code_generator": "{context_block}Problem: {task}",
}
assert CODE_GENERATOR_TASK_TEMPLATE.count(INPUT_BLOCKS["code_generator"]) == 1
_RAW_PLACEHOLDERS = {
    "web_search": WEB_SEARCH_PLACEHOLDERS,
    "code_generator": CODE_GENERATOR_PLACEHOLDERS,
}

_HEADER_LINE = re.compile(r"^[ \t]*={3,}[ \t]*([^=\n]+?)[ \t]*={3,}[ \t]*$", re.MULTILINE)
_FENCE_LINE = re.compile(r"^```[\w-]*$")
_EMPTY_MARKERS = {"", "(empty)", "empty", "(none)"}


class TeamParseError(ValueError):
    pass


def compose_subagent(unit, section):
    """The full sub-agent template for an instructions section."""
    block = INPUT_BLOCKS[unit]
    marker = "{" + INPUTS + "}"
    if marker in section:
        return section.replace(marker, block, 1)
    return section.rstrip() + "\n\n" + block


def decompose_subagent(unit, template):
    """The instructions section of a full template, with {inputs} where the inputs go."""
    block = INPUT_BLOCKS[unit]
    if template.count(block) != 1:
        raise ValueError(f"{unit} template does not hold its input block exactly once")
    return template.replace(block, "{" + INPUTS + "}")


def team_sections(bundle, subagents=None):
    """{key: text} for every section of a full-mode bundle, in document order."""
    if not is_full(bundle):
        raise ValueError("team edits need orchestrator_mode: full")
    out = {key: bundle.orchestrator.get(key) or "" for key, _ in ORCHESTRATOR_SECTIONS}
    for unit, _ in SUBAGENT_SECTIONS:
        if unit in bundle.subagents and (subagents is None or unit in subagents):
            out[unit] = decompose_subagent(unit, bundle.subagents[unit])
    return out


def render_team_document(bundle, subagents=None):
    """The team as one plain-text document, as the reflector sees and returns it."""
    chunks = []
    for key, text in team_sections(bundle, subagents).items():
        chunks.append(f"=== {HEADERS[key]} ===\n{text.strip() if text.strip() else '(empty)'}")
    return "\n\n".join(chunks)


def _normalize_header(name):
    return re.sub(r"[^A-Z0-9]+", " ", name.upper()).strip()


_KEY_BY_HEADER = {_normalize_header(h): k for k, h in HEADERS.items()}
_KEY_BY_HEADER.update({_normalize_header(k): k for k in HEADERS})


def _strip_outer_fence(text):
    lines = text.strip("\n").splitlines()
    if len(lines) >= 2 and _FENCE_LINE.match(lines[0].strip()) and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines)


def _clean_body(body):
    return _strip_outer_fence(body).strip("\n").rstrip()


def parse_team_reply(text):
    """Return ({key: text or UNCHANGED}, rationale, unknown_headers) from a reflector reply."""
    from agent_engine.utils.parsing import strip_thinking_tags

    body = _strip_outer_fence(strip_thinking_tags(text or ""))
    if not body.strip():
        raise TeamParseError("empty reflector output")
    matches = list(_HEADER_LINE.finditer(body))
    if not matches:
        raise TeamParseError("no '=== SECTION ===' headers found in the reply")
    sections, rationale, unknown = {}, "", []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = _clean_body(body[m.end() : end])
        name = _normalize_header(m.group(1))
        if name == RATIONALE:
            rationale = content.strip()
            continue
        key = _KEY_BY_HEADER.get(name)
        if key is None:
            unknown.append(m.group(1).strip())
            continue
        if key in sections:
            raise TeamParseError(f"section {HEADERS[key]!r} appears twice")
        if content.strip().strip(".").upper() in (UNCHANGED, f"({UNCHANGED})"):
            sections[key] = UNCHANGED
        elif content.strip().lower() in _EMPTY_MARKERS:
            sections[key] = ""
        else:
            sections[key] = content
    if not sections and not unknown:
        raise TeamParseError("the reply has no team section")
    return sections, rationale, unknown


def resolve_sections(sections, champion, subagents=None):
    """Full {key: text}: sections left out or UNCHANGED keep the champion's text."""
    base = team_sections(champion, subagents)
    out = dict(base)
    for key, text in sections.items():
        if text == UNCHANGED or (key in base and text.strip() == base[key].strip()):
            continue  # a verbatim copy counts as unchanged, whatever its surrounding whitespace
        out[key] = text  # a section the base lacks is kept, so validation can name it
    return out


def changed_sections(old, new):
    return [k for k in old if old[k] != new.get(k)]


def validate_team(sections, unknown_headers, enabled_tools, base):
    """Error strings for a resolved team payload; empty means valid."""
    errors = [f"unknown section {h!r} (allowed: {list(HEADERS.values())})" for h in unknown_headers]
    for unit, _ in SUBAGENT_SECTIONS:
        if unit in sections and unit not in enabled_tools:
            errors.append(f"tool {unit!r} is not enabled, so its section cannot be edited")
    if not changed_sections(base, sections):
        errors.append("no section changed")
    prompt = sections.get(FULL_PROMPT_KEY, "")
    if not prompt.strip():
        errors.append(f"{HEADERS[FULL_PROMPT_KEY]} must be non-empty")
    found = find_placeholders(prompt)
    for name in ("tools", "answer_format"):
        n = found.count(name)
        if n != 1:
            errors.append(
                f"{HEADERS[FULL_PROMPT_KEY]} must contain {{{name}}} exactly once (found {n})"
            )
    if not sections.get("turn0_prompt", "").strip():
        errors.append(f"{HEADERS['turn0_prompt']} must be non-empty")
    for key in ("turn0_prompt", "turn1n_prompt"):
        names = sorted(set(find_placeholders(sections.get(key, ""))) & {"tools", "answer_format"})
        if names:
            errors.append(f"{HEADERS[key]} must not contain {names} (only the system prompt does)")
    for unit, header in SUBAGENT_SECTIONS:
        if unit not in sections:
            continue
        text = sections[unit]
        if not text.replace("{" + INPUTS + "}", "").strip():
            errors.append(f"{header} must contain instructions")
        n = find_placeholders(text).count(INPUTS)
        if n > 1:
            errors.append(f"{header} may contain {{{INPUTS}}} at most once (found {n})")
        raw = sorted(set(find_placeholders(text)) & set(_RAW_PLACEHOLDERS[unit]))
        if raw:
            shown = ", ".join("{" + r + "}" for r in raw)
            errors.append(f"{header} must not contain {shown}; write {{{INPUTS}}} instead")
    return errors


def apply_team(champion, sections):
    """The champion with every section replaced by `sections` (a resolved payload)."""
    new = PromptBundle.from_dict(champion.to_dict())
    for key, _ in ORCHESTRATOR_SECTIONS:
        if key in sections:
            new.orchestrator[key] = sections[key]
    for unit, _ in SUBAGENT_SECTIONS:
        if unit in sections:
            new.subagents[unit] = compose_subagent(unit, sections[unit])
    return new


def team_diff(old_bundle, new_bundle, max_chars=None):
    """Unified diff of every changed section."""
    old, new = team_sections(old_bundle), team_sections(new_bundle)
    chunks = []
    for key in old:
        if old[key] == new.get(key):
            continue
        lines = difflib.unified_diff(
            old[key].splitlines(),
            new.get(key, "").splitlines(),
            f"champion/{HEADERS[key]}",
            f"candidate/{HEADERS[key]}",
            n=1,
            lineterm="",
        )
        chunks.append("\n".join(lines))
    text = "\n".join(c for c in chunks if c)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff cut)"
    return text


def output_format_text(subagents):
    """Reply format and validity rules for a team edit."""
    keys = [k for k, _ in ORCHESTRATOR_SECTIONS] + [
        u for u, _ in SUBAGENT_SECTIONS if u in subagents
    ]
    section_lines = [f"- {HEADERS[k]}: {SECTION_NOTES[k]}" for k in keys]
    return "\n".join(
        [
            "Reply in plain text (no JSON), as the team document with your changes:",
            "",
            f"=== {RATIONALE} ===",
            "<two or three sentences: which recurring problem you target, what you change in "
            "which sections, and why the changes work together>",
            "",
            *[f"=== {HEADERS[k]} ===\n<new text, or {UNCHANGED}>" for k in keys],
            "",
            "Rules. A reply that breaks them is rejected:",
            "- Use exactly these section headers, each on its own line.",
            f"- Write {UNCHANGED} as the whole body of a section you keep as it is. "
            "Change at least one section.",
            "- Each body is the complete new text of that section, not a diff.",
            "- Change sections together when they depend on each other: when a sub-agent "
            "should reply in a new format, tell the orchestrator how to use it.",
            "- In a sub-agent section, keep {inputs} once where the inputs should go (left out, "
            "they go at the end). Do not write the input names themselves, such as {query}.",
            "",
            "The sections:",
            *section_lines,
        ]
    )
