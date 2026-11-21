"""Prompt builder for constructing prompts from YAML templates.

This module provides a clean way to build prompts from templates,
separating prompt content from code.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..datasets.spec import get_spec
from ..models.base import ToolCallFormat
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Closing sentence of the tool-format block; editable through ``overrides["tool_use_note"]``.
DEFAULT_TOOL_USE_NOTE = (
    "Important: You may call tools zero, one, or multiple times as needed. "
    "Only call a tool when it would help answer the question. "
    "After receiving tool results in <tool_response></tool_response> tags, "
    "continue your reasoning with the new information."
)


class PromptBuilder:
    """Builds prompts from YAML templates."""

    def __init__(self, template_dir: Optional[Path] = None):
        """Initialize prompt builder.

        Args:
            template_dir: Directory containing YAML templates
                         (defaults to templates/ subdirectory)
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)
        self._templates: Dict[str, Dict[str, Any]] = {}

    def load_template(self, name: str) -> Dict[str, Any]:
        """Load a template from YAML file.

        Args:
            name: Template name (without .yaml extension)

        Returns:
            Template dictionary

        Raises:
            FileNotFoundError: If template file not found
        """
        if name in self._templates:
            return self._templates[name]

        template_path = self.template_dir / "system" / f"{name}.yaml"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f)

        self._templates[name] = template
        logger.debug(f"Loaded template: {name}")

        return template

    def build_system_prompt(
        self,
        dataset_name: str,
        tool_schemas: List[Dict[str, Any]],
        max_search_limit: int = 10,
        direct_tool_call: bool = True,
        baseline: bool = False,
        tool_call_format: ToolCallFormat = ToolCallFormat.JSON,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build system prompt with tools and instructions.

        When *baseline* is ``True``, load ``*_baseline`` templates and omit
        sub-goal / reasoning scaffolding from tool instructions and examples.
        *tool_call_format* controls the syntax used in format instructions and
        examples: :attr:`ToolCallFormat.JSON` (default, Qwen3-style) or
        :attr:`ToolCallFormat.PYTHONIC` (OLMo 3-style).

        *overrides* may hold any subset of ``guidance_opening``, ``guidance_closing``,
        ``example`` and ``tool_use_note``; each replaces exactly that section. It only
        applies when *tool_schemas* is non-empty and *baseline* is False. The boxed-answer
        rule (``final_instructions``) and the tool-format paragraph are never overridable.
        """
        template = self._resolve_template(dataset_name, baseline)
        use_overrides = bool(overrides) and bool(tool_schemas) and not baseline
        ov = overrides if use_overrides else {}

        # Section order mirrors MAT: base_instruction → tools → example → final_instructions
        sections = []

        if ov.get("guidance_opening") is not None:
            if ov["guidance_opening"].strip():
                sections.append(ov["guidance_opening"].strip())
        elif tool_schemas and "base_instruction_tools" in template:
            sections.append(template["base_instruction_tools"].strip())
        elif "base_instruction" in template:
            sections.append(template["base_instruction"].strip())

        if tool_schemas:
            sections.append(
                self._format_tool_schemas(
                    tool_schemas, max_search_limit, direct_tool_call,
                    baseline=baseline, tool_call_format=tool_call_format,
                    tool_use_note=ov.get("tool_use_note"),
                )
            )

        example_text = self._select_and_format_example(
            template, tool_schemas, direct_tool_call,
            baseline=baseline, tool_call_format=tool_call_format,
            example_override=ov.get("example"),
        )
        if example_text:
            sections.append(example_text)

        if "final_instructions" in template:
            sections.append(template["final_instructions"].strip())

        if tool_schemas:
            if ov.get("guidance_closing") is not None:
                if ov["guidance_closing"].strip():
                    sections.append(ov["guidance_closing"].strip())
            else:
                ft_key = self._final_tools_key(template, direct_tool_call)
                if ft_key in template:
                    sections.append(template[ft_key].strip())

        return "\n\n".join(sections)

    def _resolve_template(self, dataset_name: str, baseline: bool) -> Dict[str, Any]:
        """Load the template used for *dataset_name* (falls back to the base template)."""
        try:
            # Which template a dataset uses lives in datasets/spec.py.
            # GAIA, HLE and MuSiQue share the single-QA template; AIME, MATH500,
            # AMC, DeepMath and MATH (RL prep) share the math template.
            spec = get_spec(dataset_name.lower())
            if spec.template is None:
                # No mapping: try the raw name, so an unknown dataset still hits
                # the FileNotFoundError fallback below (and logs its warning).
                template_name = dataset_name
            else:
                template_name = f"{spec.template}_baseline" if baseline else spec.template

            return self.load_template(template_name)
        except FileNotFoundError:
            logger.warning(f"Template '{dataset_name}' not found, using base template")
            return self.load_template("base_baseline" if baseline else "base")

    @staticmethod
    def _final_tools_key(template: Dict[str, Any], direct_tool_call: bool) -> str:
        mode_key = (
            "final_instructions_tools_direct"
            if direct_tool_call
            else "final_instructions_tools_subagent"
        )
        return mode_key if mode_key in template else "final_instructions_tools"

    @staticmethod
    def _example_key(tool_schemas: List[Dict[str, Any]], direct_tool_call: bool) -> Optional[str]:
        enabled_tools = {schema.get("function", {}).get("name", "") for schema in tool_schemas}
        has_search = "web_search" in enabled_tools
        has_code = "code_generator" in enabled_tools

        if has_search and has_code:
            return "example_search_code_direct" if direct_tool_call else "example_search_code_subagent"
        if has_search:
            return "example_search_only"
        if has_code:
            return "example_code_direct" if direct_tool_call else "example_code_subagent"
        return None

    def resolve_orchestrator_parts(
        self,
        dataset_name: str,
        tool_schemas: List[Dict[str, Any]],
        direct_tool_call: bool,
    ) -> Dict[str, Any]:
        """Return the editable orchestrator parts as today's templates resolve them.

        Passing the result back as ``overrides`` to :meth:`build_system_prompt` renders the
        same prompt as passing no overrides.
        """
        template = self._resolve_template(dataset_name, baseline=False)
        ft_key = self._final_tools_key(template, direct_tool_call)
        example_key = self._example_key(tool_schemas, direct_tool_call)
        example = template.get(example_key) if example_key else None
        return {
            "guidance_opening": template.get("base_instruction_tools", "").strip(),
            "guidance_closing": template.get(ft_key, "").strip(),
            "example": example if example else None,
            "tool_use_note": DEFAULT_TOOL_USE_NOTE,
        }

    @staticmethod
    def _json_tool_call_to_single(tool_call_json: str) -> str:
        """Wrap a JSON tool-call string in the DeepSeek ``{"tool_call": {...}}`` format.

        ``'{"name": "web_search", "arguments": {"query": "foo"}}'``
        → ``'{"tool_call": {"name": "web_search", "arguments": {"query": "foo"}}}'``

        Falls back gracefully on parse failure.
        """
        try:
            obj = json.loads(tool_call_json)
            return json.dumps({"tool_call": obj}, ensure_ascii=False)
        except (json.JSONDecodeError, AttributeError):
            return f'{{"tool_call": {tool_call_json}}}'

    @staticmethod
    def _json_tool_call_to_pythonic(tool_call_json: str) -> str:
        """Convert a JSON tool-call string to OLMo 3 pythonic format.

        ``'{"name": "web_search", "arguments": {"query": "foo"}}'``
        → ``'web_search(query="foo")'``

        Falls back to the original string on parse failure.
        """
        try:
            obj = json.loads(tool_call_json)
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            arg_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
            return f"{name}({arg_str})"
        except (json.JSONDecodeError, AttributeError):
            return tool_call_json

    # Per-format strings for tool-call syntax instructions and examples.
    # Add a new entry here when onboarding a family with a different tool-call format.
    _CALL_TAG_OPEN: Dict[ToolCallFormat, str] = {
        ToolCallFormat.JSON: "<tool_call>",
        ToolCallFormat.PYTHONIC: "<function_calls>",
        ToolCallFormat.JSON_SINGLE: "",  # no XML tags for pure-JSON format
    }
    _CALL_TAG_CLOSE: Dict[ToolCallFormat, str] = {
        ToolCallFormat.JSON: "</tool_call>",
        ToolCallFormat.PYTHONIC: "</function_calls>",
        ToolCallFormat.JSON_SINGLE: "",
    }
    _CALL_PLACEHOLDER: Dict[ToolCallFormat, str] = {
        ToolCallFormat.JSON: '{"name": <function-name>, "arguments": <args-json-object>}',
        ToolCallFormat.PYTHONIC: 'function_name(arg1="value1", arg2="value2")',
        ToolCallFormat.JSON_SINGLE: '{"tool_call": {"name": "<function-name>", "arguments": {"arg1": "value1"}}}',
    }

    def _format_tool_schemas(
        self,
        schemas: List[Dict[str, Any]],
        max_search_limit: int,
        direct_tool_call: bool = False,
        baseline: bool = False,
        tool_call_format: ToolCallFormat = ToolCallFormat.JSON,
        tool_use_note: Optional[str] = None,
    ) -> str:
        """Format tool schemas and call-format instructions."""
        if not schemas:
            return ""

        tools_json = "\n".join(json.dumps(schema, indent=2) for schema in schemas)
        open_tag  = self._CALL_TAG_OPEN[tool_call_format]
        close_tag = self._CALL_TAG_CLOSE[tool_call_format]
        placeholder = self._CALL_PLACEHOLDER[tool_call_format]

        header = (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
        )
        tail = DEFAULT_TOOL_USE_NOTE if tool_use_note is None else tool_use_note

        # ── DeepSeek JSON_SINGLE: sub_goal kept for AF tracking; JSON replaces <tool_call> ──
        if tool_call_format == ToolCallFormat.JSON_SINGLE:
            direct_mode_rule = (
                "\nAfter every <tool_response>...</tool_response>, "
                "write at least one sentence of reasoning before making another tool call.\n"
                if direct_tool_call else ""
            )
            if baseline:
                return (
                    header
                    + "When calling a tool, output ONLY a valid JSON object with this exact structure:\n"
                    f"{placeholder}\n\n"
                    + tail
                )
            return (
                header
                + "For each function call, first state your specific sub-goal for this step "
                "within <sub_goal></sub_goal> tags, then output ONLY a valid JSON object with this exact structure:\n"
                f"<sub_goal>A specific, actionable goal for this step</sub_goal>\n"
                f"{placeholder}\n\n"
                "Rules:\n"
                "- Call at most one tool per turn\n"
                "- The JSON must have a \"tool_call\" key with \"name\" and \"arguments\"\n\n"
                + tail
                + direct_mode_rule
            )

        if baseline:
            return (
                header
                + f"When you need a tool, return exactly one function call within "
                f"{open_tag}{close_tag} XML tags:\n"
                f"{open_tag}\n{placeholder}\n{close_tag}\n\n"
                + tail
            )

        direct_mode_rule = (
            f"\nAfter every <tool_response>...</tool_response>, "
            f"write at least one sentence of reasoning before making another {open_tag} call.\n"
            if direct_tool_call else ""
        )
        return (
            header
            + "For each function call, first state your specific sub-goal for this step "
            f"within <sub_goal></sub_goal> tags, then return the function call within "
            f"{open_tag}{close_tag} XML tags:\n"
            f"<sub_goal>A specific, actionable goal for this step</sub_goal>\n"
            f"{open_tag}\n{placeholder}\n{close_tag}\n\n"
            + tail
            + direct_mode_rule
        )

    def _select_and_format_example(
        self,
        template: Dict[str, Any],
        tool_schemas: List[Dict[str, Any]],
        direct_tool_call: bool,
        baseline: bool = False,
        tool_call_format: ToolCallFormat = ToolCallFormat.JSON,
        example_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select and format appropriate example based on tools and mode.

        *example_override* is formatted with the same step loop instead of the template's.
        """
        if example_override is not None:
            example = example_override
        else:
            example_key = self._example_key(tool_schemas, direct_tool_call)

            if not example_key or example_key not in template:
                if "example" in template and template["example"]:
                    return self._format_example(template["example"], baseline=baseline)
                return ""

            example = template[example_key]

        open_tag  = self._CALL_TAG_OPEN[tool_call_format]
        close_tag = self._CALL_TAG_CLOSE[tool_call_format]

        if not example or "question" not in example or "steps" not in example:
            return ""

        lines = ["### EXAMPLE", ""]
        lines.append(f'Question: "{example["question"]}"')
        lines.append("")

        for step_data in example["steps"]:
            step_num = step_data.get("step")

            if step_num == "final":
                lines.append(f'**Answer:** {step_data.get("answer", "")}')
                lines.append("")
            else:
                if not baseline:
                    reasoning = step_data.get("reasoning", "")
                    if reasoning:
                        lines.append(f"**Step {step_num}:** {reasoning}")
                        lines.append("")

                    if "sub_goal" in step_data:
                        lines.append(f"<sub_goal>{step_data['sub_goal']}</sub_goal>")
                        lines.append("")

                if "tool_call" in step_data:
                    if tool_call_format == ToolCallFormat.PYTHONIC:
                        call_content = self._json_tool_call_to_pythonic(step_data["tool_call"])
                    elif tool_call_format == ToolCallFormat.JSON_SINGLE:
                        call_content = self._json_tool_call_to_single(step_data["tool_call"])
                    else:
                        call_content = step_data["tool_call"]

                    if tool_call_format == ToolCallFormat.JSON_SINGLE:
                        lines.append(call_content)
                    else:
                        lines.append(open_tag)
                        lines.append(call_content)
                        lines.append(close_tag)
                    lines.append("")

                if "tool_response" in step_data:
                    lines.append("<tool_response>")
                    lines.append(step_data["tool_response"])
                    lines.append("</tool_response>")
                    lines.append("")

        return "\n".join(lines)

    def _format_example(self, example: Dict[str, Any], baseline: bool = False) -> str:
        """Format example from template."""
        lines = ["### EXAMPLE", ""]

        if "question" in example:
            lines.append(f'**Question:** {example["question"]}')
            lines.append("")

        if not baseline and "reasoning" in example:
            lines.append(f'**Reasoning:**\n{example["reasoning"]}')
            lines.append("")

        if "answer" in example:
            lines.append(f'**Answer:** {example["answer"]}')

        return "\n".join(lines)
