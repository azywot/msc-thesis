"""Prompt builder for constructing prompts from YAML templates.

This module provides a clean way to build prompts from templates,
separating prompt content from code.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..models.base import ModelFamily, _DEEPSEEK_FAMILIES
from ..utils.logging import get_logger

logger = get_logger(__name__)


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
        direct_tool_call: bool = True,
        model_family: Optional[ModelFamily] = None,
    ) -> str:
        """Build system prompt with tools and instructions.

        Args:
            dataset_name: Name of the dataset (determines template selection).
            tool_schemas: List of tool schema dicts.
            direct_tool_call: Whether tools run in direct (orchestrator-inline) mode.
            model_family: Model family — controls tool-call tag format in prompts.
                          ``None`` defaults to Qwen3-style tags.
        """
        try:
            # GAIA, HLE, and MuSiQue share the same single‑QA prompt template.
            # AIME, MATH500, AMC share the math template.
            template_name = dataset_name
            if dataset_name.lower() in ("gaia", "hle", "musique"):
                template_name = "gaia"
            elif dataset_name.lower() in ("aime", "math500", "amc"):
                template_name = "math"

            template = self.load_template(template_name)
        except FileNotFoundError:
            logger.warning(f"Template '{dataset_name}' not found, using base template")
            template = self.load_template("base")

        # Section order mirrors MAT: base_instruction → tools → example → final_instructions
        sections = []

        if tool_schemas and "base_instruction_tools" in template:
            sections.append(template["base_instruction_tools"].strip())
        elif "base_instruction" in template:
            sections.append(template["base_instruction"].strip())

        if tool_schemas:
            sections.append(self._format_tool_schemas(tool_schemas, direct_tool_call, model_family))

        example_text = self._select_and_format_example(template, tool_schemas, direct_tool_call, model_family)
        if example_text:
            sections.append(example_text)

        if "final_instructions" in template:
            sections.append(template["final_instructions"].strip())

        if tool_schemas and "final_instructions_tools" in template:
            sections.append(template["final_instructions_tools"].strip())

        return "\n\n".join(sections)

    @staticmethod
    def _get_family_tags(
        model_family: Optional[ModelFamily],
    ) -> tuple:
        """Return (tc_open, tc_close, tr_open, tr_close) for the given model family."""
        if model_family is not None and model_family in _DEEPSEEK_FAMILIES:
            return "```json\n", "\n```", "Tool result:\n", ""
        if model_family == ModelFamily.PHI4:
            return "<|tool_call|>\n", "\n<|/tool_call|>", "<tool_response>", "</tool_response>"
        # Qwen family + default
        return "<tool_call>\n", "\n</tool_call>", "<tool_response>", "</tool_response>"

    def _format_tool_schemas(
        self,
        schemas: List[Dict[str, Any]],
        direct_tool_call: bool = False,
        model_family: Optional[ModelFamily] = None,
    ) -> str:
        """Format tool schemas with family-aware tool-call tag format."""
        if not schemas:
            return ""

        tools_json = "\n".join(json.dumps(schema, indent=2) for schema in schemas)
        tc_open, tc_close, tr_open, tr_close = self._get_family_tags(model_family)

        direct_mode_rule = ""
        if direct_tool_call:
            direct_mode_rule = (
                "\n"
                "After every tool result, "
                "write at least one sentence of reasoning before making another tool call.\n"
            )

        tr_hint = f" in {tr_open}...{tr_close} tags" if tr_close else ""

        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For each function call, first state your specific sub-goal for this step "
            "within <sub_goal></sub_goal> tags, then return the function call as follows:\n"
            "<sub_goal>A specific, actionable goal for this step</sub_goal>\n"
            f"{tc_open}"
            '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
            f"{tc_close}\n\n"
            "Important: You may call tools zero, one, or multiple times as needed. "
            "Only call a tool when it would help answer the question. "
            f"After receiving tool results{tr_hint}, "
            f"continue your reasoning with the new information.{direct_mode_rule}"
        )

    def _select_and_format_example(
        self,
        template: Dict[str, Any],
        tool_schemas: List[Dict[str, Any]],
        direct_tool_call: bool,
        model_family: Optional[ModelFamily] = None,
    ) -> str:
        """Select and format appropriate example based on tools and mode.

        Args:
            template: Template dictionary
            tool_schemas: List of enabled tool schemas
            direct_tool_call: Whether using direct mode (True) or sub-agent mode (False)
            model_family: Model family for tag style selection.

        Returns:
            Formatted example text or empty string if no example
        """
        # Determine enabled tools
        enabled_tools = {schema.get("function", {}).get("name", "") for schema in tool_schemas}
        has_search = "web_search" in enabled_tools
        has_code = "code_generator" in enabled_tools

        # Select appropriate example key
        example_key = None
        if has_search and has_code:
            example_key = "example_search_code_direct" if direct_tool_call else "example_search_code_subagent"
        elif has_search:
            example_key = "example_search_only"
        elif has_code:
            example_key = "example_code_direct" if direct_tool_call else "example_code_subagent"

        if not example_key or example_key not in template:
            if "example" in template and template["example"]:
                return self._format_example(template["example"])
            return ""

        example = template[example_key]
        if not example or "question" not in example or "steps" not in example:
            return ""

        tc_open, tc_close, tr_open, tr_close = self._get_family_tags(model_family)
        # Strip newlines from tags for inline example formatting
        tc_open, tc_close = tc_open.strip(), tc_close.strip()
        tr_open, tr_close = tr_open.strip(), tr_close.strip()

        lines = ["### EXAMPLE", ""]
        lines.append(f'Question: "{example["question"]}"')
        lines.append("")

        for step_data in example["steps"]:
            step_num = step_data.get("step")

            if step_num == "final":
                lines.append(f'**Answer:** {step_data.get("answer", "")}')
                lines.append("")
            else:
                reasoning = step_data.get("reasoning", "")
                if reasoning:
                    lines.append(f"**Step {step_num}:** {reasoning}")
                    lines.append("")

                if "sub_goal" in step_data:
                    lines.append(f"<sub_goal>{step_data['sub_goal']}</sub_goal>")
                    lines.append("")

                if "tool_call" in step_data:
                    lines.append(tc_open)
                    lines.append(step_data["tool_call"])
                    lines.append(tc_close)
                    lines.append("")

                if "tool_response" in step_data:
                    lines.append(tr_open)
                    lines.append(step_data["tool_response"])
                    if tr_close:
                        lines.append(tr_close)
                    lines.append("")

        return "\n".join(lines)

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example from template."""
        lines = ["### EXAMPLE", ""]

        if "question" in example:
            lines.append(f'**Question:** {example["question"]}')
            lines.append("")

        if "reasoning" in example:
            lines.append(f'**Reasoning:**\n{example["reasoning"]}')
            lines.append("")

        if "answer" in example:
            lines.append(f'**Answer:** {example["answer"]}')

        return "\n".join(lines)
