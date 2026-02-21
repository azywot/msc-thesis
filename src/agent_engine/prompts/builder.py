"""Prompt builder for constructing prompts from YAML templates.

This module provides a clean way to build prompts from templates,
separating prompt content from code.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SUPPORTED_TEXT_EXTS = {
    ".txt", ".md", ".log",
    ".json", ".jsonl", ".xml",
    ".csv", ".tsv",
    ".yaml", ".yml",
    ".docx", ".xlsx", ".jsonld",
    ".parquet", ".pdf", ".pdb",
    ".pptx", ".py",
}


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
        attachments: Optional[List[str]] = None,
        max_search_limit: int = 10,
        direct_tool_call: bool = True
    ) -> str:
        """Build system prompt with tools and instructions.

        Args:
            dataset_name: Dataset name (gaia, gpqa, math)
            tool_schemas: List of tool JSON schemas
            attachments: Ignored - attachment notes go into the user message via orchestrator
            max_search_limit: Maximum search attempts
            direct_tool_call: Whether tools execute directly (True) or use sub-agents (False)

        Returns:
            Complete system prompt
        """
        try:
            template = self.load_template(dataset_name)
        except FileNotFoundError:
            logger.warning(f"Template '{dataset_name}' not found, using base template")
            template = self.load_template("base")

        # Build sections — order matches multi-agent-tools:
        # base_instruction → tools → example → final_instructions
        sections = []

        # Base instruction
        if "base_instruction" in template:
            sections.append(template["base_instruction"].strip())

        # Tool descriptions (XML <tools> format matching MAT)
        if tool_schemas:
            tool_desc = self._format_tool_schemas(tool_schemas, max_search_limit, direct_tool_call)
            sections.append(tool_desc)

        # Example (before final instructions, as in MAT)
        example_text = self._select_and_format_example(template, tool_schemas, direct_tool_call)
        if example_text:
            sections.append(example_text)

        # Final instructions (Remember: block)
        if "final_instructions" in template:
            sections.append(template["final_instructions"].strip())

        return "\n\n".join(sections)

    def _format_tool_schemas(
        self,
        schemas: List[Dict[str, Any]],
        max_search_limit: int,
        direct_tool_call: bool = False,
    ) -> str:
        """Format tool schemas using the Qwen3 XML <tools> format (matches MAT).

        Args:
            schemas: List of tool JSON schemas
            max_search_limit: Maximum search attempts (currently unused in prompt text)
            direct_tool_call: Whether direct tool call mode is active

        Returns:
            Formatted tool section string
        """
        if not schemas:
            return ""

        tools_json = "\n".join(json.dumps(schema, indent=2) for schema in schemas)

        direct_mode_rule = ""
        if direct_tool_call:
            direct_mode_rule = (
                "\n"
                "After every <tool_response>...</tool_response>, "
                "write at least one sentence of reasoning before making another <tool_call>.\n"
            )

        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
            "</tool_call>\n\n"
            "Important: You may call tools zero, one, or multiple times as needed. "
            "Only call a tool when it would help answer the question. "
            "After receiving tool results in <tool_response></tool_response> tags, "
            f"continue your reasoning with the new information.{direct_mode_rule}"
        )

    def _format_attachments(self, attachments: List[str]) -> str:
        """Format attachment information in MAT's [Attachment] style.

        Args:
            attachments: List of attachment file paths (full paths or basenames)

        Returns:
            Formatted attachment note
        """
        if not attachments:
            return ""

        lines = []
        for attachment in attachments:
            fname = os.path.basename(attachment) if attachment else "attached_file"
            ext = os.path.splitext(fname)[1].lower()

            lines.append("\n[Attachment]")
            lines.append(f"- There is an attached file for this question: {fname}")

            if ext in _SUPPORTED_IMAGE_EXTS:
                lines.append(
                    "- To inspect the image, call the tool `image_inspector` with a question about the image."
                )
            elif ext in _SUPPORTED_TEXT_EXTS:
                lines.append(
                    "- To read the file, call the tool `text_inspector` (optionally with a question)."
                )
            else:
                lines.append(
                    "- The attachment type is not supported by the available inspectors in this run."
                )
            lines.append(
                "- Important: do NOT guess or provide file paths; inspectors use the attached file automatically."
            )

        return "\n".join(lines) + "\n"

    def _select_and_format_example(
        self,
        template: Dict[str, Any],
        tool_schemas: List[Dict[str, Any]],
        direct_tool_call: bool
    ) -> str:
        """Select and format appropriate example based on tools and mode.

        Args:
            template: Template dictionary
            tool_schemas: List of enabled tool schemas
            direct_tool_call: Whether using direct mode (True) or sub-agent mode (False)

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

        # Format in MAT's ### EXAMPLE style with multiline tool call/response tags
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

                if "tool_call" in step_data:
                    lines.append("<tool_call>")
                    lines.append(step_data["tool_call"])
                    lines.append("</tool_call>")
                    lines.append("")

                if "tool_response" in step_data:
                    lines.append("<tool_response>")
                    lines.append(step_data["tool_response"])
                    lines.append("</tool_response>")
                    lines.append("")

        return "\n".join(lines)

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example from template (legacy format).

        Args:
            example: Example dictionary with question, reasoning, answer

        Returns:
            Formatted example text
        """
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

    def get_user_prompt(self, question: str) -> str:
        """Build user prompt for a question.

        Args:
            question: Question text

        Returns:
            User prompt
        """
        return question
