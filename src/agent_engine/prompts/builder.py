"""Prompt builder for constructing prompts from YAML templates.

This module provides a clean way to build prompts from templates,
separating prompt content from code.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
        attachments: Optional[List[str]] = None,
        max_search_limit: int = 10,
        direct_tool_call: bool = True
    ) -> str:
        """Build system prompt with tools and instructions.

        Args:
            dataset_name: Dataset name (gaia, gpqa, math)
            tool_schemas: List of tool JSON schemas
            attachments: Optional list of attachment file names
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

        # Build sections
        sections = []

        # Base instruction
        if "base_instruction" in template:
            sections.append(template["base_instruction"])

        # Tool descriptions
        if tool_schemas:
            tool_desc = self._format_tool_schemas(tool_schemas, max_search_limit)
            sections.append(tool_desc)

        # Final instructions (includes tool usage format)
        if "final_instructions" in template:
            sections.append(template["final_instructions"])

        # Select and format example based on enabled tools and mode
        example_text = self._select_and_format_example(template, tool_schemas, direct_tool_call)
        if example_text:
            sections.append(example_text)

        # Attachment note (after examples)
        if attachments:
            attachment_note = self._format_attachments(attachments)
            sections.append(attachment_note)

        return "\n\n".join(sections)

    def _format_tool_schemas(
        self,
        schemas: List[Dict[str, Any]],
        max_search_limit: int
    ) -> str:
        """Format tool schemas as human-readable text.

        Args:
            schemas: List of tool JSON schemas
            max_search_limit: Maximum search attempts

        Returns:
            Formatted tool descriptions
        """
        lines = ["## Available Tools", ""]
        lines.append("You have access to the following tools:\n")

        for schema in schemas:
            func = schema.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {}).get("properties", {})

            lines.append(f"### {name}")
            lines.append(f"{description}\n")

            if parameters:
                lines.append("**Parameters:**")
                for param_name, param_info in parameters.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    lines.append(f"- `{param_name}` ({param_type}): {param_desc}")
                lines.append("")

        # Add usage format
        lines.append("**Usage Format:**")
        lines.append('To call a tool, use: `<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>`')
        lines.append("")

        # Add limits
        lines.append("**Limits:**")
        lines.append(f"- Maximum search attempts: {max_search_limit}")
        lines.append("- Code execution: unlimited")

        return "\n".join(lines)

    def _format_attachments(self, attachments: List[str]) -> str:
        """Format attachment information.

        Args:
            attachments: List of attachment file names

        Returns:
            Formatted attachment note
        """
        lines = ["## Attached Files", ""]
        lines.append("The following files are attached to this question:")

        for attachment in attachments:
            lines.append(f"- {attachment}")

        lines.append("\nYou can inspect these files using the text_inspector or image_inspector tools.")

        return "\n".join(lines)

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
            # Fall back to old "example" key if present
            if "example" in template and template["example"]:
                return "## Example\n" + self._format_example(template["example"])
            return ""

        example = template[example_key]
        if not example or "question" not in example or "steps" not in example:
            return ""

        # Format example in multi-step format
        lines = ["## Example", ""]
        lines.append(f"Question: \"{example['question']}\"")
        lines.append("")

        for step_data in example["steps"]:
            step_num = step_data.get("step")

            if step_num == "final":
                # Final answer
                lines.append(f"**Answer:** {step_data.get('answer', '')}")
            else:
                # Reasoning step with tool call
                lines.append(f"**Step {step_num}:** {step_data.get('reasoning', '')}")
                lines.append("")

                if "tool_call" in step_data:
                    lines.append(f"<tool_call>{step_data['tool_call']}</tool_call>")
                    lines.append("")

                if "tool_response" in step_data:
                    lines.append(f"<tool_response>{step_data['tool_response']}</tool_response>")
                    lines.append("")

        return "\n".join(lines)

    def _format_example(self, example: Dict[str, Any]) -> str:
        """Format example from template (legacy format).

        Args:
            example: Example dictionary with question, reasoning, answer

        Returns:
            Formatted example text
        """
        lines = []

        if "question" in example:
            lines.append(f"**Question:** {example['question']}")
            lines.append("")

        if "reasoning" in example:
            lines.append(f"**Reasoning:**\n{example['reasoning']}")
            lines.append("")

        if "answer" in example:
            lines.append(f"**Answer:** {example['answer']}")

        return "\n".join(lines)

    def get_user_prompt(self, question: str) -> str:
        """Build user prompt for a question.

        Args:
            question: Question text

        Returns:
            User prompt
        """
        return question
