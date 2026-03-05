"""Prompt builder for constructing prompts from YAML templates.

Matches multi-agent-tools/scripts/prompts.py and prompt_manager.py behavior exactly:
- System prompt = instruction (tools + example + reminders)
- User prompt  = dataset-specific task wrapper (mirrors get_task_instruction_*())
- Callers merge instruction + user_prompt into a single user message (no system role)

Dataset → template routing:
  gaia, hle                              → gaia.yaml   (singleqa)
  nq, triviaqa                           → gaia.yaml   (singleqa)
  hotpotqa, musique, bamboogle, 2wiki    → multiqa.yaml
  math500, aime, amc                     → math.yaml
  gpqa                                   → gpqa.yaml
  everything else                        → base.yaml   (fallback)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Single-QA datasets (use gaia/singleqa template)
_SINGLEQA_DATASETS = frozenset({"gaia", "hle", "nq", "triviaqa"})

# Multi-hop QA datasets
_MULTIQA_DATASETS = frozenset({"hotpotqa", "musique", "bamboogle", "2wiki", "2wikimultihopqa"})

# Math datasets
_MATH_DATASETS = frozenset({"math500", "aime", "amc"})

# Extract-mode mapping: mirrors old evaluate_predictions() mode parameter
DATASET_EXTRACT_MODES: Dict[str, str] = {
    # 'gen' mode: GAIA, HLE, math
    "gaia": "gen",
    "hle": "gen",
    "math500": "gen",
    "aime": "gen",
    "amc": "gen",
    # 'choose' mode: GPQA
    "gpqa": "choose",
    "medmcqa": "choose",
    "pubhealth": "choose",
    # 'qa' mode: open QA datasets
    "nq": "qa",
    "triviaqa": "qa",
    "hotpotqa": "qa",
    "musique": "qa",
    "bamboogle": "qa",
    "2wiki": "qa",
    "2wikimultihopqa": "qa",
}


class PromptBuilder:
    """Builds prompts from YAML templates."""

    def __init__(self, template_dir: Optional[Path] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)
        self._templates: Dict[str, Dict[str, Any]] = {}

    def load_template(self, name: str) -> Dict[str, Any]:
        """Load a template from YAML file."""
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

    def _resolve_template_name(self, dataset_name: str) -> str:
        """Map dataset name to template file name.

        Mirrors old prompt_manager.py routing:
          nq/triviaqa/gaia/hle   → gaia   (singleqa)
          hotpotqa/musique/…     → multiqa
          math500/aime/amc       → math
          gpqa                   → gpqa
          else                   → base
        """
        dn = dataset_name.lower()
        if dn in _SINGLEQA_DATASETS:
            return "gaia"
        if dn in _MULTIQA_DATASETS:
            return "multiqa"
        if dn in _MATH_DATASETS:
            return "math"
        return dn  # will fall back to base if file not found

    def build_system_prompt(
        self,
        dataset_name: str,
        tool_schemas: List[Dict[str, Any]],
        max_search_limit: int = 10,
        direct_tool_call: bool = True
    ) -> str:
        """Build instruction text (system prompt portion) with tools and examples.

        Section order mirrors MAT: base_instruction → tools → example → final_instructions
        """
        template_name = self._resolve_template_name(dataset_name)
        try:
            template = self.load_template(template_name)
        except FileNotFoundError:
            logger.warning(f"Template '{template_name}' not found, using base template")
            template = self.load_template("base")

        sections = []

        if "base_instruction" in template:
            sections.append(template["base_instruction"].strip())

        if tool_schemas:
            sections.append(self._format_tool_schemas(tool_schemas, max_search_limit, direct_tool_call))

        example_text = self._select_and_format_example(template, tool_schemas, direct_tool_call)
        if example_text:
            sections.append(example_text)

        if "final_instructions" in template:
            sections.append(template["final_instructions"].strip())

        return "\n\n".join(sections)

    def build_user_prompt(self, dataset_name: str, question: str) -> str:
        """Build the per-question user-role text.

        Mirrors old get_task_instruction_openqa() / _math() / _multi_choice().
        This is concatenated with build_system_prompt() output by the caller to
        form a single user message (no system role).
        """
        dn = dataset_name.lower()

        if dn == "gpqa" or dn == "medmcqa":
            # Mirrors get_task_instruction_multi_choice() (non-qwq, non-llama branch)
            return (
                'Answer the following multiple choice question. You should think step by step to solve it.\n\n'
                'Provide your final choice in the format \\boxed{YOUR_CHOICE} where YOUR_CHOICE is one of A, B, C, or D.\n\n'
                f'Question:\n{question}\n\n'
            )

        if dn in _MATH_DATASETS:
            # Mirrors get_task_instruction_math() (non-qwq branch)
            return (
                'Please answer the following math question. You should think step by step to solve it.\n\n'
                'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
                f'Question:\n{question}\n\n'
            )

        # All other datasets (gaia, hle, nq, triviaqa, hotpotqa, musique, bamboogle, 2wiki, …)
        # Mirrors get_task_instruction_openqa() (non-qwq branch)
        return (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )

    def get_extract_mode(self, dataset_name: str) -> str:
        """Return the extract_answer mode for a dataset. Mirrors old mode routing."""
        return DATASET_EXTRACT_MODES.get(dataset_name.lower(), "gen")

    def _format_tool_schemas(
        self,
        schemas: List[Dict[str, Any]],
        max_search_limit: int,
        direct_tool_call: bool = False,
    ) -> str:
        """Format tool schemas using the Qwen3 XML <tools> format (matches MAT)."""
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

    def _select_and_format_example(
        self,
        template: Dict[str, Any],
        tool_schemas: List[Dict[str, Any]],
        direct_tool_call: bool
    ) -> str:
        """Select and format appropriate example based on tools and mode."""
        enabled_tools = {schema.get("function", {}).get("name", "") for schema in tool_schemas}
        has_search = "web_search" in enabled_tools
        has_code = "code_generator" in enabled_tools

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
