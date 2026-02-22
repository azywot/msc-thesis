"""Export prompts for fine-tuning data collection.

This script exports prompt templates and generates examples for fine-tuning.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_engine.prompts import PromptBuilder
from agent_engine.tools import (
    WebSearchTool,
    CodeGeneratorTool,
    ContextManagerTool,
    TextInspectorTool,
    ImageInspectorTool,
)


def export_all_prompts(output_path: Path):
    """Export all prompt templates.

    Args:
        output_path: Output JSON file path
    """
    print("Exporting prompt templates...")

    prompt_builder = PromptBuilder()

    export_data = {
        "system_prompts": {},
        "tool_schemas": {},
        "datasets": []
    }

    # Export system prompts for each dataset
    for dataset in ["base", "gaia", "gpqa", "math"]:
        try:
            template = prompt_builder.load_template(dataset)
            export_data["system_prompts"][dataset] = template
            export_data["datasets"].append(dataset)
            print(f"  Exported {dataset} template")
        except FileNotFoundError:
            print(f"  Skipped {dataset} (not found)")

    # Export tool schemas
    tools = [
        ("web_search", WebSearchTool(serper_api_key="dummy", search_cache={})),
        ("code_generator", CodeGeneratorTool()),
        ("context_manager", ContextManagerTool()),
        ("text_inspector", TextInspectorTool()),
        ("image_inspector", ImageInspectorTool()),
    ]

    for tool_name, tool in tools:
        schema = tool.get_schema()
        export_data["tool_schemas"][tool_name] = schema
        print(f"  Exported {tool_name} schema")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nPrompts exported to: {output_path}")
    print(f"  System prompts: {len(export_data['system_prompts'])}")
    print(f"  Tool schemas: {len(export_data['tool_schemas'])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export prompts for fine-tuning")
    parser.add_argument("--output", "-o", type=str,
                       default="./experiments/prompts_export.json",
                       help="Output JSON file path")
    args = parser.parse_args()

    export_all_prompts(Path(args.output))


if __name__ == "__main__":
    main()
