"""Example: image_inspector tool — sub-agent mode.

A test PNG (a simple bar chart) is generated on the fly, then the model is
asked specific questions about its contents that can only be answered by
analysing the image with the VLM.

NOTE: image_inspector is only available in sub-agent mode (requires a VLM).

Run from msc-thesis/:
    python examples/example_image_inspector.py

The test image is regenerated each run; to inspect it manually:
    python examples/fixtures/make_test_image.py
"""

import sys
from pathlib import Path

from _common import (
    DEFAULT_CONFIG,
    build_model_providers,
    build_orchestrator,
    build_system_prompt,
    build_tools,
    print_summary,
    save_result,
)
from agent_engine.caching import CacheManager
from agent_engine.config import load_experiment_config
from agent_engine.utils import set_seed, setup_logging

OUTPUT_DIR = Path(__file__).parent.parent / "experiments/results/examples/image_inspector"
IMAGE_PATH = Path(__file__).parent / "fixtures" / "test_chart.png"


def _create_test_image(path: Path):
    """Generate a simple bar-chart PNG without external fonts."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise SystemExit("Pillow is required: pip install Pillow")

    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 480, 320
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((20, 10), "GREC 2023 - Energy Generation by Source (GWh)", fill=(30, 30, 30))

    bars = [
        ("Wind",  498, (70, 130, 180)),
        ("Solar", 187, (255, 165, 0)),
        ("Hydro",  56, (60, 179, 113)),
    ]
    bar_w, bar_gap = 80, 40
    x0 = 60
    y_top, y_bot = 50, 260
    scale = (y_bot - y_top) / 550.0

    for i, (label, value, color) in enumerate(bars):
        x = x0 + i * (bar_w + bar_gap)
        bar_h = int(value * scale)
        draw.rectangle([x, y_bot - bar_h, x + bar_w, y_bot], fill=color, outline=(50, 50, 50))
        draw.text((x + 4, y_bot - bar_h - 18), str(value), fill=(30, 30, 30))
        draw.text((x + 8, y_bot + 6), label, fill=(30, 30, 30))

    draw.line([(x0 - 5, y_bot), (x0 + 3 * (bar_w + bar_gap), y_bot)], fill=(50, 50, 50), width=2)
    img.save(path)
    return path


# Questions that require reading the bar chart visually.
QUESTION = (
    f"The image at '{IMAGE_PATH}' is a bar chart showing energy generation in GWh "
    "for three sources at Greenleaf Renewable Energy Cooperative in 2023.\n"
    "Use the image_inspector tool to analyse the chart and answer ALL of the following:\n"
    "1. Which energy source generated the most electricity, and roughly how many GWh?\n"
    "2. Which source generated the least, and roughly how many GWh?\n"
    "3. What is the approximate combined total GWh shown in the chart?\n"
    "4. What colour is used for the Solar bar?"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=OUTPUT_DIR / "example.log")
    logger.info("=== Example: image_inspector (sub-agent mode) ===")

    # Create the test image (idempotent, overwrites previous run).
    _create_test_image(IMAGE_PATH)
    logger.info(f"Test image: {IMAGE_PATH}")

    config = load_experiment_config(DEFAULT_CONFIG)
    set_seed(config.seed)

    cache_manager = CacheManager(
        config.cache_dir,
        web_tool_provider=config.tools.web_tool_provider,
        dataset_name=config.dataset.name,
    )
    enabled = ["image_inspector"]
    orchestrator_model, providers, _ = build_model_providers(config, required_roles=enabled)
    tools = build_tools(config, cache_manager, providers, enabled_tools=enabled)
    system_prompt = build_system_prompt(config, tools)
    orchestrator = build_orchestrator(config, orchestrator_model, tools, cache_manager=cache_manager)

    logger.info(f"Question: {QUESTION}")
    try:
        state = orchestrator.run(
            question=QUESTION,
            question_id=0,
            system_prompt=system_prompt,
            attachments=[str(IMAGE_PATH)],
        )
        result_path = save_result(OUTPUT_DIR, state, config)
        print_summary(logger, state, config, result_path)
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
