"""Helper script to generate the test image used by example_image_inspector.py.

Run once before the image inspector example:
    python examples/fixtures/make_test_image.py
"""
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")

OUT = Path(__file__).parent / "test_chart.png"

W, H = 480, 320
img = Image.new("RGB", (W, H), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# Title
draw.text((20, 12), "GREC 2023 — Energy Generation by Source (GWh)", fill=(30, 30, 30))

# Bar chart data
bars = [("Wind", 498, (70, 130, 180)), ("Solar", 187, (255, 165, 0)), ("Hydro", 56, (60, 179, 113))]
bar_w, bar_gap = 80, 40
x0 = 60
y_top, y_bot = 50, 260
scale = (y_bot - y_top) / 550  # 550 GWh → full height

for i, (label, value, color) in enumerate(bars):
    x = x0 + i * (bar_w + bar_gap)
    bar_h = int(value * scale)
    draw.rectangle([x, y_bot - bar_h, x + bar_w, y_bot], fill=color, outline=(50, 50, 50))
    draw.text((x + bar_w // 2 - 10, y_bot - bar_h - 18), f"{value}", fill=(30, 30, 30))
    draw.text((x + bar_w // 2 - 16, y_bot + 6), label, fill=(30, 30, 30))

# Y-axis label
draw.text((2, (y_top + y_bot) // 2 - 10), "GWh", fill=(80, 80, 80))
# Baseline
draw.line([(x0 - 5, y_bot), (x0 + 3 * (bar_w + bar_gap), y_bot)], fill=(50, 50, 50), width=2)

img.save(OUT)
print(f"Saved: {OUT}")
