#!/usr/bin/env python3
"""Generate styled figure for entry 80 with Sami text and English translation."""

import sys
from pathlib import Path
import subprocess
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path to import synthetic_image_generator
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "ocr"))
from synthetic_image_generator import find_sami_font


def find_bold_sami_font() -> Optional[str]:
    """Find a bold font that supports Sami characters."""
    # Try to find bold variants
    bold_candidates = [
        "DejaVu Sans:style=Bold",
        "DejaVu Sans Bold",
        "Liberation Sans:style=Bold",
        "Liberation Sans Bold",
        "Noto Sans:style=Bold",
        "Noto Sans Bold",
        "Arial Bold",
        "sans-serif:style=Bold"
    ]

    for font_name in bold_candidates:
        try:
            result = subprocess.run(
                ["fc-match", font_name, "file"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                if ":file=" in output:
                    font_path = output.split(":file=")[1]
                    if Path(font_path).exists():
                        return font_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # Fallback to regular font
    return find_sami_font()


def generate_bilingual_figure(
    sami_text: str,
    english_text: str,
    output_path: str,
    sami_color: tuple = (0, 51, 153),  # Darker, saturated blue
    english_color: tuple = (40, 40, 40),  # Much darker gray (almost black)
    font_size: int = 32,
    translation_font_size: int = 24,
    padding: int = 30,
    line_spacing: int = 15,
    background_color: tuple = (255, 255, 255)
):
    """
    Generate a bilingual figure with Sami text and English translation.

    Args:
        sami_text: Northern Sámi text (top)
        english_text: English translation (bottom)
        output_path: Where to save the PNG
        sami_color: RGB color for Sami text (default: darker blue)
        english_color: RGB color for English text (default: dark gray)
        font_size: Font size for Sami text
        translation_font_size: Font size for English translation
        padding: Padding around the image
        line_spacing: Space between Sami and English text
        background_color: RGB background color
    """
    # Find bold font that supports Sami characters
    font_path = find_bold_sami_font()

    if font_path:
        try:
            sami_font = ImageFont.truetype(font_path, font_size)
            english_font = ImageFont.truetype(font_path, translation_font_size)
            print(f"Using font: {font_path}")
        except Exception as e:
            print(f"Warning: Could not load font {font_path}: {e}")
            # Fallback to regular font
            font_path = find_sami_font()
            if font_path:
                sami_font = ImageFont.truetype(font_path, font_size)
                english_font = ImageFont.truetype(font_path, translation_font_size)
            else:
                sami_font = ImageFont.load_default()
                english_font = ImageFont.load_default()
    else:
        print("Warning: No Sami font found, using default")
        sami_font = ImageFont.load_default()
        english_font = ImageFont.load_default()

    # Create temporary image to measure text
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    # Measure Sami text
    sami_bbox = temp_draw.textbbox((0, 0), sami_text, font=sami_font)
    sami_width = sami_bbox[2] - sami_bbox[0]
    sami_height = sami_bbox[3] - sami_bbox[1]

    # Measure English text
    english_bbox = temp_draw.textbbox((0, 0), english_text, font=english_font)
    english_width = english_bbox[2] - english_bbox[0]
    english_height = english_bbox[3] - english_bbox[1]

    # Calculate image dimensions
    img_width = max(sami_width, english_width) + 2 * padding
    img_height = sami_height + english_height + line_spacing + 2 * padding

    # Ensure minimum dimensions
    img_width = max(img_width, 200)
    img_height = max(img_height, 80)

    # Create final image
    image = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw Sami text (centered horizontally)
    sami_x = (img_width - sami_width) // 2
    sami_y = padding
    draw.text((sami_x, sami_y), sami_text, font=sami_font, fill=sami_color)

    # Draw English text (centered horizontally, below Sami)
    english_x = (img_width - english_width) // 2
    english_y = sami_y + sami_height + line_spacing
    draw.text((english_x, english_y), english_text, font=english_font, fill=english_color)

    # Save image
    image.save(output_path)
    print(f"Generated: {output_path}")
    print(f"  Sami text: {sami_text}")
    print(f"  English: {english_text}")
    print(f"  Size: {img_width}x{img_height}")

    return image


def generate_entry_80():
    """Generate the specific figure for entry 80 used in the paper."""

    # Entry 80 from ground truth
    sami_text = "okta lea dáhpáhus go bođii dárrolaš njuorjovuopmegierragii gos ledje sámit orrume"
    english_text = "It once happened that a Norwegian settler came to the place where the Sámi were living at the Njuorjovuopme uplands"

    # Output path
    script_dir = Path(__file__).parent
    output_path = script_dir / "entry_80.png"

    print("Generating entry 80 figure for SCAI paper...")
    print()

    # Generate the figure with darker, more visible colors
    generate_bilingual_figure(
        sami_text=sami_text,
        english_text=english_text,
        output_path=str(output_path),
        sami_color=(0, 51, 153),  # Darker, saturated blue (RGB: Navy-ish)
        english_color=(30, 30, 30),  # Very dark gray (almost black)
        font_size=32,  # Larger for better visibility
        translation_font_size=24,  # Larger translation
        padding=35,
        line_spacing=20  # More spacing for clarity
    )

    print()
    print("✓ Figure ready for LaTeX!")
    print(f"  Use: \\includegraphics[width=\\textwidth]{{figures/entry_80.png}}")


if __name__ == "__main__":
    generate_entry_80()
