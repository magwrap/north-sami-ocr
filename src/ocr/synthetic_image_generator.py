"""Synthetic image generator for OCR benchmarking.

Generates clean synthetic images from text using PIL for testing OCR pipelines
on ground truth data without requiring real scanned documents.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont


def find_sami_font() -> Optional[str]:
    """
    Find a font that supports Sami special characters.

    Uses fontconfig (fc-match) for cross-platform font discovery.
    Sami characters to support: Á Â Ä Å Ï á â ä å æ ï ö ø Č č Đ đ Š š Ŧ ŧ Ŋ ŋ ž

    Returns:
        Path to font file or None if not found
    """
    # Try using fontconfig to find fonts (works on Linux, macOS)
    font_candidates = [
        "DejaVu Sans",
        "Liberation Sans",
        "Noto Sans",
        "Arial",
        "sans-serif"  # Fallback to default sans-serif
    ]

    for font_name in font_candidates:
        try:
            result = subprocess.run(
                ["fc-match", font_name, "file"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Parse output like ":file=/path/to/font.ttf"
                output = result.stdout.strip()
                if ":file=" in output:
                    font_path = output.split(":file=")[1]
                    if Path(font_path).exists():
                        return font_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # Manual fallback for common paths
    fallback_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]

    for font_path in fallback_paths:
        if Path(font_path).exists():
            return font_path

    return None


def generate_image(
    text: str,
    font_size: int = 32,
    padding: int = 20,
    max_width: int = 800,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a synthetic image from text.

    Creates a clean, single-line image with black text on white background,
    suitable for OCR model benchmarking.

    Args:
        text: Text to render in the image
        font_size: Font size in pixels (default: 32)
        padding: Padding around text in pixels (default: 20)
        max_width: Maximum image width before wrapping (default: 800)
        background_color: RGB tuple for background (default: white)
        text_color: RGB tuple for text (default: black)
        font_path: Path to font file (auto-detected if None)

    Returns:
        PIL Image object containing rendered text

    Example:
        >>> img = generate_image("Mon lean okta sápmelaš.")
        >>> img.save("output.png")
    """
    # Load font
    if font_path is None:
        font_path = find_sami_font()

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            # Fallback to default if loading fails
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Create temporary image to measure text size
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    temp_bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = temp_bbox[2] - temp_bbox[0]
    text_height = temp_bbox[3] - temp_bbox[1]

    # Create final image with calculated dimensions
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # Ensure minimum dimensions
    img_width = max(img_width, 100)
    img_height = max(img_height, 40)

    # Create final image
    image = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw text centered vertically, left-aligned horizontally
    x_pos = padding
    y_pos = padding
    draw.text((x_pos, y_pos), text, font=font, fill=text_color)

    return image


def generate_multiline_image(
    text: str,
    font_size: int = 32,
    padding: int = 20,
    max_width: int = 800,
    line_spacing: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None
) -> Image.Image:
    """
    Generate a synthetic image with text wrapping for long paragraphs.

    Automatically wraps text to fit within max_width, useful for longer
    ground truth texts.

    Args:
        text: Text to render (may contain multiple sentences)
        font_size: Font size in pixels
        padding: Padding around text in pixels
        max_width: Maximum image width before wrapping
        line_spacing: Extra spacing between lines in pixels
        background_color: RGB tuple for background
        text_color: RGB tuple for text
        font_path: Path to font file (auto-detected if None)

    Returns:
        PIL Image object with multi-line text
    """
    # Load font
    if font_path is None:
        font_path = find_sami_font()

    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    # Create temporary image for measurements
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    # Word wrapping: split text into lines that fit within max_width
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = temp_draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]

        if test_width <= (max_width - 2 * padding):
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                # Single word too long, add it anyway
                lines.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    # Calculate image dimensions
    line_heights = []
    max_line_width = 0

    for line in lines:
        bbox = temp_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)
        max_line_width = max(max_line_width, line_width)

    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)

    img_width = max_line_width + 2 * padding
    img_height = total_height + 2 * padding

    # Ensure minimum dimensions
    img_width = max(img_width, 100)
    img_height = max(img_height, 40)

    # Create final image
    image = Image.new("RGB", (img_width, img_height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw each line
    y_pos = padding
    for i, line in enumerate(lines):
        draw.text((padding, y_pos), line, font=font, fill=text_color)
        y_pos += line_heights[i] + line_spacing

    return image


def generate_from_ground_truth(
    ground_truth_path: str,
    output_dir: str,
    font_size: int = 32,
    max_width: int = 800,
    use_multiline: bool = False
) -> Dict[str, str]:
    """
    Generate synthetic images from ground truth JSON file.

    Args:
        ground_truth_path: Path to ground_truth.json file
        output_dir: Directory to save generated images
        font_size: Font size for rendering
        max_width: Maximum image width before wrapping
        use_multiline: Use multiline rendering for long texts

    Returns:
        Dictionary mapping entry IDs to generated image paths

    Example:
        >>> paths = generate_from_ground_truth(
        ...     "test_data/account_of_sami/ground_truth.json",
        ...     "test_data/account_of_sami/synthetic_images"
        ... )
        >>> print(f"Generated {len(paths)} images")
    """
    # Load ground truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate images for each entry
    image_paths = {}
    font_path = find_sami_font()

    print(f"Using font: {font_path}")
    print(f"Generating images in: {output_path}")
    print()

    for entry_id, entry_data in data.items():
        # Skip the "length" metadata field
        if entry_id == "length":
            continue

        # Get the Sami text (original)
        sami_text = entry_data.get("original", "")
        if not sami_text:
            continue

        # Generate image
        if use_multiline and len(sami_text) > 60:
            img = generate_multiline_image(
                sami_text,
                font_size=font_size,
                max_width=max_width,
                font_path=font_path
            )
        else:
            img = generate_image(
                sami_text,
                font_size=font_size,
                font_path=font_path
            )

        # Save image
        image_filename = f"entry_{entry_id}.png"
        image_path = output_path / image_filename
        img.save(image_path)

        image_paths[entry_id] = str(image_path)

        print(f"[{entry_id}] {sami_text[:50]}{'...' if len(sami_text) > 50 else ''}")
        print(f"       → {image_path} ({img.size[0]}x{img.size[1]})")

    print(f"\nGenerated {len(image_paths)} images")
    return image_paths


def test_generator():
    """Test image generation with ground truth data."""
    print("Testing synthetic image generator...")
    font_path = find_sami_font()
    print(f"Font path: {font_path}")
    print()

    # Check if ground truth exists
    gt_path = Path("test_data/account_of_sami/ground_truth.json")
    if gt_path.exists():
        print("Generating images from ground truth...")
        image_paths = generate_from_ground_truth(
            ground_truth_path=str(gt_path),
            output_dir="test_data/account_of_sami/synthetic_images",
            font_size=32,
            max_width=800
        )
        print(f"\nSuccess! Images saved to: test_data/account_of_sami/synthetic_images/")
    else:
        print(f"Ground truth not found at: {gt_path}")
        print("Testing with sample texts instead...")
        print()

        # Test cases with varying complexity
        test_texts = [
            "Mon lean okta sápmelaš.",  # Simple
            "Čilgehus dasa manin galgá okta siida atnit guovtti sajis bohccuid.",  # Medium
            "Ja áldoeallu lea váralaš ; dan galggalii sáhttit nu siivvut reainnidit go lea máŧolaš.",  # Long
        ]

        output_dir = Path("test_output/synthetic_images")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, text in enumerate(test_texts, 1):
            # Single-line version
            img = generate_image(text, font_path=font_path)
            output_path = output_dir / f"test_{i}_single.png"
            img.save(output_path)
            print(f"Generated: {output_path}")
            print(f"  Text: {text[:50]}...")
            print(f"  Size: {img.size}")

            # Multi-line version (for long text)
            if len(text) > 50:
                img_multi = generate_multiline_image(text, max_width=400, font_path=font_path)
                output_path_multi = output_dir / f"test_{i}_multi.png"
                img_multi.save(output_path_multi)
                print(f"Generated: {output_path_multi}")
                print(f"  Size: {img_multi.size}")

            print()

        print(f"Test images saved to: {output_dir}")


if __name__ == "__main__":
    test_generator()
