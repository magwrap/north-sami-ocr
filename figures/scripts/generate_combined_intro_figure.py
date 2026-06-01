#!/usr/bin/env python3
"""Generate combined figure showing Sámi alphabet and example sentence."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def generate_combined_figure(output_path: str = "combined_intro.png"):
    """
    Generate a combined figure with:
    - Top: Example sentence (Sámi + English)
    - Bottom: Sámi alphabet special characters with examples
    """
    # Create figure with two subplots (vertical layout)
    fig = plt.figure(figsize=(8, 2. ), dpi=300)

    # Create grid: top for example, bottom for alphabet
    gs = fig.add_gridspec(2, 1, height_ratios=[0.7, 1.5], hspace=0.1)

    # ===== TOP: Example Sentence =====
    ax_example = fig.add_subplot(gs[0])
    ax_example.axis('off')
    ax_example.set_xlim(0, 10)
    ax_example.set_ylim(0, 1.2)

    # Example text
    sami_text = "okta lea dáhpáhus go bođii dárrolaš njuorjovuopmegierragii"
    english_text = "It once happened that a Norwegian settler came to the uplands"

    # Sámi text (larger, blue) - positioned higher
    ax_example.text(5, 1.2, sami_text,
                   fontsize=11, fontweight='bold',
                   ha='center', va='center',
                   color='#003399',
                   family='DejaVu Sans')

    # English translation (smaller, gray, italic) - positioned lower with more spacing
    ax_example.text(5, 0.6, english_text,
                   fontsize=9, style='italic',
                   ha='center', va='center',
                   color='#404040',
                   family='DejaVu Sans')

    # ===== BOTTOM: Sámi Alphabet =====
    ax_alphabet = fig.add_subplot(gs[1])
    ax_alphabet.axis('off')
    ax_alphabet.set_xlim(0, 10)
    ax_alphabet.set_ylim(0, 5.5)

    # Northern Sámi special characters
    special_chars = [
        # ("Á/á", "áhkku"),
        # ("Č/č", "čáhci"),
        # ("Đ/đ", "ođđa"),
        # ("Ŋ/ŋ", "eaŋgals"),
        # ("Š/š", "šaldi"),
        # ("Ŧ/ŧ", "máŧolaš"),
        # ("Ž/ž", "iežá"),
        ("Á/á", "áhkku", "(grandmother)"),
        ("Č/č", "čáhci", "(water)"),
        ("Đ/đ", "ođđa", "(new)"),
        ("Ŋ/ŋ", "eaŋgals", "(english)"),
        ("Š/š", "šaldi", "(bridge)"),
        ("Ŧ/ŧ", "máŧolaš", "(possible)"),
        ("Ž/ž", "iežá", "(other)"),
        # ("Á/á", "áhkku (grandmother)"),
        # ("Č/č", "čáhci (water)"),
        # ("Đ/đ", "ođđa (new)"),
        # ("Ŋ/ŋ", "eaŋgals (english)"),
        # ("Š/š", "šaldi (bridge)"),
        # ("Ŧ/ŧ", "máŧolaš (possible)"),
        # ("Ž/ž", "iežá (other)"),
    ]

    # Draw character boxes in a single row with spacing between them
    num_chars = len(special_chars)
    box_width = 1.05  # Reduced from 1.25 to create gaps
    box_height = 2.9
    box_spacing = 0.30  # Increased from 0.15 for more spacing
    total_width = num_chars * box_width + (num_chars - 1) * box_spacing
    start_x = (10 - total_width) / 2

    for i, (char, example, translation) in enumerate(special_chars):
        x = start_x + i * (box_width + box_spacing)
        y = 2.5  # Moved up from 1.8 to avoid overlap with note

        # Character box
        rect = FancyBboxPatch((x, y), box_width, box_height,
                             boxstyle="round,pad=0.08",
                             facecolor='#e8f4f8',
                             edgecolor='#2c5282',
                             linewidth=1.2)
        ax_alphabet.add_patch(rect)

        # Character (larger, centered)
        ax_alphabet.text(x + box_width/2, y + 1.8, char,
                        fontsize=14, fontweight='bold',
                        ha='center', va='center',
                        color='#2c5282',
                        family='DejaVu Sans')

        # Example word (smaller, bottom)
        ax_alphabet.text(x + box_width/2, y + .9, example,
                        fontsize=7.5,
                        ha='center', va='center',
                        color='#4a5568',
                        style='italic',
                        family='DejaVu Sans')
        
        ax_alphabet.text(x + box_width/2, y + 0.35, translation,
                        fontsize=5.5,
                        ha='center', va='center',
                        color="#8792A2",
                        style='italic',
                        family='DejaVu Sans')

    # Add note
    note_text = "Seven diacritical characters distinguish Sámi from standard Latin alphabet"
    ax_alphabet.text(5, 1.4, note_text,
                    fontsize=9, ha='center', va='center',
                    color='#718096', style='italic')

    plt.tight_layout()

    # Save
    script_dir = Path(__file__).parent
    output_file = script_dir / output_path
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()

    print(f"✓ Generated combined figure: {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
    return output_file


if __name__ == "__main__":
    print("Generating combined introduction figure...")
    print()
    output = generate_combined_figure()
    print()
    print("✓ Figure ready for LaTeX!")
    print(f"  Use: \\includegraphics[width=\\columnwidth]{{figures/combined_intro.png}}")
