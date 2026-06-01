#!/usr/bin/env python3
"""Generate the accuracy-vs-size design space figure for the Goals chapter.

Shows existing North Sami OCR systems (TrOCR variants) and a dashed
target zone where the proposed lightweight model aims to land.
The proposed model is intentionally NOT plotted -- the Goals chapter
states the target; later chapters show whether it was hit.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Render all text through LaTeX so the figure's labels match the thesis body
# font (Computer Modern) exactly. fonttype=42 keeps the non-text glyphs as
# embedded TrueType instead of Type 3 bitmaps.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman"]

TARGET_SIZE_MIN_MB = 5
TARGET_SIZE_MAX_MB = 50
TARGET_ACC_MIN = 85.0
TARGET_ACC_MAX = 95.0


# Measured competitor numbers (mirror the macros in main.tex).
# (label, size_MB, char_accuracy_pct, latency_sec_per_image, annotation_offset)
COMPETITORS = [
    ("TrOCR (smi+synth)",      350, 91.56, 12.41, ( 10,  6)),
    ("TrOCR (smi+pred+synth)", 350, 84.15, 12.08, ( 10,  0)),
    ("TrOCR (smi only)",       350, 75.88, 16.23, ( 10, -6)),
]


# Colour palette (consistent with other figures in the report).
COLOR_TROCR   = "#B23A48"   # warm red for the heavyweight family
COLOR_TARGET  = "#1F6FB2"   # blue for the target zone
COLOR_AXIS    = "#404040"
COLOR_GRID    = "#E0E0E0"


def generate_design_space(output_dir: Path) -> None:
    # Match the rendered width (0.8 * \textwidth ≈ 5.2 in) so LaTeX does not
    # downscale the figure -- otherwise every fontsize gets multiplied by ~0.74
    # and text starts to look soft.
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=300)

    # --- target zone (drawn first so points sit on top) ---------------------
    target = Rectangle(
        (TARGET_SIZE_MIN_MB, TARGET_ACC_MIN),
        TARGET_SIZE_MAX_MB - TARGET_SIZE_MIN_MB,
        TARGET_ACC_MAX - TARGET_ACC_MIN,
        linewidth=1.8,
        linestyle=(0, (6, 4)),
        edgecolor=COLOR_TARGET,
        facecolor=COLOR_TARGET,
        alpha=0.10,
        zorder=1,
    )
    ax.add_patch(target)

    target_cx = (TARGET_SIZE_MIN_MB * TARGET_SIZE_MAX_MB) ** 0.5  # log-mid
    ax.text(
        target_cx, (TARGET_ACC_MIN + TARGET_ACC_MAX) / 2,
        r"Target zone" "\n" r"(lightweight $\times$ competitive)",
        ha="center", va="center",
        fontsize=9, fontweight="bold",
        color=COLOR_TARGET,
        zorder=2,
    )

    # --- competitor points --------------------------------------------------
    for label, size_mb, acc, latency, (dx, dy) in COMPETITORS:
        ax.scatter(
            size_mb, acc,
            s=140, marker="o",
            facecolor=COLOR_TROCR, edgecolor="white", linewidth=1.2,
            zorder=4,
        )
        ax.annotate(
            f"{label}\n{latency:.1f}s / image",
            xy=(size_mb, acc),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8, color=COLOR_AXIS,
            ha="left", va="center",
            zorder=5,
        )

    # --- axes ---------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(3, 800)
    ax.set_ylim(70, 96)
    ax.set_xlabel("Model size (MB, log scale)", fontsize=11, color=COLOR_AXIS)
    ax.set_ylabel(r"Character accuracy (\%)", fontsize=11, color=COLOR_AXIS)

    ax.grid(True, which="major", linestyle=":", color=COLOR_GRID, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)

    # log-scale tick labels readable as MB rather than 10^x
    ax.set_xticks([5, 10, 50, 100, 350, 700])
    ax.set_xticklabels(["5", "10", "50", "100", "350", "700"])

    # --- annotation: phone-deployable band ----------------------------------
    ax.axvline(100, color=COLOR_AXIS, linewidth=0.6, linestyle="--", alpha=0.4, zorder=0)
    ax.text(
        95, 71.2, r"phone-deployable  $\rightarrow$  $\leftarrow$  server-only",
        fontsize=8.5, color=COLOR_AXIS, alpha=0.7,
        ha="center", va="bottom",
    )

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = output_dir / f"design_space.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"  wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    plt.close(fig)


def main() -> None:
    figures_dir = Path(__file__).resolve().parents[1]
    generate_design_space(figures_dir)


if __name__ == "__main__":
    main()
