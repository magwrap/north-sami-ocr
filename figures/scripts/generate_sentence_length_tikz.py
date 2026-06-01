#!/usr/bin/env python3
"""Generate TikZ scatter plot comparing sentence length vs accuracy across models."""

import json
from pathlib import Path
import numpy as np
import jiwer

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "src" / "ocr" / "results" / "predictions"

# Model configurations: (file_name, display_name, color_hex)
MODELS = [
    ("trocr_smi_synth", "TrOCR Smi+Synth", "2c5282"),      # Blue
    ("trocr_smi_pred_synth", "TrOCR Smi+Pred+Synth", "38a169"),  # Green
    ("ctc_simple", "CTC Simple", "d69e2e"),                 # Yellow
]


def load_predictions(model_name: str):
    """Load predictions and compute accuracy/word count pairs."""
    jsonl_path = PREDICTIONS_DIR / f"{model_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found")

    results = []
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line.strip())
            prediction = sample.get("prediction", "")
            reference = sample.get("reference", "")
            if not reference:
                continue
            cer = jiwer.cer(reference, prediction)
            word_count = len(reference.split())
            accuracy = max(0, 100 - cer * 100)  # Clamp to 0
            results.append((word_count, accuracy))

    return results


def compute_trend_coefficients(data):
    """Compute polynomial trend line coefficients (degree 2)."""
    words = np.array([d[0] for d in data])
    accuracy = np.array([d[1] for d in data])
    coeffs = np.polyfit(words, accuracy, 2)
    return coeffs


def generate_tikz(models_data):
    """Generate standalone TikZ document with multi-model comparison."""

    # Generate coordinate strings for each model
    def coords_str(points):
        return " ".join(f"({w},{a:.2f})" for w, a in points)

    # Generate trend line coordinates
    def trend_coords(coeffs, x_min=1, x_max=52):
        x_vals = np.linspace(x_min, x_max, 50)
        p = np.poly1d(coeffs)
        return " ".join(f"({x:.1f},{p(x):.2f})" for x in x_vals)

    # Build plot commands for each model
    scatter_plots = []
    trend_plots = []

    for model_name, display_name, color_hex in MODELS:
        data = models_data[model_name]["data"]
        coeffs = models_data[model_name]["coeffs"]
        color_name = f"color{model_name.replace('_', '')}"

        # Scatter plot
        scatter_plots.append(rf"""% {display_name} scatter
\addplot[
    only marks,
    mark=*,
    mark size=1.8pt,
    {color_name},
    opacity=0.6,
] coordinates {{ {coords_str(data)} }};
\addlegendentry{{{display_name}}}""")

        # Trend line
        trend_plots.append(rf"""% {display_name} trend
\addplot[
    {color_name},
    dashed,
    thick,
    opacity=0.8,
] coordinates {{ {trend_coords(coeffs)} }};""")

    # Color definitions
    color_defs = "\n".join(
        rf"\definecolor{{color{name.replace('_', '')}}}{{HTML}}{{{hex_code}}}"
        for name, _, hex_code in MODELS
    )

    tikz = rf"""\documentclass[tikz, border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

% Model colors
{color_defs}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=12cm,
    height=8cm,
    xlabel={{Sentence Length (words)}},
    ylabel={{Character Accuracy (\%)}},
    xlabel style={{font=\sffamily}},
    ylabel style={{font=\sffamily}},
    xticklabel style={{font=\sffamily}},
    yticklabel style={{font=\sffamily}},
    xmin=0, xmax=55,
    ymin=-5, ymax=105,
    xtick={{0,10,20,30,40,50}},
    ytick={{0,20,40,60,80,100}},
    grid=major,
    grid style={{gray!30, dashed}},
    title={{\textbf{{Model Comparison: Accuracy vs. Sentence Length}}}},
    title style={{font=\sffamily}},
    legend style={{
        at={{(0.02,0.02)}},
        anchor=south west,
        font=\sffamily\small,
        draw=gray!50,
        fill=white,
        fill opacity=0.9,
    }},
    clip=true,
]

{chr(10).join(scatter_plots)}

{chr(10).join(trend_plots)}

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    return tikz


def main():
    print("Loading predictions for all models...")
    models_data = {}

    for model_name, display_name, _ in MODELS:
        print(f"  Loading {display_name}...")
        data = load_predictions(model_name)
        coeffs = compute_trend_coefficients(data)
        models_data[model_name] = {"data": data, "coeffs": coeffs}
        print(f"    {len(data)} samples, coeffs: {coeffs}")

    print("\nGenerating TikZ...")
    tikz = generate_tikz(models_data)

    output_path = SCRIPT_DIR / "model_comparison_sentence_length.tex"
    with open(output_path, "w") as f:
        f.write(tikz)
    print(f"Generated: {output_path}")

    # Compile to PDF if pdflatex available
    import subprocess
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", output_path.name],
            cwd=SCRIPT_DIR,
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"Compiled: {output_path.with_suffix('.pdf')}")
        else:
            print("PDF compilation failed (pdflatex error)")
            print(result.stderr.decode()[-500:])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("PDF compilation skipped (pdflatex not available or timeout)")


if __name__ == "__main__":
    main()
