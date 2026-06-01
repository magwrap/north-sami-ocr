#!/usr/bin/env python3
"""Generate figures for the SCAI paper from benchmark data."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import jiwer

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "src" / "ocr" / "results" / "benchmark_results.json"
PREDICTIONS_DIR = PROJECT_ROOT / "src" / "ocr" / "results" / "predictions"
PIPELINE_RESULTS_DIR = PROJECT_ROOT / "pipeline_benchmark_results"
FIGURES_DIR = SCRIPT_DIR


def load_benchmark_results():
    """Load benchmark results from JSON file."""
    if not BENCHMARK_PATH.exists():
        print(f"Warning: {BENCHMARK_PATH} not found. Using fallback data.")
        return None
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def load_pipeline_results(model_name: str = "ctc_simple"):
    """Load per-sample results from the latest pipeline benchmark run.

    Args:
        model_name: Model name to filter results (default: ctc_simple)

    Returns:
        List of per-sample results with id, cer, ground_truth_sami, etc.
    """
    if not PIPELINE_RESULTS_DIR.exists():
        print(f"Warning: {PIPELINE_RESULTS_DIR} not found.")
        return None

    # Find all results for the specified model, sorted by timestamp (newest first)
    model_dirs = sorted(
        [d for d in PIPELINE_RESULTS_DIR.iterdir()
         if d.is_dir() and model_name in d.name],
        reverse=True
    )

    if not model_dirs:
        print(f"Warning: No pipeline results found for {model_name}")
        return None

    # Load the most recent results
    results_path = model_dirs[0] / "results.json"
    if not results_path.exists():
        print(f"Warning: {results_path} not found")
        return None

    print(f"Loading pipeline results from: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    return data.get("per_sample_results", [])


def load_predictions_from_jsonl(model_name: str = "ctc_simple"):
    """Load per-sample predictions from JSONL files in src/ocr/results/predictions.

    Args:
        model_name: Model name (e.g., "ctc_simple", "trocr_smi")

    Returns:
        List of dicts with id, prediction, reference, cer, word_count
    """
    jsonl_path = PREDICTIONS_DIR / f"{model_name}.jsonl"

    if not jsonl_path.exists():
        print(f"Warning: {jsonl_path} not found")
        return None

    print(f"Loading predictions from: {jsonl_path}")

    results = []
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line.strip())

            # Compute CER
            prediction = sample.get("prediction", "")
            reference = sample.get("reference", "")

            cer = jiwer.cer(reference, prediction) if reference else 0.0
            word_count = len(reference.split())

            results.append({
                "id": sample.get("id", ""),
                "prediction": prediction,
                "reference": reference,
                "cer": cer,
                "word_count": word_count
            })

    print(f"Loaded {len(results)} samples from {model_name}")
    return results


def generate_sami_alphabet_figure():
    """Generate figure showing Northern Sámi special characters."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # Northern Sámi special characters
    special_chars = [
        ("Á/á", "áhkku (grandmother)"),
        ("Č/č", "čáhci (water)"),
        ("Đ/đ", "ođđa (new)"),
        ("Ŋ/ŋ", "eaŋgals (english)"),
        ("Š/š", "šaldi (bridge)"),
        ("Ŧ/ŧ", "máŧolaš (possible)"),
        ("Ž/ž", "iežá (other)"),
    ]

    # Layout
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 8.0, "Northern Sámi Special Characters", fontsize=14, fontweight='bold',
            ha='center', va='center')

    # Draw character boxes
    for i, (char, example) in enumerate(special_chars):
        row = i // 4
        col = i % 4
        x = 1.2 + col * 2.2
        y = 5.5 - row * 2.5

        # Character box
        rect = mpatches.FancyBboxPatch((x, y), 1.5, 1.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#e8f4f8',
                                        edgecolor='#2c5282',
                                        linewidth=1.5)
        ax.add_patch(rect)

        # Character
        ax.text(x + 0.75, y + 1.1, char, fontsize=18, fontweight='bold',
                ha='center', va='center', color='#2c5282')

        # Example word (smaller)
        ax.text(x + 0.75, y + 0.35, example, fontsize=7,
                ha='center', va='center', color='#4a5568', style='italic')

    # Add note about Latin base
    # ax.text(5,2, "Based on Latin alphabet with additional diacritics for unique Sámi phonemes",
    #         fontsize=9, ha='center', va='center', color='#718096')

    plt.tight_layout()
    output_path = FIGURES_DIR / 'sami_alphabet.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {output_path}")


def generate_sentence_length_analysis():
    """Generate scatter plot showing character accuracy vs sentence length."""
    # Load data from JSONL predictions
    per_sample = load_predictions_from_jsonl("ctc_simple")

    if per_sample is None:
        print("Warning: No prediction data found. Using fallback data.")
        # Fallback hardcoded data (now using accuracy = 100 - CER)
        data = [
            {"id": "1", "words": 4, "accuracy": 100.0, "quality": "Perfect"},
            {"id": "91", "words": 14, "accuracy": 100.0, "quality": "Perfect"},
            {"id": "997", "words": 36, "accuracy": 51.2, "quality": "Failed"},
            {"id": "3052", "words": 48, "accuracy": 35.7, "quality": "Failed"},
        ]
    else:
        # Process loaded data: already has cer and word_count
        data = []
        for sample in per_sample:
            cer_val = sample.get("cer", 0) * 100  # Convert to percentage
            accuracy = 100 - cer_val  # Character accuracy = 100 - CER

            # Categorize quality based on accuracy
            if accuracy == 100:
                quality = "Perfect"
            elif accuracy >= 95:
                quality = "Excellent"
            elif accuracy >= 80:
                quality = "Good"
            else:
                quality = "Poor"

            data.append({
                "id": sample.get("id", ""),
                "words": sample.get("word_count", 0),
                "accuracy": accuracy,
                "quality": quality
            })

        print(f"Processed {len(data)} samples for sentence length analysis")

    words = np.array([d["words"] for d in data])
    accuracy = np.array([d["accuracy"] for d in data])

    # Color by quality (green for high accuracy, red for low)
    colors = []
    for d in data:
        if d["quality"] == "Perfect":
            colors.append('#38a169')  # green
        elif d["quality"] == "Excellent":
            colors.append('#d69e2e')  # yellow
        elif d["quality"] == "Good":
            colors.append('#dd6b20')  # orange
        else:
            colors.append('#e53e3e')  # red

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Scatter plot
    scatter = ax.scatter(words, accuracy, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1)

    # Trend line (polynomial fit)
    z = np.polyfit(words, accuracy, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(words), max(words), 100)
    ax.plot(x_trend, p(x_trend), 'r--', alpha=0.6, linewidth=2, label='Trend')

    # Labels
    ax.set_xlabel('Sentence Length (words)', fontsize=11)
    ax.set_ylabel('Character Accuracy (%)', fontsize=11)
    ax.set_title('OCR Character Accuracy vs. Sentence Length', fontsize=12, fontweight='bold')

    # Custom legend (inverted - high accuracy is good)
    legend_elements = [
        mpatches.Patch(facecolor='#38a169', edgecolor='white', label='Perfect (100% Acc.)'),
        mpatches.Patch(facecolor='#d69e2e', edgecolor='white', label='Excellent (≥95%)'),
        mpatches.Patch(facecolor='#dd6b20', edgecolor='white', label='Good (≥80%)'),
        mpatches.Patch(facecolor='#e53e3e', edgecolor='white', label='Poor (<80%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(30, 105)  # Inverted range for accuracy
    ax.set_xlim(0, 55)

    # Annotation for threshold
    ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5)
    ax.text(52, 79, '80% threshold', fontsize=8, color='gray', va='top', ha='right')

    plt.tight_layout()
    output_path = FIGURES_DIR / 'sentence_length_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {output_path}")


def generate_multi_model_sentence_length_comparison():
    """Compare top 3 models (trocr_smi_synth, trocr_smi_pred_synth, ctc_simple) on sentence length."""
    models = [
        ("trocr_smi_synth", "TrOCR Smi+Synth", '#2c5282'),
        ("trocr_smi_pred_synth", "TrOCR Smi+Pred+Synth", '#38a169'),
        ("ctc_simple", "CTC Simple", '#d69e2e'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for model_name, label, color in models:
        # Load predictions
        samples = load_predictions_from_jsonl(model_name)
        if samples is None:
            print(f"Skip {model_name}: no data")
            continue

        # Extract word count and accuracy
        words = []
        accuracy = []
        for s in samples:
            cer_val = s.get("cer", 0) * 100
            acc = 100 - cer_val
            words.append(s.get("word_count", 0))
            accuracy.append(acc)

        # Scatter plot
        ax.scatter(words, accuracy, label=label, alpha=0.6, s=40, color=color, edgecolors='white', linewidth=0.5)

        # Trend line
        words_arr = np.array(words)
        acc_arr = np.array(accuracy)
        if len(words_arr) > 2:
            z = np.polyfit(words_arr, acc_arr, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(words_arr), max(words_arr), 100)
            ax.plot(x_trend, p(x_trend), '--', alpha=0.7, linewidth=2, color=color)

    # Labels
    ax.set_xlabel('Sentence Length (words)', fontsize=11)
    ax.set_ylabel('Character Accuracy (%)', fontsize=11)
    ax.set_title('Model Comparison: Accuracy vs. Sentence Length', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-5, 105)
    ax.set_xlim(0, 55)

    plt.tight_layout()
    output_path = FIGURES_DIR / 'model_comparison_sentence_length.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {output_path}")


def generate_benchmark_comparison():
    """Generate bar chart comparing OCR model performance from benchmark data."""
    results = load_benchmark_results()

    if results is None:
        # Fallback data if benchmark file not found
        results = [
            {"model": "ctc_simple", "metrics": {"cer": 0.1228, "wer": 0.3755}},
            {"model": "ctc_vgg16", "metrics": {"cer": 0.7375, "wer": 0.9891}},
            {"model": "ctc_vgg19", "metrics": {"cer": 0.7162, "wer": 0.9797}},
            {"model": "ctc_resnet50", "metrics": {"cer": 0.7501, "wer": 0.9876}},
        ]

    # Filter successful models only
    results = [r for r in results if r.get("status") == "success" or "metrics" in r]

    # Extract data
    models = [r["model"] for r in results]
    cer_values = [r["metrics"]["cer"] * 100 for r in results]  # Convert to percentage
    wer_values = [r["metrics"]["wer"] * 100 for r in results]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Color palette
    colors = ['#2c5282', '#38a169', '#d69e2e', '#e53e3e', '#805ad5']
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]

    # CER subplot
    x = np.arange(len(models))
    bars1 = ax1.bar(x, cer_values, color=bar_colors, edgecolor='white', linewidth=1)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Character Error Rate (%)', fontsize=11)
    ax1.set_title('CER by Model', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, val in zip(bars1, cer_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # WER subplot
    bars2 = ax2.bar(x, wer_values, color=bar_colors, edgecolor='white', linewidth=1)
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('Word Error Rate (%)', fontsize=11)
    ax2.set_title('WER by Model', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, val in zip(bars2, wer_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = FIGURES_DIR / 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Generated: {output_path}")

    # Also print summary for quick reference
    print("\nBenchmark Summary:")
    print("-" * 50)
    for model, cer, wer in zip(models, cer_values, wer_values):
        print(f"  {model:20s}  CER: {cer:6.2f}%  WER: {wer:6.2f}%")


def generate_all():
    """Generate all figures."""
    print("Generating SCAI paper figures...")
    print(f"Reading benchmark data from: {BENCHMARK_PATH}")
    print()

    generate_sami_alphabet_figure()
    generate_sentence_length_analysis()
    generate_multi_model_sentence_length_comparison()
    generate_benchmark_comparison()

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    generate_all()
