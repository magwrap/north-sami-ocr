#!/usr/bin/env python3
"""Visualize OCR benchmark results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_results(results_dir: str) -> pd.DataFrame:
    """Load benchmark results from JSON."""
    json_path = Path(results_dir) / "benchmark_results.json"
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for r in data:
        if r["status"] == "success":
            m = r["metrics"]
            row = {
                "model": r["model"],
                "cer": m["cer"] * 100,
                "wer": m["wer"] * 100,
                "accuracy": m["accuracy"] * 100,
                "normalized_accuracy": m.get("normalized_accuracy", -1) * 100 if m.get("normalized_accuracy", -1) >= 0 else -1,
                "samples": r["num_samples"],
                "hf_count": r.get("hf_count", 0),
                "book_count": r.get("book_count", 0)
            }

            # Add per-dataset metrics if available
            for prefix in ['hf', 'book']:
                for metric in ['cer', 'wer', 'accuracy', 'normalized_accuracy']:
                    key = f"{prefix}_{metric}"
                    val = m.get(key, -1)
                    row[key] = val * 100 if val >= 0 else -1

            rows.append(row)

    return pd.DataFrame(rows).sort_values("cer")


def plot_metrics(df: pd.DataFrame, output_dir: str):
    """Generate bar charts for CER, WER, and accuracy."""
    output_path = Path(output_dir)

    # Filter to models with CER < 50% for readability
    df_good = df[df["cer"] < 50].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # CER
    axes[0].barh(df_good["model"], df_good["cer"], color="steelblue")
    axes[0].set_xlabel("CER (%)")
    axes[0].set_title("Character Error Rate (lower is better)")
    axes[0].invert_yaxis()

    # WER
    axes[1].barh(df_good["model"], df_good["wer"], color="coral")
    axes[1].set_xlabel("WER (%)")
    axes[1].set_title("Word Error Rate (lower is better)")
    axes[1].invert_yaxis()

    # Accuracy
    axes[2].barh(df_good["model"], df_good["accuracy"], color="seagreen")
    axes[2].set_xlabel("Accuracy (%)")
    axes[2].set_title("Exact Match Accuracy (higher is better)")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path / "benchmark_metrics.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path / 'benchmark_metrics.png'}")
    plt.close()


def plot_comparison(df: pd.DataFrame, output_dir: str):
    """Generate grouped bar chart comparing all models."""
    output_path = Path(output_dir)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    width = 0.25

    ax.bar([i - width for i in x], df["cer"], width, label="CER", color="steelblue")
    ax.bar(x, df["wer"], width, label="WER", color="coral")
    ax.bar([i + width for i in x], df["accuracy"], width, label="Accuracy", color="seagreen")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("OCR Benchmark: Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path / "benchmark_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path / 'benchmark_comparison.png'}")
    plt.close()


def plot_per_dataset_comparison(df: pd.DataFrame, output_dir: str):
    """Generate per-dataset comparison charts (HF vs Book)."""
    output_path = Path(output_dir)

    # Filter to models with per-dataset data
    df_filtered = df[(df["hf_count"] > 0) & (df["book_count"] > 0) & (df["hf_cer"] >= 0)].copy()

    if len(df_filtered) == 0:
        print("No models with per-dataset metrics, skipping per_dataset_comparison.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Dataset Comparison: HF Validation vs Book", fontsize=14, fontweight="bold")

    x = range(len(df_filtered))
    width = 0.35

    # CER
    axes[0, 0].bar([i - width/2 for i in x], df_filtered["hf_cer"], width, label="HF Validation", color="steelblue")
    axes[0, 0].bar([i + width/2 for i in x], df_filtered["book_cer"], width, label="Book", color="coral")
    axes[0, 0].set_ylabel("CER (%)")
    axes[0, 0].set_title("Character Error Rate (lower is better)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df_filtered["model"], rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    # WER
    axes[0, 1].bar([i - width/2 for i in x], df_filtered["hf_wer"], width, label="HF Validation", color="steelblue")
    axes[0, 1].bar([i + width/2 for i in x], df_filtered["book_wer"], width, label="Book", color="coral")
    axes[0, 1].set_ylabel("WER (%)")
    axes[0, 1].set_title("Word Error Rate (lower is better)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df_filtered["model"], rotation=45, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Exact Accuracy
    axes[1, 0].bar([i - width/2 for i in x], df_filtered["hf_accuracy"], width, label="HF Validation", color="steelblue")
    axes[1, 0].bar([i + width/2 for i in x], df_filtered["book_accuracy"], width, label="Book", color="coral")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_title("Exact Match Accuracy (higher is better)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df_filtered["model"], rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Normalized Accuracy
    axes[1, 1].bar([i - width/2 for i in x], df_filtered["hf_normalized_accuracy"], width, label="HF Validation", color="steelblue")
    axes[1, 1].bar([i + width/2 for i in x], df_filtered["book_normalized_accuracy"], width, label="Book", color="coral")
    axes[1, 1].set_ylabel("Normalized Accuracy (%)")
    axes[1, 1].set_title("Normalized Accuracy (higher is better)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(df_filtered["model"], rotation=45, ha="right")
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "per_dataset_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path / 'per_dataset_comparison.png'}")
    plt.close()


def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Generate comparison between exact and normalized accuracy."""
    output_path = Path(output_dir)

    # Filter to models with normalized accuracy data
    df_filtered = df[df["normalized_accuracy"] >= 0].copy()

    if len(df_filtered) == 0:
        print("No models with normalized accuracy, skipping accuracy_comparison.png")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df_filtered))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], df_filtered["accuracy"], width, label="Exact Match", color="seagreen")
    bars2 = ax.bar([i + width/2 for i in x], df_filtered["normalized_accuracy"], width, label="Normalized", color="mediumseagreen")

    # Annotate significant deltas (>1%)
    for i, (exact, norm) in enumerate(zip(df_filtered["accuracy"], df_filtered["normalized_accuracy"])):
        delta = norm - exact
        if delta > 1:
            ax.text(i, max(exact, norm) + 1, f"+{delta:.1f}%", ha='center', va='bottom', fontsize=8, color='darkgreen')

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison: Exact Match vs Normalized")
    ax.set_xticks(x)
    ax.set_xticklabels(df_filtered["model"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path / 'accuracy_comparison.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize OCR benchmark results")
    parser.add_argument("--results-dir", default="src/ocr/results", help="Results directory")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} model results\n")
    print(df.to_string(index=False))
    print()

    plot_metrics(df, args.results_dir)
    plot_comparison(df, args.results_dir)
    plot_per_dataset_comparison(df, args.results_dir)
    plot_accuracy_comparison(df, args.results_dir)


if __name__ == "__main__":
    main()
