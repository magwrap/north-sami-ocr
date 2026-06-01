#!/usr/bin/env python3
"""
Translation Benchmark Visualization

Creates publication-quality plots from translation benchmark results:
- Metrics comparison (BLEU, chrF, TER) across modes
- Statistical significance visualization
- Per-sample analysis

Usage:
    python src/translation/visualize.py --result-dir src/translation/results/2026-04-21_230703_translation_benchmark
    python src/translation/visualize.py --result-dir src/translation/results/2026-04-21_230703_translation_benchmark --output plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize translation benchmark results"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to benchmark results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: result-dir/plots)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs"
    )
    return parser.parse_args()


def load_results(result_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load all benchmark results from directory.

    Args:
        result_dir: Path to results directory

    Returns:
        Tuple of (metadata, all_results, comparison)
    """
    with open(result_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    all_results = {}
    for mode in ['baseline', 'simple_split', 'full_split']:
        with open(result_dir / f'results_{mode}.json', 'r') as f:
            all_results[mode] = json.load(f)

    with open(result_dir / 'comparison.json', 'r') as f:
        comparison = json.load(f)

    return metadata, all_results, comparison


def setup_style():
    """Set up matplotlib style for publication-quality plots."""
    # Use seaborn style with custom tweaks
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Custom color palette
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        color=['#2E86AB', '#A23B72', '#F18F01']
    )
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.bbox'] = 'tight'


def plot_metrics_comparison(
    all_results: Dict,
    comparison: Dict,
    output_path: Path,
    fmt: str,
    dpi: int
):
    """
    Create bar chart comparing metrics across modes.

    Args:
        all_results: Dictionary with results for each mode
        comparison: Comparison statistics
        output_path: Output file path
        fmt: File format
        dpi: DPI for raster outputs
    """
    modes = ['baseline', 'simple_split', 'full_split']
    mode_labels = ['Baseline', 'Simple Split\n(7 conjunctions)', 'Full Split\n(25+ conjunctions)']
    metrics = ['bleu', 'chrf', 'ter']
    metric_labels = ['BLEU ↑', 'chrF ↑', 'TER ↓']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Get values for each mode
        values = [all_results[mode]['aggregate_metrics'][metric] for mode in modes]

        # Create bars
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax.bar(range(len(modes)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Annotate bars with values
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()

            # For TER, show percentage
            if metric == 'ter':
                label = f'{value:.3f}\n({value*100:.1f}%)'
            else:
                label = f'{value:.2f}'

            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom', fontsize=10)


        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(mode_labels, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Adjust y-axis to leave room for annotations
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi)
    plt.close()

    print(f"  ✓ Metrics comparison saved to {output_path}")


def plot_split_statistics(
    all_results: Dict,
    output_path: Path,
    fmt: str,
    dpi: int
):
    """
    Visualize sentence splitting statistics.

    Args:
        all_results: Dictionary with results for each mode
        output_path: Output file path
        fmt: File format
        dpi: DPI for raster outputs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Average segments per sample
    modes = ['simple_split', 'full_split']
    mode_labels = ['Simple Split', 'Full Split']
    colors = ['#A23B72', '#F18F01']

    avg_segments = [
        all_results[mode]['aggregate_metrics']['avg_segments_per_sample']
        for mode in modes
    ]

    bars1 = ax1.bar(mode_labels, avg_segments, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.axhline(y=1.0, color='#2E86AB', linestyle='--', linewidth=2, label='Baseline (no split)')

    for bar, value in zip(bars1, avg_segments):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Average Segments per Sample', fontsize=11, fontweight='bold')
    ax1.set_title('Splitting Granularity', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Samples split vs unchanged
    samples_split = [
        all_results[mode]['aggregate_metrics']['samples_split']
        for mode in modes
    ]
    samples_unchanged = [
        all_results[mode]['aggregate_metrics']['samples_unchanged']
        for mode in modes
    ]

    x = np.arange(len(modes))
    width = 0.35

    bars_split = ax2.bar(x - width/2, samples_split, width, label='Split',
                         color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    bars_unchanged = ax2.bar(x + width/2, samples_unchanged, width, label='Unchanged',
                            color='#CCCCCC', alpha=0.8, edgecolor='black', linewidth=1)

    ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax2.set_title('Splitting Coverage', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_labels)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate bars
    for bars in [bars_split, bars_unchanged]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi)
    plt.close()

    print(f"  ✓ Split statistics saved to {output_path}")


def plot_per_sample_comparison(
    all_results: Dict,
    output_path: Path,
    fmt: str,
    dpi: int,
    metric: str = 'bleu'
):
    """
    Create scatter plot comparing per-sample metrics.

    Args:
        all_results: Dictionary with results for each mode
        output_path: Output file path
        fmt: File format
        dpi: DPI for raster outputs
        metric: Metric to compare ('bleu', 'chrf', 'ter')
    """
    # Extract per-sample BLEU scores
    # Note: We need to compute per-sample metrics from predictions and references
    from benchmark_data.evaluation.metrics import compute_bleu, compute_chrf, compute_ter

    baseline_samples = all_results['baseline']['per_sample_results']
    simple_samples = all_results['simple_split']['per_sample_results']
    full_samples = all_results['full_split']['per_sample_results']

    # Compute per-sample scores
    baseline_scores = []
    simple_scores = []
    full_scores = []

    metric_func = {
        'bleu': compute_bleu,
        'chrf': compute_chrf,
        'ter': compute_ter
    }[metric]

    for b, s, f in zip(baseline_samples, simple_samples, full_samples):
        # Compute individual sample scores
        b_score = metric_func([b['reference']], [b['prediction']])[metric]
        s_score = metric_func([s['reference']], [s['prediction']])[metric]
        f_score = metric_func([f['reference']], [f['prediction']])[metric]

        if metric == 'ter':
            b_score /= 100.0
            s_score /= 100.0
            f_score /= 100.0

        baseline_scores.append(b_score)
        simple_scores.append(s_score)
        full_scores.append(f_score)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Baseline vs Simple Split
    ax1.scatter(baseline_scores, simple_scores, alpha=0.6, s=50, color='#A23B72', edgecolor='black', linewidth=0.5)

    # Add diagonal line (y=x)
    lims = [
        min(min(baseline_scores), min(simple_scores)),
        max(max(baseline_scores), max(simple_scores))
    ]
    ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x (equal performance)')

    ax1.set_xlabel(f'Baseline {metric.upper()}', fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'Simple Split {metric.upper()}', fontsize=11, fontweight='bold')
    ax1.set_title('Baseline vs Simple Split (per sample)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Plot 2: Baseline vs Full Split
    ax2.scatter(baseline_scores, full_scores, alpha=0.6, s=50, color='#F18F01', edgecolor='black', linewidth=0.5)
    ax2.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='y=x (equal performance)')

    ax2.set_xlabel(f'Baseline {metric.upper()}', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'Full Split {metric.upper()}', fontsize=11, fontweight='bold')
    ax2.set_title('Baseline vs Full Split (per sample)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi)
    plt.close()

    print(f"  ✓ Per-sample comparison saved to {output_path}")


def main():
    args = parse_args()

    print("=== Translation Benchmark Visualization ===")
    print(f"Results: {args.result_dir}")
    print()

    # Load results
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return

    print("[1/4] Loading results...")
    metadata, all_results, comparison = load_results(result_dir)
    print(f"  Dataset: {metadata['data_path']}")
    print(f"  Samples: {metadata['total_samples']}")
    print(f"  Timestamp: {metadata['timestamp']}")
    print()

    # Set up output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = result_dir / 'plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/4] Setting up plotting style...")
    setup_style()
    print()

    print(f"[3/4] Generating plots...")

    # Plot 1: Metrics comparison
    plot_metrics_comparison(
        all_results,
        comparison,
        output_dir / f'metrics_comparison.{args.format}',
        args.format,
        args.dpi
    )

    # Plot 2: Split statistics
    plot_split_statistics(
        all_results,
        output_dir / f'split_statistics.{args.format}',
        args.format,
        args.dpi
    )

    # Plot 3: Per-sample comparison
    plot_per_sample_comparison(
        all_results,
        output_dir / f'per_sample_bleu.{args.format}',
        args.format,
        args.dpi,
        metric='bleu'
    )

    print()
    print("[4/4] Summary")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: {args.format}")
    print(f"  Plots generated: 3")
    print()
    print("Visualization complete!")


if __name__ == "__main__":
    main()
