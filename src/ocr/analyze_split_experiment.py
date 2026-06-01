#!/usr/bin/env python3
"""Analyze results from the conjunction-based split experiment.

Generates visualizations and statistical analysis comparing original
vs split OCR performance.

Usage:
    python analyze_split_experiment.py --results results/split_experiment
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> Tuple[Dict, List[Dict]]:
    """Load experiment results from JSON files."""
    results_path = Path(results_dir)

    with open(results_path / "split_experiment_summary.json") as f:
        summary = json.load(f)

    with open(results_path / "split_experiment_details.json") as f:
        details = json.load(f)

    return summary, details


def create_length_vs_cer_plot(details: List[Dict], output_path: Path):
    """
    Create scatter plot showing CER vs sentence length for original and split.

    This is the key visualization showing if splitting helps for long sentences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data
    lengths = [d["original_length"] for d in details]
    original_cer = [d["original_cer"] * 100 for d in details]
    split_cer = [d["split_cer"] * 100 for d in details]
    was_split = [d["was_split"] for d in details]

    # Left plot: Original OCR
    ax1 = axes[0]
    ax1.scatter(lengths, original_cer, alpha=0.5, s=20, c='blue', label='Original')
    ax1.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='80 char threshold')

    # Add trend line
    z = np.polyfit(lengths, original_cer, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(lengths), max(lengths), 100)
    ax1.plot(x_trend, p(x_trend), 'b-', alpha=0.8, linewidth=2, label=f'Trend')

    ax1.set_xlabel("Sentence Length (characters)")
    ax1.set_ylabel("Character Error Rate (%)")
    ax1.set_title("Original OCR: CER vs Sentence Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, min(100, max(original_cer) * 1.1))

    # Right plot: Comparison for long sentences
    ax2 = axes[1]

    # Filter to only long sentences that were split
    long_and_split = [(d["original_length"], d["original_cer"]*100, d["split_cer"]*100)
                      for d in details if d["was_split"]]

    if long_and_split:
        ls_lengths, ls_orig, ls_split = zip(*long_and_split)

        ax2.scatter(ls_lengths, ls_orig, alpha=0.6, s=30, c='blue', marker='o', label='Original')
        ax2.scatter(ls_lengths, ls_split, alpha=0.6, s=30, c='green', marker='^', label='Split')

        # Connect pairs with lines
        for i in range(len(ls_lengths)):
            ax2.plot([ls_lengths[i], ls_lengths[i]], [ls_orig[i], ls_split[i]],
                    'gray', alpha=0.3, linewidth=0.5)

        ax2.axvline(x=80, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel("Sentence Length (characters)")
        ax2.set_ylabel("Character Error Rate (%)")
        ax2.set_title("Split vs Original for Long Sentences")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, min(100, max(max(ls_orig), max(ls_split)) * 1.1))

    plt.tight_layout()
    plt.savefig(output_path / "cer_vs_length.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "cer_vs_length.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: cer_vs_length.png/pdf")


def create_improvement_histogram(details: List[Dict], output_path: Path):
    """
    Create histogram of CER improvement for split sentences.

    Positive = split helped, Negative = split hurt.
    """
    # Filter to only split sentences
    split_results = [d for d in details if d["was_split"]]

    if not split_results:
        print("  No split sentences - skipping improvement histogram")
        return

    improvements = [d["cer_improvement"] * 100 for d in split_results]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create histogram
    bins = np.linspace(min(improvements) - 1, max(improvements) + 1, 30)
    n, bins, patches = ax.hist(improvements, bins=bins, edgecolor='black', alpha=0.7)

    # Color bars by improvement direction
    for i, patch in enumerate(patches):
        if bins[i] >= 0:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)

    # Add statistics
    mean_imp = np.mean(improvements)
    positive_count = sum(1 for x in improvements if x > 0)
    negative_count = sum(1 for x in improvements if x < 0)

    ax.axvline(x=mean_imp, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {mean_imp:+.2f}%')

    ax.set_xlabel("CER Improvement (percentage points)")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"CER Improvement Distribution for Split Sentences\n"
                f"(Improved: {positive_count}, Degraded: {negative_count})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "improvement_histogram.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "improvement_histogram.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: improvement_histogram.png/pdf")


def create_comparison_bar_chart(summary: Dict, output_path: Path):
    """
    Create bar chart comparing CER for short vs long sentences.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Categories and values
    categories = ['Short (<= 80)', 'Long (> 80)\nOriginal', 'Long (> 80)\nSplit']
    values = [
        summary["short_original_cer"] * 100,
        summary["long_original_cer"] * 100,
        summary["long_split_cer"] * 100
    ]
    colors = ['steelblue', 'coral', 'green']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12)

    ax.set_ylabel("Character Error Rate (%)")
    ax.set_title(f"CER Comparison by Sentence Length\n"
                f"(Short: n={summary['short_count']}, Long: n={summary['long_count']})")
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement annotation
    if summary["long_count"] > 0:
        improvement = (summary["long_original_cer"] - summary["long_split_cer"]) * 100
        ax.annotate(f'Improvement: {improvement:+.2f}%',
                   xy=(2.1, summary["long_split_cer"] * 100 * 0.93),
                   xytext=(1.9, summary["long_original_cer"] * 100 * 0.8),
                   fontsize=11,
                   arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(output_path / "comparison_bar_chart.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "comparison_bar_chart.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_bar_chart.png/pdf")


def create_segment_count_analysis(details: List[Dict], output_path: Path):
    """
    Analyze how the number of segments affects CER improvement.
    """
    split_results = [d for d in details if d["was_split"]]

    if not split_results:
        print("  No split sentences - skipping segment analysis")
        return

    # Group by number of segments
    by_segments = {}
    for d in split_results:
        n = d["num_segments"]
        if n not in by_segments:
            by_segments[n] = []
        by_segments[n].append(d["cer_improvement"] * 100)

    fig, ax = plt.subplots(figsize=(10, 5))

    segment_counts = sorted(by_segments.keys())
    means = [np.mean(by_segments[n]) for n in segment_counts]
    stds = [np.std(by_segments[n]) for n in segment_counts]
    counts = [len(by_segments[n]) for n in segment_counts]

    x = np.arange(len(segment_counts))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue',
                  edgecolor='black', alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in segment_counts])
    ax.set_xlabel("Number of Segments")
    ax.set_ylabel("Mean CER Improvement (percentage points)")
    ax.set_title("CER Improvement by Number of Segments")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / "segment_count_analysis.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "segment_count_analysis.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved: segment_count_analysis.png/pdf")


def compute_statistical_significance(details: List[Dict]) -> Dict:
    """
    Compute statistical significance of CER improvement using paired t-test.
    """
    from scipy import stats

    split_results = [d for d in details if d["was_split"]]

    if len(split_results) < 2:
        return {"error": "Not enough split samples for statistical test"}

    original_cer = [d["original_cer"] for d in split_results]
    split_cer = [d["split_cer"] for d in split_results]

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(original_cer, split_cer)

    # Effect size (Cohen's d)
    differences = np.array(original_cer) - np.array(split_cer)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)

    # Convert numpy types to Python native types for JSON serialization
    return {
        "n_samples": len(split_results),
        "mean_original_cer": float(np.mean(original_cer)),
        "mean_split_cer": float(np.mean(split_cer)),
        "mean_improvement": float(np.mean(differences)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
    }


def generate_latex_table(summary: Dict, stats: Dict, output_path: Path):
    """Generate LaTeX table for paper inclusion."""

    latex = r"""\begin{table}[h]
\centering
\caption{Conjunction-based Splitting Experiment Results}
\label{tab:split_experiment}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Original} & \textbf{Split} \\
\midrule
Overall CER (\%) & """ + f"{summary['overall_original_cer']*100:.2f}" + r" & " + f"{summary['overall_split_cer']*100:.2f}" + r""" \\
Overall WER (\%) & """ + f"{summary['overall_original_wer']*100:.2f}" + r" & " + f"{summary['overall_split_wer']*100:.2f}" + r""" \\
\midrule
Long sentences CER (\%) & """ + f"{summary['long_original_cer']*100:.2f}" + r" & " + f"{summary['long_split_cer']*100:.2f}" + r""" \\
Long sentences WER (\%) & """ + f"{summary['long_original_wer']*100:.2f}" + r" & " + f"{summary['long_split_wer']*100:.2f}" + r""" \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\footnotesize{Total samples: """ + str(summary['total_samples']) + r""", Split: """ + str(summary['samples_split']) + r"""}
"""

    if "p_value" in stats:
        latex += r"""
\footnotesize{Paired t-test: $t = """ + f"{stats['t_statistic']:.3f}" + r"$, $p = " + f"{stats['p_value']:.4f}" + r"$, Cohen's $d = " + f"{stats['cohens_d']:.3f}" + r"""$}
"""

    latex += r"""\end{table}
"""

    with open(output_path / "results_table.tex", "w") as f:
        f.write(latex)

    print(f"  Saved: results_table.tex")


def generate_markdown_report(summary: Dict, details: List[Dict], stats: Dict, output_path: Path):
    """Generate Markdown report summarizing the experiment."""

    split_results = [d for d in details if d["was_split"]]
    improved = sum(1 for d in split_results if d["cer_improvement"] > 0)
    degraded = sum(1 for d in split_results if d["cer_improvement"] < 0)

    report = f"""# Conjunction-Based Split Experiment Results

## Summary

| Metric | Value |
|--------|-------|
| Total samples | {summary['total_samples']} |
| Samples split | {summary['samples_split']} ({100*summary['samples_split']/summary['total_samples']:.1f}%) |
| Short sentences (≤80 chars) | {summary['short_count']} |
| Long sentences (>80 chars) | {summary['long_count']} |

## Overall Metrics

| Metric | Original | Split | Improvement |
|--------|----------|-------|-------------|
| CER | {summary['overall_original_cer']*100:.2f}% | {summary['overall_split_cer']*100:.2f}% | {(summary['overall_original_cer']-summary['overall_split_cer'])*100:+.2f}% |
| WER | {summary['overall_original_wer']*100:.2f}% | {summary['overall_split_wer']*100:.2f}% | {(summary['overall_original_wer']-summary['overall_split_wer'])*100:+.2f}% |

## Long Sentence Analysis (>80 chars)

| Metric | Original | Split | Improvement |
|--------|----------|-------|-------------|
| CER | {summary['long_original_cer']*100:.2f}% | {summary['long_split_cer']*100:.2f}% | {(summary['long_original_cer']-summary['long_split_cer'])*100:+.2f}% |
| WER | {summary['long_original_wer']*100:.2f}% | {summary['long_split_wer']*100:.2f}% | {(summary['long_original_wer']-summary['long_split_wer'])*100:+.2f}% |

## Split Outcome Distribution

- Samples improved by splitting: **{improved}** ({100*improved/len(split_results):.1f}%)
- Samples degraded by splitting: **{degraded}** ({100*degraded/len(split_results):.1f}%)
"""

    if "p_value" in stats:
        sig = "Yes" if stats['significant_at_05'] else "No"
        report += f"""
## Statistical Significance

| Test | Value |
|------|-------|
| Paired t-statistic | {stats['t_statistic']:.4f} |
| p-value | {stats['p_value']:.6f} |
| Cohen's d | {stats['cohens_d']:.4f} |
| Significant at α=0.05? | {sig} |
"""

    # Add some example cases
    if split_results:
        best = max(split_results, key=lambda d: d["cer_improvement"])
        worst = min(split_results, key=lambda d: d["cer_improvement"])

        report += f"""
## Example Cases

### Best Improvement

- **Sample:** {best['sample_id']}
- **Length:** {best['original_length']} chars
- **Original CER:** {best['original_cer']*100:.1f}% → **Split CER:** {best['split_cer']*100:.1f}%
- **Improvement:** {best['cer_improvement']*100:+.1f}%

Ground truth:
> {best['ground_truth'][:150]}{'...' if len(best['ground_truth']) > 150 else ''}

### Worst Degradation

- **Sample:** {worst['sample_id']}
- **Length:** {worst['original_length']} chars
- **Original CER:** {worst['original_cer']*100:.1f}% → **Split CER:** {worst['split_cer']*100:.1f}%
- **Change:** {worst['cer_improvement']*100:+.1f}%

Ground truth:
> {worst['ground_truth'][:150]}{'...' if len(worst['ground_truth']) > 150 else ''}
"""

    with open(output_path / "experiment_report.md", "w") as f:
        f.write(report)

    print(f"  Saved: experiment_report.md")


def analyze_results(results_dir: str, output_subdir: str = "figures"):
    """Run full analysis and generate all outputs."""

    print("=" * 70)
    print("ANALYZING SPLIT EXPERIMENT RESULTS")
    print("=" * 70)

    results_path = Path(results_dir)
    output_path = results_path / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {results_path}")
    summary, details = load_results(results_dir)

    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Split samples: {summary['samples_split']}")

    print("\nGenerating figures...")
    create_length_vs_cer_plot(details, output_path)
    create_improvement_histogram(details, output_path)
    create_comparison_bar_chart(summary, output_path)
    create_segment_count_analysis(details, output_path)

    print("\nComputing statistics...")
    try:
        stats = compute_statistical_significance(details)
        if "p_value" in stats:
            print(f"  Paired t-test: t={stats['t_statistic']:.4f}, p={stats['p_value']:.6f}")
            print(f"  Effect size (Cohen's d): {stats['cohens_d']:.4f}")
            print(f"  Significant at α=0.05: {stats['significant_at_05']}")
    except ImportError:
        print("  scipy not available - skipping statistical tests")
        stats = {}

    print("\nGenerating reports...")
    generate_latex_table(summary, stats, output_path)
    generate_markdown_report(summary, details, stats, output_path)

    # Save stats
    if stats:
        with open(output_path / "statistical_analysis.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: statistical_analysis.json")

    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {output_path}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze split experiment results")
    parser.add_argument("--results", type=str, default="results/split_experiment",
                       help="Path to results directory")
    parser.add_argument("--output-subdir", type=str, default="figures",
                       help="Subdirectory for figures (default: figures)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_results(args.results, args.output_subdir)
