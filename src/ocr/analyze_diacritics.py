#!/usr/bin/env python3
"""Analyze per-character accuracy for Sámi-specific diacritics.

Computes accuracy for the special characters: á, č, đ, ŋ, š, ŧ, ž
and their uppercase variants.

Usage:
    python analyze_diacritics.py --predictions results/predictions/ctc_simple.jsonl
    python analyze_diacritics.py --compare ctc_simple trocr_smi_synth
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Sámi-specific diacritical characters
SAMI_DIACRITICS = {
    'á': 'Á', 'č': 'Č', 'đ': 'Đ', 'ŋ': 'Ŋ', 'š': 'Š', 'ŧ': 'Ŧ', 'ž': 'Ž'
}

ALL_DIACRITICS = set(SAMI_DIACRITICS.keys()) | set(SAMI_DIACRITICS.values())


def load_predictions(filepath: str) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def align_strings(ref: str, pred: str) -> List[Tuple[str, str]]:
    """
    Align reference and prediction strings using dynamic programming.
    Returns list of (ref_char, pred_char) pairs.
    Uses Levenshtein alignment to handle insertions/deletions.
    """
    m, n = len(ref), len(pred)

    # DP table for edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # Backtrack to get alignment
    alignments = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and (ref[i-1] == pred[j-1] or
                                  dp[i][j] == dp[i-1][j-1] + 1):
            alignments.append((ref[i-1], pred[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignments.append(('', pred[j-1]))  # Insertion
            j -= 1
        else:
            alignments.append((ref[i-1], ''))  # Deletion
            i -= 1

    return list(reversed(alignments))


def analyze_diacritics(predictions: List[Dict]) -> Dict:
    """
    Analyze per-character accuracy for Sámi diacritics.

    Returns dict with:
    - per_char_stats: {char: {correct, total, accuracy, confusions}}
    - overall_stats: aggregate statistics
    """
    per_char_stats = {char: {'correct': 0, 'total': 0, 'confusions': defaultdict(int)}
                      for char in ALL_DIACRITICS}

    for sample in predictions:
        ref = sample['reference']
        pred = sample['prediction']

        alignments = align_strings(ref, pred)

        for ref_char, pred_char in alignments:
            if ref_char in ALL_DIACRITICS:
                per_char_stats[ref_char]['total'] += 1
                if ref_char == pred_char:
                    per_char_stats[ref_char]['correct'] += 1
                else:
                    per_char_stats[ref_char]['confusions'][pred_char] += 1

    # Compute accuracies
    for char, stats in per_char_stats.items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total'] * 100
        else:
            stats['accuracy'] = None

        # Convert confusions to regular dict for JSON serialization
        stats['confusions'] = dict(stats['confusions'])

    # Compute overall statistics
    total_correct = sum(s['correct'] for s in per_char_stats.values())
    total_chars = sum(s['total'] for s in per_char_stats.values())

    overall_stats = {
        'total_diacritic_chars': total_chars,
        'total_correct': total_correct,
        'overall_accuracy': total_correct / total_chars * 100 if total_chars > 0 else 0
    }

    return {
        'per_char_stats': per_char_stats,
        'overall_stats': overall_stats
    }


def generate_latex_table(results: Dict, model_name: str = "Model") -> str:
    """Generate LaTeX table for paper inclusion."""
    stats = results['per_char_stats']

    # Group lowercase and uppercase
    rows = []
    for lower, upper in SAMI_DIACRITICS.items():
        lower_acc = stats[lower]['accuracy']
        upper_acc = stats[upper]['accuracy']
        lower_n = stats[lower]['total']
        upper_n = stats[upper]['total']

        if lower_acc is not None or upper_acc is not None:
            # Combined accuracy (weighted by count)
            total_n = lower_n + upper_n
            if total_n > 0:
                combined_correct = stats[lower]['correct'] + stats[upper]['correct']
                combined_acc = combined_correct / total_n * 100
            else:
                combined_acc = None

            rows.append((lower, upper, combined_acc, total_n))

    latex = r"""\begin{table}[t]
\centering
\caption{Per-Character Accuracy for S\'{a}mi Diacritics}
\label{tab:diacritics}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Character} & \textbf{Accuracy (\%)} & \textbf{Count} \\
\midrule
"""

    for lower, upper, acc, count in rows:
        acc_str = f"{acc:.1f}" if acc is not None else "N/A"
        latex += f"{lower} / {upper} & {acc_str} & {count} \\\\\n"

    overall_acc = results['overall_stats']['overall_accuracy']
    total_count = results['overall_stats']['total_diacritic_chars']

    latex += r"""\midrule
\textbf{Overall} & """ + f"{overall_acc:.1f}" + r""" & """ + str(total_count) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_comparison_table(results_dict: Dict[str, Dict]) -> str:
    """Generate LaTeX table comparing multiple models."""
    models = list(results_dict.keys())

    latex = r"""\begin{table}[t]
\centering
\caption{Per-Character Accuracy for S\'{a}mi Diacritics}
\label{tab:diacritics}
\small
\begin{tabular}{@{}l""" + "c" * len(models) + r"""@{}}
\toprule
\textbf{Character} & """ + " & ".join([f"\\textbf{{{m}}}" for m in models]) + r""" \\
\midrule
"""

    for lower, upper in SAMI_DIACRITICS.items():
        row = f"{lower} / {upper}"
        for model in models:
            stats = results_dict[model]['per_char_stats']
            lower_n = stats[lower]['total']
            upper_n = stats[upper]['total']
            total_n = lower_n + upper_n

            if total_n > 0:
                combined_correct = stats[lower]['correct'] + stats[upper]['correct']
                acc = combined_correct / total_n * 100
                row += f" & {acc:.1f}\\%"
            else:
                row += " & N/A"

        latex += row + r" \\" + "\n"

    # Overall row
    latex += r"\midrule" + "\n" + r"\textbf{Overall}"
    for model in models:
        acc = results_dict[model]['overall_stats']['overall_accuracy']
        latex += f" & {acc:.1f}\\%"
    latex += r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def print_confusion_analysis(results: Dict, top_n: int = 3):
    """Print top confusions for each diacritic character."""
    print("\n" + "=" * 60)
    print("TOP CONFUSIONS FOR SAMI DIACRITICS")
    print("=" * 60)

    for char in sorted(ALL_DIACRITICS):
        stats = results['per_char_stats'][char]
        if stats['total'] == 0:
            continue

        acc = stats['accuracy']
        confusions = stats['confusions']

        print(f"\n'{char}': {acc:.1f}% accuracy ({stats['correct']}/{stats['total']})")

        if confusions:
            sorted_conf = sorted(confusions.items(), key=lambda x: -x[1])[:top_n]
            for confused_char, count in sorted_conf:
                display = repr(confused_char) if confused_char == '' else f"'{confused_char}'"
                print(f"  -> {display}: {count} times")


def main():
    parser = argparse.ArgumentParser(description="Analyze Sámi diacritic accuracy")
    parser.add_argument("--predictions", type=str,
                       help="Path to predictions JSONL file")
    parser.add_argument("--compare", nargs="+",
                       help="Model names to compare (looks in results/predictions/)")
    parser.add_argument("--output-dir", type=str, default="results/diacritics",
                       help="Output directory for results")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.compare:
        # Compare multiple models
        results_dict = {}
        predictions_dir = Path("src/ocr/results/predictions")

        for model in args.compare:
            filepath = predictions_dir / f"{model}.jsonl"
            if not filepath.exists():
                print(f"Warning: {filepath} not found, skipping")
                continue

            print(f"Analyzing {model}...")
            predictions = load_predictions(str(filepath))
            results_dict[model] = analyze_diacritics(predictions)

        # Generate comparison table
        latex_table = generate_comparison_table(results_dict)

        with open(output_path / "diacritics_comparison.tex", "w") as f:
            f.write(latex_table)
        print(f"\nSaved: {output_path}/diacritics_comparison.tex")

        # Print results
        print("\n" + "=" * 60)
        print("DIACRITIC ACCURACY COMPARISON")
        print("=" * 60)

        for model, results in results_dict.items():
            print(f"\n{model}:")
            print(f"  Overall: {results['overall_stats']['overall_accuracy']:.1f}%")
            print_confusion_analysis(results)

        # Save JSON results
        with open(output_path / "diacritics_analysis.json", "w") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"\nSaved: {output_path}/diacritics_analysis.json")
        print("\nLaTeX table:")
        print(latex_table)

    elif args.predictions:
        # Single model analysis
        print(f"Loading predictions from: {args.predictions}")
        predictions = load_predictions(args.predictions)
        print(f"Loaded {len(predictions)} samples")

        results = analyze_diacritics(predictions)

        # Print results
        print("\n" + "=" * 60)
        print("DIACRITIC ACCURACY ANALYSIS")
        print("=" * 60)

        print(f"\nOverall diacritic accuracy: {results['overall_stats']['overall_accuracy']:.1f}%")
        print(f"Total diacritic characters: {results['overall_stats']['total_diacritic_chars']}")

        print("\nPer-character accuracy:")
        for char in sorted(ALL_DIACRITICS):
            stats = results['per_char_stats'][char]
            if stats['total'] > 0:
                print(f"  '{char}': {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")

        print_confusion_analysis(results)

        # Generate LaTeX table
        latex_table = generate_latex_table(results)

        with open(output_path / "diacritics_table.tex", "w") as f:
            f.write(latex_table)

        # Save JSON
        with open(output_path / "diacritics_analysis.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nSaved results to: {output_path}")
        print("\nLaTeX table:")
        print(latex_table)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
