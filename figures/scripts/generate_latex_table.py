#!/usr/bin/env python3
"""
Generate LaTeX table and macros from benchmark results.

Outputs:
1. LaTeX table code for copy-paste into paper
2. LaTeX \\newcommand macros for inline metric citations

Usage:
    python scai_paper/figures/generate_latex_table.py
"""

import json
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "src" / "ocr" / "results" / "benchmark_results.json"


# Model display names and sizes (approximate)
MODEL_INFO = {
    "ctc_simple": {"display": "ctc\\_simple (ours)", "size": "23MB", "type": "CNN+CTC"},
    "ctc_vgg16": {"display": "ctc\\_vgg16", "size": "$\\sim$550MB", "type": "CNN+CTC"},
    "ctc_vgg19": {"display": "ctc\\_vgg19", "size": "$\\sim$550MB", "type": "CNN+CTC"},
    "ctc_resnet50": {"display": "ctc\\_resnet50", "size": "$\\sim$100MB", "type": "CNN+CTC"},
    "ctc_resnet101": {"display": "ctc\\_resnet101", "size": "$\\sim$180MB", "type": "CNN+CTC"},
    # TrOCR models (if benchmarked)
    "trocr_smi": {"display": "trocr\\_smi", "size": "$\\sim$350MB", "type": "Transformer"},
    "trocr_smi_nor": {"display": "trocr\\_smi\\_nor", "size": "$\\sim$350MB", "type": "Transformer"},
    "trocr_smi_pred": {"display": "trocr\\_smi\\_pred", "size": "$\\sim$350MB", "type": "Transformer"},
    "trocr_smi_synth": {"display": "trocr\\_smi\\_synth", "size": "$\\sim$350MB", "type": "Transformer"},
    "trocr_smi_nor_pred": {"display": "trocr\\_smi\\_nor\\_pred", "size": "$\\sim$350MB", "type": "Transformer"},
    "trocr_smi_pred_synth": {"display": "trocr\\_smi\\_pred\\_synth", "size": "$\\sim$350MB", "type": "Transformer"},
    "trocr_smi_nor_pred_synth": {"display": "trocr\\_smi\\_nor\\_pred\\_synth", "size": "$\\sim$350MB", "type": "Transformer"},
}

# Training-data labels following the IISA paper convention:
# smi=Sami data, nor=+Norwegian, pred=+auto-transcribed, synth=+synthetic pre-training.
TRAINING_DATA = {
    "trocr_smi": "Smi only",
    "trocr_smi_nor": "Smi+Nor",
    "trocr_smi_pred": "Smi+Pred",
    "trocr_smi_synth": "Smi+Synth",
    "trocr_smi_nor_pred": "Smi+Nor+Pred",
    "trocr_smi_pred_synth": "Smi+Pred+Synth",
    "trocr_smi_nor_pred_synth": "Smi+Nor+Pred+Synth",
    "ctc_simple": "Smi+Synth",
    "ctc_vgg16": "Smi+Synth",
    "ctc_vgg19": "Smi+Synth",
    "ctc_resnet50": "Smi+Synth",
    "ctc_resnet101": "Smi+Synth",
}


def load_benchmark_results():
    """Load benchmark results from JSON file."""
    if not BENCHMARK_PATH.exists():
        print(f"Error: {BENCHMARK_PATH} not found")
        return None
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def generate_latex_table(results: list) -> str:
    """Generate LaTeX table code from benchmark results."""
    # Filter successful models
    successful = [r for r in results if r.get("status") == "success"]

    # Sort by CER (best first)
    successful.sort(key=lambda x: x["metrics"]["cer"])

    # Find best CER for bolding
    best_cer = min(r["metrics"]["cer"] for r in successful)
    best_wer = min(r["metrics"]["wer"] for r in successful)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{OCR benchmark results. Lower CER/WER is better.}",
        "\\label{tab:results}",
        "\\begin{tabular}{@{}lcccr@{}}",
        "\\toprule",
        "Model & Type & CER (\\%) & WER (\\%) & Size \\\\",
        "\\midrule",
    ]

    for r in successful:
        model = r["model"]
        info = MODEL_INFO.get(model, {"display": model.replace("_", "\\_"), "size": "?", "type": "?"})

        cer = r["metrics"]["cer"] * 100
        wer = r["metrics"]["wer"] * 100

        # Bold best values
        cer_str = f"\\textbf{{{cer:.2f}}}" if abs(cer - best_cer * 100) < 0.01 else f"{cer:.2f}"
        wer_str = f"\\textbf{{{wer:.2f}}}" if abs(wer - best_wer * 100) < 0.01 else f"{wer:.2f}"

        # Bold best model name
        display = f"\\textbf{{{info['display']}}}" if model == "ctc_simple" else info["display"]

        lines.append(f"{display} & {info['type']} & {cer_str} & {wer_str} & {info['size']} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_extended_iisa_table(results: list) -> str:
    """Generate extended IISA-style table with all 11 models (7 TrOCR + ctc_simple +
    3 failed ImageNet backbones). Mirrors the column layout of Table 1 in iisa_paper/
    main.tex, adding Exact Acc and keeping all variants instead of the top 5."""
    successful = [r for r in results if r.get("status") == "success"]
    successful.sort(key=lambda x: x["metrics"]["cer"])

    best_cer = min(r["metrics"]["cer"] for r in successful)
    best_wer = min(r["metrics"]["wer"] for r in successful)
    best_exact = max(r["metrics"]["accuracy"] for r in successful)
    # Fastest among successful models
    best_time = min(r["inference_time"] / r["num_samples"] for r in successful)

    def fmt(value, best, precision=2, suffix=""):
        # Tolerance scales with display precision so 0.031 vs 0.034 are not both bolded.
        tol = 0.5 * 10 ** (-precision)
        return (f"\\textbf{{{value:.{precision}f}{suffix}}}"
                if abs(value - best) < tol
                else f"{value:.{precision}f}{suffix}")

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Extended OCR benchmark on \\testSamples{} test samples: all 7 TrOCR variants from the National Library of Norway, the proposed lightweight \\texttt{ctc\\_simple}, and the three ImageNet-pretrained CTC backbones that failed to converge. Bold marks the best value in each column. Lower CER/WER is better, higher Exact Match Accuracy is better.}",
        "\\label{tab:extended_results}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Training Data} & \\textbf{CER↓} & \\textbf{WER↓} & \\textbf{Exact↑} & \\textbf{sec / img} & \\textbf{Size} \\\\",
        " & & \\textbf{(\\%)} & \\textbf{(\\%)} & \\textbf{(\\%)} & & \\textbf{(MB)} \\\\",
        "\\midrule",
    ]

    for r in successful:
        model = r["model"]
        info = MODEL_INFO.get(model, {"display": model.replace("_", "\\_"), "size": "?"})
        training = TRAINING_DATA.get(model, "?")

        cer = r["metrics"]["cer"] * 100
        wer = r["metrics"]["wer"] * 100
        exact = r["metrics"]["accuracy"] * 100
        sec = r["inference_time"] / r["num_samples"]

        display = (f"\\textbf{{{info['display']}}}"
                   if model == "ctc_simple"
                   else info["display"])

        # Strip trailing "MB" so the column header "(MB)" carries the unit.
        size_numeric = info["size"].replace("MB", "").strip()

        lines.append(
            f"{display} & {training} & "
            f"{fmt(cer, best_cer * 100)} & "
            f"{fmt(wer, best_wer * 100)} & "
            f"{fmt(exact, best_exact * 100)} & "
            f"{fmt(sec, best_time, precision=3)} & "
            f"{size_numeric} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\footnotesize",
        "$^*$Training-data tags: \\texttt{smi}=S\\'{a}mi data, \\texttt{nor}=+Norwegian, "
        "\\texttt{pred}=+auto-transcribed, \\texttt{synth}=+synthetic pre-training. "
        "The three \\texttt{ctc\\_vgg*}/\\texttt{ctc\\_resnet*} variants additionally use "
        "ImageNet-pretrained CNN backbones (see Section~\\ref{sec:main-result} for the "
        "representation-problem discussion).",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_macros(results: list) -> str:
    """Generate LaTeX \\newcommand macros for benchmark metrics."""
    lines = [
        "% ============================================================================",
        "% BENCHMARK RESULTS - Auto-generated from benchmark_results.json",
        "% ============================================================================",
    ]

    # Get total samples from first result
    total_samples = results[0].get("num_samples", 0) if results else 0
    lines.append(f"\\newcommand{{\\benchmarkSamples}}{{{total_samples}}}  % Total benchmark samples")
    lines.append("")

    for r in results:
        if r.get("status") != "success":
            continue

        model = r["model"]
        cer = r["metrics"]["cer"] * 100
        wer = r["metrics"]["wer"] * 100

        # Create macro names (e.g., ctc_simple -> ctcSimpleCER)
        parts = model.split("_")
        macro_name = parts[0] + "".join(p.capitalize() for p in parts[1:])

        lines.append(f"\\newcommand{{\\{macro_name}CER}}{{{cer:.2f}}}  % {model}")
        lines.append(f"\\newcommand{{\\{macro_name}WER}}{{{wer:.2f}}}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("LaTeX Table Generator for SCAI Paper")
    print("=" * 60)
    print(f"\nReading: {BENCHMARK_PATH}")

    results = load_benchmark_results()
    if results is None:
        return

    # Generate table
    print("\n" + "=" * 60)
    print("LATEX TABLE (copy-paste into main.tex)")
    print("=" * 60 + "\n")
    print(generate_latex_table(results))

    # Generate extended IISA-style table (all 11 models)
    print("\n" + "=" * 60)
    print("EXTENDED IISA TABLE (all 11 models, for thesis_report)")
    print("=" * 60 + "\n")
    print(generate_extended_iisa_table(results))

    # Generate macros
    print("\n" + "=" * 60)
    print("LATEX MACROS (paste in preamble)")
    print("=" * 60 + "\n")
    print(generate_latex_macros(results))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.get("status") == "success"]
    successful.sort(key=lambda x: x["metrics"]["cer"])

    print(f"\nTotal models: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Benchmark samples: {results[0].get('num_samples', '?')}")
    print("\nRanking by CER:")
    for i, r in enumerate(successful, 1):
        cer = r["metrics"]["cer"] * 100
        wer = r["metrics"]["wer"] * 100
        print(f"  {i}. {r['model']:20s}  CER: {cer:6.2f}%  WER: {wer:6.2f}%")


if __name__ == "__main__":
    main()
