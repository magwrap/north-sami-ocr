#!/usr/bin/env python3
"""Benchmark experiment: Conjunction-based sentence splitting for North Sámi OCR.

Tests the hypothesis that long sentences (>80 chars) degrade OCR accuracy
due to 800px width constraint, and that splitting at conjunction boundaries
may improve accuracy by keeping segments short.

Usage:
    cd src/ocr && python benchmark_split_experiment.py
    python benchmark_split_experiment.py --model ctc_simple --weights trained_models/queue_20250426_201006/ctc_simple/checkpoint_best.pt
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image

# Add parent to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from sami_text_splitter import split_at_conjunctions, rejoin_segments, SplitResult
from synthetic_image_generator import generate_image, find_sami_font
from train_utils import compute_cer, compute_wer


@dataclass
class SampleResult:
    """Result for a single sample comparing original vs split OCR."""
    sample_id: str
    ground_truth: str
    original_length: int

    # Original (full sentence) OCR
    original_pred: str
    original_cer: float
    original_wer: float

    # Split OCR (if applicable)
    was_split: bool
    num_segments: int
    segment_lengths: List[int]
    segment_predictions: List[str]
    split_pred: str  # rejoined prediction
    split_cer: float
    split_wer: float

    # Improvement metrics
    cer_improvement: float  # positive = split is better
    wer_improvement: float


@dataclass
class ExperimentResults:
    """Aggregated experiment results."""
    model_name: str
    total_samples: int
    samples_split: int
    samples_not_split: int

    # Overall metrics
    overall_original_cer: float
    overall_original_wer: float
    overall_split_cer: float
    overall_split_wer: float

    # Long sentences only (>80 chars)
    long_count: int
    long_original_cer: float
    long_original_wer: float
    long_split_cer: float
    long_split_wer: float

    # Short sentences (<= 80 chars) - should be unchanged
    short_count: int
    short_original_cer: float
    short_original_wer: float

    # Per-sample results
    sample_results: List[SampleResult]


def load_test_data(hf_sample_size: int = 1000, seed: int = 42, include_book: bool = True) -> List[Dict]:
    """Load test samples from HuggingFace dataset + book data."""
    from datasets import load_dataset

    test_data = []

    # Load HF validation split
    print("Loading HuggingFace validation split...")
    dataset = load_dataset("Sprakbanken/synthetic_sami_ocr_data", split="validation")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(hf_sample_size, len(dataset)))
    indices.sort()

    for i in indices:
        test_data.append({
            "id": f"hf_{i}",
            "image": dataset[i]["image"],
            "text": dataset[i]["text"],
            "source": "hf"
        })
    print(f"  Loaded {len(test_data)} HF samples")

    # Load book data
    if include_book:
        book_dir = Path(__file__).parent.parent.parent / "test_data" / "account_of_sami"
        gt_path = book_dir / "ground_truth.json"
        images_dir = book_dir / "synthetic_images"

        if gt_path.exists():
            print("Loading 'Account of Sami' book data...")
            with open(gt_path) as f:
                ground_truth = json.load(f)

            book_count = 0
            for entry_id, entry in ground_truth.items():
                if entry_id == "length":
                    continue

                # Use original_formatted or original
                text = entry.get("original_formatted", entry.get("original", "")).strip()
                if not text:
                    continue

                img_path = images_dir / f"entry_{entry_id}.png"
                if img_path.exists():
                    test_data.append({
                        "id": f"book_{entry_id}",
                        "image": Image.open(img_path),
                        "text": text,
                        "source": "book"
                    })
                    book_count += 1

            print(f"  Loaded {book_count} book samples")

    print(f"Total: {len(test_data)} test samples")
    return test_data


def run_ocr(pipeline, image: Image.Image) -> str:
    """Run OCR on an image and return the prediction."""
    try:
        return pipeline.recognize(image)
    except Exception as e:
        print(f"    OCR error: {e}")
        return ""


def run_split_ocr(
    pipeline,
    text: str,
    split_result: SplitResult,
    font_path: Optional[str] = None
) -> Tuple[List[str], str]:
    """
    Run OCR on split segments and return individual predictions + rejoined result.

    Generates fresh synthetic images for each segment to test OCR on shorter inputs.
    """
    segment_predictions = []

    for segment in split_result.segments:
        # Generate image for this segment
        seg_image = generate_image(segment, font_path=font_path)

        # Run OCR
        pred = run_ocr(pipeline, seg_image)
        segment_predictions.append(pred)

    # Rejoin predictions
    rejoined = rejoin_segments(segment_predictions)
    return segment_predictions, rejoined


def process_sample(
    pipeline,
    sample: Dict,
    font_path: Optional[str],
    length_threshold: int = 80
) -> SampleResult:
    """Process a single sample with both original and split OCR."""
    sample_id = sample["id"]
    ground_truth = sample["text"]
    image = sample["image"]
    original_length = len(ground_truth)

    # Run original OCR
    original_pred = run_ocr(pipeline, image)
    original_cer = compute_cer(original_pred, ground_truth)
    original_wer = compute_wer(original_pred, ground_truth)

    # Check if we should split
    split_result = split_at_conjunctions(ground_truth)
    was_split = len(split_result.segments) > 1

    if was_split:
        # Run split OCR
        segment_preds, split_pred = run_split_ocr(
            pipeline, ground_truth, split_result, font_path
        )
        split_cer = compute_cer(split_pred, ground_truth)
        split_wer = compute_wer(split_pred, ground_truth)
        segment_lengths = [len(s) for s in split_result.segments]
    else:
        # No split - use original results
        segment_preds = [original_pred]
        split_pred = original_pred
        split_cer = original_cer
        split_wer = original_wer
        segment_lengths = [original_length]

    return SampleResult(
        sample_id=sample_id,
        ground_truth=ground_truth,
        original_length=original_length,
        original_pred=original_pred,
        original_cer=original_cer,
        original_wer=original_wer,
        was_split=was_split,
        num_segments=len(split_result.segments),
        segment_lengths=segment_lengths,
        segment_predictions=segment_preds,
        split_pred=split_pred,
        split_cer=split_cer,
        split_wer=split_wer,
        cer_improvement=original_cer - split_cer,
        wer_improvement=original_wer - split_wer
    )


def aggregate_results(
    model_name: str,
    sample_results: List[SampleResult],
    length_threshold: int = 80
) -> ExperimentResults:
    """Aggregate individual sample results into experiment summary."""
    total = len(sample_results)
    split_count = sum(1 for r in sample_results if r.was_split)

    # Overall metrics
    overall_orig_cer = sum(r.original_cer for r in sample_results) / total
    overall_orig_wer = sum(r.original_wer for r in sample_results) / total
    overall_split_cer = sum(r.split_cer for r in sample_results) / total
    overall_split_wer = sum(r.split_wer for r in sample_results) / total

    # Long sentences (>80 chars)
    long_results = [r for r in sample_results if r.original_length > length_threshold]
    long_count = len(long_results)
    if long_count > 0:
        long_orig_cer = sum(r.original_cer for r in long_results) / long_count
        long_orig_wer = sum(r.original_wer for r in long_results) / long_count
        long_split_cer = sum(r.split_cer for r in long_results) / long_count
        long_split_wer = sum(r.split_wer for r in long_results) / long_count
    else:
        long_orig_cer = long_orig_wer = long_split_cer = long_split_wer = 0.0

    # Short sentences (<=80 chars)
    short_results = [r for r in sample_results if r.original_length <= length_threshold]
    short_count = len(short_results)
    if short_count > 0:
        short_orig_cer = sum(r.original_cer for r in short_results) / short_count
        short_orig_wer = sum(r.original_wer for r in short_results) / short_count
    else:
        short_orig_cer = short_orig_wer = 0.0

    return ExperimentResults(
        model_name=model_name,
        total_samples=total,
        samples_split=split_count,
        samples_not_split=total - split_count,
        overall_original_cer=overall_orig_cer,
        overall_original_wer=overall_orig_wer,
        overall_split_cer=overall_split_cer,
        overall_split_wer=overall_split_wer,
        long_count=long_count,
        long_original_cer=long_orig_cer,
        long_original_wer=long_orig_wer,
        long_split_cer=long_split_cer,
        long_split_wer=long_split_wer,
        short_count=short_count,
        short_original_cer=short_orig_cer,
        short_original_wer=short_orig_wer,
        sample_results=sample_results
    )


def print_summary(results: ExperimentResults):
    """Print experiment summary to console."""
    print("\n" + "=" * 70)
    print("SPLIT EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"\nModel: {results.model_name}")
    print(f"Total samples: {results.total_samples}")
    print(f"  - Split: {results.samples_split} ({100*results.samples_split/results.total_samples:.1f}%)")
    print(f"  - Not split: {results.samples_not_split}")

    print(f"\nOverall Metrics:")
    print(f"  Original CER: {results.overall_original_cer*100:.2f}%")
    print(f"  Split CER:    {results.overall_split_cer*100:.2f}%")
    print(f"  Improvement:  {(results.overall_original_cer - results.overall_split_cer)*100:+.2f}%")

    print(f"\nLong Sentences (>{80} chars, n={results.long_count}):")
    print(f"  Original CER: {results.long_original_cer*100:.2f}%")
    print(f"  Split CER:    {results.long_split_cer*100:.2f}%")
    print(f"  Improvement:  {(results.long_original_cer - results.long_split_cer)*100:+.2f}%")

    print(f"\nShort Sentences (<=80 chars, n={results.short_count}):")
    print(f"  Original CER: {results.short_original_cer*100:.2f}%")

    # Find best/worst improvements
    if results.samples_split > 0:
        split_results = [r for r in results.sample_results if r.was_split]
        best = max(split_results, key=lambda r: r.cer_improvement)
        worst = min(split_results, key=lambda r: r.cer_improvement)

        print(f"\nBest improvement (CER): {best.cer_improvement*100:+.2f}% on sample {best.sample_id}")
        print(f"  Original: {best.original_cer*100:.1f}% -> Split: {best.split_cer*100:.1f}%")

        if worst.cer_improvement < 0:
            print(f"\nWorst regression: {worst.cer_improvement*100:+.2f}% on sample {worst.sample_id}")
            print(f"  Original: {worst.original_cer*100:.1f}% -> Split: {worst.split_cer*100:.1f}%")

    print("=" * 70)


def save_results(results: ExperimentResults, output_dir: str):
    """Save experiment results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary JSON
    summary = {
        "model_name": results.model_name,
        "total_samples": results.total_samples,
        "samples_split": results.samples_split,
        "samples_not_split": results.samples_not_split,
        "overall_original_cer": results.overall_original_cer,
        "overall_original_wer": results.overall_original_wer,
        "overall_split_cer": results.overall_split_cer,
        "overall_split_wer": results.overall_split_wer,
        "long_count": results.long_count,
        "long_original_cer": results.long_original_cer,
        "long_original_wer": results.long_original_wer,
        "long_split_cer": results.long_split_cer,
        "long_split_wer": results.long_split_wer,
        "short_count": results.short_count,
        "short_original_cer": results.short_original_cer,
        "short_original_wer": results.short_original_wer,
    }

    with open(output_path / "split_experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed per-sample results
    sample_data = []
    for r in results.sample_results:
        sample_data.append({
            "sample_id": r.sample_id,
            "ground_truth": r.ground_truth,
            "original_length": r.original_length,
            "original_pred": r.original_pred,
            "original_cer": r.original_cer,
            "original_wer": r.original_wer,
            "was_split": r.was_split,
            "num_segments": r.num_segments,
            "segment_lengths": r.segment_lengths,
            "segment_predictions": r.segment_predictions,
            "split_pred": r.split_pred,
            "split_cer": r.split_cer,
            "split_wer": r.split_wer,
            "cer_improvement": r.cer_improvement,
            "wer_improvement": r.wer_improvement,
        })

    with open(output_path / "split_experiment_details.json", "w") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    # Save CSV for easy analysis
    with open(output_path / "split_experiment_comparison.csv", "w") as f:
        f.write("sample_id,length,was_split,num_segments,original_cer,split_cer,cer_improvement,original_wer,split_wer\n")
        for r in results.sample_results:
            f.write(f"{r.sample_id},{r.original_length},{r.was_split},{r.num_segments},"
                    f"{r.original_cer:.6f},{r.split_cer:.6f},{r.cer_improvement:.6f},"
                    f"{r.original_wer:.6f},{r.split_wer:.6f}\n")

    print(f"\nResults saved to {output_path}/")


def run_experiment(
    model_name: str = "ctc_simple",
    weights_path: Optional[str] = None,
    hf_sample_size: int = 1000,
    include_book: bool = True,
    output_dir: str = "results/split_experiment",
    seed: int = 42
) -> ExperimentResults:
    """Run the full split experiment."""

    print("=" * 70)
    print("CONJUNCTION-BASED SPLIT EXPERIMENT")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    if weights_path:
        print(f"Weights: {weights_path}")
    print(f"HF samples: {hf_sample_size}")
    print(f"Include book data: {include_book}")

    # Initialize pipeline
    from pipeline import OCRPipeline
    pipeline = OCRPipeline(model_name, weights_path=weights_path)

    # Load test data
    print("\n" + "-" * 40)
    test_data = load_test_data(hf_sample_size, seed, include_book)

    # Find font for synthetic image generation
    font_path = find_sami_font()
    print(f"\nUsing font: {font_path}")

    # Process all samples
    print("\n" + "-" * 40)
    print("Processing samples...")

    sample_results = []
    start_time = time.time()

    for i, sample in enumerate(test_data):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_data) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(test_data)} ({rate:.1f} samples/s, ETA: {eta:.0f}s)")

        result = process_sample(pipeline, sample, font_path)
        sample_results.append(result)

    total_time = time.time() - start_time
    print(f"\nProcessed {len(test_data)} samples in {total_time:.1f}s ({len(test_data)/total_time:.1f} samples/s)")

    # Aggregate results
    results = aggregate_results(model_name, sample_results)

    # Print and save
    print_summary(results)
    save_results(results, output_dir)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run conjunction-based split experiment")
    parser.add_argument("--model", type=str, default="ctc_simple",
                       help="Model name (default: ctc_simple)")
    parser.add_argument("--weights", type=str,
                       help="Path to model weights (required for custom models)")
    parser.add_argument("--hf-samples", type=int, default=1000,
                       help="Number of HuggingFace validation samples (default: 1000)")
    parser.add_argument("--no-book", action="store_true",
                       help="Exclude book test data")
    parser.add_argument("--output-dir", type=str, default="results/split_experiment",
                       help="Output directory (default: results/split_experiment)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-discover weights if not provided
    weights_path = args.weights
    if weights_path is None and args.model != "trocr_smi_synth":
        # Try to find trained weights
        candidates = list(Path("trained_models").glob(f"*/{args.model}/checkpoint_best.pt"))
        if candidates:
            weights_path = str(candidates[-1])  # Use most recent
            print(f"Auto-discovered weights: {weights_path}")
        elif args.model.startswith("trocr_"):
            pass  # TrOCR models don't need weights
        else:
            print(f"Error: No weights found for {args.model}")
            print("Either specify --weights or train the model first")
            sys.exit(1)

    run_experiment(
        model_name=args.model,
        weights_path=weights_path,
        hf_sample_size=args.hf_samples,
        include_book=not args.no_book,
        output_dir=args.output_dir,
        seed=args.seed
    )
