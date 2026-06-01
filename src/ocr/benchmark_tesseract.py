#!/usr/bin/env python3
"""
Benchmark Tesseract OCR baseline on Sámi synthetic dataset.

This script evaluates Tesseract 5 with the Northern Sámi language model (sme)
on the same test set used for neural OCR benchmarks, enabling direct comparison.

Usage:
    # From project root, within nix develop shell:
    python src/ocr/benchmark_tesseract.py

    # With options:
    python src/ocr/benchmark_tesseract.py --sample 100 --lang sme

Requirements:
    - Tesseract 5 with sme.traineddata (installed via flake.nix)
    - pytesseract Python package (in requirements.txt)
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

# Reuse existing utilities
from train_utils import compute_cer, compute_wer, compute_normalized_accuracy


def check_tesseract_installation():
    """Verify Tesseract is installed and sme language is available."""
    try:
        # Check tesseract version
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True, text=True, check=True
        )
        version_line = result.stdout.split('\n')[0]
        print(f"Tesseract: {version_line}")

        # Check available languages
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True, text=True, check=True
        )
        available_langs = result.stdout.strip().split('\n')[1:]  # Skip header

        return available_langs

    except FileNotFoundError:
        print("ERROR: Tesseract not found. Run 'nix develop' to enter the dev shell.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Tesseract check failed: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Tesseract OCR on Sámi dataset"
    )
    parser.add_argument(
        "--lang", type=str, default="sme",
        help="Tesseract language code (default: sme for Northern Sámi)"
    )
    parser.add_argument(
        "--sample", type=int, default=1000,
        help="Number of HF validation samples (default: 1000)"
    )
    parser.add_argument(
        "--no-book-data", action="store_true",
        help="Exclude 'Account of Sami' book test data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="src/ocr/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--psm", type=int, default=7,
        help="Tesseract page segmentation mode (default: 7 = single text line)"
    )
    parser.add_argument(
        "--oem", type=int, default=3,
        help="Tesseract OCR engine mode (default: 3 = LSTM only)"
    )
    return parser.parse_args()


def load_test_data(sample_size: int, seed: int, include_book_data: bool = True):
    """Load test samples from HuggingFace dataset + book synthetic data."""
    from datasets import load_dataset
    from PIL import Image

    test_data = []

    # Load HF validation split (unseen during training)
    print("Loading HF validation split...")
    dataset = load_dataset("Sprakbanken/synthetic_sami_ocr_data", split="validation")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
    indices.sort()

    for i in indices:
        test_data.append({
            "id": f"hf_{i}",
            "image": dataset[i]["image"],
            "text": dataset[i]["text"]
        })
    print(f"Loaded {len(test_data)} samples from HF validation")

    # Add book synthetic data (completely unseen source)
    if include_book_data:
        book_dir = Path(__file__).parent.parent.parent / "test_data" / "account_of_sami"
        ground_truth_path = book_dir / "ground_truth.json"
        images_dir = book_dir / "synthetic_images"

        if ground_truth_path.exists():
            print("Loading 'Account of Sami' book test data...")
            with open(ground_truth_path) as f:
                ground_truth = json.load(f)

            for entry_id, entry in ground_truth.items():
                img_path = images_dir / f"entry_{entry_id}.png"
                if img_path.exists():
                    test_data.append({
                        "id": f"book_{entry_id}",
                        "image": Image.open(img_path),
                        "text": entry["original_formatted"].strip()
                    })

            book_count = len([d for d in test_data if d['id'].startswith('book_')])
            print(f"Added {book_count} book samples")

    print(f"Total: {len(test_data)} test samples")
    return test_data


def recognize_with_tesseract(image, lang: str, psm: int, oem: int) -> str:
    """
    Run Tesseract OCR on a PIL Image.

    Args:
        image: PIL Image
        lang: Tesseract language code
        psm: Page segmentation mode (7 = single line)
        oem: OCR engine mode (3 = LSTM)

    Returns:
        Recognized text string
    """
    import pytesseract

    # Configure Tesseract
    config = f"--psm {psm} --oem {oem}"

    # Convert to RGB if necessary (Tesseract prefers RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Run OCR
    text = pytesseract.image_to_string(image, lang=lang, config=config)

    # Clean up: remove trailing newlines, normalize whitespace
    text = text.strip()

    return text


def main():
    args = parse_args()

    print("=" * 60)
    print("Tesseract OCR Baseline Benchmark")
    print("=" * 60)

    # Check installation
    available_langs = check_tesseract_installation()

    # Check if requested language is available
    if args.lang not in available_langs:
        print(f"\nWARNING: Language '{args.lang}' not found!")
        print(f"Available languages: {', '.join(available_langs)}")

        # Try fallback options
        sami_langs = [l for l in available_langs if l.startswith('sm')]
        if sami_langs:
            args.lang = sami_langs[0]
            print(f"Using fallback: {args.lang}")
        else:
            print("\nTo install Northern Sámi language data:")
            print("  1. Download sme.traineddata from:")
            print("     https://github.com/tesseract-ocr/tessdata_best/blob/main/sme.traineddata")
            print("  2. Place it in your TESSDATA_PREFIX directory")
            print("\nAlternatively, try with --lang eng for a baseline comparison.")
            sys.exit(1)

    print(f"Using language: {args.lang}")
    print(f"PSM (page segmentation): {args.psm}")
    print(f"OEM (engine mode): {args.oem}")

    # Load test data
    print("\n" + "-" * 40)
    test_data = load_test_data(
        args.sample,
        args.seed,
        include_book_data=not args.no_book_data
    )

    # Run benchmark
    print("\n" + "-" * 40)
    print("Running Tesseract OCR...")

    predictions = []
    references = []
    cache_data = []

    start_time = time.time()

    for i, item in enumerate(test_data):
        try:
            pred = recognize_with_tesseract(
                item["image"],
                lang=args.lang,
                psm=args.psm,
                oem=args.oem
            )
        except Exception as e:
            print(f"  Warning: Failed on {item['id']}: {e}")
            pred = ""

        predictions.append(pred)
        references.append(item["text"])
        cache_data.append({
            "id": item["id"],
            "prediction": pred,
            "reference": item["text"]
        })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(test_data)} ({rate:.1f} samples/s)")

    total_time = time.time() - start_time
    print(f"Inference: {total_time:.1f}s ({len(test_data)/total_time:.1f} samples/s)")

    # Compute metrics
    print("\n" + "-" * 40)
    print("Computing metrics...")

    total_cer, total_wer, exact = 0.0, 0.0, 0
    hf_cer, hf_wer, hf_exact, hf_count = 0.0, 0.0, 0, 0
    book_cer, book_wer, book_exact, book_count = 0.0, 0.0, 0, 0
    total_normalized = 0
    hf_normalized, book_normalized = 0, 0

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        cer = compute_cer(pred, ref)
        wer = compute_wer(pred, ref)
        norm_acc = compute_normalized_accuracy(pred, ref)

        total_cer += cer
        total_wer += wer
        total_normalized += norm_acc
        if pred == ref:
            exact += 1

        # Track per-dataset metrics
        item_id = test_data[i]["id"]
        if item_id.startswith("hf_"):
            hf_cer += cer
            hf_wer += wer
            hf_normalized += norm_acc
            if pred == ref:
                hf_exact += 1
            hf_count += 1
        elif item_id.startswith("book_"):
            book_cer += cer
            book_wer += wer
            book_normalized += norm_acc
            if pred == ref:
                book_exact += 1
            book_count += 1

    n = len(predictions)

    # Overall metrics
    metrics = {
        "model": f"tesseract_{args.lang}",
        "language": args.lang,
        "psm": args.psm,
        "oem": args.oem,
        "total_samples": n,
        "cer": total_cer / n * 100,  # As percentage
        "wer": total_wer / n * 100,
        "accuracy": exact / n * 100,
        "normalized_accuracy": total_normalized / n * 100,
        "inference_time_total": total_time,
        "inference_time_per_sample": total_time / n,
        "samples_per_second": n / total_time,
    }

    # Per-dataset breakdown
    if hf_count > 0:
        metrics["hf_cer"] = hf_cer / hf_count * 100
        metrics["hf_wer"] = hf_wer / hf_count * 100
        metrics["hf_accuracy"] = hf_exact / hf_count * 100
        metrics["hf_normalized_accuracy"] = hf_normalized / hf_count * 100
        metrics["hf_count"] = hf_count

    if book_count > 0:
        metrics["book_cer"] = book_cer / book_count * 100
        metrics["book_wer"] = book_wer / book_count * 100
        metrics["book_accuracy"] = book_exact / book_count * 100
        metrics["book_normalized_accuracy"] = book_normalized / book_count * 100
        metrics["book_count"] = book_count

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Tesseract Baseline")
    print("=" * 60)
    print(f"Language:      {args.lang}")
    print(f"Total Samples: {n}")
    print("-" * 40)
    print(f"CER:           {metrics['cer']:.2f}%")
    print(f"WER:           {metrics['wer']:.2f}%")
    print(f"Exact Match:   {metrics['accuracy']:.2f}%")
    print(f"Norm. Acc:     {metrics['normalized_accuracy']:.2f}%")
    print("-" * 40)
    print(f"Time/sample:   {metrics['inference_time_per_sample']:.3f}s")
    print(f"Throughput:    {metrics['samples_per_second']:.1f} samples/s")

    if hf_count > 0 and book_count > 0:
        print("\n--- Per-Dataset Breakdown ---")
        print(f"HuggingFace (n={hf_count}):")
        print(f"  CER: {metrics['hf_cer']:.2f}%, WER: {metrics['hf_wer']:.2f}%")
        print(f"Book (n={book_count}):")
        print(f"  CER: {metrics['book_cer']:.2f}%, WER: {metrics['book_wer']:.2f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = output_dir / f"tesseract_{args.lang}_results.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save predictions (for analysis)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)
    predictions_path = predictions_dir / f"tesseract_{args.lang}.jsonl"
    with open(predictions_path, 'w') as f:
        for item in cache_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Predictions saved to: {predictions_path}")

    # Print comparison hint
    print("\n" + "=" * 60)
    print("COMPARISON WITH NEURAL MODELS:")
    print("-" * 40)
    print("For reference (from benchmark_results.json):")
    print("  trocr_smi_synth:  CER  8.55%")
    print("  ctc_simple:       CER 12.24%")
    print(f"  tesseract_{args.lang}:      CER {metrics['cer']:.2f}%")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
