#!/usr/bin/env python3
"""Benchmark OCR models on Sami synthetic dataset."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Allow running as script or module
# if __package__ is None or __package__ == "":
#     sys.path.insert(0, str(Path(__file__).parent))
#     from pipeline import OCRPipeline, list_models
#     from train_utils import compute_cer, compute_wer, compute_normalized_accuracy
# else:
from pipeline import OCRPipeline, list_models
from train_utils import compute_cer, compute_wer, compute_normalized_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark OCR models")
    parser.add_argument(
        "--models", nargs="+",
        help="Models to benchmark (default: all TrOCR + trained models)"
    )
    parser.add_argument(
        "--weights", type=str,
        help="Path to model weights (overrides auto-discovery)"
    )
    parser.add_argument(
        "--sample", type=int, default=1000,
        help="Number of HF validation samples (default: 1000)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable prediction caching"
    )
    parser.add_argument(
        "--no-book-data", action="store_true",
        help="Exclude 'Account of Sami' book test data"
    )
    parser.add_argument(
        "--no-trained", action="store_true",
        help="Skip auto-discovery of trained models"
    )
    parser.add_argument(
        "--output-dir", type=str, default="src/ocr/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
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
        test_data.append({"id": f"hf_{i}", "image": dataset[i]["image"], "text": dataset[i]["text"]})
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

            print(f"Added {len([d for d in test_data if d['id'].startswith('book_')])} book samples")

    print(f"Total: {len(test_data)} test samples")
    return test_data


def get_cache_path(output_dir: str, model_name: str) -> Path:
    """Get cache file path."""
    cache_dir = Path(output_dir) / "predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model_name}.jsonl"


def load_cache(cache_path: Path):
    """Load cached predictions."""
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return [json.loads(line) for line in f]


def save_cache(predictions: list, cache_path: Path):
    """Save predictions to cache."""
    with open(cache_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def discover_trained_models(base_dir: str = "trained_models") -> dict:
    """
    Auto-discover trained model checkpoints.

    Returns:
        Dict mapping model_name -> checkpoint_path
    """
    trained_models = {}
    base_path = Path(base_dir)

    if not base_path.exists():
        return trained_models

    # Search all queue directories
    for queue_dir in base_path.iterdir():
        if not queue_dir.is_dir():
            continue

        # Search each model subdirectory
        for model_dir in queue_dir.iterdir():
            if not model_dir.is_dir():
                continue

            checkpoint = model_dir / "checkpoint_best.pt"
            if checkpoint.exists():
                model_name = model_dir.name
                trained_models[model_name] = str(checkpoint)

    return trained_models


def evaluate_model(model_name: str, test_data: list, output_dir: str, use_cache: bool, weights_path: str = None):
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    cache_path = get_cache_path(output_dir, model_name)

    # Try cache
    if use_cache:
        cached = load_cache(cache_path)
        if cached:
            print(f"Using cached predictions ({len(cached)} samples)")
            predictions = [c["prediction"] for c in cached]
            references = [c["reference"] for c in cached]
            inference_time = 0.0
        else:
            cached = None
    else:
        cached = None

    # Run inference if not cached
    if cached is None:
        try:
            pipeline = OCRPipeline(model_name, weights_path=weights_path)
        except Exception as e:
            print(f"Failed to load: {e}")
            return {"model": model_name, "status": "failed", "error": str(e)}

        predictions = []
        references = []
        cache_data = []

        start = time.time()
        for i, item in enumerate(test_data):
            try:
                pred = pipeline.recognize(item["image"])
            except Exception as e:
                pred = ""
            predictions.append(pred)
            references.append(item["text"])
            cache_data.append({"id": item["id"], "prediction": pred, "reference": item["text"]})

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(test_data)}")

        inference_time = time.time() - start
        print(f"Inference: {inference_time:.1f}s ({len(test_data)/inference_time:.1f} samples/s)")

        if use_cache:
            save_cache(cache_data, cache_path)

    # Compute metrics (overall + per-dataset breakdown)
    total_cer, total_wer, exact = 0.0, 0.0, 0
    hf_cer, hf_wer, hf_exact, hf_count = 0.0, 0.0, 0, 0
    book_cer, book_wer, book_exact, book_count = 0.0, 0.0, 0, 0
    total_normalized = 0
    hf_normalized = 0
    book_normalized = 0

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
        item_id = test_data[i]["id"] if i < len(test_data) else ""
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
    metrics = {
        "cer": total_cer / n,
        "wer": total_wer / n,
        "accuracy": exact / n,
        "normalized_accuracy": total_normalized / n
    }

    # Add per-dataset metrics if both sources present
    if hf_count > 0 and book_count > 0:
        metrics["hf_cer"] = hf_cer / hf_count
        metrics["hf_wer"] = hf_wer / hf_count
        metrics["hf_accuracy"] = hf_exact / hf_count
        metrics["hf_normalized_accuracy"] = hf_normalized / hf_count
        metrics["book_cer"] = book_cer / book_count
        metrics["book_wer"] = book_wer / book_count
        metrics["book_accuracy"] = book_exact / book_count
        metrics["book_normalized_accuracy"] = book_normalized / book_count

    print(f"Overall - CER: {metrics['cer']*100:.2f}% | WER: {metrics['wer']*100:.2f}% | Acc: {metrics['accuracy']*100:.1f}% | Norm Acc: {metrics['normalized_accuracy']*100:.1f}%")
    if "hf_cer" in metrics:
        print(f"HF Val  - CER: {metrics['hf_cer']*100:.2f}% | WER: {metrics['hf_wer']*100:.2f}% | Acc: {metrics['hf_accuracy']*100:.1f}% | Norm Acc: {metrics['hf_normalized_accuracy']*100:.1f}%")
        print(f"Book    - CER: {metrics['book_cer']*100:.2f}% | WER: {metrics['book_wer']*100:.2f}% | Acc: {metrics['book_accuracy']*100:.1f}% | Norm Acc: {metrics['book_normalized_accuracy']*100:.1f}%")

    return {
        "model": model_name,
        "status": "success",
        "num_samples": n,
        "hf_count": hf_count,
        "book_count": book_count,
        "metrics": metrics,
        "inference_time": inference_time
    }


def print_results_table(results: list):
    """Print formatted results table."""
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        print("\nNo successful evaluations.")
        return

    successful.sort(key=lambda x: x["metrics"]["cer"])

    print("\n" + "=" * 70)
    print("OCR BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<30} {'CER':>12} {'WER':>12} {'Accuracy':>12}")
    print("-" * 70)

    for r in successful:
        m = r["metrics"]
        print(f"{r['model']:<30} {m['cer']*100:>11.2f}% {m['wer']*100:>11.2f}% {m['accuracy']*100:>11.2f}%")

    print("=" * 70)

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  {r['model']}: {r.get('error', 'Unknown')}")


def save_results(results: list, output_dir: str):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV with expanded per-dataset columns
    csv_path = output_path / "benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("model,num_samples,cer,wer,accuracy,normalized_accuracy,")
        f.write("hf_cer,hf_wer,hf_accuracy,hf_normalized_accuracy,")
        f.write("book_cer,book_wer,book_accuracy,book_normalized_accuracy,")
        f.write("hf_count,book_count,inference_time\n")

        for r in sorted(results, key=lambda x: x.get("metrics", {}).get("cer", 999)):
            if r["status"] == "success":
                m = r["metrics"]
                # Overall metrics
                row = [
                    r['model'],
                    str(r['num_samples']),
                    f"{m['cer']:.6f}",
                    f"{m['wer']:.6f}",
                    f"{m['accuracy']:.6f}",
                    f"{m.get('normalized_accuracy', -1):.6f}"
                ]

                # Per-dataset metrics (use -1 sentinel if missing)
                for prefix in ['hf', 'book']:
                    for metric in ['cer', 'wer', 'accuracy', 'normalized_accuracy']:
                        key = f"{prefix}_{metric}"
                        row.append(f"{m.get(key, -1):.6f}")

                # Sample counts
                row.append(str(r.get('hf_count', 0)))
                row.append(str(r.get('book_count', 0)))
                row.append(f"{r['inference_time']:.2f}")

                f.write(",".join(row) + "\n")
    print(f"\nSaved: {csv_path}")

    # JSON
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    args = parse_args()

    # Discover trained models
    trained_checkpoints = {}
    if not args.no_trained:
        trained_checkpoints = discover_trained_models()
        if trained_checkpoints:
            print(f"\nDiscovered trained models: {list(trained_checkpoints.keys())}")

    # Get models to test
    available = list_models()
    models_to_test = []

    if args.models:
        # User specified models
        for m in args.models:
            if m in available:
                models_to_test.append((m, trained_checkpoints.get(m, args.weights)))
            else:
                print(f"Skipping unavailable: {m}")
    else:
        # Test all available TrOCR models
        trocr_models = [m for m in available if m.startswith("trocr_")]
        for m in trocr_models:
            models_to_test.append((m, None))

        # Test all discovered trained models
        if not args.no_trained:
            for model_name, checkpoint_path in trained_checkpoints.items():
                if model_name in available:
                    models_to_test.append((model_name, checkpoint_path))

    print(f"\nModels to test ({len(models_to_test)}):")
    for model_name, weights in models_to_test:
        if weights:
            print(f"  {model_name} (weights: {weights})")
        else:
            print(f"  {model_name}")

    # Load test data
    test_data = load_test_data(args.sample, args.seed, include_book_data=not args.no_book_data)

    # Evaluate each model
    results = []
    for model_name, weights_path in models_to_test:
        result = evaluate_model(model_name, test_data, args.output_dir, not args.no_cache, weights_path)
        results.append(result)

    # Output
    print_results_table(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
