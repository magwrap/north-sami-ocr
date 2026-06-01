#!/usr/bin/env python3
"""
OCR + Translation Pipeline with Benchmarking

Performs OCR on Sami text images and translates to English using TartuNLP API.
Supports both pre-trained TrOCR models and custom trained models.

Default mode runs benchmarks on all synthetic images with OCR and translation metrics.

Usage:
    # List available models
    python src/ocr_translate_pipeline.py --list-models

    # Benchmark mode with interactive model selection (DEFAULT)
    python src/ocr_translate_pipeline.py --interactive

    # Benchmark mode with explicit model
    python src/ocr_translate_pipeline.py --model trocr_smi_pred_synth

    # Single-image mode (for individual images)
    python src/ocr_translate_pipeline.py --image test.jpg --model trocr_smi_pred_synth --single-image

    # Using trained model for benchmarking
    python src/ocr_translate_pipeline.py \\
        --model crnn_vgg16 \\
        --weights trained_models/2024-03-19_queue/crnn_vgg16/checkpoint_best.pt
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Add src/ocr to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ocr.pipeline import OCRPipeline, ModularOCRModel, list_models
from ocr.train_utils import compute_cer, compute_wer, compute_accuracy, clean_text
from translation.sme_eng import translate_to_english

from metrics import compute_bleu, compute_chrf, compute_ter


def discover_trained_models(base_dir: Path = Path("trained_models")) -> list[dict]:
    """
    Scan for trained models in trained_models/**/checkpoint_best.pt

    Returns:
        List of dicts with keys: name (architecture), path (checkpoint path),
        experiment (parent directory name)
    """
    if not base_dir.exists():
        return []

    models = []
    for checkpoint in base_dir.glob("*/*/checkpoint_best.pt"):
        architecture = checkpoint.parent.name
        experiment = checkpoint.parent.parent.name
        models.append({
            "name": architecture,
            "path": str(checkpoint),
            "experiment": experiment
        })

    return sorted(models, key=lambda x: (x["experiment"], x["name"]))


def get_trocr_models() -> list[str]:
    """Get list of available TrOCR models from pipeline registry."""
    all_models = list_models()
    return [m for m in all_models if m.startswith("trocr_")]


def list_all_models() -> dict:
    """
    Combine TrOCR and trained models into categorized dict.

    Returns:
        {"trocr": [...], "trained": [...]}
    """
    return {
        "trocr": get_trocr_models(),
        "trained": discover_trained_models()
    }


def load_model(model_name: str, weights_path: Optional[str] = None):
    """
    Load model using hybrid approach.

    Args:
        model_name: Model identifier (e.g., "trocr_smi_pred_synth" or "crnn_vgg16")
        weights_path: Path to checkpoint file (required for non-TrOCR models)

    Returns:
        Model instance with .recognize() method
    """
    # TrOCR models: use OCRPipeline (no weights needed)
    if model_name.startswith("trocr_"):
        return OCRPipeline(model_name=model_name)

    # Trained models: use ModularOCRModel directly (requires weights_path)
    if weights_path is None:
        raise ValueError(
            f"Model '{model_name}' requires --weights parameter. "
            f"Trained models need explicit checkpoint paths."
        )

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    return ModularOCRModel(model_id=model_name, weights_path=weights_path)


def recognize_text(model, image_path: str) -> str:
    """
    Run OCR inference.

    Args:
        model: OCRPipeline or ModularOCRModel instance
        image_path: Path to image file

    Returns:
        Recognized text string
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    return model.recognize(image)



def interactive_select_model() -> tuple[str, Optional[str]]:
    """
    Prompt user to select from available models.

    Returns:
        Tuple of (model_name, weights_path)
    """
    models = list_all_models()

    print("\n=== Available Models ===\n")

    # Display TrOCR models
    print("TrOCR Models (pre-trained):")
    options = []
    for i, model in enumerate(models["trocr"], start=1):
        print(f"  {i}. {model}")
        options.append(("trocr", model, None))

    # Display trained models
    if models["trained"]:
        print("\nTrained Models:")
        start_idx = len(options) + 1
        for i, model_info in enumerate(models["trained"], start=start_idx):
            print(f"  {i}. {model_info['name']} ({model_info['experiment']})")
            options.append(("trained", model_info["name"], model_info["path"]))

    # Prompt for selection
    print()
    while True:
        try:
            choice = input(f"Select model (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                _, model_name, weights_path = options[idx]
                return model_name, weights_path
            else:
                print(f"Invalid choice. Please enter 1-{len(options)}")
        except (ValueError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)


def display_models_list():
    """Display all available models and exit."""
    models = list_all_models()

    print("\n=== Available OCR Models ===\n")

    print("TrOCR Models (pre-trained from HuggingFace):")
    for model in models["trocr"]:
        print(f"  • {model}")

    if models["trained"]:
        print("\nTrained Models (from train_queue.py):")
        for model_info in models["trained"]:
            print(f"  • {model_info['name']}")
            print(f"    Experiment: {model_info['experiment']}")
            print(f"    Weights: {model_info['path']}")
    else:
        print("\nTrained Models: (none found)")
        print("  Run 'python src/ocr/train_queue.py' to train custom models")

    print()


def load_ground_truth(json_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth JSON with Sami text and English translations.

    Args:
        json_path: Path to ground_truth.json file

    Returns:
        Dictionary mapping entry IDs to {original, translation}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Remove the 'length' key if present
    if 'length' in data:
        del data['length']

    return data


def discover_synthetic_images(
    images_dir: str,
    ground_truth: Dict[str, Dict[str, str]]
) -> List[Tuple[str, str, str, str]]:
    """
    Scan for entry_*.png files and match with ground truth.

    Args:
        images_dir: Directory containing synthetic images
        ground_truth: Ground truth dictionary from load_ground_truth()

    Returns:
        List of (id, image_path, gt_sami, gt_english) tuples
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images_data = []
    for image_file in sorted(images_path.glob("entry_*.png")):
        # Extract ID from filename (e.g., "entry_1.png" -> "1")
        entry_id = image_file.stem.replace("entry_", "")

        # Check if ID exists in ground truth
        if entry_id in ground_truth:
            gt_entry = ground_truth[entry_id]
            images_data.append((
                entry_id,
                str(image_file),
                gt_entry["original"],
                gt_entry["translation"]
            ))

    return images_data


def process_batch(
    model,
    images_data: List[Tuple[str, str, str, str]]
) -> List[Dict]:
    """
    Process all images: OCR → Translation → Metrics.

    Args:
        model: Loaded OCR model instance
        images_data: List of (id, image_path, gt_sami, gt_english) tuples

    Returns:
        List of result dictionaries with predictions and metrics
    """
    results = []
    total = len(images_data)

    print(f"[2/3] Processing {total} images and translations...")

    for i, (entry_id, image_path, gt_sami, gt_english) in enumerate(images_data, 1):
        # Progress indicator
        progress_pct = (i / total) * 100
        bar_length = 30
        filled = int(bar_length * i / total)
        bar = "=" * filled + ">" + " " * (bar_length - filled - 1)
        print(f"\r  Progress: [{bar}] {i}/{total} ({progress_pct:.1f}%)", end="", flush=True)

        try:
            # Step 1: OCR
            image = Image.open(image_path)
            predicted_sami = model.recognize(image)

            # Step 2: Translate
            predicted_english = translate_to_english(predicted_sami)

            # clean the text
            gt_sami = clean_text(gt_sami)
            gt_english = clean_text(gt_english)
            predicted_sami = clean_text(predicted_sami)
            predicted_english = clean_text(predicted_english)

            # Step 3: Compute OCR metrics
            cer = compute_cer(predicted_sami, gt_sami)
            wer = compute_wer(predicted_sami, gt_sami)
            accuracy = compute_accuracy(predicted_sami, gt_sami)

            # Store result
            results.append({
                "id": entry_id,
                "image_path": image_path,
                "ground_truth_sami": gt_sami,
                "predicted_sami": predicted_sami,
                "cer": cer,
                "wer": wer,
                "accuracy": accuracy,
                "ground_truth_english": gt_english,
                "predicted_english": predicted_english
            })

            # print result
            print(f"ID: {entry_id}")
            print(f"  Image: {image_path}")
            print(f"  GT Sami: {gt_sami}")
            print(f"  Pred Sami: {predicted_sami}")
            print(f"  CER: {cer:.4f}, WER: {wer:.4f}, Accuracy: {accuracy:.2f}")
            print(f"  GT English: {gt_english}")
            print(f"  Pred English: {predicted_english}")
            print()

        except Exception as e:
            print(f"\n  Warning: Failed to process {image_path}: {e}")
            # Store partial result with error
            results.append({
                "id": entry_id,
                "image_path": image_path,
                "ground_truth_sami": gt_sami,
                "predicted_sami": f"ERROR: {e}",
                "cer": 1.0,
                "wer": 1.0,
                "accuracy": 0.0,
                "ground_truth_english": gt_english,
                "predicted_english": ""
            })

    print()  # New line after progress bar
    return results


def aggregate_results(per_sample_results: List[Dict]) -> Dict:
    """
    Compute system-level aggregate metrics.

    Args:
        per_sample_results: List of per-sample result dictionaries

    Returns:
        Dictionary with aggregate OCR and translation metrics
    """
    print("[3/3] Computing aggregate metrics...")

    # Aggregate OCR metrics (average)
    mean_cer = sum(r["cer"] for r in per_sample_results) / len(per_sample_results)
    mean_wer = sum(r["wer"] for r in per_sample_results) / len(per_sample_results)
    accuracy_rate = sum(r["accuracy"] for r in per_sample_results) / len(per_sample_results)
    exact_matches = sum(1 for r in per_sample_results if r["accuracy"] == 1.0)

    # Aggregate translation metrics (corpus-level)
    all_references = [r["ground_truth_english"] for r in per_sample_results]
    all_predictions = [r["predicted_english"] for r in per_sample_results]

    # Compute corpus-level translation metrics
    bleu_result = compute_bleu(all_references, all_predictions)
    chrf_result = compute_chrf(all_references, all_predictions)
    ter_result = compute_ter(all_references, all_predictions)

    return {
        "ocr": {
            "mean_cer": mean_cer,
            "mean_wer": mean_wer,
            "accuracy": accuracy_rate,
            "exact_matches": exact_matches
        },
        "translation": {
            "bleu": bleu_result["bleu"],
            "chrf": chrf_result["chrf"],
            "ter": ter_result["ter"]
        }
    }


def save_results(
    results: Dict,
    per_sample_results: List[Dict],
    output_dir: str,
    model_name: str,
    weights_path: Optional[str],
    images_dir: str,
    ground_truth_file: str
):
    """
    Save results to CSV and JSON files.

    Args:
        results: Aggregate metrics dictionary
        per_sample_results: List of per-sample results
        output_dir: Base output directory
        model_name: Name of the model used
        weights_path: Path to weights (if applicable)
        images_dir: Directory containing test images
        ground_truth_file: Path to ground truth JSON
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = Path(output_dir) / f"{timestamp}_{model_name}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save CSV (per-sample breakdown)
    # csv_path = output_path / "results.csv"
    # with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    #     if per_sample_results:
    #         writer = csv.DictWriter(f, fieldnames=per_sample_results[0].keys())
    #         writer.writeheader()
    #         writer.writerows(per_sample_results)


    # print per-sample results to console
    # for sample_result in per_sample_results:
    #     print(f"ID: {sample_result['id']}")
    #     print(f"  Image: {sample_result['image_path']}")
    #     print(f"  GT Sami: {sample_result['ground_truth_sami']}")
    #     print(f"  Pred Sami: {sample_result['predicted_sami']}")
    #     print(f"  CER: {sample_result['cer']:.4f}, WER: {sample_result['wer']:.4f}, Accuracy: {sample_result['accuracy']:.2f}")
    #     print(f"  GT English: {sample_result['ground_truth_english']}")
    #     print(f"  Pred English: {sample_result['predicted_english']}")
    #     print()

    # Save JSON (complete results with metadata)
    json_path = output_path / "results.json"
    json_data = {
        "metadata": {
            "model": model_name,
            "weights_path": weights_path,
            "images_dir": images_dir,
            "ground_truth_file": ground_truth_file,
            "timestamp": timestamp,
            "total_samples": len(per_sample_results)
        },
        "aggregate_metrics": results,
        "per_sample_results": per_sample_results
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {json_path}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="OCR + Translation Pipeline for Northern Sami",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (only used with --single-image)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., trocr_smi_pred_synth, crnn_vgg16)"
    )

    parser.add_argument(
        "--weights",
        type=str,
        help="Path to checkpoint file (required for non-TrOCR models)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Interactive mode: prompts for model selection (DEFAULT)"
    )

    parser.add_argument(
        "--single-image",
        action="store_true",
        help="Process a single image instead of running benchmark mode"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="pipeline_benchmark_results",
        help="Output directory for pipeline benchmark results (default: pipeline_benchmark_results/)"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Handle --list-models flag
    if args.list_models:
        display_models_list()
        return

    # Model selection: interactive vs explicit
    if args.interactive and not args.model:
        model_name, weights_path = interactive_select_model()
    else:
        if not args.model:
            print("Error: Either --model or --interactive is required")
            print("Use --list-models to see available models")
            sys.exit(1)
        model_name = args.model
        weights_path = args.weights

    # Branch: Single-image mode vs Benchmark mode
    if args.single_image:
        # ===== SINGLE-IMAGE MODE (Original Behavior) =====
        if not args.image:
            print("Error: --image is required in single-image mode")
            print("Usage: python src/ocr_translate_pipeline.py --image path/to/image.jpg --single-image [options]")
            sys.exit(1)

        # Display pipeline header
        print("\n=== OCR + Translation Pipeline ===")
        print(f"Model: {model_name}")
        if weights_path:
            print(f"Weights: {weights_path}")
        print(f"Image: {args.image}")
        print()

        try:
            # Step 1: Load model
            print("[1/3] Loading model...")
            model = load_model(model_name, weights_path)

            # Step 2: Run OCR
            print("[2/3] Running OCR...")
            sami_text = recognize_text(model, args.image)
            print(f"Northern Sami: {sami_text}")
            print()

            # Step 3: Translate
            print("[3/3] Translating to English...")
            english_text = translate_to_english(sami_text)
            print(f"English: {english_text}")
            print()

            print("Done!")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        # ===== BENCHMARK MODE (New Default Behavior) =====
        ground_truth_file = "test_data/account_of_sami/ground_truth.json"
        images_dir = "test_data/account_of_sami/synthetic_images/"

        try:
            # Load ground truth and discover images
            ground_truth = load_ground_truth(ground_truth_file)
            images_data = discover_synthetic_images(images_dir, ground_truth)

            if not images_data:
                print(f"Error: No synthetic images found in {images_dir}")
                sys.exit(1)

            # Display benchmark header
            print("\n=== OCR + Translation Benchmark ===")
            print(f"Model: {model_name}")
            if weights_path:
                print(f"Weights: {weights_path}")
            print(f"Dataset: {images_dir}")
            print(f"Total Images: {len(images_data)}")
            print()

            # Step 1: Load model
            print("[1/3] Loading model...")
            model = load_model(model_name, weights_path)

            # Step 2: Process all images
            per_sample_results = process_batch(model, images_data)

            # Step 3: Aggregate metrics
            aggregate_metrics = aggregate_results(per_sample_results)

            # Display results
            print("\n=== Results ===\n")
            print("OCR Performance:")
            print(f"  CER:      {aggregate_metrics['ocr']['mean_cer']:.4f} ({aggregate_metrics['ocr']['mean_cer']*100:.2f}% error)")
            print(f"  WER:      {aggregate_metrics['ocr']['mean_wer']:.4f} ({aggregate_metrics['ocr']['mean_wer']*100:.2f}% error)")
            print(f"  Accuracy: {aggregate_metrics['ocr']['accuracy']*100:.2f}% ({aggregate_metrics['ocr']['exact_matches']}/{len(images_data)} exact matches)")
            print()
            print("Translation Performance:")
            print(f"  BLEU:     {aggregate_metrics['translation']['bleu']:.2f}")
            print(f"  chrF:     {aggregate_metrics['translation']['chrf']:.2f}")
            print(f"  TER:      {aggregate_metrics['translation']['ter']:.4f} ({aggregate_metrics['translation']['ter']:.2f}% edit rate)")
            print()

            # Save results
            save_results(
                aggregate_metrics,
                per_sample_results,
                args.output_dir,
                model_name,
                weights_path,
                images_dir,
                ground_truth_file
            )

            print("\nBenchmark complete!")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
