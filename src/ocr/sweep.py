#!/usr/bin/env python3
"""Hyperparameter sweep for OCR models.

Grid search over learning rate, batch size, and image width.
Results are logged to a CSV for analysis.

Examples:
    # Sweep different architectures
    python sweep.py --model crnn_simple --epochs 20
    python sweep.py --model crnn_vgg16 --epochs 20
    python sweep.py --model ctc_resnet50 --epochs 20
    python sweep.py --model transformer_simple --epochs 20
"""

import argparse
import csv
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for OCR training")
    parser.add_argument(
        "--model",
        type=str,
        default="crnn_simple",
        help="Architecture to sweep. Examples: crnn_simple, crnn_vgg16, ctc_vgg16, transformer_vgg16, etc. "
             "See models/__init__.py for all available architectures.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[1e-4, 5e-4, 1e-3],
        help="Learning rates to try",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Batch sizes to try",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="Image widths to try (simple models only)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs per configuration",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5000,
        help="Samples to use (for faster sweeps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/ocr/sweep_results",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without running",
    )
    return parser.parse_args()


def run_training(model: str, config: dict, output_dir: Path, seed: int) -> dict:
    """Run a single training configuration and return metrics."""
    # Use train_unified.py with architecture presets
    cmd = [
        sys.executable,
        "src/ocr/train_unified.py",
        "--architecture", model,
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--img-width", str(config.get("img_width", 800)),
        "--sample", str(config["sample"]),
        "--seed", str(seed),
        "--patience", "10",
        "--output-dir", str(output_dir / "weights"),
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        elapsed = time.time() - start_time

        # Parse output for best CER
        best_cer = None
        best_acc = None
        for line in result.stdout.split("\n"):
            if "Best CER:" in line:
                try:
                    best_cer = float(line.split("Best CER:")[-1].strip())
                except ValueError:
                    pass
            if "Saved best model (CER:" in line:
                try:
                    cer_str = line.split("CER:")[-1].strip().rstrip(")")
                    best_cer = float(cer_str)
                except ValueError:
                    pass

        return {
            **config,
            "status": "success" if result.returncode == 0 else "failed",
            "best_cer": best_cer,
            "elapsed_seconds": elapsed,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            **config,
            "status": "timeout",
            "best_cer": None,
            "elapsed_seconds": 3600,
            "returncode": -1,
        }
    except Exception as e:
        return {
            **config,
            "status": "error",
            "best_cer": None,
            "elapsed_seconds": 0,
            "returncode": -1,
            "error": str(e),
        }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate configurations
    # Sweep img_width only for SimpleCNN models
    if "simple" in args.model:
        configs = [
            {
                "lr": lr,
                "batch_size": bs,
                "img_width": iw,
                "epochs": args.epochs,
                "sample": args.sample,
            }
            for lr, bs, iw in itertools.product(args.lr, args.batch_size, args.img_width)
        ]
    else:
        # For VGG/ResNet, use fixed img_width
        configs = [
            {
                "lr": lr,
                "batch_size": bs,
                "img_width": 800,  # Fixed for larger models
                "epochs": args.epochs,
                "sample": args.sample,
            }
            for lr, bs in itertools.product(args.lr, args.batch_size)
        ]

    print(f"Hyperparameter Sweep for {args.model}")
    print(f"Training script: train_unified.py")
    print(f"Total configurations: {len(configs)}")
    print(f"Learning rates: {args.lr}")
    print(f"Batch sizes: {args.batch_size}")
    if "simple" in args.model:
        print(f"Image widths: {args.img_width}")
    print(f"Epochs per config: {args.epochs}")
    print(f"Samples: {args.sample}")
    print()

    if args.dry_run:
        print("Configurations to run:")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. lr={cfg['lr']}, bs={cfg['batch_size']}, iw={cfg['img_width']}")
        return

    # Run sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"sweep_{args.model}_{timestamp}.csv"

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Configuration {i}/{len(configs)}")
        print(f"lr={config['lr']}, batch_size={config['batch_size']}, img_width={config['img_width']}")
        print(f"{'='*60}")

        result = run_training(args.model, config, output_dir, args.seed)
        result["model"] = args.model
        result["config_id"] = i
        results.append(result)

        # Save intermediate results
        fieldnames = ["config_id", "model", "lr", "batch_size", "img_width", "epochs",
                      "sample", "status", "best_cer", "elapsed_seconds"]
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"Status: {result['status']}, Best CER: {result['best_cer']}")

    # Print summary
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success" and r["best_cer"] is not None]
    if successful:
        best = min(successful, key=lambda x: x["best_cer"])
        print(f"\nBest configuration:")
        print(f"  Learning rate: {best['lr']}")
        print(f"  Batch size: {best['batch_size']}")
        print(f"  Image width: {best['img_width']}")
        print(f"  Best CER: {best['best_cer']:.4f}")
    else:
        print("\nNo successful runs found.")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
