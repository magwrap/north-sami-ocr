#!/usr/bin/env python3
"""
Training queue for all OCR model architectures.

Trains all 15 model combinations with centralized experiment management,
status tracking, and summary generation.

Usage:
    python src/ocr/train_queue.py --test             # 5 epochs, 500 samples
    python src/ocr/train_queue.py --fast             # Only simple models 
    python src/ocr/train_queue.py --fast --multi-gpu # Fast mode with all GPUs 
    python src/ocr/train_queue.py                    # Production: 100 epochs, all 15 models
    python src/ocr/train_queue.py --status           # Show progress
    python src/ocr/train_queue.py --resume           # Resume interrupted queue
    python src/ocr/train_queue.py --models ctc_simple crnn_vgg16  
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Architecture training order (prioritized by expected performance/speed)
# Tier 1: High Performance + Fast (SimpleCNN variants - proven backbone)
# Tier 2: Moderate Performance + Speed (VGG + encoders - pretrained with sequence modeling)
# Tier 3: Lower Priority (VGG pure CTC retry - less promising)
# Tier 4: Slow Models (ResNet50 variants - 10x slower, only if VGG shows promise)
# Tier 5: Very Slow (ResNet101 - only if ResNet50 exceptional)
ARCHITECTURES = [
    # Tier 1: High Performance + Fast (SimpleCNN variants)
    # "ctc_simple", # already successfully trained 
    "crnn_simple", "transformer_simple",

    # Tier 2: Moderate Performance + Speed (VGG + encoders)
    "crnn_vgg16", "crnn_vgg19",
    "transformer_vgg16", "transformer_vgg19",

    # Tier 3: Lower Priority (VGG pure CTC retry)
    "ctc_vgg16", "ctc_vgg19",

    # Tier 4: Slow Models (ResNet50 variants)
    "crnn_resnet50", "transformer_resnet50", "ctc_resnet50",

    # Tier 5: Very Slow (ResNet101 - only if ResNet50 promising)
    "crnn_resnet101", "transformer_resnet101", "ctc_resnet101",
]

# Fast mode: only SimpleCNN backbone models (fastest training, ~5-9M params each)
FAST_ARCHITECTURES = ["ctc_simple", "crnn_simple", "transformer_simple"]

# Hyperparameters by backbone (adjusted for transfer learning and batch norm stability)
HYPERPARAMS = {
    "simple": {
        "lr": 1e-4,
        "batch_size": 32,
        "patience": 15,
        "warmup_epochs": 0,  # No warmup needed for from-scratch training
        "backbone_lr_mult": 1.0,  # No differential LR (trained from scratch)
        "weight_decay": 1e-4,
    },
    "vgg16": {
        "lr": 3e-5,  # Lower LR for pretrained model (was 1e-4)
        "batch_size": 32,  # Increase for stable batch norm (was 16)
        "patience": 25,  # More patience for slower convergence (was 15)
        "warmup_epochs": 5,  # Gradual LR warmup
        "backbone_lr_mult": 0.1,  # 10x lower LR for pretrained backbone
        "weight_decay": 3e-4,  # Stronger regularization for large model
    },
    "vgg19": {
        "lr": 3e-5,
        "batch_size": 32,  # Increase (was 16)
        "patience": 25,
        "warmup_epochs": 5,
        "backbone_lr_mult": 0.1,
        "weight_decay": 3e-4,
    },
    "resnet50": {
        "lr": 1e-4,  # Increase from 5e-5 (was too conservative)
        "batch_size": 16,  # Increase (was 8) - memory constraint
        "patience": 30,  # More patience
        "warmup_epochs": 10,  # Longer warmup for large model
        "backbone_lr_mult": 0.05,  # 20x lower LR for backbone (massive model)
        "weight_decay": 5e-4,  # Strong regularization
    },
    "resnet101": {
        "lr": 1e-4,
        "batch_size": 12,  # Lower than resnet50 (larger model)
        "patience": 35,
        "warmup_epochs": 15,
        "backbone_lr_mult": 0.03,  # Even lower for huge model
        "weight_decay": 5e-4,
    },
}

# CER thresholds for recommendations
CER_THRESHOLDS = {
    "excellent": 0.1,
    "good": 0.3,
    "needs_sweep": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training queue for OCR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: 5 epochs, 500 samples"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: train only SimpleCNN models (ctc/crnn/transformer_simple)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current queue status"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted training queue"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Train only specific models (e.g., ctc_simple crnn_vgg16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Override sample size"
    )
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Use all available GPUs with DataParallel"
    )
    return parser.parse_args()


def get_backbone(arch: str) -> str:
    """Extract backbone name from architecture."""
    for backbone in ["simple", "vgg16", "vgg19", "resnet50", "resnet101"]:
        if backbone in arch:
            return backbone
    return "simple"


def get_hyperparams(arch: str) -> dict:
    """Get hyperparameters for architecture based on backbone."""
    backbone = get_backbone(arch)
    return HYPERPARAMS.get(backbone, HYPERPARAMS["simple"])


def find_latest_experiment(base_dir: Path) -> Path | None:
    """Find the most recent experiment directory."""
    if not base_dir.exists():
        return None
    dirs = (list(base_dir.glob("*_queue")) +
            list(base_dir.glob("*_test")) +
            list(base_dir.glob("*_fast")))
    dirs = sorted(dirs, reverse=True)
    return dirs[0] if dirs else None


def create_experiment_dir(test_mode: bool, fast_mode: bool = False) -> Path:
    """Create timestamped experiment directory."""
    base_dir = Path("trained_models")
    base_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    suffix = "test" if test_mode else ("fast" if fast_mode else "queue")
    exp_dir = base_dir / f"{timestamp}_{suffix}"

    # Handle existing directory by adding counter
    counter = 1
    original = exp_dir
    while exp_dir.exists():
        exp_dir = Path(f"{original}_{counter}")
        counter += 1

    exp_dir.mkdir(parents=True)
    return exp_dir


def init_status(exp_dir: Path, models: list[str], mode: str) -> dict:
    """Initialize status.json for the queue."""
    status = {
        "started_at": datetime.now().isoformat(),
        "mode": mode,
        "total": len(models),
        "completed": 0,
        "failed": 0,
        "running": None,
        "models": {m: {"status": "pending"} for m in models},
    }
    save_status(exp_dir, status)
    return status


def load_status(exp_dir: Path) -> dict | None:
    """Load status.json from experiment directory."""
    status_path = exp_dir / "status.json"
    if status_path.exists():
        return json.loads(status_path.read_text())
    return None


def save_status(exp_dir: Path, status: dict):
    """Save status.json to experiment directory."""
    status_path = exp_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2))


def save_config(model_dir: Path, arch: str, cmd: list[str], hp: dict):
    """Save model config to JSON."""
    config = {
        "architecture": arch,
        "command": " ".join(cmd),
        "hyperparams": hp,
        "started_at": datetime.now().isoformat(),
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))


def parse_training_log(log_path: Path) -> dict:
    """Extract CER/WER from training log."""
    result = {"cer": None, "wer": None, "success": False}

    if not log_path.exists():
        return result

    content = log_path.read_text()

    # Look for final result: "Training complete. Best CER: 0.1234"
    final_match = re.search(r"Training complete\. Best CER: ([\d.]+)", content)
    if final_match:
        result["cer"] = float(final_match.group(1))
        result["success"] = True

    # Look for per-epoch metrics: "CER: 0.1234 | WER: 0.2345"
    epoch_matches = re.findall(r"CER: ([\d.]+) \| WER: ([\d.]+)", content)
    if epoch_matches:
        # Use last epoch's WER if final result found
        result["wer"] = float(epoch_matches[-1][1])
        if result["cer"] is None:
            result["cer"] = float(epoch_matches[-1][0])
            result["success"] = True

    return result


def get_recommendation(cer: float | None) -> str:
    """Get recommendation based on CER threshold."""
    if cer is None:
        return "failed"
    if cer < CER_THRESHOLDS["excellent"]:
        return "excellent"
    if cer < CER_THRESHOLDS["good"]:
        return "good"
    if cer < CER_THRESHOLDS["needs_sweep"]:
        return "needs_sweep"
    return "needs_investigation"


def train_model(arch: str, exp_dir: Path, epochs: int, sample: int | None, multi_gpu: bool = False) -> dict:
    """Train a single model and return results."""
    model_dir = exp_dir / arch
    model_dir.mkdir(exist_ok=True)
    tb_dir = model_dir / "tensorboard"

    hp = get_hyperparams(arch)

    cmd = [
        sys.executable, "src/ocr/train_unified.py",
        "-a", arch,
        "--epochs", str(epochs),
        "--batch-size", str(hp["batch_size"]),
        "--lr", str(hp["lr"]),
        "--patience", str(hp["patience"]),
        "--weight-decay", str(hp.get("weight_decay", 1e-4)),
        "--warmup-epochs", str(hp.get("warmup_epochs", 0)),
        "--backbone-lr-mult", str(hp.get("backbone_lr_mult", 1.0)),
        "--tensorboard",
        "--log-dir", str(tb_dir),
        "--output-dir", str(model_dir),
        "--experiment-name", "checkpoint",
    ]

    if sample:
        cmd.extend(["--sample", str(sample)])

    if multi_gpu:
        cmd.append("--multi-gpu")

    save_config(model_dir, arch, cmd, hp)

    # Run training with log capture
    log_path = model_dir / "train.log"
    start_time = datetime.now()

    print(f"\n{'='*60}")
    print(f"Training: {arch}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_path}")
    print(f"{'='*60}\n")

    with open(log_path, "w") as f:
        process = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            cwd=Path.cwd()
        )

    duration = (datetime.now() - start_time).total_seconds()

    # Parse results
    result = parse_training_log(log_path)
    result["duration_s"] = int(duration)
    result["return_code"] = process.returncode

    return result


def update_status_running(exp_dir: Path, status: dict, arch: str):
    """Update status when starting a model."""
    status["running"] = arch
    status["models"][arch] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
    }
    save_status(exp_dir, status)


def update_status_complete(exp_dir: Path, status: dict, arch: str, result: dict):
    """Update status when model completes."""
    model_status = {
        "status": "completed" if result["success"] else "failed",
        "duration_s": result["duration_s"],
    }
    if result["cer"] is not None:
        model_status["cer"] = result["cer"]
    if result["wer"] is not None:
        model_status["wer"] = result["wer"]

    status["models"][arch] = model_status
    status["running"] = None

    if result["success"]:
        status["completed"] += 1
    else:
        status["failed"] += 1

    save_status(exp_dir, status)


def run_queue(args):
    """Main queue execution loop."""
    test_mode = args.test
    fast_mode = args.fast
    epochs = args.epochs or (5 if test_mode else 100)
    sample = args.sample or (500 if test_mode else None)

    # Select model set: explicit --models > --fast > all
    if args.models:
        models = args.models
    elif fast_mode:
        models = FAST_ARCHITECTURES
    else:
        models = ARCHITECTURES

    # Validate model names
    invalid = [m for m in models if m not in ARCHITECTURES]
    if invalid:
        print(f"Error: Unknown models: {invalid}")
        print(f"Available: {ARCHITECTURES}")
        sys.exit(1)

    if args.resume:
        exp_dir = find_latest_experiment(Path("trained_models"))
        if not exp_dir:
            print("Error: No trained_models directory found to resume")
            sys.exit(1)
        status = load_status(exp_dir)
        if not status:
            print(f"Error: No status.json found in {exp_dir}")
            sys.exit(1)
        print(f"Resuming training models: {exp_dir}")
        # Filter to only pending models
        models = [m for m in models if status["models"].get(m, {}).get("status") == "pending"]
    else:
        exp_dir = create_experiment_dir(test_mode, fast_mode)
        mode = "test" if test_mode else ("fast" if fast_mode else "production")
        status = init_status(exp_dir, models, mode)
        print(f"Created experiment: {exp_dir}")

    print(f"\nQueue: {len(models)} models")
    print(f"Epochs: {epochs}, Sample: {sample or 'full'}")
    print(f"Status file: {exp_dir}/status.json\n")

    for arch in models:
        update_status_running(exp_dir, status, arch)

        result = train_model(arch, exp_dir, epochs, sample, args.multi_gpu)

        update_status_complete(exp_dir, status, arch, result)

        rec = get_recommendation(result["cer"])
        cer_str = f"{result['cer']:.4f}" if result["cer"] else "N/A"
        print(f"\n✓ {arch}: CER={cer_str}, recommendation={rec}")

    generate_summary(exp_dir, status)


def generate_summary(exp_dir: Path, status: dict):
    """Generate final summary with recommendations."""
    summary = {
        "experiment": str(exp_dir),
        "started_at": status["started_at"],
        "completed_at": datetime.now().isoformat(),
        "total": status["total"],
        "completed": status["completed"],
        "failed": status["failed"],
        "models": {},
    }

    # Collect results with recommendations
    for arch, model_status in status["models"].items():
        cer = model_status.get("cer")
        rec = get_recommendation(cer)
        summary["models"][arch] = {
            **model_status,
            "recommendation": rec,
        }

    # Save summary JSON
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print summary table
    print("\n" + "=" * 80)
    print(f"TRAINING QUEUE SUMMARY - {exp_dir.name}")
    print("=" * 80)
    print(f"{'Model':<25} {'CER':>8} {'WER':>8} {'Status':<12} Recommendation")
    print("-" * 80)

    # Sort by CER (completed models first)
    sorted_models = sorted(
        summary["models"].items(),
        key=lambda x: (x[1].get("cer") is None, x[1].get("cer", 999))
    )

    rec_counts = {"excellent": 0, "good": 0, "needs_sweep": 0, "needs_investigation": 0, "failed": 0}

    for arch, data in sorted_models:
        cer = f"{data['cer']:.4f}" if data.get("cer") else "N/A"
        wer = f"{data['wer']:.4f}" if data.get("wer") else "N/A"
        status_str = data.get("status", "pending")
        rec = data.get("recommendation", "pending")
        print(f"{arch:<25} {cer:>8} {wer:>8} {status_str:<12} {rec}")

        if rec in rec_counts:
            rec_counts[rec] += 1

    print("=" * 80)
    print(f"Summary: {status['completed']} completed | {status['failed']} failed | "
          f"{status['total'] - status['completed'] - status['failed']} pending")

    rec_str = " | ".join(f"{k}: {v}" for k, v in rec_counts.items() if v > 0)
    if rec_str:
        print(f"  {rec_str}")
    print("=" * 80)


def show_status(args):
    """Display current queue status."""
    exp_dir = find_latest_experiment(Path("trained_models"))
    if not exp_dir:
        print("No experiment directory found.")
        return

    status = load_status(exp_dir)
    if not status:
        print(f"No status.json found in {exp_dir}")
        return

    print(f"\nExperiment: {exp_dir}")
    print(f"Mode: {status['mode']}")
    print(f"Started: {status['started_at']}")
    print(f"Progress: {status['completed']}/{status['total']} completed, {status['failed']} failed")

    if status["running"]:
        print(f"Currently running: {status['running']}")

    print(f"\n{'Model':<25} {'Status':<12} {'CER':>8} {'WER':>8}")
    print("-" * 55)

    for arch, data in status["models"].items():
        cer = f"{data['cer']:.4f}" if data.get("cer") else "-"
        wer = f"{data['wer']:.4f}" if data.get("wer") else "-"
        print(f"{arch:<25} {data['status']:<12} {cer:>8} {wer:>8}")


def main():
    args = parse_args()

    if args.status:
        show_status(args)
    else:
        run_queue(args)


if __name__ == "__main__":
    main()
