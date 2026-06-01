#!/usr/bin/env python3
"""
Unified training script for modular OCR architectures.

Supports all combinations of backbones (VGG, ResNet, SimpleCNN) and
encoders (BiLSTM, Transformer, None).

Examples:
    # Architecture presets
    python train_unified.py -a crnn_vgg16 --epochs 50
    python train_unified.py -a ctc_resnet50 --epochs 50
    python train_unified.py -a transformer_vgg16 --epochs 50

    # Custom combinations
    python train_unified.py --backbone vgg16 --encoder none --epochs 50
    python train_unified.py --backbone resnet50 --encoder transformer --transformer-layers 6

    # Quick testing
    python train_unified.py -a crnn_simple --sample 1000 --epochs 5
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as a script from src/ocr directory
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import shared training utilities
from src.ocr.train_utils import compute_cer, compute_wer, ctc_decode
from src.ocr.base_trainer import OCRTrainer

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train modular OCR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture presets:
  CRNN (BiLSTM):     crnn_vgg16, crnn_vgg19, crnn_resnet50, crnn_resnet101, crnn_simple
  CNN-CTC (no RNN):  ctc_vgg16, ctc_vgg19, ctc_resnet50, ctc_resnet101, ctc_simple
  Transformer:       transformer_vgg16, transformer_vgg19, transformer_resnet50,
                     transformer_resnet101, transformer_simple

Examples:
  %(prog)s -a crnn_vgg16 --epochs 50
  %(prog)s --backbone resnet50 --encoder transformer --epochs 50
  %(prog)s -a crnn_simple --sample 1000 --epochs 5  # Quick test
        """
    )

    # Architecture selection
    arch_group = parser.add_argument_group("Architecture")
    arch_group.add_argument(
        "-a", "--architecture",
        type=str,
        default=None,
        help="Architecture preset (e.g., crnn_vgg16, ctc_resnet50, transformer_simple)",
    )
    arch_group.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["vgg16", "vgg19", "resnet50", "resnet101", "simple_cnn"],
        help="CNN backbone (overrides preset)",
    )
    arch_group.add_argument(
        "--encoder",
        type=str,
        default=None,
        choices=["bilstm", "transformer", "none"],
        help="Sequence encoder (overrides preset)",
    )
    arch_group.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained backbone weights",
    )

    # Model hyperparameters
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden size for projection and LSTM (default: 256)",
    )
    model_group.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )
    model_group.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.2,
        help="LSTM dropout (default: 0.2)",
    )
    model_group.add_argument(
        "--transformer-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)",
    )
    model_group.add_argument(
        "--transformer-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    model_group.add_argument(
        "--transformer-d-model",
        type=int,
        default=256,
        help="Transformer model dimension (default: 256)",
    )
    model_group.add_argument(
        "--transformer-ff-dim",
        type=int,
        default=1024,
        help="Transformer feedforward dimension (default: 1024)",
    )
    model_group.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.1,
        help="Transformer dropout (default: 0.1)",
    )
    model_group.add_argument(
        "--leaky-relu",
        action="store_true",
        help="Use LeakyReLU in SimpleCNN backbone",
    )

    # Training hyperparameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train_group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "rmsprop"],
        help="Optimizer (default: adamw)",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    train_group.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step", "none"],
        help="Learning rate scheduler (default: plateau)",
    )
    train_group.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Gradient clipping max norm (default: 5.0)",
    )
    train_group.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    train_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs with linear LR increase (default: 0)",
    )
    train_group.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help="Learning rate multiplier for backbone (vs head layers) (default: 1.0)",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--img-height",
        type=int,
        default=32,
        help="Input image height (default: 32)",
    )
    data_group.add_argument(
        "--img-width",
        type=int,
        default=800,
        help="Input image width (default: 800)",
    )
    data_group.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only N samples for quick testing",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Data loader workers (default: 4)",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: src/ocr/weights/)",
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for checkpoint file (default: architecture name)",
    )
    output_group.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    output_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory",
    )
    output_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    output_group.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs with DataParallel",
    )

    args = parser.parse_args()

    # Validate architecture specification
    if args.architecture is None and (args.backbone is None or args.encoder is None):
        parser.error("Must specify --architecture OR both --backbone and --encoder")

    return args


# Metric functions (compute_cer, compute_wer, ctc_decode) are now imported from train_utils
# Training/validation functions (train_epoch, validate) are now in OCRTrainer class from base_trainer


def create_optimizer(args, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from args with optional differential learning rates."""
    # Split parameters into backbone and head for differential LR
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Create parameter groups with different learning rates
    backbone_lr = args.lr * args.backbone_lr_mult
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': args.lr}
    ]

    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        return torch.optim.Adam(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        return torch.optim.SGD(
            param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9
        )
    elif args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            param_groups, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_scheduler(args, optimizer: torch.optim.Optimizer):
    """Create learning rate scheduler from args with optional warmup."""
    # Handle warmup + main scheduler combination
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=args.warmup_epochs
        )

        # Create main scheduler based on type (avoiding recursion!)
        if args.scheduler == "plateau":
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        elif args.scheduler == "cosine":
            # Adjust T_max to account for warmup epochs
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs
            )
        elif args.scheduler == "step":
            main_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif args.scheduler == "none":
            main_scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {args.scheduler}")

        # For plateau scheduler, we can't use SequentialLR (needs .step(metric))
        # Return tuple for special handling in training loop
        if args.scheduler == "plateau":
            return (warmup_scheduler, main_scheduler)
        elif main_scheduler is None:
            return warmup_scheduler
        else:
            # Chain warmup → main for other schedulers
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )

    # No warmup: return main scheduler directly
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif args.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Import model creation functions
    from src.ocr.models import (
        create_ocr_model,
        save_checkpoint,
        list_architectures,
        ARCHITECTURE_PRESETS,
    )

    # Resolve architecture
    if args.architecture:
        if args.architecture not in ARCHITECTURE_PRESETS:
            print(f"Unknown architecture: {args.architecture}")
            print(f"Available: {list_architectures()}")
            sys.exit(1)
        preset = ARCHITECTURE_PRESETS[args.architecture]
        backbone = args.backbone or preset["backbone"]
        encoder = args.encoder or preset["encoder"]
        arch_name = args.architecture
    else:
        backbone = args.backbone
        encoder = args.encoder
        arch_name = f"{encoder}_{backbone}" if encoder != "bilstm" else f"crnn_{backbone}"

    print(f"Architecture: {arch_name} (backbone={backbone}, encoder={encoder})")

    # Initialize TensorBoard writer
    writer = None
    if args.tensorboard:
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
        else:
            if args.log_dir:
                log_dir = args.log_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = f"runs/{arch_name}_{timestamp}"
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not found. Install with: pip install datasets")
        sys.exit(1)

    dataset = load_dataset("Sprakbanken/synthetic_sami_ocr_data", split="train")
    print(f"Loaded {len(dataset)} samples")

    # Sample if requested
    if args.sample:
        dataset = dataset.select(range(min(args.sample, len(dataset))))
        print(f"Using {len(dataset)} samples for quick test")

    # Build charset
    from src.ocr.dataset import build_charset, SamiOCRDataset, collate_fn

    print("Building character set...")
    charset = build_charset(dataset)
    print(f"Character set ({len(charset)} chars): {repr(charset)}")

    # Create character mappings
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
    char_to_idx["<blank>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_hf = dataset.select(train_indices)
    val_hf = dataset.select(val_indices)

    print(f"Train: {len(train_hf)}, Val: {len(val_hf)}")

    # Create datasets
    train_dataset = SamiOCRDataset(
        train_hf,
        char_to_idx,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=True,
    )
    val_dataset = SamiOCRDataset(
        val_hf,
        char_to_idx,
        img_height=args.img_height,
        img_width=args.img_width,
        augment=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    num_classes = len(charset) + 1
    print(f"Creating model: backbone={backbone}, encoder={encoder}")

    model = create_ocr_model(
        backbone=backbone,
        encoder=encoder,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        pretrained=not args.no_pretrained,
        # BiLSTM options
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        # Transformer options
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        # SimpleCNN options
        leaky_relu=args.leaky_relu,
    )
    model.to(device)

    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)

    # Handle warmup + plateau tuple case
    warmup_scheduler = None
    main_scheduler = scheduler
    if isinstance(scheduler, tuple):
        warmup_scheduler, main_scheduler = scheduler
        scheduler = warmup_scheduler  # Start with warmup

    # Create unified trainer
    trainer = OCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        idx_to_char=idx_to_char,
        logits_format="btc",  # Standard (batch, time, classes) format
    )

    # Architecture config for checkpoint
    architecture_config = {
        "backbone": backbone,
        "encoder": encoder,
        "hidden_size": args.hidden_size,
        "num_classes": num_classes,
    }
    if encoder == "bilstm":
        architecture_config["lstm_layers"] = args.lstm_layers
        architecture_config["lstm_dropout"] = args.lstm_dropout
    elif encoder == "transformer":
        architecture_config["transformer_layers"] = args.transformer_layers
        architecture_config["transformer_nhead"] = args.transformer_heads
        architecture_config["transformer_d_model"] = args.transformer_d_model
        architecture_config["transformer_ff_dim"] = args.transformer_ff_dim

    # Training config
    train_config = {
        "img_height": args.img_height,
        "img_width": args.img_width,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
    }

    # Training loop
    best_cer = float("inf")
    patience_counter = 0
    exp_name = args.experiment_name or arch_name
    checkpoint_path = output_dir / f"{exp_name}_best.pt"

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = trainer.train_epoch(grad_clip=args.grad_clip)

        # Validate
        val_loss, val_cer, val_wer, val_acc = trainer.validate()

        # Update scheduler (handle warmup + plateau special case)
        if warmup_scheduler is not None:
            # Warmup + plateau case: manual switching
            if epoch <= args.warmup_epochs:
                warmup_scheduler.step()
            else:
                main_scheduler.step(val_cer)
        elif args.scheduler == "plateau":
            trainer.step_scheduler(metric=val_cer)
        else:
            trainer.step_scheduler()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"CER: {val_cer:.4f} | "
            f"WER: {val_wer:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Metrics/CER", val_cer, epoch)
            writer.add_scalar("Metrics/WER", val_wer, epoch)
            writer.add_scalar("Metrics/Accuracy", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0

            metrics = {
                "cer": val_cer,
                "wer": val_wer,
                "accuracy": val_acc,
                "val_loss": val_loss,
            }

            # Unwrap DataParallel if needed
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            save_checkpoint(
                path=str(checkpoint_path),
                model=model_to_save,
                charset=charset,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                architecture_config=architecture_config,
                config=train_config,
                metrics=metrics,
                epoch=epoch,
            )
            print(f"  -> Saved best model (CER: {val_cer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.patience} epochs)")
                break

    print("-" * 80)
    print(f"Training complete. Best CER: {best_cer:.4f}")
    print(f"Model saved to: {checkpoint_path}")

    # Close TensorBoard writer
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
