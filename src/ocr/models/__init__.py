"""
Modular OCR model architectures.

This package provides a flexible system for building OCR models with
different backbone and encoder combinations.

Example usage:
    # Use architecture preset
    from src.ocr.models import create_ocr_model
    model = create_ocr_model("crnn_vgg16", num_classes=100)

    # Custom combination
    model = create_ocr_model(
        backbone="resnet50",
        encoder="transformer",
        num_classes=100,
        transformer_layers=6,
    )

    # Load from checkpoint
    model, char_to_idx, idx_to_char, config = load_checkpoint("weights/model.pt")
"""

import torch
from pathlib import Path

from .backbones import (
    create_backbone,
    list_backbones,
    BACKBONE_REGISTRY,
)
from .encoders import (
    create_encoder,
    list_encoders,
    ENCODER_REGISTRY,
)
from .ocr_model import OCRModel, build_ocr_model


# Architecture presets: maps name -> (backbone, encoder)
ARCHITECTURE_PRESETS = {
    # CRNN variants (with BiLSTM)
    "crnn_vgg16": {"backbone": "vgg16", "encoder": "bilstm"},
    "crnn_vgg19": {"backbone": "vgg19", "encoder": "bilstm"},
    "crnn_resnet50": {"backbone": "resnet50", "encoder": "bilstm"},
    "crnn_resnet101": {"backbone": "resnet101", "encoder": "bilstm"},
    "crnn_simple": {"backbone": "simple_cnn", "encoder": "bilstm"},

    # CNN-CTC variants (no RNN, direct CNN to CTC)
    "ctc_vgg16": {"backbone": "vgg16", "encoder": "none"},
    "ctc_vgg19": {"backbone": "vgg19", "encoder": "none"},
    "ctc_resnet50": {"backbone": "resnet50", "encoder": "none"},
    "ctc_resnet101": {"backbone": "resnet101", "encoder": "none"},
    "ctc_simple": {"backbone": "simple_cnn", "encoder": "none"},

    # Transformer variants
    "transformer_vgg16": {"backbone": "vgg16", "encoder": "transformer"},
    "transformer_vgg19": {"backbone": "vgg19", "encoder": "transformer"},
    "transformer_resnet50": {"backbone": "resnet50", "encoder": "transformer"},
    "transformer_resnet101": {"backbone": "resnet101", "encoder": "transformer"},
    "transformer_simple": {"backbone": "simple_cnn", "encoder": "transformer"},
}


def list_architectures() -> list[str]:
    """List all architecture preset names."""
    return list(ARCHITECTURE_PRESETS.keys())


def create_ocr_model(
    architecture: str = None,
    backbone: str = None,
    encoder: str = None,
    num_classes: int = 100,
    hidden_size: int = 256,
    pretrained: bool = True,
    **kwargs,
) -> OCRModel:
    """
    Create an OCR model from architecture preset or custom components.

    Args:
        architecture: Architecture preset name (e.g., 'crnn_vgg16')
        backbone: Backbone name (overrides preset)
        encoder: Encoder name (overrides preset)
        num_classes: Number of output classes (charset + 1 for CTC blank)
        hidden_size: Hidden size for projection and LSTM
        pretrained: Use pretrained backbone weights
        **kwargs: Additional arguments passed to build_ocr_model

    Returns:
        Configured OCRModel instance

    Examples:
        # Use preset
        model = create_ocr_model("crnn_vgg16", num_classes=100)

        # Custom combination
        model = create_ocr_model(
            backbone="resnet50",
            encoder="transformer",
            num_classes=100,
        )

        # Override preset
        model = create_ocr_model(
            architecture="crnn_vgg16",
            encoder="transformer",  # Override to use transformer
            num_classes=100,
        )
    """
    # Resolve architecture preset
    if architecture:
        if architecture not in ARCHITECTURE_PRESETS:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Available: {list_architectures()}"
            )
        preset = ARCHITECTURE_PRESETS[architecture]
        backbone = backbone or preset["backbone"]
        encoder = encoder or preset["encoder"]

    # Validate required components
    if not backbone:
        raise ValueError("Must specify either 'architecture' or 'backbone'")
    if not encoder:
        raise ValueError("Must specify either 'architecture' or 'encoder'")

    return build_ocr_model(
        backbone_name=backbone,
        encoder_name=encoder,
        num_classes=num_classes,
        hidden_size=hidden_size,
        pretrained=pretrained,
        **kwargs,
    )


def create_model_with_charset(
    architecture: str = None,
    backbone: str = None,
    encoder: str = None,
    charset: str = None,
    hidden_size: int = 256,
    pretrained: bool = True,
    **kwargs,
) -> tuple[OCRModel, dict, dict]:
    """
    Create an OCR model with character mappings.

    Args:
        architecture: Architecture preset name
        backbone: Backbone name
        encoder: Encoder name
        charset: String of all characters in vocabulary
        hidden_size: Hidden size
        pretrained: Use pretrained weights
        **kwargs: Additional model arguments

    Returns:
        Tuple of (model, char_to_idx, idx_to_char)
    """
    if charset is None:
        raise ValueError("charset is required")

    # CTC blank token at index 0
    char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}
    char_to_idx["<blank>"] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    num_classes = len(charset) + 1  # +1 for CTC blank

    model = create_ocr_model(
        architecture=architecture,
        backbone=backbone,
        encoder=encoder,
        num_classes=num_classes,
        hidden_size=hidden_size,
        pretrained=pretrained,
        **kwargs,
    )

    return model, char_to_idx, idx_to_char


def save_checkpoint(
    path: str,
    model: OCRModel,
    charset: str,
    char_to_idx: dict,
    idx_to_char: dict,
    architecture_config: dict,
    config: dict,
    metrics: dict = None,
    epoch: int = None,
    optimizer_state: dict = None,
    scheduler_state: dict = None,
):
    """
    Save model checkpoint with all necessary metadata.

    Args:
        path: Output path
        model: The OCRModel instance
        charset: Character set string
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        architecture_config: Dict with 'backbone', 'encoder', etc.
        config: Training config (img_height, img_width, etc.)
        metrics: Optional metrics dict (cer, wer, accuracy)
        epoch: Optional current epoch
        optimizer_state: Optional optimizer state dict
        scheduler_state: Optional scheduler state dict
    """
    checkpoint = {
        "architecture": architecture_config,
        "model_state_dict": model.state_dict(),
        "charset": charset,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "config": config,
    }

    if metrics is not None:
        checkpoint["metrics"] = metrics
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: str = "cpu",
    strict: bool = True,
) -> tuple[OCRModel, dict, dict, dict]:
    """
    Load model from checkpoint.

    Args:
        path: Checkpoint path
        device: Device to load model to
        strict: Strict state dict loading

    Returns:
        Tuple of (model, char_to_idx, idx_to_char, config)
    """
    checkpoint = torch.load(path, map_location=device)

    # Get architecture config
    arch_config = checkpoint.get("architecture", {})
    config = checkpoint.get("config", {})

    # Get charset info
    charset = checkpoint["charset"]
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]

    # Reconstruct model
    num_classes = len(charset) + 1

    # Support both new format (architecture dict) and legacy format
    if "backbone" in arch_config:
        model = create_ocr_model(
            backbone=arch_config["backbone"],
            encoder=arch_config.get("encoder", "bilstm"),
            num_classes=num_classes,
            hidden_size=arch_config.get("hidden_size", 256),
            pretrained=False,  # Loading weights, don't need pretrained
            # Pass through encoder-specific options
            lstm_layers=arch_config.get("lstm_layers", 2),
            transformer_layers=arch_config.get("transformer_layers", 4),
            transformer_nhead=arch_config.get("transformer_nhead", 8),
            transformer_d_model=arch_config.get("transformer_d_model", 256),
        )
    else:
        # Legacy format: assume CRNN with backbone from config
        backbone = config.get("backbone", "vgg16")
        hidden_size = config.get("hidden_size", 256)
        model = create_ocr_model(
            backbone=backbone,
            encoder="bilstm",
            num_classes=num_classes,
            hidden_size=hidden_size,
            pretrained=False,
        )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        # Legacy format: load individual component state dicts
        if "cnn" in checkpoint:
            model.backbone.load_state_dict(checkpoint["cnn"])
        if "map_to_seq" in checkpoint:
            model.map_to_seq.load_state_dict(checkpoint["map_to_seq"])
        if "rnn" in checkpoint:
            # Legacy CRNN had single rnn, new has encoder.rnn
            if hasattr(model.encoder, "rnn"):
                model.encoder.rnn.load_state_dict(checkpoint["rnn"])
        if "fc" in checkpoint:
            model.fc.load_state_dict(checkpoint["fc"])

    model.to(device)
    model.eval()

    return model, char_to_idx, idx_to_char, config


__all__ = [
    # Main creation functions
    "create_ocr_model",
    "create_model_with_charset",
    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    # Listing functions
    "list_architectures",
    "list_backbones",
    "list_encoders",
    # Presets
    "ARCHITECTURE_PRESETS",
    # Registries
    "BACKBONE_REGISTRY",
    "ENCODER_REGISTRY",
    # Core classes
    "OCRModel",
]
