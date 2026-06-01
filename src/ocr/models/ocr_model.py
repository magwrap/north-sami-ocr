"""Modular OCR model composed of backbone + encoder + CTC head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone
from .encoders import create_encoder


class OCRModel(nn.Module):
    """
    Modular OCR model: Backbone -> Pool -> Project -> Encoder -> FC -> CTC

    Supports arbitrary combinations of:
    - Backbones: VGG16, VGG19, ResNet50, ResNet101, SimpleCNN
    - Encoders: BiLSTM, Transformer, None (pure CNN-CTC)
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        num_classes: int,
        hidden_size: int = 256,
    ):
        """
        Initialize OCR model.

        Args:
            backbone: CNN backbone module (must have `out_channels` attribute)
            encoder: Sequence encoder module (must have `output_size` property)
            num_classes: Number of output classes (charset + 1 for CTC blank)
            hidden_size: Hidden size for projection layer
        """
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Adaptive pooling to reduce height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        # Project from backbone channels to hidden size
        self.map_to_seq = nn.Linear(backbone.out_channels, hidden_size)

        # Final classification layer
        self.fc = nn.Linear(encoder.output_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)
               Grayscale (C=1) will be expanded to 3 channels

        Returns:
            Log probabilities of shape (B, T, num_classes)
        """
        # Handle grayscale input by repeating to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # CNN features: (B, C, H, W) -> (B, cnn_out, H', W')
        features = self.backbone(x)

        # Pool height to 1: (B, cnn_out, H', W') -> (B, cnn_out, 1, W')
        features = self.pool(features)

        # Reshape for sequence: (B, cnn_out, 1, W') -> (B, W', cnn_out)
        features = features.squeeze(2).permute(0, 2, 1)

        # Project to hidden size: (B, W', cnn_out) -> (B, W', hidden_size)
        features = self.map_to_seq(features)

        # Encode sequence: (B, W', hidden_size) -> (B, W', enc_out)
        encoded = self.encoder(features)

        # Classification: (B, W', enc_out) -> (B, W', num_classes)
        logits = self.fc(encoded)

        # Log softmax for CTC loss
        return F.log_softmax(logits, dim=2)

    def get_config(self) -> dict:
        """Return model configuration for checkpoint saving."""
        return {
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
        }


def build_ocr_model(
    backbone_name: str,
    encoder_name: str,
    num_classes: int,
    hidden_size: int = 256,
    pretrained: bool = True,
    # BiLSTM options
    lstm_layers: int = 2,
    lstm_dropout: float = 0.2,
    # Transformer options
    transformer_d_model: int = 256,
    transformer_nhead: int = 8,
    transformer_layers: int = 4,
    transformer_ff_dim: int = 1024,
    transformer_dropout: float = 0.1,
    # SimpleCNN options
    leaky_relu: bool = False,
) -> OCRModel:
    """
    Build an OCR model from component names.

    Args:
        backbone_name: Name of backbone ('vgg16', 'resnet50', 'simple_cnn', etc.)
        encoder_name: Name of encoder ('bilstm', 'transformer', 'none')
        num_classes: Number of output classes
        hidden_size: Hidden size for projection and LSTM
        pretrained: Use pretrained backbone weights
        lstm_layers: Number of LSTM layers
        lstm_dropout: LSTM dropout rate
        transformer_d_model: Transformer model dimension
        transformer_nhead: Number of attention heads
        transformer_layers: Number of transformer layers
        transformer_ff_dim: Feedforward dimension
        transformer_dropout: Transformer dropout rate
        leaky_relu: Use LeakyReLU in SimpleCNN

    Returns:
        Configured OCRModel instance
    """
    # Create backbone
    backbone_kwargs = {"pretrained": pretrained}
    if backbone_name == "simple_cnn":
        backbone_kwargs["leaky_relu"] = leaky_relu
    backbone = create_backbone(backbone_name, **backbone_kwargs)

    # Create encoder
    if encoder_name == "bilstm":
        encoder = create_encoder(
            encoder_name,
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
    elif encoder_name == "transformer":
        encoder = create_encoder(
            encoder_name,
            input_size=hidden_size,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
        )
    else:  # 'none' or other
        encoder = create_encoder(encoder_name, input_size=hidden_size)

    return OCRModel(backbone, encoder, num_classes, hidden_size)
