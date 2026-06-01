"""Sequence encoder modules for OCR models."""

import math
import torch
import torch.nn as nn


# Encoder registry
ENCODER_REGISTRY: dict[str, type[nn.Module]] = {}


def register_encoder(name: str):
    """Decorator to register an encoder class."""
    def decorator(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def create_encoder(name: str, **kwargs) -> nn.Module:
    """
    Create an encoder by name.

    Args:
        name: Encoder name (e.g., 'bilstm', 'transformer', 'none')
        **kwargs: Additional arguments passed to encoder constructor

    Returns:
        Encoder module with `output_size` property
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: {name}. Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[name](**kwargs)


def list_encoders() -> list[str]:
    """List all registered encoder names."""
    return list(ENCODER_REGISTRY.keys())


@register_encoder("bilstm")
class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for sequence modeling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize BiLSTM encoder.

        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden size (output will be 2x for bidirectional)
            num_layers: Number of LSTM layers
            dropout: Dropout probability between layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def output_size(self) -> int:
        """Output feature dimension (2x hidden_size for bidirectional)."""
        return self.hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, input_size)

        Returns:
            Output tensor of shape (B, T, hidden_size * 2)
        """
        output, _ = self.rnn(x)
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Args:
            x: Input tensor of shape (B, T, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


@register_encoder("transformer")
class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer encoder.

        Args:
            input_size: Input feature dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model

        # Project input to d_model if needed
        self.input_proj = nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def output_size(self) -> int:
        """Output feature dimension."""
        return self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, input_size)

        Returns:
            Output tensor of shape (B, T, d_model)
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x


@register_encoder("none")
class NoEncoder(nn.Module):
    """Pass-through encoder for CNN-CTC (no sequence modeling)."""

    def __init__(self, input_size: int, **kwargs):
        """
        Initialize NoEncoder.

        Args:
            input_size: Input feature dimension (preserved in output)
            **kwargs: Ignored for compatibility
        """
        super().__init__()
        self._output_size = input_size

    @property
    def output_size(self) -> int:
        """Output feature dimension (same as input)."""
        return self._output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass-through forward."""
        return x
