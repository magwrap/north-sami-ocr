"""CNN backbone architectures for OCR models."""

import torch
import torch.nn as nn
from torchvision import models


# Backbone registry
BACKBONE_REGISTRY: dict[str, type[nn.Module]] = {}


def register_backbone(name: str):
    """Decorator to register a backbone class."""
    def decorator(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def create_backbone(name: str, **kwargs) -> nn.Module:
    """
    Create a backbone by name.

    Args:
        name: Backbone name (e.g., 'vgg16', 'resnet50', 'simple_cnn')
        **kwargs: Additional arguments passed to backbone constructor

    Returns:
        Backbone module with `out_channels` attribute
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone: {name}. Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name](**kwargs)


def list_backbones() -> list[str]:
    """List all registered backbone names."""
    return list(BACKBONE_REGISTRY.keys())


@register_backbone("vgg16")
class VGG16Backbone(nn.Module):
    """VGG16 backbone (ImageNet pretrained)."""

    out_channels = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg.features.children()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


@register_backbone("vgg19")
class VGG19Backbone(nn.Module):
    """VGG19 backbone (ImageNet pretrained)."""

    out_channels = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        vgg = models.vgg19(weights=weights)
        self.features = nn.Sequential(*list(vgg.features.children()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


@register_backbone("resnet50")
class ResNet50Backbone(nn.Module):
    """ResNet50 backbone (ImageNet pretrained)."""

    out_channels = 2048

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


@register_backbone("resnet101")
class ResNet101Backbone(nn.Module):
    """ResNet101 backbone (ImageNet pretrained)."""

    out_channels = 2048

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        resnet = models.resnet101(weights=weights)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


@register_backbone("simple_cnn")
class SimpleCNNBackbone(nn.Module):
    """
    Lightweight 7-layer CNN backbone from crnn-pytorch.

    Not pretrained - trained from scratch.
    Architecture: 7 conv layers with batch norm and pooling.
    """

    out_channels = 512

    def __init__(self, pretrained: bool = False, leaky_relu: bool = False):
        """
        Initialize SimpleCNN backbone.

        Args:
            pretrained: Ignored (no pretrained weights available)
            leaky_relu: Use LeakyReLU instead of ReLU
        """
        super().__init__()

        channels = [64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        layers = []

        # Input: grayscale or 3-channel (we'll handle conversion in OCRModel)
        in_channels = 3

        def make_conv_block(in_ch, out_ch, kernel_size, padding, batch_norm=False):
            block = [nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding)]
            if batch_norm:
                block.append(nn.BatchNorm2d(out_ch))
            if leaky_relu:
                block.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                block.append(nn.ReLU(inplace=True))
            return block

        # Layer 0: conv + pool
        layers.extend(make_conv_block(in_channels, channels[0], kernel_sizes[0], paddings[0]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Layer 1: conv + pool
        layers.extend(make_conv_block(channels[0], channels[1], kernel_sizes[1], paddings[1]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Layers 2-3: conv, conv
        layers.extend(make_conv_block(channels[1], channels[2], kernel_sizes[2], paddings[2]))
        layers.extend(make_conv_block(channels[2], channels[3], kernel_sizes[3], paddings[3]))
        layers.append(nn.MaxPool2d(kernel_size=(2, 1)))

        # Layers 4-5: conv + bn, conv + bn
        layers.extend(make_conv_block(channels[3], channels[4], kernel_sizes[4], paddings[4], batch_norm=True))
        layers.extend(make_conv_block(channels[4], channels[5], kernel_sizes[5], paddings[5], batch_norm=True))
        layers.append(nn.MaxPool2d(kernel_size=(2, 1)))

        # Layer 6: conv (2x2 kernel, no padding)
        layers.extend(make_conv_block(channels[5], channels[6], kernel_sizes[6], paddings[6]))

        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
