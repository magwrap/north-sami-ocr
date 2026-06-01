#!/usr/bin/env python3
"""Generate architecture visualizations for trained OCR models."""

import torch
from pathlib import Path
from torchviz import make_dot
from torchsummary import summary
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "ocr"))

from models.ocr_model import build_ocr_model


def load_checkpoint_config(checkpoint_path):
    """Extract model config from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})
    return config


def visualize_model(model_name, backbone, encoder, output_dir):
    """Generate architecture visualization for a model."""
    print(f"\n{'='*60}")
    print(f"Visualizing: {model_name}")
    print(f"  Backbone: {backbone}")
    print(f"  Encoder: {encoder}")
    print(f"{'='*60}\n")

    # Build model (dummy num_classes, will be overwritten by checkpoint)
    model = build_ocr_model(
        backbone_name=backbone,
        encoder_name=encoder,
        num_classes=50,  # placeholder
        hidden_size=256,
        pretrained=False  # don't need pretrained weights for architecture viz
    )
    model.eval()

    # Create dummy input (batch_size=1, channels=1, height=64, width=256)
    dummy_input = torch.randn(1, 1, 64, 256)

    # Generate graphviz diagram
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    graph_path = output_dir / f"{model_name}_graph"
    dot.render(graph_path, cleanup=True)
    print(f"✓ Graph saved: {graph_path}.png")

    # Generate text summary
    summary_path = output_dir / f"{model_name}_summary.txt"
    with open(summary_path, 'w') as f:
        # Redirect stdout to file
        old_stdout = sys.stdout
        sys.stdout = f

        print(f"Model: {model_name}")
        print(f"Backbone: {backbone}")
        print(f"Encoder: {encoder}")
        print("\n" + "="*80 + "\n")

        try:
            # torchsummary expects (C, H, W) not (B, C, H, W)
            summary(model, (1, 64, 256), device="cpu")
        except Exception as e:
            print(f"Summary generation failed: {e}")
            # Manual summary
            print("\nModel Architecture:")
            print(model)
            print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        sys.stdout = old_stdout

    print(f"✓ Summary saved: {summary_path}")

    return output.shape


def main():
    """Generate visualizations for all trained models."""
    # Define models to visualize
    models = [
        ("ctc_simple", "simple_cnn", "bilstm"),
        ("ctc_vgg16", "vgg16", "bilstm"),
        ("ctc_vgg19", "vgg19", "bilstm"),
        ("ctc_resnet50", "resnet50", "bilstm"),
    ]

    # Create output directory
    output_dir = Path(__file__).parent / "architecture_visualization"
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Generate visualizations
    results = []
    for model_name, backbone, encoder in models:
        try:
            output_shape = visualize_model(model_name, backbone, encoder, output_dir)
            results.append((model_name, "✓", str(output_shape)))
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append((model_name, "✗", str(e)))

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    for model_name, status, info in results:
        print(f"{status} {model_name:20s} {info}")

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
