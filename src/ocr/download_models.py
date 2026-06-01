"""Fetch pretrained CTC OCR checkpoints from HuggingFace Hub.

Models land in `trained_models/<exp_dir>/<arch>/checkpoint_best.pt`, matching
the layout produced by train_queue.py so benchmark/inference scripts find them
without further configuration.

Usage:
    python src/ocr/download_models.py                # all models
    python src/ocr/download_models.py --arch ctc_vgg16   # one model
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


HF_USER = "magwrap"
EXP_DIR = "2026-03-28_queue"

MODELS: dict[str, str] = {
    "ctc_vgg16": f"{HF_USER}/sami-ocr-ctc-vgg16",
    "ctc_vgg19": f"{HF_USER}/sami-ocr-ctc-vgg19",
    "ctc_resnet50": f"{HF_USER}/sami-ocr-ctc-resnet50",
}

FILES = ("checkpoint_best.pt", "config.json")


def fetch_model(arch: str, repo_id: str, dest_root: Path) -> Path:
    """Download all FILES for one architecture into trained_models/<EXP_DIR>/<arch>/.

    Returns the directory the files were written to.
    """
    dest = dest_root / EXP_DIR / arch
    dest.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        cached = hf_hub_download(repo_id=repo_id, filename=fname)
        target = dest / fname
        shutil.copy(cached, target)
        print(f"  {arch}/{fname}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        choices=list(MODELS),
        help="Fetch a single architecture. Default: all.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("trained_models"),
        help="Destination root (default: ./trained_models)",
    )
    args = parser.parse_args()

    archs = [args.arch] if args.arch else list(MODELS)
    for arch in archs:
        out = fetch_model(arch, MODELS[arch], args.dest)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
