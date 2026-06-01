"""Fetch the pretrained CNN-CTC OCR checkpoint from HuggingFace Hub.

The model lands in `trained_models/<exp_dir>/ctc_simple/checkpoint_best.pt`,
matching the layout produced by train_queue.py so benchmark/inference scripts
find it without further configuration.

Usage:
    python src/ocr/download_models.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "magwrap/cnn-ctc-ocr-sme"
ARCH = "ctc_simple"
EXP_DIR = "2026-03-28_queue"
FILES = ("checkpoint_best.pt", "config.json")


def fetch_model(repo_id: str, arch: str, dest_root: Path) -> Path:
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
        "--dest",
        type=Path,
        default=Path("trained_models"),
        help="Destination root (default: ./trained_models)",
    )
    args = parser.parse_args()

    out = fetch_model(REPO_ID, ARCH, args.dest)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
