# Northern Sami OCR + Translation

OCR for Northern Sami text with automatic English translation
([TartuNLP](https://api.tartunlp.ai), `sme → eng`).
Supports pre-trained TrOCR models from Sprakbanken and custom-trained modular
CTC architectures.

## Quick Start

```bash
nix develop                                    # enter dev shell (auto-installs deps)

# Benchmark all synthetic images in test_data/account_of_sami/ (default mode)
python src/ocr_translate_pipeline.py --model trocr_smi_pred_synth

# OCR + translate a single image
python src/ocr_translate_pipeline.py --single-image \
    --image test_data/test1.jpg --model trocr_smi_pred_synth

# Interactive model picker
python src/ocr_translate_pipeline.py --list-models
python src/ocr_translate_pipeline.py --single-image --image test_data/test1.jpg
```

## Models

**TrOCR (pre-trained, no training needed):** 7 variants from
[Sprakbanken](https://huggingface.co/Sprakbanken) — `trocr_smi`, `trocr_smi_nor`,
`trocr_smi_pred`, `trocr_smi_nor_pred`, `trocr_smi_synth`,
`trocr_smi_pred_synth` (best), `trocr_smi_nor_pred_synth`.

**Modular CTC (custom-trained):** pluggable `Backbone → Encoder → CTC` —
15 combinations of {SimpleCNN, VGG16, VGG19, ResNet50, ResNet101} ×
{None, BiLSTM, Transformer}, e.g. `ctc_simple`, `crnn_vgg16`,
`transformer_resnet50`.

Trained checkpoints live in `trained_models/<date>_queue/<arch>/checkpoint_best.pt`
and are auto-discovered by `--list-models`.

### Pretrained CTC checkpoints (HuggingFace Hub)

Three CTC variants trained on `Sprakbanken/synthetic_sami_ocr_data` are
published on the Hub:

- [`magwrap/sami-ocr-ctc-vgg16`](https://huggingface.co/magwrap/sami-ocr-ctc-vgg16) (~57 MB)
- [`magwrap/sami-ocr-ctc-vgg19`](https://huggingface.co/magwrap/sami-ocr-ctc-vgg19) (~77 MB)
- [`magwrap/sami-ocr-ctc-resnet50`](https://huggingface.co/magwrap/sami-ocr-ctc-resnet50) (~92 MB)

Fetch them into the expected layout with:

```bash
python src/ocr/download_models.py            # all three
python src/ocr/download_models.py --arch ctc_vgg16
```

## Training

```bash
python src/ocr/train_queue.py --test       # smoke test: 5 epochs, 500 samples
python src/ocr/train_queue.py --fast       # SimpleCNN variants only (~3-4h on V100)
python src/ocr/train_queue.py              # all 15 architectures, 100 epochs
python src/ocr/train_queue.py --status     # progress / summary
python src/ocr/train_queue.py --resume     # resume interrupted queue
python src/ocr/train_queue.py --models crnn_vgg16 --epochs 50 --sample 10000

# single architecture, direct
python src/ocr/train_unified.py -a crnn_simple --epochs 5 --sample 500
```

Hyperparameters (LR, batch size, patience, warmup) are chosen per backbone in
`train_queue.py`. Each run writes `checkpoint_best.pt`, `config.json`,
`train.log`, plus an aggregated `status.json` and `summary.json` with
CER/WER metrics.

## Inference with a Trained Model

```bash
python src/ocr_translate_pipeline.py --single-image \
    --image test_data/test1.jpg \
    --model crnn_vgg16 \
    --weights trained_models/2026-03-28_queue/crnn_vgg16/checkpoint_best.pt
```

## Project Structure

```
sami_ocr/
├── src/
│   ├── ocr_translate_pipeline.py   # OCR + translation CLI (entry point)
│   ├── metrics.py                  # BLEU, chrF, TER for translation
│   ├── ocr/
│   │   ├── pipeline.py             # OCR model registry (TrOCR + Modular)
│   │   ├── train_queue.py          # Train all architectures
│   │   ├── train_unified.py        # Train a single model
│   │   ├── benchmark.py            # OCR-only benchmark
│   │   ├── models/                 # backbones.py, encoders.py, ocr_model.py
│   │   └── ...                     # dataset, synthetic data, sweep, analysis
│   └── translation/
│       ├── sme_eng.py              # TartuNLP API client
│       └── benchmark.py            # Translation-only benchmark
├── test_data/                      # test1-4.{jpg,png}, account_of_sami/, ...
├── trained_models/                 # YYYY-MM-DD_queue/<arch>/checkpoint_best.pt
├── flake.nix                       # Nix dev environment (PyTorch)
└── requirements.txt
```

## Cloud Training

See [cloud-setup.md](./cloud-setup.md). Recommended flow:

```bash
python src/ocr/train_queue.py --test                      # 2-3 min validation
python src/ocr/train_queue.py --fast                      # SimpleCNN models
nohup python src/ocr/train_queue.py > training.log 2>&1 & # full queue, detached
tail -f training.log
```

## Data

Training data is fetched on demand from
[`Sprakbanken/synthetic_sami_ocr_data`](https://huggingface.co/datasets/Sprakbanken/synthetic_sami_ocr_data)
and is not stored in the repository.

Benchmark corpus (`benchmark_data/`) is sourced from publicly available
Northern Sami / English material — see `benchmark_data/about-data.md` for
the per-source URLs and licensing notes (Muitalus, Giellatekno, Bokselskap).

## Reproducibility

- `flake.nix` pins Python 3.11 and PyTorch. `nix develop` is the supported
  entry point.
- `requirements.txt` ships exact pins captured from the working dev shell
  (`pip freeze`), so a non-Nix Python 3.11 environment can also be reproduced
  with `pip install -r requirements.txt`.
- Retraining from scratch: see the **Training** section above. Reuse the
  HuggingFace dataset identifier — no manual download step.
- Evaluation: `python src/ocr/benchmark.py` discovers all checkpoints under
  `trained_models/` and reports CER/WER.

## Troubleshooting

- **No Nix?** Use a Python 3.11 venv:
  `python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- **CUDA mismatch:** the pinned `torch==2.11.0` wheel targets CUDA 13. For
  other CUDA versions, install a matching PyTorch build *before*
  `pip install -r requirements.txt`.
- **HuggingFace download fails:** set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`
  and check `~/.cache/huggingface/` permissions.

## License

This project is released under the [MIT License](./LICENSE).

## Citation

If you use this work, please cite the thesis:

```bibtex
@thesis{musiol2026samiocr,
  author = {Jan Musiol},
  title  = {Digitalization of Endangered Scripts: An OCR-Translation Pipeline
            for Low-Resource S\'{a}mi Languages},
  school = {University of Southern Denmark,
            The Maersk Mc-Kinney Moeller Institute},
  year   = {2026},
  type   = {Bachelor's thesis}
}
```

Related work / acknowledgements:

- Dataset: [Sprakbanken Northern Sami OCR](https://huggingface.co/Sprakbanken)
- Translation backend: [TartuNLP](https://api.tartunlp.ai)
