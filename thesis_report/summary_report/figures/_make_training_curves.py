"""Parse train.log files and plot val CER + train loss across all CTC variants."""
import re
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
LOGS = {
    "ctc_simple (7-layer SimpleCNN, from scratch)": REPO / "trained_models/2026-03-27_queue_1/ctc_simple/train.log",
    "ctc_vgg16 (ImageNet-pretrained)":              REPO / "trained_models/2026-03-28_queue/ctc_vgg16/train.log",
    "ctc_vgg19 (ImageNet-pretrained)":              REPO / "trained_models/2026-03-28_queue/ctc_vgg19/train.log",
    "ctc_resnet50 (ImageNet-pretrained)":           REPO / "trained_models/2026-03-28_queue/ctc_resnet50/train.log",
}
COLORS = {
    "ctc_simple (7-layer SimpleCNN, from scratch)": "#1f77b4",
    "ctc_vgg16 (ImageNet-pretrained)":              "#2ca02c",
    "ctc_vgg19 (ImageNet-pretrained)":              "#d62728",
    "ctc_resnet50 (ImageNet-pretrained)":           "#9467bd",
}
OUT = Path(__file__).parent / "training_curves.pdf"

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+)\s*\|\s*CER:\s*([\d.]+)"
)


def parse(path: Path):
    epochs, train_loss, val_cer = [], [], []
    for line in path.read_text().splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_cer.append(float(m.group(4)))
    return epochs, train_loss, val_cer


fig, (ax_loss, ax_cer) = plt.subplots(1, 2, figsize=(11, 4))
for label, path in LOGS.items():
    ep, tl, vc = parse(path)
    ax_loss.plot(ep, tl, label=label, color=COLORS[label], linewidth=1.8)
    ax_cer.plot(ep, vc, label=label, color=COLORS[label], linewidth=1.8)

for ax, title, ylabel in (
    (ax_loss, "Training loss vs. epoch", "Train CTC loss"),
    (ax_cer, "Validation CER vs. epoch", "Validation CER"),
):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(left=1)

ax_cer.axhline(0.10, linestyle=":", color="grey", linewidth=1)
ax_cer.text(2, 0.12, "10% CER (rough usability threshold)", fontsize=8, color="grey")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
