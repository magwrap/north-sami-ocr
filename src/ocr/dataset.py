"""Dataset utilities for OCR training."""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SamiOCRDataset(Dataset):
    """
    Dataset wrapper for HuggingFace Sami OCR dataset.

    Expects dataset with 'image' and 'text' columns.
    """

    def __init__(
        self,
        hf_dataset,
        char_to_idx: dict,
        img_height: int = 32,
        img_width: int = 2048,
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            hf_dataset: HuggingFace dataset object
            char_to_idx: Character to index mapping (index 0 reserved for CTC blank)
            img_height: Target image height
            img_width: Target image width (will be padded/cropped)
            augment: Whether to apply data augmentation
        """
        self.dataset = hf_dataset
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.img_width = img_width

        # Base transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.02, 0.02),
                    scale=(0.98, 1.02),
                    fill=255,
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single item.

        Returns:
            Tuple of (image_tensor, target_tensor, target_length)
        """
        item = self.dataset[idx]
        image = item["image"]
        text = item["text"]

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Resize maintaining aspect ratio, then pad/crop to target width
        image = self._resize_and_pad(image)

        # Apply transforms
        image_tensor = self.transform(image)

        # Encode text to indices (skip unknown characters)
        target = []
        for char in text:
            if char in self.char_to_idx:
                target.append(self.char_to_idx[char])

        target_tensor = torch.tensor(target, dtype=torch.long)

        return image_tensor, target_tensor, len(target)

    def _resize_and_pad(self, image: Image.Image) -> Image.Image:
        """Resize image to target height maintaining aspect ratio, then pad/crop width."""
        # Calculate new width maintaining aspect ratio
        w, h = image.size
        new_h = self.img_height
        new_w = int(w * (new_h / h))

        # Resize
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Pad or crop to target width
        if new_w < self.img_width:
            # Pad with white on the right
            padded = Image.new("RGB", (self.img_width, self.img_height), (255, 255, 255))
            padded.paste(image, (0, 0))
            return padded
        elif new_w > self.img_width:
            # Crop from the right
            return image.crop((0, 0, self.img_width, self.img_height))
        else:
            return image


def build_charset(hf_dataset, min_freq: int = 1) -> str:
    """
    Build character set from dataset.

    Args:
        hf_dataset: HuggingFace dataset with 'text' column
        min_freq: Minimum character frequency to include

    Returns:
        String of unique characters sorted alphabetically
    """
    from collections import Counter

    char_counts = Counter()
    for item in hf_dataset:
        char_counts.update(item["text"])

    # Filter by frequency and sort
    chars = [char for char, count in char_counts.items() if count >= min_freq]
    chars = sorted(chars)

    return "".join(chars)


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader handling variable-length sequences.

    Args:
        batch: List of (image, target, target_length) tuples

    Returns:
        Tuple of:
            - images: (B, C, H, W)
            - targets: Concatenated targets for CTC loss
            - target_lengths: (B,) tensor of target lengths
            - input_lengths: (B,) tensor of input sequence lengths
    """
    images, targets, target_lengths = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Concatenate targets (CTC loss expects flat tensor)
    targets = torch.cat(targets, dim=0)

    # Convert lengths to tensors
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Input lengths will be computed in training based on model output
    # For now, use a placeholder (will be set based on actual output width)
    input_lengths = torch.full((len(batch),), -1, dtype=torch.long)

    return images, targets, target_lengths, input_lengths


def get_input_length(model_output: torch.Tensor) -> torch.Tensor:
    """
    Get input lengths for CTC loss based on model output.

    Args:
        model_output: Model output tensor of shape (B, T, C)

    Returns:
        Tensor of input lengths (B,)
    """
    batch_size, seq_len, _ = model_output.shape
    return torch.full((batch_size,), seq_len, dtype=torch.long)
