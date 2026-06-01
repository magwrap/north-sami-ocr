"""Base training class for OCR models.

This module provides a unified training framework that eliminates code duplication
across different OCR training scripts.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional

from train_utils import compute_cer, compute_wer, ctc_decode


class OCRTrainer:
    """Unified training framework for OCR models.

    Supports both standard (B,T,C) and legacy (T,B,C) tensor formats,
    with configurable training and validation logic.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (typically nn.CTCLoss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on (cpu or cuda)
        idx_to_char: Index to character mapping for decoding
        logits_format: Model output format ("btc" or "tbc")
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        idx_to_char: Dict[int, str],
        logits_format: str = "btc",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.idx_to_char = idx_to_char
        self.logits_format = logits_format

        # Pre-filter idx_to_char for target decoding
        self.idx_to_char_clean = {k: v for k, v in idx_to_char.items() if v != "<blank>"}

    def train_epoch(self, grad_clip: float = 5.0) -> float:
        """
        Train for one epoch.

        Args:
            grad_clip: Gradient clipping value

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Unpack batch (handle both 3 and 4 element tuples)
            if len(batch) == 4:
                images, targets, target_lengths, _ = batch
            else:
                images, targets, target_lengths = batch

            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images)

            # Get input lengths from model output
            if self.logits_format == "btc":
                batch_size, seq_len, _ = logits.shape
                input_lengths = torch.full(
                    (batch_size,), seq_len, dtype=torch.long, device=self.device
                )
                # CTC loss expects (T, B, C) format
                logits_ctc = logits.permute(1, 0, 2)
            else:  # tbc format
                seq_len, batch_size, _ = logits.shape
                input_lengths = torch.full(
                    (batch_size,), seq_len, dtype=torch.long, device=self.device
                )
                logits_ctc = logits

            # Compute loss
            loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN/Inf loss detected, skipping batch")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self) -> Tuple[float, float, float, float]:
        """
        Validate model and compute metrics.

        Returns:
            Tuple of (loss, cer, wer, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_cer = 0.0
        total_wer = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Unpack batch (handle both 3 and 4 element tuples)
                if len(batch) == 4:
                    images, targets, target_lengths, _ = batch
                else:
                    images, targets, target_lengths = batch

                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward pass
                logits = self.model(images)

                # Get input lengths
                if self.logits_format == "btc":
                    batch_size, seq_len, _ = logits.shape
                    input_lengths = torch.full(
                        (batch_size,), seq_len, dtype=torch.long, device=self.device
                    )
                    logits_ctc = logits.permute(1, 0, 2)
                else:  # tbc format
                    seq_len, batch_size, _ = logits.shape
                    input_lengths = torch.full(
                        (batch_size,), seq_len, dtype=torch.long, device=self.device
                    )
                    logits_ctc = logits

                # Compute loss
                loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                # Decode predictions
                pred_strings = ctc_decode(logits, self.idx_to_char, format=self.logits_format)

                # Decode targets
                target_list = targets.cpu().tolist()
                target_lengths_list = target_lengths.cpu().tolist()
                offset = 0
                for i, length in enumerate(target_lengths_list):
                    target_indices = target_list[offset : offset + length]
                    target_str = "".join(
                        self.idx_to_char_clean.get(idx, "") for idx in target_indices
                    )
                    offset += length

                    pred_str = pred_strings[i]
                    total_cer += compute_cer(pred_str, target_str)
                    total_wer += compute_wer(pred_str, target_str)
                    if pred_str == target_str:
                        correct += 1
                    total += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_cer = total_cer / max(total, 1)
        avg_wer = total_wer / max(total, 1)
        accuracy = correct / max(total, 1)

        return avg_loss, avg_cer, avg_wer, accuracy

    def step_scheduler(self, metric: Optional[float] = None):
        """
        Step the learning rate scheduler.

        Args:
            metric: Validation metric for schedulers like ReduceLROnPlateau
        """
        if self.scheduler is None:
            return

        # Check if scheduler needs a metric (e.g., ReduceLROnPlateau)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires a metric")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
