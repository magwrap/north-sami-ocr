"""Shared training utilities for OCR models.

This module provides common metric computation and decoding functions
used across different OCR training scripts.
"""

import torch
from typing import Dict, List
import string


def compute_cer(pred: str, target: str) -> float:
    """
    Compute Character Error Rate using edit distance.

    Args:
        pred: Predicted text string
        target: Ground truth text string

    Returns:
        Character Error Rate (0.0 = perfect, 1.0 = completely wrong)
    """
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0

    # Levenshtein distance using dynamic programming
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n] / len(target)


def compute_wer(pred: str, target: str) -> float:
    """
    Compute Word Error Rate using edit distance on words.

    Args:
        pred: Predicted text string
        target: Ground truth text string

    Returns:
        Word Error Rate (0.0 = perfect, 1.0 = completely wrong)
    """
    pred_words = pred.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0

    # Levenshtein distance on word sequences
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == target_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n] / len(target_words)


def ctc_decode(
    logits: torch.Tensor,
    idx_to_char: Dict[int, str],
    format: str = "btc",
    blank_idx: int = 0
) -> List[str]:
    """
    Greedy CTC decoding with flexible tensor format support.

    Args:
        logits: Predictions in one of two formats:
            - "btc": (batch, time, classes) - Standard format
            - "tbc": (time, batch, classes) - Legacy format
        idx_to_char: Index to character mapping dictionary
        format: Tensor format ("btc" or "tbc")
        blank_idx: Index of the CTC blank token (default: 0)

    Returns:
        List of decoded strings (one per batch element)
    """
    # Handle both tensor formats by converting to (B, T, C)
    if format == "tbc":
        logits = logits.permute(1, 0, 2)  # (T, B, C) -> (B, T, C)
    elif format != "btc":
        raise ValueError(f"Unknown format '{format}'. Must be 'btc' or 'tbc'")

    # Greedy decoding: take argmax over class dimension
    predictions = logits.argmax(dim=2)  # (B, T)
    batch_size = predictions.shape[0]
    results = []

    # Decode each sequence in the batch
    for b in range(batch_size):
        seq = predictions[b].cpu().tolist()
        chars = []
        prev = -1

        for idx in seq:
            # CTC blank removal + collapse repeated characters
            if idx != prev and idx != blank_idx:
                # Skip blank tokens and unknown indices
                if idx in idx_to_char and idx_to_char[idx] != "<blank>":
                    chars.append(idx_to_char[idx])
            prev = idx

        results.append("".join(chars))

    return results


def compute_accuracy(pred: str, target: str) -> float:
    # return 1.0 if pred == target else 0.0
    return compute_token_accuracy(pred, target)


def compute_token_accuracy(pred: str, target: str) -> float:
    pred_tokens = pred.split()
    target_tokens = target.split()
    correct = sum(p == t for p, t in zip(pred_tokens, target_tokens))
    return correct / max(len(pred_tokens), len(target_tokens))


def clean_text(text: str) -> str:
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text.lower().strip()


def compute_normalized_accuracy(pred: str, target: str) -> float:
    """
    Binary accuracy after removing punctuation/whitespace.

    Args:
        pred: Predicted text string
        target: Ground truth text string

    Returns:
        1.0 if normalized strings match, 0.0 otherwise
    """
    pred_clean = clean_text(pred)
    target_clean = clean_text(target)
    return 1.0 if pred_clean == target_clean else 0.0