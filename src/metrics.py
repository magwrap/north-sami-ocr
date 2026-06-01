"""
Translation evaluation metrics: BLEU, chrF, and TER.

Corpus-level metrics for evaluating machine translation quality.
Uses sacrebleu when available, falls back to basic implementations otherwise.
"""

from typing import List, Union
import re
from collections import Counter


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenization."""
    return re.findall(r'\w+|[^\w\s]', text.lower())


# Try to use sacrebleu for accurate implementations
try:
    import sacrebleu as _sb
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    lowercase: bool = True,
) -> dict:
    """
    Compute corpus-level BLEU score.

    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        lowercase: Whether to lowercase before scoring

    Returns:
        Dict with 'bleu' score (0-100 scale)
    """
    if _HAS_SACREBLEU:
        # sacrebleu.corpus_bleu expects: hypotheses, [references]
        # references should be wrapped in a list (for multiple references per sample)
        score = _sb.corpus_bleu(hypotheses, [references], lowercase=lowercase)
        return {"bleu": score.score}

    # Fallback: average sentence-level BLEU
    if len(references) == 0:
        return {"bleu": 0.0}

    total_score = 0.0
    for hyp, ref in zip(hypotheses, references):
        total_score += _sentence_bleu(hyp, ref, lowercase)

    return {"bleu": total_score / len(references)}


def _sentence_bleu(hypothesis: str, reference: str, lowercase: bool = True) -> float:
    """Basic sentence-level BLEU for fallback."""
    import math

    hyp_tokens = _tokenize(hypothesis) if lowercase else hypothesis.split()
    ref_tokens = _tokenize(reference) if lowercase else reference.split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Compute n-gram precisions (1-4)
    precisions = []
    for n in range(1, 5):
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))

        if sum(hyp_ngrams.values()) == 0:
            precisions.append(0.0)
            continue

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)
        precisions.append(clipped / sum(hyp_ngrams.values()))

    if 0.0 in precisions:
        return 0.0

    log_precision = sum(math.log(p) for p in precisions) / 4

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

    return bp * math.exp(log_precision) * 100


def compute_chrf(
    references: List[str],
    hypotheses: List[str],
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2.0,
) -> dict:
    """
    Compute corpus-level chrF score (character n-gram F-score).

    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        char_order: Maximum character n-gram order
        word_order: Maximum word n-gram order (0 = chrF, 2 = chrF++)
        beta: F-score beta parameter

    Returns:
        Dict with 'chrf' score (0-100 scale)
    """
    if _HAS_SACREBLEU:
        score = _sb.corpus_chrf(hypotheses, [references], char_order=char_order,
                                 word_order=word_order, beta=beta)
        return {"chrf": score.score}

    # Fallback: average sentence-level chrF
    if len(references) == 0:
        return {"chrf": 0.0}

    total_score = 0.0
    for hyp, ref in zip(hypotheses, references):
        total_score += _sentence_chrf(hyp, ref, char_order, beta)

    return {"chrf": total_score / len(references)}


def _sentence_chrf(hypothesis: str, reference: str, char_order: int = 6, beta: float = 2.0) -> float:
    """Basic sentence-level chrF for fallback."""
    def char_ngrams(text: str, n: int) -> Counter:
        text = text.replace(' ', '')
        return Counter(text[i:i+n] for i in range(len(text) - n + 1))

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for n in range(1, char_order + 1):
        hyp_ngrams = char_ngrams(hypothesis, n)
        ref_ngrams = char_ngrams(reference, n)

        hyp_total = sum(hyp_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        if hyp_total == 0 or ref_total == 0:
            continue

        overlap = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)

        total_precision += overlap / hyp_total
        total_recall += overlap / ref_total
        count += 1

    if count == 0:
        return 0.0

    avg_precision = total_precision / count
    avg_recall = total_recall / count

    if avg_precision + avg_recall == 0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1 + beta_sq) * avg_precision * avg_recall / (beta_sq * avg_precision + avg_recall)

    return f_score * 100


def compute_ter(
    references: List[str],
    hypotheses: List[str],
    lowercase: bool = True,
) -> dict:
    """
    Compute corpus-level Translation Edit Rate (TER).

    TER measures the number of edits needed to transform hypothesis into reference,
    normalized by reference length. Lower is better.

    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        lowercase: Whether to lowercase before scoring

    Returns:
        Dict with 'ter' score (0-100 scale, lower is better)
    """
    if _HAS_SACREBLEU:
        score = _sb.corpus_ter(hypotheses, [references], normalized=True,
                                no_punct=False, asian_support=False, case_sensitive=not lowercase)
        return {"ter": score.score}

    # Fallback: average sentence-level TER
    if len(references) == 0:
        return {"ter": 0.0}

    total_score = 0.0
    for hyp, ref in zip(hypotheses, references):
        total_score += _sentence_ter(hyp, ref, lowercase)

    return {"ter": total_score / len(references)}


def _sentence_ter(hypothesis: str, reference: str, lowercase: bool = True) -> float:
    """Basic sentence-level TER using word-level Levenshtein distance."""
    hyp_tokens = _tokenize(hypothesis) if lowercase else hypothesis.lower().split()
    ref_tokens = _tokenize(reference) if lowercase else reference.lower().split()

    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 100.0

    # Levenshtein distance
    m, n = len(hyp_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[m][n]
    return (edit_distance / len(ref_tokens)) * 100


__all__ = ["compute_bleu", "compute_chrf", "compute_ter"]
